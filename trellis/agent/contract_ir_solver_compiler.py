"""Structural ContractIR solver compiler for Phase 3 shadow-mode dispatch.

This module binds the Phase 2 ``ContractIR`` surface onto a bounded set of
checked pricing helpers and raw kernels. The compiler is intentionally narrow:
it only covers the first migrated family wave and fails closed for everything
else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from importlib import import_module
from math import isclose
from types import MappingProxyType
from typing import Callable, Mapping

from trellis.agent.contract_ir import (
    Annuity,
    Constant,
    ContractIR,
    ContinuousInterval,
    FiniteSchedule,
    Gt,
    Indicator,
    LinearBasket,
    Lt,
    Max,
    Mul,
    Scaled,
    Singleton,
    Spot,
    Strike,
    Sub,
    SwapRate,
    VarianceObservable,
)
from trellis.agent.contract_pattern import (
    ConstantPattern,
    ContractPattern,
    ExercisePattern,
    PayoffPattern,
    SpotPattern,
    StrikePattern,
    UnderlyingPattern,
    Wildcard,
)
from trellis.agent.contract_pattern_eval import evaluate_pattern
from trellis.agent.contract_ir_solver_registry import (
    ContractIRSolverDeclaration,
    ContractIRSolverMaterialization,
    ContractIRSolverMarketRequirements,
    ContractIRSolverProvenance,
    ContractIRSolverRegistry,
    ContractIRSolverSelectionAuthority,
    build_contract_ir_solver_registry,
)
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.sensitivity_support import (
    normalize_requested_measures,
    normalize_requested_outputs,
)
from trellis.agent.valuation_context import ValuationContext
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _string_tuple(values) -> tuple[str, ...]:
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _float_tuple(values) -> tuple[float, ...]:
    if not values:
        return ()
    result: list[float] = []
    for value in values:
        result.append(float(value))
    return tuple(result)


def _import_ref(ref: str) -> object:
    module_name, _, symbol = str(ref or "").rpartition(".")
    if not module_name or not symbol:
        raise ValueError(f"Invalid import ref {ref!r}")
    module = import_module(module_name)
    return getattr(module, symbol)


def _normalized_frequency(value: object | None, *, default: Frequency | None) -> Frequency | None:
    if value in {None, ""}:
        return default
    if isinstance(value, Frequency):
        return value
    text = str(value).strip()
    if text.startswith("Frequency."):
        text = text.split(".", 1)[1]
    token = text.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "annual": "ANNUAL",
        "yearly": "ANNUAL",
        "semiannual": "SEMI_ANNUAL",
        "semi_annual": "SEMI_ANNUAL",
        "quarterly": "QUARTERLY",
        "monthly": "MONTHLY",
    }
    member = aliases.get(token)
    if member is None:
        return default
    return getattr(Frequency, member)


def _normalized_day_count(
    value: object | None,
    *,
    default: DayCountConvention,
) -> DayCountConvention:
    if value in {None, ""}:
        return default
    if isinstance(value, DayCountConvention):
        return value
    text = str(value).strip()
    if text.startswith("DayCountConvention."):
        text = text.split(".", 1)[1]
    token = text.strip().upper().replace("-", "_").replace("/", "_")
    for member_name in (
        token,
        token.replace("ACTUAL", "ACT"),
        token.replace("30E_360", "THIRTY_E_360"),
        token.replace("30_360", "THIRTY_360"),
    ):
        if hasattr(DayCountConvention, member_name):
            return getattr(DayCountConvention, member_name)
    return default


def _parse_float_grid(value: object | None) -> tuple[float, ...]:
    if value in {None, ""}:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    text = str(value).strip()
    if not text:
        return ()
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _market_identity(valuation_context: ValuationContext | None, market_state: MarketState) -> str:
    if valuation_context is not None and valuation_context.market_snapshot_handle:
        return valuation_context.market_snapshot_handle
    provenance = dict(getattr(market_state, "market_provenance", None) or {})
    for key in ("market_identity", "snapshot_id", "market_snapshot_id", "scenario_id"):
        value = provenance.get(key)
        if value:
            return str(value).strip()
    return "market_state"


def _market_overlay_identity(market_state: MarketState) -> str:
    provenance = dict(getattr(market_state, "market_provenance", None) or {})
    for key in ("overlay_identity", "overlay_id", "scenario_overlay", "market_scenario_id"):
        value = provenance.get(key)
        if value:
            return str(value).strip()
    return ""


def _resolve_spot(market_state: MarketState, underlier_id: str) -> float:
    spots = dict(getattr(market_state, "underlier_spots", None) or {})
    if underlier_id in spots:
        return float(spots[underlier_id])
    if market_state.spot is not None:
        return float(market_state.spot)
    raise ValueError(f"MarketState does not expose spot for underlier {underlier_id!r}")


def _equity_option_expiry_years(
    market_state: MarketState,
    expiry_date: date,
    *,
    day_count: DayCountConvention = DayCountConvention.ACT_365,
) -> float:
    settlement = getattr(market_state, "settlement", None)
    if settlement is None:
        raise ValueError("MarketState.settlement is required for structural compiler timing")
    return max(float(year_fraction(settlement, expiry_date, day_count)), 0.0)


def _discount_factor(market_state: MarketState, maturity: float) -> float:
    if market_state.discount is None:
        raise ValueError("Structural compiler requires market_state.discount")
    return float(market_state.discount.discount(max(float(maturity), 0.0)))


def _vol_at(market_state: MarketState, maturity: float, strike: float) -> float:
    if market_state.vol_surface is None:
        raise ValueError("Structural compiler requires market_state.vol_surface")
    return float(market_state.vol_surface.black_vol(max(float(maturity), 1e-6), float(strike)))


def _rate_curve_frequency(schedule: FiniteSchedule | None) -> Frequency | None:
    if schedule is None or len(schedule.dates) < 2:
        return None
    months = {
        (right.year - left.year) * 12 + (right.month - left.month)
        for left, right in zip(schedule.dates, schedule.dates[1:])
    }
    if len(months) != 1:
        return None
    month_step = next(iter(months))
    if month_step == 1:
        return Frequency.MONTHLY
    if month_step == 3:
        return Frequency.QUARTERLY
    if month_step == 6:
        return Frequency.SEMI_ANNUAL
    if month_step == 12:
        return Frequency.ANNUAL
    return None


class ContractIRSolverCompileError(ValueError):
    """Base error for structural ContractIR compilation failures."""


class ContractIRSolverNoMatchError(ContractIRSolverCompileError):
    """Raised when no structural declaration is admissible."""


class ContractIRSolverAmbiguityError(ContractIRSolverCompileError):
    """Raised when multiple admissible declarations survive selection."""


@dataclass(frozen=True)
class CashSettlementTerms:
    """Generic cash-scaling and payout-convention terms."""

    notional: float = 1.0
    payout_currency: str = ""
    settlement_kind: str = "cash"


@dataclass(frozen=True)
class AccrualConventionTerms:
    """Generic accrual, payment-frequency, and day-count terms."""

    day_count: DayCountConvention = DayCountConvention.ACT_365
    fixed_leg_day_count: DayCountConvention = DayCountConvention.ACT_360
    float_leg_day_count: DayCountConvention = DayCountConvention.ACT_360
    payment_frequency: Frequency | None = None


@dataclass(frozen=True)
class FloatingRateReferenceTerms:
    """Generic floating-rate index and multi-curve reference terms."""

    rate_index: str = ""
    discount_curve_name: str = ""
    forecast_curve_name: str = ""


@dataclass(frozen=True)
class QuoteGridTerms:
    """Generic explicit quote-grid terms used by bounded observable helpers."""

    replication_strikes: tuple[float, ...] = ()
    replication_volatilities: tuple[float, ...] = ()


@dataclass(frozen=True)
class ContractIRTermEnvironment:
    """Normalized reusable term groups kept separate from ContractIR structure."""

    cash_settlement: CashSettlementTerms = field(default_factory=CashSettlementTerms)
    accrual_conventions: AccrualConventionTerms = field(default_factory=AccrualConventionTerms)
    floating_rate_reference: FloatingRateReferenceTerms = field(default_factory=FloatingRateReferenceTerms)
    quote_grid: QuoteGridTerms = field(default_factory=QuoteGridTerms)
    raw_term_fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_term_fields", _freeze_mapping(self.raw_term_fields))

    def present_group_names(self) -> tuple[str, ...]:
        return (
            "cash_settlement",
            "accrual_conventions",
            "floating_rate_reference",
            "quote_grid",
        )


def build_contract_ir_term_environment(contract=None) -> ContractIRTermEnvironment:
    """Build the normalized non-structural term environment from a semantic contract."""

    product = getattr(contract, "product", None)
    conventions = getattr(product, "conventions", None)
    raw_term_fields = dict(getattr(product, "term_fields", {}) or {})
    notional = float(raw_term_fields.get("notional") or 1.0)
    payout_currency = str(
        getattr(conventions, "payment_currency", None)
        or getattr(conventions, "reporting_currency", None)
        or raw_term_fields.get("payment_currency")
        or ""
    ).strip()
    settlement_kind = str(raw_term_fields.get("settlement_kind") or "cash").strip().lower() or "cash"
    day_count_token = (
        raw_term_fields.get("day_count")
        or raw_term_fields.get("fixed_leg_day_count")
        or getattr(conventions, "day_count_convention", None)
    )
    payment_frequency = _normalized_frequency(
        raw_term_fields.get("payment_frequency") or raw_term_fields.get("swap_frequency"),
        default=None,
    )
    fixed_leg_day_count = _normalized_day_count(
        raw_term_fields.get("fixed_leg_day_count") or day_count_token,
        default=DayCountConvention.ACT_360,
    )
    float_leg_day_count = _normalized_day_count(
        raw_term_fields.get("float_leg_day_count") or day_count_token,
        default=DayCountConvention.ACT_360,
    )
    generic_day_count = _normalized_day_count(day_count_token, default=DayCountConvention.ACT_365)
    return ContractIRTermEnvironment(
        cash_settlement=CashSettlementTerms(
            notional=notional,
            payout_currency=payout_currency,
            settlement_kind=settlement_kind,
        ),
        accrual_conventions=AccrualConventionTerms(
            day_count=generic_day_count,
            fixed_leg_day_count=fixed_leg_day_count,
            float_leg_day_count=float_leg_day_count,
            payment_frequency=payment_frequency,
        ),
        floating_rate_reference=FloatingRateReferenceTerms(
            rate_index=str(raw_term_fields.get("rate_index") or "").strip(),
            discount_curve_name=str(raw_term_fields.get("discount_curve_name") or "").strip(),
            forecast_curve_name=str(raw_term_fields.get("forecast_curve_name") or "").strip(),
        ),
        quote_grid=QuoteGridTerms(
            replication_strikes=_parse_float_grid(raw_term_fields.get("replication_strikes")),
            replication_volatilities=_parse_float_grid(raw_term_fields.get("replication_volatilities")),
        ),
        raw_term_fields=raw_term_fields,
    )


@dataclass(frozen=True)
class ContractIRCompilerDecision:
    """Deterministic structural solver decision emitted in Phase 3 shadow mode."""

    declaration_id: str
    requested_method: str
    requested_outputs: tuple[str, ...]
    callable_ref: str
    call_style: str
    call_kwargs: Mapping[str, object] = field(default_factory=dict)
    value_scale: float = 1.0
    match_bindings: Mapping[str, object] = field(default_factory=dict)
    consumed_term_groups: tuple[str, ...] = ()
    market_identity: str = ""
    market_overlay_identity: str = ""
    resolved_market_coordinates: tuple[str, ...] = ()
    validation_bundle_id: str = ""
    helper_refs: tuple[str, ...] = ()
    pricing_kernel_refs: tuple[str, ...] = ()
    callable: Callable[..., float] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_kwargs", _freeze_mapping(self.call_kwargs))
        object.__setattr__(self, "match_bindings", _freeze_mapping(self.match_bindings))
        object.__setattr__(self, "requested_outputs", _string_tuple(self.requested_outputs))
        object.__setattr__(self, "consumed_term_groups", _string_tuple(self.consumed_term_groups))
        object.__setattr__(self, "resolved_market_coordinates", _string_tuple(self.resolved_market_coordinates))
        object.__setattr__(self, "helper_refs", _string_tuple(self.helper_refs))
        object.__setattr__(self, "pricing_kernel_refs", _string_tuple(self.pricing_kernel_refs))


@dataclass(frozen=True)
class ContractIRSolverShadowRecord:
    """Shadow-mode comparison record attached to semantic blueprints."""

    declaration_id: str
    callable_ref: str
    requested_method: str
    market_identity: str = ""
    market_overlay_identity: str = ""
    resolved_market_coordinates: tuple[str, ...] = ()
    legacy_route_id: str = ""
    legacy_route_family: str = ""
    legacy_route_modules: tuple[str, ...] = ()


def execute_contract_ir_solver_decision(decision: ContractIRCompilerDecision) -> float:
    """Execute one bound structural solver decision."""

    callable_obj = decision.callable or _import_ref(decision.callable_ref)
    result = float(callable_obj(**dict(decision.call_kwargs)))
    return float(decision.value_scale) * result


def _normalize_requested_method(
    preferred_method: str | None,
    valuation_context: ValuationContext | None,
) -> str:
    if preferred_method:
        return normalize_method(preferred_method)
    if valuation_context is not None and valuation_context.model_spec:
        return normalize_method(valuation_context.model_spec)
    return "analytical"


def _normalize_requested_output_tuple(
    requested_outputs,
    valuation_context: ValuationContext | None,
) -> tuple[str, ...]:
    explicit = normalize_requested_outputs(requested_outputs)
    if explicit:
        return explicit
    if valuation_context is not None:
        implicit = normalize_requested_outputs(valuation_context.requested_outputs)
        if implicit:
            return implicit
    return ("price",)


def compile_contract_ir_solver(
    contract_ir: ContractIR,
    *,
    term_environment: ContractIRTermEnvironment | None = None,
    valuation_context: ValuationContext | None = None,
    market_state: MarketState | None = None,
    preferred_method: str | None = None,
    requested_outputs: tuple[str, ...] | list[str] | None = None,
    registry: ContractIRSolverRegistry | None = None,
) -> ContractIRCompilerDecision:
    """Compile a ``ContractIR`` into one bound structural solver call."""

    if market_state is None:
        raise ContractIRSolverCompileError(
            "compile_contract_ir_solver requires a bound MarketState in Phase 3"
        )

    selected_registry = registry or default_contract_ir_solver_registry()
    normalized_terms = term_environment or ContractIRTermEnvironment()
    method = _normalize_requested_method(preferred_method, valuation_context)
    outputs = _normalize_requested_output_tuple(requested_outputs, valuation_context)
    measures = normalize_requested_measures(outputs)
    candidates: list[tuple[int, ContractIRSolverDeclaration, Mapping[str, object], Mapping[str, object]]] = []

    for registered in selected_registry.selection_order():
        declaration = registered.declaration
        match = evaluate_pattern(declaration.authority.contract_pattern, contract_ir)
        if not match.ok:
            continue
        if declaration.authority.admissible_methods and method not in declaration.authority.admissible_methods:
            continue
        supported_outputs = set(declaration.outputs.supported_outputs or ("price",))
        if any(output not in supported_outputs for output in outputs if output != "price" and output not in measures):
            continue
        supported_measures = set(declaration.outputs.supported_measures)
        if any(str(measure) not in supported_measures for measure in measures):
            continue
        required_capabilities = set(declaration.market_requirements.required_capabilities)
        if required_capabilities - set(market_state.available_capabilities):
            continue
        missing_groups = set(declaration.authority.required_term_groups) - set(normalized_terms.present_group_names())
        if missing_groups:
            continue
        adapter = _import_ref(declaration.materialization.adapter_ref) if declaration.materialization.adapter_ref else None
        try:
            adapter_payload = (
                adapter(
                    contract_ir=contract_ir,
                    term_environment=normalized_terms,
                    valuation_context=valuation_context,
                    market_state=market_state,
                    bindings=dict(match.bindings),
                )
                if adapter is not None
                else {"call_kwargs": {}}
            )
        except (TypeError, ValueError):
            continue
        candidates.append((declaration.precedence, declaration, dict(match.bindings), dict(adapter_payload)))

    if not candidates:
        raise ContractIRSolverNoMatchError(
            "No admissible structural ContractIR solver declaration was found for "
            f"method {method!r} and outputs {outputs!r}."
        )

    top_precedence = max(item[0] for item in candidates)
    top = [item for item in candidates if item[0] == top_precedence]
    if len(top) > 1:
        raise ContractIRSolverAmbiguityError(
            "Multiple structural ContractIR solver declarations remained admissible "
            f"at precedence {top_precedence}: {[item[1].provenance.declaration_id for item in top]}"
        )

    _precedence, declaration, bindings, adapter_payload = top[0]
    callable_ref = declaration.materialization.callable_ref
    callable_obj = _import_ref(callable_ref)
    return ContractIRCompilerDecision(
        declaration_id=declaration.provenance.declaration_id,
        requested_method=method,
        requested_outputs=outputs,
        callable_ref=callable_ref,
        call_style=declaration.materialization.call_style,
        call_kwargs=dict(adapter_payload.get("call_kwargs") or {}),
        value_scale=float(adapter_payload.get("value_scale", 1.0)),
        match_bindings=dict(bindings),
        consumed_term_groups=declaration.authority.required_term_groups,
        market_identity=_market_identity(valuation_context, market_state),
        market_overlay_identity=_market_overlay_identity(market_state),
        resolved_market_coordinates=tuple(adapter_payload.get("resolved_market_coordinates") or ()),
        validation_bundle_id=declaration.provenance.validation_bundle_id,
        helper_refs=declaration.provenance.helper_refs,
        pricing_kernel_refs=declaration.provenance.pricing_kernel_refs,
        callable=callable_obj,
    )


def shadow_record_from_decision(
    decision: ContractIRCompilerDecision,
    *,
    legacy_route_id: str = "",
    legacy_route_family: str = "",
    legacy_route_modules: tuple[str, ...] = (),
) -> ContractIRSolverShadowRecord:
    """Project one structural decision onto the compact shadow-mode summary surface."""

    return ContractIRSolverShadowRecord(
        declaration_id=decision.declaration_id,
        callable_ref=decision.callable_ref,
        requested_method=decision.requested_method,
        market_identity=decision.market_identity,
        market_overlay_identity=decision.market_overlay_identity,
        resolved_market_coordinates=decision.resolved_market_coordinates,
        legacy_route_id=str(legacy_route_id or ""),
        legacy_route_family=str(legacy_route_family or ""),
        legacy_route_modules=_string_tuple(legacy_route_modules),
    )


def _zero_carry_black76_adapter(
    *,
    contract_ir: ContractIR,
    term_environment: ContractIRTermEnvironment,
    valuation_context: ValuationContext | None,
    market_state: MarketState,
    bindings: dict[str, object],
    call: bool,
) -> dict[str, object]:
    expiry = contract_ir.exercise.schedule
    if not isinstance(expiry, Singleton):
        raise ValueError("Black76 vanilla adapter requires European singleton exercise")
    strike = float(bindings["k"])
    underlier = str(bindings["u"])
    maturity = _equity_option_expiry_years(
        market_state,
        expiry.t,
        day_count=term_environment.accrual_conventions.day_count,
    )
    if maturity <= 0.0:
        return {
            "call_kwargs": {
                "F": _resolve_spot(market_state, underlier),
                "K": strike,
                "sigma": 0.0,
                "T": 0.0,
            },
            "value_scale": float(term_environment.cash_settlement.notional),
            "resolved_market_coordinates": ("spot", "discount_curve", "black_vol_surface"),
        }
    df = _discount_factor(market_state, maturity)
    spot = _resolve_spot(market_state, underlier)
    sigma = _vol_at(market_state, maturity, strike)
    forward = spot / max(df, 1e-12)
    return {
        "call_kwargs": {
            "F": forward,
            "K": strike,
            "sigma": sigma,
            "T": maturity,
        },
        "value_scale": float(term_environment.cash_settlement.notional) * df,
        "resolved_market_coordinates": ("spot", "discount_curve", "black_vol_surface"),
    }


def _extract_indicator_orientation(predicate: object, underlier: str, strike: float) -> str:
    if isinstance(predicate, Gt):
        if isinstance(predicate.lhs, Spot) and isinstance(predicate.rhs, Strike):
            if predicate.lhs.underlier_id == underlier and isclose(float(predicate.rhs.value), strike):
                return "call"
    if isinstance(predicate, Lt):
        if isinstance(predicate.lhs, Spot) and isinstance(predicate.rhs, Strike):
            if predicate.lhs.underlier_id == underlier and isclose(float(predicate.rhs.value), strike):
                return "put"
    raise ValueError("Indicator predicate is not a supported digital strike comparison")


def _digital_cash_adapter(
    *,
    contract_ir: ContractIR,
    term_environment: ContractIRTermEnvironment,
    valuation_context: ValuationContext | None,
    market_state: MarketState,
    bindings: dict[str, object],
    call: bool,
) -> dict[str, object]:
    expiry = contract_ir.exercise.schedule
    if not isinstance(expiry, Singleton):
        raise ValueError("Digital adapter requires European singleton exercise")
    payout = float(bindings["cash_payoff"])
    predicate = bindings["predicate"]
    underlier = ""
    strike = 0.0
    if isinstance(predicate, (Gt, Lt)) and isinstance(predicate.lhs, Spot) and isinstance(predicate.rhs, Strike):
        underlier = predicate.lhs.underlier_id
        strike = float(predicate.rhs.value)
    orientation = _extract_indicator_orientation(predicate, underlier, strike)
    if (orientation == "call") is not call:
        raise ValueError("Digital predicate orientation does not match declaration")
    maturity = _equity_option_expiry_years(
        market_state,
        expiry.t,
        day_count=term_environment.accrual_conventions.day_count,
    )
    df = _discount_factor(market_state, maturity)
    spot = _resolve_spot(market_state, underlier)
    sigma = _vol_at(market_state, maturity, strike)
    forward = spot / max(df, 1e-12)
    return {
        "call_kwargs": {
            "F": forward,
            "K": strike,
            "sigma": sigma,
            "T": maturity,
        },
        "value_scale": float(term_environment.cash_settlement.notional) * payout * df,
        "resolved_market_coordinates": ("spot", "discount_curve", "black_vol_surface"),
    }


def _digital_asset_adapter(
    *,
    contract_ir: ContractIR,
    term_environment: ContractIRTermEnvironment,
    valuation_context: ValuationContext | None,
    market_state: MarketState,
    bindings: dict[str, object],
    call: bool,
) -> dict[str, object]:
    expiry = contract_ir.exercise.schedule
    if not isinstance(expiry, Singleton):
        raise ValueError("Digital adapter requires European singleton exercise")
    underlier = str(bindings["u"])
    predicate = bindings["predicate"]
    strike = 0.0
    if isinstance(predicate, (Gt, Lt)) and isinstance(predicate.lhs, Spot) and isinstance(predicate.rhs, Strike):
        strike = float(predicate.rhs.value)
    orientation = _extract_indicator_orientation(predicate, underlier, strike)
    if (orientation == "call") is not call:
        raise ValueError("Digital predicate orientation does not match declaration")
    maturity = _equity_option_expiry_years(
        market_state,
        expiry.t,
        day_count=term_environment.accrual_conventions.day_count,
    )
    df = _discount_factor(market_state, maturity)
    spot = _resolve_spot(market_state, underlier)
    sigma = _vol_at(market_state, maturity, strike)
    forward = spot / max(df, 1e-12)
    return {
        "call_kwargs": {
            "F": forward,
            "K": strike,
            "sigma": sigma,
            "T": maturity,
        },
        "value_scale": float(term_environment.cash_settlement.notional) * df,
        "resolved_market_coordinates": ("spot", "discount_curve", "black_vol_surface"),
    }


@dataclass(frozen=True)
class _StructuralSwaptionSpec:
    notional: float
    strike: float
    expiry_date: date
    swap_start: date
    swap_end: date
    swap_frequency: Frequency
    day_count: DayCountConvention
    rate_index: str | None
    is_payer: bool


def _swaption_helper_adapter(
    *,
    contract_ir: ContractIR,
    term_environment: ContractIRTermEnvironment,
    valuation_context: ValuationContext | None,
    market_state: MarketState,
    bindings: dict[str, object],
    is_payer: bool,
) -> dict[str, object]:
    expiry = contract_ir.exercise.schedule
    schedule = bindings.get("schedule")
    if not isinstance(expiry, Singleton) or not isinstance(schedule, FiniteSchedule):
        raise ValueError("Swaption adapter requires European exercise and a finite annuity schedule")
    strike = float(bindings["k"])
    inferred_frequency = _rate_curve_frequency(schedule)
    # The current ContractIR decomposition uses coarse yearly coupon dates for
    # swaptions. Preserve the checked helper contract by defaulting to the
    # market-standard semi-annual fixed leg unless a more specific convention is
    # explicitly carried in the generic term environment or the structural
    # schedule is already finer than annual.
    swap_frequency = term_environment.accrual_conventions.payment_frequency
    if swap_frequency is None and inferred_frequency not in {None, Frequency.ANNUAL}:
        swap_frequency = inferred_frequency
    if swap_frequency is None:
        swap_frequency = Frequency.SEMI_ANNUAL
    day_count = term_environment.accrual_conventions.fixed_leg_day_count
    rate_index = term_environment.floating_rate_reference.rate_index or None
    spec = _StructuralSwaptionSpec(
        notional=float(term_environment.cash_settlement.notional),
        strike=strike,
        expiry_date=expiry.t,
        swap_start=expiry.t,
        swap_end=schedule.dates[-1],
        swap_frequency=swap_frequency,
        day_count=day_count,
        rate_index=rate_index,
        is_payer=is_payer,
    )
    coordinates = ["discount_curve", "black_vol_surface", "forward_curve"]
    if rate_index:
        coordinates.append(f"forecast_curve:{rate_index}")
    return {
        "call_kwargs": {
            "market_state": market_state,
            "spec": spec,
        },
        "value_scale": 1.0,
        "resolved_market_coordinates": tuple(coordinates),
    }


@dataclass(frozen=True)
class _StructuralBasketSpec:
    notional: float
    underliers: str
    strike: float
    expiry_date: date
    correlation: str
    weights: str | None
    spots: str | None
    vols: str | None
    dividend_yields: str | None
    basket_style: str
    option_type: str
    day_count: DayCountConvention


def _basket_helper_adapter(
    *,
    contract_ir: ContractIR,
    term_environment: ContractIRTermEnvironment,
    valuation_context: ValuationContext | None,
    market_state: MarketState,
    bindings: dict[str, object],
    option_type: str,
) -> dict[str, object]:
    expiry = contract_ir.exercise.schedule
    if not isinstance(expiry, Singleton):
        raise ValueError("Basket adapter requires European singleton exercise")
    weights = (float(bindings["w1"]), float(bindings["w2"]))
    underliers = (str(bindings["u1"]), str(bindings["u2"]))
    if len(set(underliers)) != 2:
        raise ValueError("Basket adapter requires two distinct underliers")
    basket_style = "spread" if weights[0] * weights[1] < 0.0 else "weighted_sum"
    spec = _StructuralBasketSpec(
        notional=float(term_environment.cash_settlement.notional),
        underliers=",".join(underliers),
        strike=float(bindings["k"]),
        expiry_date=expiry.t,
        correlation="",
        weights=",".join(str(weight) for weight in weights),
        spots=None,
        vols=None,
        dividend_yields=None,
        basket_style=basket_style,
        option_type=option_type,
        day_count=term_environment.accrual_conventions.day_count,
    )
    return {
        "call_kwargs": {
            "market_state": market_state,
            "spec": spec,
        },
        "value_scale": 1.0,
        "resolved_market_coordinates": (
            "spot",
            "discount_curve",
            "black_vol_surface",
            "model_parameters.correlation_matrix",
        ),
    }


@dataclass(frozen=True)
class _StructuralVarianceSwapSpec:
    notional: float
    spot: float
    strike_variance: float
    expiry_date: date
    realized_variance: float = 0.0
    replication_strikes: str | None = None
    replication_volatilities: str | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


def _variance_helper_adapter(
    *,
    contract_ir: ContractIR,
    term_environment: ContractIRTermEnvironment,
    valuation_context: ValuationContext | None,
    market_state: MarketState,
    bindings: dict[str, object],
) -> dict[str, object]:
    interval = bindings.get("interval")
    if not isinstance(interval, ContinuousInterval):
        raise ValueError("Variance adapter requires a continuous observation interval")
    underlier = str(bindings["u"])
    replication_strikes = term_environment.quote_grid.replication_strikes
    replication_volatilities = term_environment.quote_grid.replication_volatilities
    if replication_volatilities and not replication_strikes:
        raise ValueError("Variance adapter requires replication_strikes with replication_volatilities")
    if replication_strikes and replication_volatilities and len(replication_strikes) != len(replication_volatilities):
        raise ValueError("Variance quote grid lengths must match")
    spec = _StructuralVarianceSwapSpec(
        notional=float(bindings["notional"]),
        spot=_resolve_spot(market_state, underlier),
        strike_variance=float(bindings["k"]),
        expiry_date=interval.t_end,
        replication_strikes=",".join(str(value) for value in replication_strikes) or None,
        replication_volatilities=",".join(str(value) for value in replication_volatilities) or None,
        day_count=term_environment.accrual_conventions.day_count,
    )
    coordinates = ["spot", "discount_curve", "black_vol_surface"]
    if replication_strikes:
        coordinates.append("quote_grid.replication_strikes")
    if replication_volatilities:
        coordinates.append("quote_grid.replication_volatilities")
    return {
        "call_kwargs": {
            "market_state": market_state,
            "spec": spec,
        },
        "value_scale": 1.0,
        "resolved_market_coordinates": tuple(coordinates),
    }


def _pattern_max_ramp(lhs, rhs) -> ContractPattern:
    return ContractPattern(
        payoff=PayoffPattern(
            kind="max",
            args=(
                PayoffPattern(kind="sub", args=(lhs, rhs)),
                ConstantPattern(value=0.0),
            ),
        ),
        exercise=ExercisePattern(style="european"),
    )


def _weighted_spot(weight_name: str, underlier_name: str):
    return PayoffPattern(
        kind="scaled",
        args=(
            ConstantPattern(value=Wildcard(weight_name)),
            SpotPattern(underlier=Wildcard(underlier_name)),
        ),
    )


def _default_registry() -> ContractIRSolverRegistry:
    declarations = (
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=_pattern_max_ramp(
                        SpotPattern(underlier=Wildcard("u")),
                        StrikePattern(value=Wildcard("k")),
                    ).payoff,
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="equity_diffusion"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.black.black76_call",
                call_style="raw_kernel_kwargs",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._vanilla_call_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="black76_vanilla_call",
                validation_bundle_id="vanilla_option_contract",
                pricing_kernel_refs=("trellis.models.black.black76_call",),
            ),
            precedence=50,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=_pattern_max_ramp(
                        StrikePattern(value=Wildcard("k")),
                        SpotPattern(underlier=Wildcard("u")),
                    ).payoff,
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="equity_diffusion"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.black.black76_put",
                call_style="raw_kernel_kwargs",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._vanilla_put_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="black76_vanilla_put",
                validation_bundle_id="vanilla_option_contract",
                pricing_kernel_refs=("trellis.models.black.black76_put",),
            ),
            precedence=49,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=PayoffPattern(
                        kind="mul",
                        args=(
                            ConstantPattern(value=Wildcard("cash_payoff")),
                            PayoffPattern(kind="indicator", args=(Wildcard("predicate"),)),
                        ),
                    ),
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="equity_diffusion"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.black.black76_cash_or_nothing_call",
                call_style="raw_kernel_kwargs",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._digital_cash_call_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="black76_cash_digital_call",
                validation_bundle_id="digital_option_contract",
                pricing_kernel_refs=("trellis.models.black.black76_cash_or_nothing_call",),
            ),
            precedence=48,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=PayoffPattern(
                        kind="mul",
                        args=(
                            ConstantPattern(value=Wildcard("cash_payoff")),
                            PayoffPattern(kind="indicator", args=(Wildcard("predicate"),)),
                        ),
                    ),
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="equity_diffusion"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.black.black76_cash_or_nothing_put",
                call_style="raw_kernel_kwargs",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._digital_cash_put_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="black76_cash_digital_put",
                validation_bundle_id="digital_option_contract",
                pricing_kernel_refs=("trellis.models.black.black76_cash_or_nothing_put",),
            ),
            precedence=47,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=PayoffPattern(
                        kind="mul",
                        args=(
                            SpotPattern(underlier=Wildcard("u")),
                            PayoffPattern(kind="indicator", args=(Wildcard("predicate"),)),
                        ),
                    ),
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="equity_diffusion"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.black.black76_asset_or_nothing_call",
                call_style="raw_kernel_kwargs",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._digital_asset_call_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="black76_asset_digital_call",
                validation_bundle_id="digital_option_contract",
                pricing_kernel_refs=("trellis.models.black.black76_asset_or_nothing_call",),
            ),
            precedence=46,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=PayoffPattern(
                        kind="mul",
                        args=(
                            SpotPattern(underlier=Wildcard("u")),
                            PayoffPattern(kind="indicator", args=(Wildcard("predicate"),)),
                        ),
                    ),
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="equity_diffusion"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.black.black76_asset_or_nothing_put",
                call_style="raw_kernel_kwargs",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._digital_asset_put_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="black76_asset_digital_put",
                validation_bundle_id="digital_option_contract",
                pricing_kernel_refs=("trellis.models.black.black76_asset_or_nothing_put",),
            ),
            precedence=45,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=PayoffPattern(
                        kind="scaled",
                        args=(
                            PayoffPattern(kind="annuity", args=(Wildcard("u"), Wildcard("schedule"))),
                            PayoffPattern(
                                kind="max",
                                args=(
                                    PayoffPattern(
                                        kind="sub",
                                        args=(
                                            PayoffPattern(kind="swap_rate", args=(Wildcard("u"), Wildcard("schedule"))),
                                            StrikePattern(value=Wildcard("k")),
                                        ),
                                    ),
                                    ConstantPattern(value=0.0),
                                ),
                            ),
                        ),
                    ),
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="interest_rate"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions", "floating_rate_reference"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.rate_style_swaption.price_swaption_black76",
                call_style="helper_call",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._payer_swaption_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="helper_swaption_payer_black76",
                validation_bundle_id="rate_style_swaption_contract",
                helper_refs=("trellis.models.rate_style_swaption.price_swaption_black76",),
            ),
            precedence=44,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=PayoffPattern(
                        kind="scaled",
                        args=(
                            PayoffPattern(kind="annuity", args=(Wildcard("u"), Wildcard("schedule"))),
                            PayoffPattern(
                                kind="max",
                                args=(
                                    PayoffPattern(
                                        kind="sub",
                                        args=(
                                            StrikePattern(value=Wildcard("k")),
                                            PayoffPattern(kind="swap_rate", args=(Wildcard("u"), Wildcard("schedule"))),
                                        ),
                                    ),
                                    ConstantPattern(value=0.0),
                                ),
                            ),
                        ),
                    ),
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="interest_rate"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions", "floating_rate_reference"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.rate_style_swaption.price_swaption_black76",
                call_style="helper_call",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._receiver_swaption_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="helper_swaption_receiver_black76",
                validation_bundle_id="rate_style_swaption_contract",
                helper_refs=("trellis.models.rate_style_swaption.price_swaption_black76",),
            ),
            precedence=43,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=PayoffPattern(
                        kind="max",
                        args=(
                            PayoffPattern(
                                kind="sub",
                                args=(
                                    PayoffPattern(
                                        kind="linear_basket",
                                        args=(
                                            _weighted_spot("w1", "u1"),
                                            _weighted_spot("w2", "u2"),
                                        ),
                                    ),
                                    StrikePattern(value=Wildcard("k")),
                                ),
                            ),
                            ConstantPattern(value=0.0),
                        ),
                    ),
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="equity_diffusion"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.basket_option.price_basket_option_analytical",
                call_style="helper_call",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._basket_call_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="helper_basket_option_call",
                validation_bundle_id="basket_option_contract",
                helper_refs=("trellis.models.basket_option.price_basket_option_analytical",),
            ),
            precedence=42,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=PayoffPattern(
                        kind="max",
                        args=(
                            PayoffPattern(
                                kind="sub",
                                args=(
                                    StrikePattern(value=Wildcard("k")),
                                    PayoffPattern(
                                        kind="linear_basket",
                                        args=(
                                            _weighted_spot("w1", "u1"),
                                            _weighted_spot("w2", "u2"),
                                        ),
                                    ),
                                ),
                            ),
                            ConstantPattern(value=0.0),
                        ),
                    ),
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="equity_diffusion"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("cash_settlement", "accrual_conventions"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.basket_option.price_basket_option_analytical",
                call_style="helper_call",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._basket_put_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="helper_basket_option_put",
                validation_bundle_id="basket_option_contract",
                helper_refs=("trellis.models.basket_option.price_basket_option_analytical",),
            ),
            precedence=41,
        ),
        ContractIRSolverDeclaration(
            authority=ContractIRSolverSelectionAuthority(
                contract_pattern=ContractPattern(
                    payoff=PayoffPattern(
                        kind="scaled",
                        args=(
                            ConstantPattern(value=Wildcard("notional")),
                            PayoffPattern(
                                kind="sub",
                                args=(
                                    PayoffPattern(
                                        kind="variance_observable",
                                        args=(Wildcard("u"), Wildcard("interval")),
                                    ),
                                    StrikePattern(value=Wildcard("k")),
                                ),
                            ),
                        ),
                    ),
                    exercise=ExercisePattern(style="european"),
                    underlying=UnderlyingPattern(kind="equity_diffusion"),
                ),
                admissible_methods=("analytical",),
                required_term_groups=("accrual_conventions", "quote_grid"),
            ),
            materialization=ContractIRSolverMaterialization(
                callable_ref="trellis.models.analytical.equity_exotics.price_equity_variance_swap_analytical",
                call_style="helper_call",
                adapter_ref="trellis.agent.contract_ir_solver_compiler._variance_swap_adapter",
            ),
            provenance=ContractIRSolverProvenance(
                declaration_id="helper_equity_variance_swap",
                validation_bundle_id="variance_swap_contract",
                helper_refs=("trellis.models.analytical.equity_exotics.price_equity_variance_swap_analytical",),
            ),
            precedence=40,
        ),
    )
    return build_contract_ir_solver_registry(declarations)


def _vanilla_call_adapter(**kwargs) -> dict[str, object]:
    return _zero_carry_black76_adapter(call=True, **kwargs)


def _vanilla_put_adapter(**kwargs) -> dict[str, object]:
    return _zero_carry_black76_adapter(call=False, **kwargs)


def _digital_cash_call_adapter(**kwargs) -> dict[str, object]:
    return _digital_cash_adapter(call=True, **kwargs)


def _digital_cash_put_adapter(**kwargs) -> dict[str, object]:
    return _digital_cash_adapter(call=False, **kwargs)


def _digital_asset_call_adapter(**kwargs) -> dict[str, object]:
    return _digital_asset_adapter(call=True, **kwargs)


def _digital_asset_put_adapter(**kwargs) -> dict[str, object]:
    return _digital_asset_adapter(call=False, **kwargs)


def _payer_swaption_adapter(**kwargs) -> dict[str, object]:
    return _swaption_helper_adapter(is_payer=True, **kwargs)


def _receiver_swaption_adapter(**kwargs) -> dict[str, object]:
    return _swaption_helper_adapter(is_payer=False, **kwargs)


def _basket_call_adapter(**kwargs) -> dict[str, object]:
    return _basket_helper_adapter(option_type="call", **kwargs)


def _basket_put_adapter(**kwargs) -> dict[str, object]:
    return _basket_helper_adapter(option_type="put", **kwargs)


def _variance_swap_adapter(**kwargs) -> dict[str, object]:
    return _variance_helper_adapter(**kwargs)


_DEFAULT_REGISTRY: ContractIRSolverRegistry | None = None


def default_contract_ir_solver_registry() -> ContractIRSolverRegistry:
    """Return the cached first-wave structural declaration registry."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = _default_registry()
    return _DEFAULT_REGISTRY


__all__ = [
    "AccrualConventionTerms",
    "CashSettlementTerms",
    "ContractIRCompilerDecision",
    "ContractIRSolverAmbiguityError",
    "ContractIRSolverCompileError",
    "ContractIRSolverNoMatchError",
    "ContractIRSolverShadowRecord",
    "ContractIRTermEnvironment",
    "FloatingRateReferenceTerms",
    "QuoteGridTerms",
    "build_contract_ir_term_environment",
    "compile_contract_ir_solver",
    "default_contract_ir_solver_registry",
    "execute_contract_ir_solver_decision",
    "shadow_record_from_decision",
]
