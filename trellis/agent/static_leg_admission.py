"""Selection surface for bounded static leg lowerings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from trellis.conventions.day_count import DayCountConvention
from trellis.core.types import Frequency
from trellis.instruments.swap import SwapSpec
from trellis.agent.static_leg_contract import (
    CmsRateIndex,
    CouponLeg,
    FixedCouponFormula,
    FloatingCouponFormula,
    KnownCashflowLeg,
    OvernightRateIndex,
    PeriodRateOptionStripLeg,
    SignedLeg,
    StaticLegContractIR,
    TermRateIndex,
)


class StaticLegLoweringNoMatchError(ValueError):
    """Raised when no static-leg lowering declaration is admissible."""


class StaticLegLoweringAmbiguityError(ValueError):
    """Raised when multiple static-leg lowering declarations survive selection."""


@dataclass(frozen=True)
class StaticLegLoweringDeclaration:
    declaration_id: str
    matcher: Callable[[StaticLegContractIR], bool]
    callable_ref: str
    adapter_ref: str = ""
    validation_bundle_id: str = ""
    required_capabilities: tuple[str, ...] = ()
    cashflow_engine_refs: tuple[str, ...] = ()
    helper_refs: tuple[str, ...] = ()
    supported_methods: tuple[str, ...] = ()
    precedence: int = 0


@dataclass(frozen=True)
class StaticLegLoweringSelection:
    declaration_id: str
    callable_ref: str
    adapter_ref: str
    validation_bundle_id: str
    required_capabilities: tuple[str, ...]
    cashflow_engine_refs: tuple[str, ...]
    helper_refs: tuple[str, ...]
    method: str | None = None


def _is_fixed_float_swap(contract: StaticLegContractIR) -> bool:
    if len(contract.legs) != 2:
        return False
    if not all(isinstance(signed_leg.leg, CouponLeg) for signed_leg in contract.legs):
        return False
    formulas = [signed_leg.leg.coupon_formula for signed_leg in contract.legs]
    return any(isinstance(formula, FixedCouponFormula) for formula in formulas) and any(
        isinstance(formula, FloatingCouponFormula) for formula in formulas
    )


def _is_basis_swap(contract: StaticLegContractIR) -> bool:
    if len(contract.legs) != 2:
        return False
    if not all(isinstance(signed_leg.leg, CouponLeg) for signed_leg in contract.legs):
        return False
    if not all(
        isinstance(signed_leg.leg.coupon_formula, FloatingCouponFormula)
        for signed_leg in contract.legs
    ):
        return False
    for signed_leg in contract.legs:
        leg = signed_leg.leg
        if len(leg.notional_schedule.steps) != 1:
            return False
        formula = leg.coupon_formula
        if not isinstance(formula.rate_index, (OvernightRateIndex, TermRateIndex)):
            return False
    return True


def _is_fixed_coupon_bond(contract: StaticLegContractIR) -> bool:
    if not contract.legs:
        return False
    if any(signed_leg.direction != "receive" for signed_leg in contract.legs):
        return False
    has_fixed_coupon_leg = any(
        isinstance(signed_leg.leg, CouponLeg)
        and isinstance(signed_leg.leg.coupon_formula, FixedCouponFormula)
        for signed_leg in contract.legs
    )
    has_redemption_leg = any(
        isinstance(signed_leg.leg, KnownCashflowLeg)
        for signed_leg in contract.legs
    )
    return has_fixed_coupon_leg and has_redemption_leg


def _is_period_rate_option_strip(contract: StaticLegContractIR) -> bool:
    if len(contract.legs) != 1 or contract.legs[0].direction != "receive":
        return False
    leg = contract.legs[0].leg
    return (
        isinstance(leg, PeriodRateOptionStripLeg)
        and len(leg.option_periods) > 1
        and len(leg.notional_schedule.steps) == 1
        and isinstance(leg.rate_index, (OvernightRateIndex, TermRateIndex))
    )


def _unimplemented_static_leg_lowering(**kwargs):
    raise NotImplementedError(
        "Static-leg lowering is selection-only in the first closure slice."
    )


def _resolve_ref(ref: str) -> object:
    module_name, _, attr_name = ref.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid dotted ref: {ref!r}")
    module = import_module(module_name)
    return getattr(module, attr_name)


def _normalize_method_name(method: str | None) -> str | None:
    normalized = str(method or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def _frequency_enum(token: str) -> Frequency:
    normalized = str(token or "").strip().lower().replace("-", "_")
    mapping = {
        "annual": Frequency.ANNUAL,
        "semiannual": Frequency.SEMI_ANNUAL,
        "quarterly": Frequency.QUARTERLY,
        "monthly": Frequency.MONTHLY,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unsupported static-leg payment frequency {token!r}"
        ) from exc


def _day_count_enum(token: str) -> DayCountConvention:
    normalized = str(token or "").strip().upper().replace("_", "/")
    mapping = {
        "ACT/360": DayCountConvention.ACT_360,
        "ACT/365": DayCountConvention.ACT_365,
        "ACT/ACT": DayCountConvention.ACT_ACT,
        "30/360": DayCountConvention.THIRTY_360,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unsupported static-leg day count {token!r}"
        ) from exc


def _constant_notional_amount(leg: CouponLeg | PeriodRateOptionStripLeg) -> float:
    if len(leg.notional_schedule.steps) != 1:
        raise NotImplementedError(
            "Static-leg materialization only supports constant-notional legs in the first slice."
        )
    return float(leg.notional_schedule.steps[0].amount)


def _rate_index_name(index: object) -> str | None:
    if isinstance(index, OvernightRateIndex):
        return index.name
    if isinstance(index, TermRateIndex):
        return index.name
    if isinstance(index, CmsRateIndex):
        return index.curve_id
    return None


def _rate_index_identifier(index: object) -> str | None:
    if isinstance(index, OvernightRateIndex):
        return index.name
    if isinstance(index, TermRateIndex):
        return f"{index.name}-{index.tenor}"
    if isinstance(index, CmsRateIndex):
        return f"{index.curve_id}-{index.tenor}"
    return None


def _fixed_float_swap_adapter(
    contract: StaticLegContractIR,
    *,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    del normalized_terms
    fixed_leg = next(
        signed_leg
        for signed_leg in contract.legs
        if isinstance(signed_leg.leg, CouponLeg)
        and isinstance(signed_leg.leg.coupon_formula, FixedCouponFormula)
    )
    floating_leg = next(
        signed_leg
        for signed_leg in contract.legs
        if isinstance(signed_leg.leg, CouponLeg)
        and isinstance(signed_leg.leg.coupon_formula, FloatingCouponFormula)
    )
    fixed_coupon_leg = fixed_leg.leg
    floating_coupon_leg = floating_leg.leg
    fixed_formula = fixed_coupon_leg.coupon_formula
    floating_formula = floating_coupon_leg.coupon_formula
    notional = _constant_notional_amount(fixed_coupon_leg)
    floating_notional = _constant_notional_amount(floating_coupon_leg)
    if abs(notional - floating_notional) > 1e-12:
        raise NotImplementedError(
            "Static-leg swap materialization requires matching constant notionals."
        )
    return {
        "call_kwargs": {
            "spec": SwapSpec(
                notional=notional,
                fixed_rate=fixed_formula.rate,
                start_date=fixed_coupon_leg.coupon_periods[0].accrual_start,
                end_date=fixed_coupon_leg.coupon_periods[-1].accrual_end,
                fixed_frequency=_frequency_enum(fixed_coupon_leg.payment_frequency),
                float_frequency=_frequency_enum(floating_coupon_leg.payment_frequency),
                fixed_day_count=_day_count_enum(fixed_coupon_leg.day_count),
                float_day_count=_day_count_enum(floating_coupon_leg.day_count),
                rate_index=_rate_index_name(floating_formula.rate_index),
                is_payer=fixed_leg.direction == "pay",
            )
        }
    }


def _fixed_coupon_bond_adapter(
    contract: StaticLegContractIR,
    *,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    del normalized_terms
    coupon_leg = next(
        signed_leg.leg
        for signed_leg in contract.legs
        if isinstance(signed_leg.leg, CouponLeg)
        and isinstance(signed_leg.leg.coupon_formula, FixedCouponFormula)
    )
    notional = _constant_notional_amount(coupon_leg)
    maturity_date = max(
        max(period.payment_date for period in coupon_leg.coupon_periods),
        max(
            cashflow.payment_date
            for signed_leg in contract.legs
            if isinstance(signed_leg.leg, KnownCashflowLeg)
            for cashflow in signed_leg.leg.cashflows
        ),
    )
    return {
        "call_kwargs": {
            "notional": notional,
            "coupon": coupon_leg.coupon_formula.rate,
            "maturity_date": maturity_date,
            "issue_date": coupon_leg.coupon_periods[0].accrual_start,
            "frequency": _frequency_enum(coupon_leg.payment_frequency).value,
            "day_count": _day_count_enum(coupon_leg.day_count),
        }
    }


def _basis_swap_adapter(
    contract: StaticLegContractIR,
    *,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    del normalized_terms
    pay_leg = next(signed_leg.leg for signed_leg in contract.legs if signed_leg.direction == "pay")
    receive_leg = next(
        signed_leg.leg for signed_leg in contract.legs if signed_leg.direction == "receive"
    )
    pay_formula = pay_leg.coupon_formula
    receive_formula = receive_leg.coupon_formula
    if not isinstance(pay_formula, FloatingCouponFormula) or not isinstance(
        receive_formula, FloatingCouponFormula
    ):
        raise TypeError("Basis swap lowering requires floating coupon formulas on both legs")

    basis_period_ref = "trellis.models.rate_basis_swap.BasisSwapFloatingLegPeriod"
    basis_leg_ref = "trellis.models.rate_basis_swap.BasisSwapFloatingLegSpec"
    basis_spec_ref = "trellis.models.rate_basis_swap.RateBasisSwapSpec"
    basis_period_cls = _resolve_ref(basis_period_ref)
    basis_leg_cls = _resolve_ref(basis_leg_ref)
    basis_spec_cls = _resolve_ref(basis_spec_ref)

    def _leg_spec(leg: CouponLeg, formula: FloatingCouponFormula):
        return basis_leg_cls(
            notional=_constant_notional_amount(leg),
            periods=tuple(
                basis_period_cls(
                    accrual_start=period.accrual_start,
                    accrual_end=period.accrual_end,
                    payment_date=period.payment_date,
                    fixing_date=period.fixing_date,
                )
                for period in leg.coupon_periods
            ),
            day_count=_day_count_enum(leg.day_count),
            rate_index=_rate_index_identifier(formula.rate_index),
            spread=formula.spread,
        )

    return {
        "call_kwargs": {
            "spec": basis_spec_cls(
                pay_leg=_leg_spec(pay_leg, pay_formula),
                receive_leg=_leg_spec(receive_leg, receive_formula),
            )
        }
    }


def _period_rate_option_strip_call_kwargs(
    contract: StaticLegContractIR,
    *,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    strip_leg = contract.legs[0].leg
    if not isinstance(strip_leg, PeriodRateOptionStripLeg):
        raise TypeError("Expected PeriodRateOptionStripLeg for strip-lowering materialization")

    cap_floor_period_cls = _resolve_ref("trellis.models.rate_cap_floor.CapFloorPeriod")
    call_kwargs: dict[str, object] = {
        "instrument_class": "cap" if strip_leg.option_side == "call" else "floor",
        "periods": tuple(
            cap_floor_period_cls(
                start_date=period.accrual_start,
                end_date=period.accrual_end,
                payment_date=period.payment_date,
                fixing_date=period.fixing_date,
            )
            for period in strip_leg.option_periods
        ),
        "notional": _constant_notional_amount(strip_leg),
        "strike": strip_leg.strike,
        "start_date": strip_leg.option_periods[0].accrual_start,
        "end_date": strip_leg.option_periods[-1].accrual_end,
        "frequency": _frequency_enum(strip_leg.payment_frequency),
        "day_count": _day_count_enum(strip_leg.day_count),
        "rate_index": _rate_index_identifier(strip_leg.rate_index),
    }

    terms = dict(normalized_terms or {})
    for key in ("calendar_name", "business_day_adjustment"):
        if terms.get(key) is not None:
            call_kwargs[key] = terms[key]
    return call_kwargs


def _period_rate_option_strip_analytical_adapter(
    contract: StaticLegContractIR,
    *,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    call_kwargs = _period_rate_option_strip_call_kwargs(
        contract,
        normalized_terms=normalized_terms,
    )
    terms = dict(normalized_terms or {})
    for key in ("model", "shift", "sabr"):
        if terms.get(key) is not None:
            call_kwargs[key] = terms[key]
    return {"call_kwargs": call_kwargs}


def _period_rate_option_strip_monte_carlo_adapter(
    contract: StaticLegContractIR,
    *,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    call_kwargs = _period_rate_option_strip_call_kwargs(
        contract,
        normalized_terms=normalized_terms,
    )
    terms = dict(normalized_terms or {})
    for key in (
        "n_paths",
        "seed",
        "n_steps",
        "mean_reversion",
        "sigma",
        "discount_curve",
        "forward_curve",
        "vol",
    ):
        if terms.get(key) is not None:
            call_kwargs[key] = terms[key]
    return {"call_kwargs": call_kwargs}


def _unimplemented_static_leg_materialization(
    contract: StaticLegContractIR,
    *,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    del contract, normalized_terms
    raise NotImplementedError(
        "No checked executable lowering is landed yet for this static-leg family."
    )


_DECLARATIONS = (
    StaticLegLoweringDeclaration(
        declaration_id="static_leg_fixed_float_swap",
        matcher=_is_fixed_float_swap,
        callable_ref="trellis.instruments.swap.SwapPayoff",
        adapter_ref="trellis.agent.static_leg_admission._fixed_float_swap_adapter",
        validation_bundle_id="static_leg_fixed_float_swap_contract",
        required_capabilities=("discount_curve", "forward_curve"),
        helper_refs=("trellis.instruments.swap.SwapPayoff", "trellis.instruments.swap.SwapSpec"),
        precedence=30,
    ),
    StaticLegLoweringDeclaration(
        declaration_id="static_leg_period_rate_option_strip_analytical",
        matcher=_is_period_rate_option_strip,
        callable_ref="trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical",
        adapter_ref="trellis.agent.static_leg_admission._period_rate_option_strip_analytical_adapter",
        validation_bundle_id="static_leg_period_rate_option_strip_contract",
        required_capabilities=("discount_curve", "forward_curve", "black_vol_surface"),
        helper_refs=("trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical",),
        supported_methods=("analytical",),
        precedence=25,
    ),
    StaticLegLoweringDeclaration(
        declaration_id="static_leg_period_rate_option_strip_monte_carlo",
        matcher=_is_period_rate_option_strip,
        callable_ref="trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo",
        adapter_ref="trellis.agent.static_leg_admission._period_rate_option_strip_monte_carlo_adapter",
        validation_bundle_id="static_leg_period_rate_option_strip_contract",
        required_capabilities=("discount_curve", "forward_curve", "black_vol_surface"),
        helper_refs=("trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo",),
        supported_methods=("monte_carlo",),
        precedence=24,
    ),
    StaticLegLoweringDeclaration(
        declaration_id="static_leg_basis_swap",
        matcher=_is_basis_swap,
        callable_ref="trellis.models.rate_basis_swap.price_rate_basis_swap",
        adapter_ref="trellis.agent.static_leg_admission._basis_swap_adapter",
        validation_bundle_id="static_leg_basis_swap_contract",
        required_capabilities=("discount_curve", "forward_curve"),
        helper_refs=(
            "trellis.models.rate_basis_swap.price_rate_basis_swap",
            "trellis.models.contingent_cashflows.coupon_cashflow_pv",
        ),
        precedence=20,
    ),
    StaticLegLoweringDeclaration(
        declaration_id="static_leg_fixed_coupon_bond",
        matcher=_is_fixed_coupon_bond,
        callable_ref="trellis.instruments.bond.Bond",
        adapter_ref="trellis.agent.static_leg_admission._fixed_coupon_bond_adapter",
        validation_bundle_id="static_leg_fixed_coupon_bond_contract",
        required_capabilities=("discount_curve",),
        helper_refs=("trellis.instruments.bond.Bond",),
        precedence=10,
    ),
)


def default_static_leg_lowering_declarations() -> tuple[StaticLegLoweringDeclaration, ...]:
    return _DECLARATIONS


def select_static_leg_lowering(
    contract_ir: StaticLegContractIR,
    *,
    requested_method: str | None = None,
) -> StaticLegLoweringSelection:
    """Select one bounded static-leg lowering declaration."""

    normalized_method = _normalize_method_name(requested_method)
    matches = [
        declaration
        for declaration in default_static_leg_lowering_declarations()
        if declaration.matcher(contract_ir)
        and (
            normalized_method is None
            or not declaration.supported_methods
            or normalized_method in declaration.supported_methods
        )
    ]
    if not matches:
        raise StaticLegLoweringNoMatchError(
            "No admissible static-leg lowering declaration was found."
        )
    top_precedence = max(declaration.precedence for declaration in matches)
    top = [declaration for declaration in matches if declaration.precedence == top_precedence]
    if len(top) > 1:
        raise StaticLegLoweringAmbiguityError(
            f"Multiple static-leg lowering declarations remained admissible: {[item.declaration_id for item in top]}"
        )
    chosen = top[0]
    return StaticLegLoweringSelection(
        declaration_id=chosen.declaration_id,
        callable_ref=chosen.callable_ref,
        adapter_ref=chosen.adapter_ref,
        validation_bundle_id=chosen.validation_bundle_id,
        required_capabilities=chosen.required_capabilities,
        cashflow_engine_refs=chosen.cashflow_engine_refs,
        helper_refs=chosen.helper_refs,
        method=normalized_method or (chosen.supported_methods[0] if chosen.supported_methods else None),
    )


def materialize_static_leg_lowering(
    contract_ir: StaticLegContractIR,
    *,
    selection: StaticLegLoweringSelection | None = None,
    requested_method: str | None = None,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Materialize the bounded lowering payload for a selected static-leg family."""

    chosen = selection or select_static_leg_lowering(
        contract_ir,
        requested_method=requested_method,
    )
    adapter = _resolve_ref(chosen.adapter_ref)
    payload = adapter(contract_ir, normalized_terms=normalized_terms)
    if not isinstance(payload, dict):
        raise TypeError("Static-leg materialization adapter must return a dict payload")
    return {
        "callable_ref": chosen.callable_ref,
        **payload,
    }


__all__ = [
    "StaticLegLoweringAmbiguityError",
    "StaticLegLoweringDeclaration",
    "StaticLegLoweringNoMatchError",
    "StaticLegLoweringSelection",
    "default_static_leg_lowering_declarations",
    "materialize_static_leg_lowering",
    "select_static_leg_lowering",
]
