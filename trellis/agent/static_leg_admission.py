"""Selection surface for bounded static leg lowerings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from trellis.conventions.day_count import DayCountConvention
from trellis.core.date_utils import build_payment_timeline
from trellis.core.types import Frequency
from trellis.instruments.swap import SwapSpec
from trellis.agent.static_leg_contract import (
    CmsRateIndex,
    ConditionalAccrualLeg,
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
from trellis.agent.semantic_observables import (
    BetweenPredicate,
    RateIndexObservable,
    predicate_support_blockers,
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


@dataclass(frozen=True)
class StaticLegAdmissionBlocker:
    blocker_id: str
    reason: str
    required_ticket: str = ""


def _is_fixed_float_swap(contract: StaticLegContractIR) -> bool:
    if len(contract.legs) != 2:
        return False
    if not all(isinstance(signed_leg.leg, CouponLeg) for signed_leg in contract.legs):
        return False
    formulas = [signed_leg.leg.coupon_formula for signed_leg in contract.legs]
    return (
        any(isinstance(formula, FixedCouponFormula) for formula in formulas)
        and any(isinstance(formula, FloatingCouponFormula) for formula in formulas)
        and not static_coupon_obligation_admission_blockers(contract)
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
    return not static_coupon_obligation_admission_blockers(contract)


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
    return (
        has_fixed_coupon_leg
        and has_redemption_leg
        and not static_coupon_obligation_admission_blockers(contract)
    )


def _is_period_rate_option_strip(contract: StaticLegContractIR) -> bool:
    if len(contract.legs) != 1:
        return False
    leg = contract.legs[0].leg
    return (
        isinstance(leg, PeriodRateOptionStripLeg)
        and len(leg.option_periods) > 1
        and len(leg.notional_schedule.steps) == 1
        and isinstance(leg.rate_index, (OvernightRateIndex, TermRateIndex))
    )


def _is_conditional_range_accrual(contract: StaticLegContractIR) -> bool:
    return not conditional_range_accrual_admission_blockers(contract)


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


def _constant_notional_amount(
    leg: CouponLeg | ConditionalAccrualLeg | PeriodRateOptionStripLeg,
) -> float:
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


def _metadata_mapping(value: object) -> dict[str, object]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _merged_metadata_mapping(*values: object) -> dict[str, object]:
    merged: dict[str, object] = {}
    for value in values:
        if isinstance(value, Mapping):
            merged.update(value)
    return merged


def _admission_blocker(
    blocker_id: str,
    reason: str,
    *,
    required_ticket: str = "",
) -> StaticLegAdmissionBlocker:
    return StaticLegAdmissionBlocker(
        blocker_id=blocker_id,
        reason=reason,
        required_ticket=required_ticket,
    )


def static_coupon_obligation_admission_blockers(
    contract: StaticLegContractIR,
) -> tuple[StaticLegAdmissionBlocker, ...]:
    """Return exact blockers for bounded explicit coupon obligations."""

    coupon_legs = tuple(
        signed_leg.leg
        for signed_leg in contract.legs
        if isinstance(signed_leg.leg, CouponLeg)
    )
    if not coupon_legs:
        return ()

    blockers: list[StaticLegAdmissionBlocker] = []
    if any(
        not isinstance(signed_leg.leg, (CouponLeg, KnownCashflowLeg))
        for signed_leg in contract.legs
    ):
        blockers.append(
            _admission_blocker(
                "static_coupon_obligation_extra_leg_unsupported",
                "Explicit coupon execution admits coupon legs and known cashflows only.",
            )
        )

    currencies = set(contract.currencies)
    if len(currencies) != 1:
        blockers.append(
            _admission_blocker(
                "static_coupon_obligation_single_currency_required",
                "Explicit coupon execution requires one contract currency.",
            )
        )
    if contract.settlement.settlement_kind != "cash":
        blockers.append(
            _admission_blocker(
                "static_coupon_obligation_cash_settlement_required",
                "Explicit coupon execution admits cash settlement only.",
            )
        )
    if contract.settlement.settlement_lag_days != 0:
        blockers.append(
            _admission_blocker(
                "static_coupon_obligation_zero_settlement_lag_required",
                "Coupon payment dates must include settlement lag explicitly.",
            )
        )

    for leg in coupon_legs:
        steps = tuple(leg.notional_schedule.steps)
        if len(steps) != 1:
            blockers.append(
                _admission_blocker(
                    "static_coupon_obligation_constant_notional_required",
                    "Explicit coupon execution requires one constant notional step per leg.",
                )
            )
        else:
            step = steps[0]
            first_period = leg.coupon_periods[0]
            last_period = leg.coupon_periods[-1]
            if float(step.amount) <= 0.0:
                blockers.append(
                    _admission_blocker(
                        "static_coupon_obligation_positive_notional_required",
                        "Explicit coupon execution requires positive notionals.",
                    )
                )
            if (
                step.start_date > first_period.accrual_start
                or step.end_date < last_period.accrual_end
            ):
                blockers.append(
                    _admission_blocker(
                        "static_coupon_obligation_notional_coverage_required",
                        "The constant notional step must cover every explicit coupon period.",
                    )
                )

        try:
            _day_count_enum(leg.day_count)
        except NotImplementedError:
            blockers.append(
                _admission_blocker(
                    "static_coupon_obligation_day_count_unsupported",
                    f"Unsupported explicit coupon day count {leg.day_count!r}.",
                )
            )

        formula = leg.coupon_formula
        if isinstance(formula, FixedCouponFormula):
            continue
        if not isinstance(formula, FloatingCouponFormula):
            blockers.append(
                _admission_blocker(
                    "static_coupon_obligation_formula_unsupported",
                    "Explicit coupon execution admits fixed and floating coupon formulas only.",
                )
            )
            continue
        index = formula.rate_index
        if isinstance(index, TermRateIndex):
            continue
        if isinstance(index, OvernightRateIndex):
            if index.compounding != "simple":
                blockers.append(
                    _admission_blocker(
                        "static_coupon_obligation_simple_overnight_required",
                        "Overnight coupon execution currently admits simple period rates only.",
                    )
                )
            continue
        blockers.append(
            _admission_blocker(
                "static_coupon_obligation_rate_index_unsupported",
                "Explicit floating coupons require a term or overnight rate index.",
            )
        )

    unique: dict[str, StaticLegAdmissionBlocker] = {}
    for blocker in blockers:
        unique.setdefault(blocker.blocker_id, blocker)
    return tuple(unique.values())


def _coupon_leg_requires_explicit_period_execution(leg: CouponLeg) -> bool:
    try:
        timeline = build_payment_timeline(
            leg.coupon_periods[0].accrual_start,
            leg.coupon_periods[-1].accrual_end,
            _frequency_enum(leg.payment_frequency),
            day_count=_day_count_enum(leg.day_count),
        )
    except (NotImplementedError, ValueError):
        return True

    explicit_periods = tuple(
        (period.accrual_start, period.accrual_end, period.payment_date)
        for period in leg.coupon_periods
    )
    regular_periods = tuple(
        (period.start_date, period.end_date, period.payment_date)
        for period in timeline.periods
    )
    if explicit_periods != regular_periods:
        return True
    if isinstance(leg.coupon_formula, FloatingCouponFormula):
        return any(
            period.fixing_date is not None
            and period.fixing_date != period.accrual_start
            for period in leg.coupon_periods
        )
    return False


def _is_explicit_coupon_obligation_contract(contract: StaticLegContractIR) -> bool:
    if static_coupon_obligation_admission_blockers(contract):
        return False
    coupon_legs = tuple(
        signed_leg.leg
        for signed_leg in contract.legs
        if isinstance(signed_leg.leg, CouponLeg)
    )
    if not coupon_legs or not all(
        isinstance(signed_leg.leg, (CouponLeg, KnownCashflowLeg))
        for signed_leg in contract.legs
    ):
        return False
    if _is_basis_swap(contract):
        return False
    if _is_fixed_float_swap(contract):
        return any(
            _coupon_leg_requires_explicit_period_execution(leg)
            for leg in coupon_legs
        )
    if _is_fixed_coupon_bond(contract):
        return any(
            _coupon_leg_requires_explicit_period_execution(leg)
            for leg in coupon_legs
        )
    return True


def _coupon_obligation_required_capabilities(
    contract: StaticLegContractIR,
) -> tuple[str, ...]:
    capabilities = ["discount_curve"]
    if any(
        isinstance(signed_leg.leg, CouponLeg)
        and isinstance(signed_leg.leg.coupon_formula, FloatingCouponFormula)
        for signed_leg in contract.legs
    ):
        capabilities.append("forward_curve")
    return tuple(capabilities)


def _observable_support_admission_blockers(
    predicate,
) -> tuple[StaticLegAdmissionBlocker, ...]:
    return tuple(
        _admission_blocker(
            blocker.blocker_id,
            blocker.reason,
            required_ticket=blocker.required_ticket,
        )
        for blocker in predicate_support_blockers(predicate)
    )


def _conditional_range_accrual_signed_leg(
    contract: StaticLegContractIR,
) -> SignedLeg | None:
    matches = [
        signed_leg
        for signed_leg in contract.legs
        if isinstance(signed_leg.leg, ConditionalAccrualLeg)
    ]
    return matches[0] if len(matches) == 1 else None


def conditional_range_accrual_admission_blockers(
    contract: StaticLegContractIR,
) -> tuple[StaticLegAdmissionBlocker, ...]:
    """Return blockers for the checked single-index range-accrual route."""

    blockers: list[StaticLegAdmissionBlocker] = []
    contract_metadata = _metadata_mapping(contract.metadata)
    conditional_signed_legs = [
        signed_leg
        for signed_leg in contract.legs
        if isinstance(signed_leg.leg, ConditionalAccrualLeg)
    ]
    if len(conditional_signed_legs) != 1:
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_single_coupon_leg_required",
                "Range-accrual admission requires exactly one conditional accrual coupon leg.",
            )
        )
        return tuple(blockers)

    signed_leg = conditional_signed_legs[0]
    leg = signed_leg.leg
    leg_metadata = _metadata_mapping(leg.metadata)
    semantic_family = str(
        leg_metadata.get("semantic_family")
        or contract_metadata.get("semantic_family")
        or ""
    ).strip()
    if semantic_family != "range_accrual":
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_semantic_family_required",
                "Conditional accrual route admission is restricted to range_accrual legs.",
            )
        )
    callability = _merged_metadata_mapping(
        contract_metadata.get("callability"),
        leg_metadata.get("callability"),
    )
    if callability:
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_callability_pending",
                (
                    "Callable range accrual requires a dynamic exercise wrapper "
                    "before the checked static-leg route may be used."
                ),
                required_ticket="QUA-1117",
            )
        )
    dynamic_features = _merged_metadata_mapping(
        contract_metadata.get("dynamic_features"),
        leg_metadata.get("dynamic_features"),
    )
    interruption_events = (
        dynamic_features.get("interruption_events")
        or dynamic_features.get("interruptions")
        or dynamic_features.get("accrual_interruptions")
        or ()
    )
    if interruption_events:
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_interruption_state_pending",
                (
                    "Interrupted range accrual requires dynamic event-state "
                    "admission before the checked static-leg route may be used."
                ),
                required_ticket="QUA-1120",
            )
        )
    barrier_state = (
        dynamic_features.get("barrier_state")
        or dynamic_features.get("barrier_events")
        or dynamic_features.get("knockout_condition")
        or dynamic_features.get("knock_out")
        or dynamic_features.get("knock_in")
        or {}
    )
    if barrier_state:
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_barrier_state_pending",
                (
                    "Barrier-style range accrual requires dynamic event-state "
                    "admission before the checked static-leg route may be used."
                ),
                required_ticket="QUA-1120",
            )
        )
    if signed_leg.direction != "receive":
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_receive_leg_required",
                "The checked range-accrual route admits receive-side coupon legs only.",
            )
        )
    unsupported_legs = [
        item.leg
        for item in contract.legs
        if not isinstance(item.leg, (ConditionalAccrualLeg, KnownCashflowLeg))
    ]
    if unsupported_legs:
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_extra_leg_unsupported",
                "The checked route admits only the conditional coupon leg and optional principal redemption.",
            )
        )
    principal_legs = [
        item
        for item in contract.legs
        if isinstance(item.leg, KnownCashflowLeg)
    ]
    if len(principal_legs) > 1:
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_single_principal_leg_required",
                "The checked route admits at most one principal redemption leg.",
            )
        )
    if any(len(item.leg.cashflows) != 1 for item in principal_legs):
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_single_principal_cashflow_required",
                "The checked route admits exactly one principal redemption cashflow when a principal leg is present.",
            )
        )
    if any(item.direction != "receive" for item in principal_legs):
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_receive_principal_required",
                "Principal redemption must be receive-side for the checked route.",
            )
        )
    if len(principal_legs) == 1 and len(principal_legs[0].leg.cashflows) == 1:
        principal_cashflow = principal_legs[0].leg.cashflows[0]
        final_coupon_payment_date = leg.accrual_periods[-1].payment_date
        if principal_cashflow.payment_date != final_coupon_payment_date:
            blockers.append(
                _admission_blocker(
                    "conditional_range_accrual_principal_maturity_payment_required",
                    "Principal redemption must pay on the final coupon payment date for the checked route.",
                )
            )
    if not isinstance(leg.coupon_formula, FixedCouponFormula):
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_fixed_coupon_required",
                "The checked range-accrual route admits fixed coupon-if-in-range formulas only.",
            )
        )
    if len(leg.notional_schedule.steps) != 1:
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_constant_notional_required",
                "The checked range-accrual route admits one constant notional schedule.",
            )
        )
    elif float(leg.notional_schedule.steps[0].amount) <= 0.0:
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_positive_notional_required",
                "The checked range-accrual route requires positive notional.",
            )
        )
    if leg.settlement_rule != "coupon_period_cash_settlement":
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_coupon_settlement_required",
                "The checked range-accrual route admits coupon_period_cash_settlement only.",
            )
        )
    if any(
        period.fixing_date is not None
        and period.fixing_date != period.observation_date
        for period in leg.accrual_periods
    ):
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_observation_fixing_identity_required",
                "The checked route requires each fixing date to match its observation date.",
            )
        )
    if leg.accrual_counter_ref != "in_range_coupon_count":
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_counter_ref_unsupported",
                "The checked range-accrual route admits the in_range_coupon_count counter only.",
            )
        )
    condition = leg.accrual_condition
    support_blockers = (
        _observable_support_admission_blockers(condition)
        if condition is not None
        else ()
    )
    if not isinstance(condition, BetweenPredicate):
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_between_predicate_required",
                "The checked range-accrual route admits a single between predicate.",
            )
        )
        blockers.extend(support_blockers)
    elif support_blockers:
        blockers.extend(support_blockers)
    elif not isinstance(condition.observable, RateIndexObservable):
        blockers.append(
            _admission_blocker(
                "conditional_range_accrual_rate_index_required",
                "The checked range-accrual route admits one rate-index observable.",
            )
        )
    return tuple(blockers)


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


def _coupon_obligation_execution_adapter(
    contract: StaticLegContractIR,
    *,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from trellis.execution import compile_static_leg_execution_ir

    call_kwargs: dict[str, object] = {
        "execution_ir": compile_static_leg_execution_ir(
            contract,
            fail_on_unsupported=True,
        )
    }
    if normalized_terms:
        call_kwargs["execution_terms"] = dict(normalized_terms)
    return {"call_kwargs": call_kwargs}


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
    return {
        "call_kwargs": call_kwargs,
        "result_multiplier": (
            1.0 if contract.legs[0].direction == "receive" else -1.0
        ),
    }


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
    return {
        "call_kwargs": call_kwargs,
        "result_multiplier": (
            1.0 if contract.legs[0].direction == "receive" else -1.0
        ),
    }


def _range_accrual_reference_index(observable: RateIndexObservable) -> str:
    if observable.tenor:
        return f"{observable.index_name}-{observable.tenor}"
    return observable.index_name


def _range_accrual_principal_redemption(
    contract: StaticLegContractIR,
    *,
    notional: float,
) -> float:
    principal_legs = [
        signed_leg.leg
        for signed_leg in contract.legs
        if isinstance(signed_leg.leg, KnownCashflowLeg)
    ]
    if not principal_legs:
        return 0.0
    cashflows = principal_legs[0].cashflows
    if not cashflows:
        return 0.0
    if len(cashflows) != 1:
        raise NotImplementedError(
            "Static-leg range-accrual materialization admits one principal cashflow."
        )
    return float(cashflows[-1].amount) / notional


def _range_accrual_adapter(
    contract: StaticLegContractIR,
    *,
    normalized_terms: Mapping[str, object] | None = None,
) -> dict[str, object]:
    del normalized_terms
    signed_leg = _conditional_range_accrual_signed_leg(contract)
    if signed_leg is None or conditional_range_accrual_admission_blockers(contract):
        raise NotImplementedError(
            "Static-leg range-accrual materialization requires an admitted conditional accrual leg."
        )
    leg = signed_leg.leg
    if not isinstance(leg, ConditionalAccrualLeg):
        raise TypeError("Expected ConditionalAccrualLeg for range-accrual materialization")
    condition = leg.accrual_condition
    if not isinstance(condition, BetweenPredicate) or not isinstance(
        condition.observable,
        RateIndexObservable,
    ):
        raise TypeError("Expected rate-index between predicate for range-accrual materialization")
    notional = _constant_notional_amount(leg)
    range_accrual_spec_cls = _resolve_ref("trellis.models.range_accrual.RangeAccrualSpec")
    return {
        "call_kwargs": {
            "spec": range_accrual_spec_cls(
                reference_index=_range_accrual_reference_index(condition.observable),
                notional=notional,
                coupon_rate=leg.coupon_formula.rate,
                lower_bound=condition.lower_bound,
                upper_bound=condition.upper_bound,
                observation_dates=tuple(
                    period.observation_date for period in leg.accrual_periods
                ),
                accrual_start_dates=tuple(
                    period.accrual_start for period in leg.accrual_periods
                ),
                payment_dates=tuple(
                    period.payment_date for period in leg.accrual_periods
                ),
                principal_redemption=_range_accrual_principal_redemption(
                    contract,
                    notional=notional,
                ),
                inclusive_lower=condition.inclusive_lower,
                inclusive_upper=condition.inclusive_upper,
                day_count=_day_count_enum(leg.day_count),
            )
        }
    }


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
        declaration_id="static_leg_coupon_obligations",
        matcher=_is_explicit_coupon_obligation_contract,
        callable_ref="trellis.core.payoff.ExecutionBackedPayoff",
        adapter_ref=(
            "trellis.agent.static_leg_admission._coupon_obligation_execution_adapter"
        ),
        validation_bundle_id="static_coupon_obligation_execution_v1",
        helper_refs=(
            "trellis.execution.compile_static_leg_execution_ir",
            "trellis.execution.price_static_leg_execution_ir",
        ),
        precedence=40,
    ),
    StaticLegLoweringDeclaration(
        declaration_id="static_leg_range_accrual_discounted",
        matcher=_is_conditional_range_accrual,
        callable_ref="trellis.models.range_accrual.price_range_accrual",
        adapter_ref="trellis.agent.static_leg_admission._range_accrual_adapter",
        validation_bundle_id="range_accrual_discounted_cashflow_v1",
        required_capabilities=("discount_curve", "forward_curve", "fixing_history"),
        cashflow_engine_refs=(
            "trellis.models.contingent_cashflows.coupon_cashflow_pv",
            "trellis.models.contingent_cashflows.principal_payment_pv",
        ),
        helper_refs=("trellis.models.range_accrual.price_range_accrual",),
        supported_methods=("analytical",),
        precedence=35,
    ),
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
        coupon_blockers = static_coupon_obligation_admission_blockers(contract_ir)
        if coupon_blockers:
            details = "; ".join(
                f"{blocker.blocker_id}: {blocker.reason}" for blocker in coupon_blockers
            )
            raise StaticLegLoweringNoMatchError(details)
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
    required_capabilities = chosen.required_capabilities
    if chosen.declaration_id == "static_leg_coupon_obligations":
        required_capabilities = _coupon_obligation_required_capabilities(contract_ir)
    return StaticLegLoweringSelection(
        declaration_id=chosen.declaration_id,
        callable_ref=chosen.callable_ref,
        adapter_ref=chosen.adapter_ref,
        validation_bundle_id=chosen.validation_bundle_id,
        required_capabilities=required_capabilities,
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
    "StaticLegAdmissionBlocker",
    "StaticLegLoweringDeclaration",
    "StaticLegLoweringNoMatchError",
    "StaticLegLoweringSelection",
    "conditional_range_accrual_admission_blockers",
    "default_static_leg_lowering_declarations",
    "materialize_static_leg_lowering",
    "select_static_leg_lowering",
    "static_coupon_obligation_admission_blockers",
]
