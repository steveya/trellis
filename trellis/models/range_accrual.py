"""Deterministic range-accrual pricing helpers for the first checked desk route."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Mapping

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.curves.forward_curve import ForwardCurve
from trellis.models.contingent_cashflows import (
    CouponAccrual,
    PrincipalPayment,
    coupon_cashflow_pv,
    principal_payment_pv,
)

ROUTE_ID = "range_accrual_discounted_cashflow_v1"
DEFAULT_SCENARIO_SHIFTS_BPS = (-100.0, -50.0, 50.0, 100.0)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _string_tuple(values) -> tuple[str, ...]:
    if not values:
        return ()
    items: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in items:
            items.append(text)
    return tuple(items)


def _normalize_date_tuple(values, *, field_name: str) -> tuple[date, ...]:
    if not values:
        return ()
    normalized: list[date] = []
    for value in values:
        if isinstance(value, date):
            normalized.append(value)
            continue
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"Range accrual {field_name} contains an empty date.")
        normalized.append(date.fromisoformat(text))
    return tuple(normalized)


def _normalize_fixing_history(fixing_history) -> Mapping[date, float]:
    if not fixing_history:
        return MappingProxyType({})
    if isinstance(fixing_history, Mapping):
        normalized = {}
        for key, value in fixing_history.items():
            fixing_date = key if isinstance(key, date) else date.fromisoformat(str(key).strip())
            normalized[fixing_date] = float(value)
        return MappingProxyType(normalized)
    normalized = {}
    for item in fixing_history:
        fixing_date = item.get("date")
        fixing_value = item.get("value")
        if fixing_date in {None, ""}:
            continue
        normalized[date.fromisoformat(str(fixing_date).strip())] = float(fixing_value)
    return MappingProxyType(normalized)


def _inferred_accrual_starts(observation_dates: tuple[date, ...]) -> tuple[date, ...]:
    if len(observation_dates) == 1:
        raise ValueError(
            "Range accrual pricing requires `accrual_start_dates` when only one observation date is provided."
        )
    first_gap = observation_dates[1] - observation_dates[0]
    return (observation_dates[0] - first_gap, *observation_dates[:-1])


def _is_strictly_increasing(values: tuple[date, ...]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def _in_range(rate: float, *, lower_bound: float, upper_bound: float, inclusive_lower: bool, inclusive_upper: bool) -> bool:
    lower_ok = rate >= lower_bound if inclusive_lower else rate > lower_bound
    upper_ok = rate <= upper_bound if inclusive_upper else rate < upper_bound
    return lower_ok and upper_ok


@dataclass(frozen=True)
class RangeAccrualSpec:
    """Typed deterministic range-accrual pricing inputs."""

    reference_index: str
    notional: float
    coupon_rate: float
    lower_bound: float
    upper_bound: float
    observation_dates: tuple[date, ...]
    accrual_start_dates: tuple[date, ...] = ()
    payment_dates: tuple[date, ...] = ()
    principal_redemption: float = 1.0
    inclusive_lower: bool = True
    inclusive_upper: bool = True
    day_count: DayCountConvention = DayCountConvention.ACT_365
    assumptions: tuple[str, ...] = ()

    def __post_init__(self):
        reference_index = str(self.reference_index or "").strip().upper()
        observation_dates = _normalize_date_tuple(self.observation_dates, field_name="observation_dates")
        if not observation_dates:
            raise ValueError("Range accrual pricing requires at least one observation date.")
        if not _is_strictly_increasing(observation_dates):
            raise ValueError("Range accrual observation_dates must be strictly increasing.")

        accrual_start_dates = _normalize_date_tuple(self.accrual_start_dates, field_name="accrual_start_dates")
        payment_dates = _normalize_date_tuple(self.payment_dates, field_name="payment_dates")
        assumptions = list(_string_tuple(self.assumptions))

        if not reference_index:
            raise ValueError("Range accrual pricing requires a reference_index.")
        if float(self.notional) <= 0.0:
            raise ValueError("Range accrual pricing requires a positive notional.")
        if float(self.lower_bound) > float(self.upper_bound):
            raise ValueError("Range accrual pricing requires lower_bound <= upper_bound.")

        if not accrual_start_dates:
            accrual_start_dates = _inferred_accrual_starts(observation_dates)
            assumptions.append(
                "Inferred accrual_start_dates from the observation cadence because no explicit accrual schedule was provided."
            )
        if len(accrual_start_dates) != len(observation_dates):
            raise ValueError("Range accrual pricing requires one accrual_start_date per observation date.")
        if not payment_dates:
            payment_dates = observation_dates
            assumptions.append(
                "Defaulted payment_dates to observation_dates because no explicit payment schedule was provided."
            )
        if len(payment_dates) != len(observation_dates):
            raise ValueError("Range accrual pricing requires one payment_date per observation date.")
        if not _is_strictly_increasing(payment_dates):
            raise ValueError("Range accrual payment_dates must be strictly increasing.")
        if any(start >= end for start, end in zip(accrual_start_dates, observation_dates)):
            raise ValueError("Range accrual accrual_start_dates must be earlier than observation_dates.")
        if any(payment < observation for payment, observation in zip(payment_dates, observation_dates)):
            raise ValueError("Range accrual payment_dates must be on or after the observation_dates.")

        object.__setattr__(self, "reference_index", reference_index)
        object.__setattr__(self, "notional", float(self.notional))
        object.__setattr__(self, "coupon_rate", float(self.coupon_rate))
        object.__setattr__(self, "lower_bound", float(self.lower_bound))
        object.__setattr__(self, "upper_bound", float(self.upper_bound))
        object.__setattr__(self, "observation_dates", observation_dates)
        object.__setattr__(self, "accrual_start_dates", accrual_start_dates)
        object.__setattr__(self, "payment_dates", payment_dates)
        object.__setattr__(self, "principal_redemption", float(self.principal_redemption))
        object.__setattr__(self, "assumptions", tuple(assumptions))


@dataclass(frozen=True)
class RangeAccrualScenario:
    """One scenario ladder point."""

    name: str
    shift_bps: float
    price: float
    pnl: float

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name or "").strip())
        object.__setattr__(self, "shift_bps", float(self.shift_bps))
        object.__setattr__(self, "price", float(self.price))
        object.__setattr__(self, "pnl", float(self.pnl))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shift_bps": self.shift_bps,
            "price": self.price,
            "pnl": self.pnl,
        }


@dataclass(frozen=True)
class RangeAccrualRisk:
    """First risk summary for the checked route."""

    parallel_curve_pv01: float

    def __post_init__(self):
        object.__setattr__(self, "parallel_curve_pv01", float(self.parallel_curve_pv01))

    def to_dict(self) -> dict[str, object]:
        return {
            "parallel_curve_pv01": self.parallel_curve_pv01,
        }


@dataclass(frozen=True)
class RangeAccrualValidationCheck:
    """One deterministic validation or reference check."""

    check_id: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "check_id", str(self.check_id or "").strip())
        object.__setattr__(self, "status", str(self.status or "").strip())
        object.__setattr__(self, "details", _freeze_mapping(self.details))

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class RangeAccrualValidationBundle:
    """Deterministic validation output for the checked route."""

    route_id: str
    checks: tuple[RangeAccrualValidationCheck, ...]
    reference_metrics: Mapping[str, object] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "route_id", str(self.route_id or "").strip())
        object.__setattr__(self, "checks", tuple(self.checks or ()))
        object.__setattr__(self, "reference_metrics", _freeze_mapping(self.reference_metrics))
        object.__setattr__(self, "assumptions", _string_tuple(self.assumptions))

    def to_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "checks": [check.to_dict() for check in self.checks],
            "reference_metrics": dict(self.reference_metrics),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class RangeAccrualResult:
    """Structured checked-route result."""

    price: float
    coupon_leg_pv: float
    principal_leg_pv: float
    observed_coupon_count: int
    projected_coupon_count: int
    risk: RangeAccrualRisk
    scenarios: tuple[RangeAccrualScenario, ...]
    validation_bundle: RangeAccrualValidationBundle

    def __post_init__(self):
        object.__setattr__(self, "price", float(self.price))
        object.__setattr__(self, "coupon_leg_pv", float(self.coupon_leg_pv))
        object.__setattr__(self, "principal_leg_pv", float(self.principal_leg_pv))
        object.__setattr__(self, "observed_coupon_count", int(self.observed_coupon_count))
        object.__setattr__(self, "projected_coupon_count", int(self.projected_coupon_count))
        object.__setattr__(self, "scenarios", tuple(self.scenarios or ()))

    def to_dict(self) -> dict[str, object]:
        return {
            "price": self.price,
            "coupon_leg_pv": self.coupon_leg_pv,
            "principal_leg_pv": self.principal_leg_pv,
            "observed_coupon_count": self.observed_coupon_count,
            "projected_coupon_count": self.projected_coupon_count,
            "risk": self.risk.to_dict(),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "validation_bundle": self.validation_bundle.to_dict(),
        }


def _shift_curve(curve, shift_bps: float):
    if not hasattr(curve, "shift"):
        raise ValueError("Range accrual scenario ladder requires shiftable curves.")
    return curve.shift(float(shift_bps))


def _base_projection(
    spec: RangeAccrualSpec,
    *,
    as_of: date,
    discount_curve,
    forecast_curve,
    fixing_history: Mapping[date, float],
) -> dict[str, object]:
    normalized_fixings = _normalize_fixing_history(fixing_history)
    missing_fixings = [
        observation_date.isoformat()
        for observation_date in spec.observation_dates
        if observation_date <= as_of and observation_date not in normalized_fixings
    ]
    if missing_fixings:
        raise ValueError(
            "Missing fixing history for observed coupon dates: "
            + ", ".join(missing_fixings)
        )

    forward_curve = ForwardCurve(forecast_curve)
    coupon_leg_pv = 0.0
    max_coupon_leg_pv = 0.0
    observed_coupon_count = 0
    projected_coupon_count = 0

    for accrual_start, observation_date, payment_date in zip(
        spec.accrual_start_dates,
        spec.observation_dates,
        spec.payment_dates,
    ):
        if observation_date <= as_of:
            observed_coupon_count += 1
            observed_rate = float(normalized_fixings[observation_date])
        else:
            projected_coupon_count += 1
            observation_time = year_fraction(as_of, observation_date, spec.day_count)
            observed_rate = forward_curve.forward_rate(0.0, observation_time, compounding="simple")

        if payment_date <= as_of:
            continue

        payment_time = year_fraction(as_of, payment_date, spec.day_count)
        discount_factor = float(discount_curve.discount(payment_time))
        accrual = year_fraction(accrual_start, observation_date, spec.day_count)
        coupon = CouponAccrual(
            notional=spec.notional,
            rate=spec.coupon_rate,
            accrual=accrual,
            discount_factor=discount_factor,
            weight=1.0 if _in_range(
                observed_rate,
                lower_bound=spec.lower_bound,
                upper_bound=spec.upper_bound,
                inclusive_lower=spec.inclusive_lower,
                inclusive_upper=spec.inclusive_upper,
            ) else 0.0,
        )
        coupon_leg_pv += coupon_cashflow_pv(coupon)
        max_coupon_leg_pv += coupon_cashflow_pv(
            CouponAccrual(
                notional=spec.notional,
                rate=spec.coupon_rate,
                accrual=accrual,
                discount_factor=discount_factor,
                weight=1.0,
            )
        )

    principal_leg_pv = 0.0
    maturity_date = spec.payment_dates[-1]
    if maturity_date > as_of:
        maturity_time = year_fraction(as_of, maturity_date, spec.day_count)
        principal_leg_pv = principal_payment_pv(
            PrincipalPayment(
                scheduled_principal=spec.notional * spec.principal_redemption,
                discount_factor=float(discount_curve.discount(maturity_time)),
            )
        )

    return {
        "coupon_leg_pv": float(coupon_leg_pv),
        "principal_leg_pv": float(principal_leg_pv),
        "price": float(coupon_leg_pv + principal_leg_pv),
        "observed_coupon_count": observed_coupon_count,
        "projected_coupon_count": projected_coupon_count,
        "required_fixings": observed_coupon_count,
        "available_fixings": len(normalized_fixings),
        "max_coupon_leg_pv": float(max_coupon_leg_pv),
    }


def price_range_accrual(
    spec: RangeAccrualSpec,
    *,
    as_of: date,
    discount_curve,
    forecast_curve,
    fixing_history=None,
    scenario_shifts_bps: tuple[float, ...] = DEFAULT_SCENARIO_SHIFTS_BPS,
    assumptions=(),
) -> RangeAccrualResult:
    """Price one checked range-accrual trade with deterministic scenario and validation output."""
    base = _base_projection(
        spec,
        as_of=as_of,
        discount_curve=discount_curve,
        forecast_curve=forecast_curve,
        fixing_history=fixing_history,
    )

    price = float(base["price"])
    price_up_1bp = _base_projection(
        spec,
        as_of=as_of,
        discount_curve=_shift_curve(discount_curve, 1.0),
        forecast_curve=_shift_curve(forecast_curve, 1.0),
        fixing_history=fixing_history,
    )["price"]
    scenarios = tuple(
        RangeAccrualScenario(
            name=f"rates_{'up' if shift_bps > 0 else 'down'}_{abs(int(shift_bps))}bp",
            shift_bps=shift_bps,
            price=_base_projection(
                spec,
                as_of=as_of,
                discount_curve=_shift_curve(discount_curve, shift_bps),
                forecast_curve=_shift_curve(forecast_curve, shift_bps),
                fixing_history=fixing_history,
            )["price"],
            pnl=_base_projection(
                spec,
                as_of=as_of,
                discount_curve=_shift_curve(discount_curve, shift_bps),
                forecast_curve=_shift_curve(forecast_curve, shift_bps),
                fixing_history=fixing_history,
            )["price"]
            - price,
        )
        for shift_bps in scenario_shifts_bps
    )

    validation_bundle = RangeAccrualValidationBundle(
        route_id=ROUTE_ID,
        checks=(
            RangeAccrualValidationCheck(
                check_id="schedule_monotonic",
                status="passed",
                details={
                    "observation_count": len(spec.observation_dates),
                    "payment_count": len(spec.payment_dates),
                },
            ),
            RangeAccrualValidationCheck(
                check_id="historical_fixing_coverage",
                status="passed",
                details={
                    "required_fixings": int(base["required_fixings"]),
                    "available_fixings": int(base["available_fixings"]),
                },
            ),
            RangeAccrualValidationCheck(
                check_id="coupon_leg_reference_bound",
                status="passed" if base["coupon_leg_pv"] <= base["max_coupon_leg_pv"] + 1e-10 else "failed",
                details={
                    "coupon_leg_pv": float(base["coupon_leg_pv"]),
                    "max_coupon_leg_pv": float(base["max_coupon_leg_pv"]),
                },
            ),
            RangeAccrualValidationCheck(
                check_id="pv_reconciliation",
                status="passed" if abs(price - (base["coupon_leg_pv"] + base["principal_leg_pv"])) <= 1e-10 else "failed",
                details={
                    "price": price,
                    "coupon_leg_pv": float(base["coupon_leg_pv"]),
                    "principal_leg_pv": float(base["principal_leg_pv"]),
                },
            ),
        ),
        reference_metrics={
            "max_coupon_leg_pv": float(base["max_coupon_leg_pv"]),
        },
        assumptions=(*spec.assumptions, *_string_tuple(assumptions)),
    )

    return RangeAccrualResult(
        price=price,
        coupon_leg_pv=float(base["coupon_leg_pv"]),
        principal_leg_pv=float(base["principal_leg_pv"]),
        observed_coupon_count=int(base["observed_coupon_count"]),
        projected_coupon_count=int(base["projected_coupon_count"]),
        risk=RangeAccrualRisk(
            parallel_curve_pv01=price - float(price_up_1bp),
        ),
        scenarios=scenarios,
        validation_bundle=validation_bundle,
    )


__all__ = [
    "DEFAULT_SCENARIO_SHIFTS_BPS",
    "ROUTE_ID",
    "RangeAccrualResult",
    "RangeAccrualRisk",
    "RangeAccrualScenario",
    "RangeAccrualSpec",
    "RangeAccrualValidationBundle",
    "RangeAccrualValidationCheck",
    "price_range_accrual",
]
