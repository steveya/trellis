"""Bounded static leg-based contract semantics.

This module is the first post-Phase-4 static-leg semantic surface. It is
deliberately narrow: it captures static coupon and cashflow obligations for
plain swaps and coupon bonds without callability, interruption, or target
state. Dynamic wrappers belong in the later event/state/control layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Mapping

from trellis.agent.contract_ir import CurveQuote, SurfaceQuote


class StaticLegIRWellFormednessError(ValueError):
    """Raised when a static leg contract violates a local semantic invariant."""


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _require_text(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise StaticLegIRWellFormednessError(f"{label} must be a non-empty string")
    return text


def _as_float(value: float | int, *, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive only
        raise StaticLegIRWellFormednessError(f"{label} must be numeric") from exc


@dataclass(frozen=True)
class NotionalStep:
    start_date: date
    end_date: date
    amount: float

    def __post_init__(self) -> None:
        if not isinstance(self.start_date, date) or not isinstance(self.end_date, date):
            raise StaticLegIRWellFormednessError(
                "NotionalStep requires date boundaries"
            )
        if self.start_date >= self.end_date:
            raise StaticLegIRWellFormednessError(
                "NotionalStep requires start_date < end_date"
            )
        object.__setattr__(self, "amount", _as_float(self.amount, label="NotionalStep.amount"))


@dataclass(frozen=True)
class NotionalSchedule:
    steps: tuple[NotionalStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise StaticLegIRWellFormednessError("NotionalSchedule must be non-empty")
        previous_end: date | None = None
        for step in self.steps:
            if not isinstance(step, NotionalStep):
                raise StaticLegIRWellFormednessError(
                    "NotionalSchedule.steps must contain NotionalStep values"
                )
            if previous_end is not None and step.start_date < previous_end:
                raise StaticLegIRWellFormednessError(
                    "NotionalSchedule steps must be non-overlapping and ordered"
                )
            previous_end = step.end_date

    @property
    def initial_notional(self) -> float:
        return float(self.steps[0].amount)


@dataclass(frozen=True)
class TermRateIndex:
    name: str
    tenor: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, label="TermRateIndex.name"))
        object.__setattr__(self, "tenor", _require_text(self.tenor, label="TermRateIndex.tenor"))


@dataclass(frozen=True)
class OvernightRateIndex:
    name: str
    compounding: str = "simple"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _require_text(self.name, label="OvernightRateIndex.name"),
        )
        object.__setattr__(
            self,
            "compounding",
            _require_text(
                self.compounding,
                label="OvernightRateIndex.compounding",
            ).lower(),
        )


@dataclass(frozen=True)
class CmsRateIndex:
    curve_id: str
    tenor: str
    convention: str = "par_rate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "curve_id", _require_text(self.curve_id, label="CmsRateIndex.curve_id"))
        object.__setattr__(self, "tenor", _require_text(self.tenor, label="CmsRateIndex.tenor"))
        object.__setattr__(
            self,
            "convention",
            _require_text(self.convention, label="CmsRateIndex.convention").lower(),
        )


RateIndex = TermRateIndex | OvernightRateIndex | CmsRateIndex


@dataclass(frozen=True)
class FixedCouponFormula:
    rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate", _as_float(self.rate, label="FixedCouponFormula.rate"))


@dataclass(frozen=True)
class FloatingCouponFormula:
    rate_index: RateIndex
    spread: float = 0.0
    gearing: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.rate_index, (TermRateIndex, OvernightRateIndex, CmsRateIndex)):
            raise StaticLegIRWellFormednessError(
                "FloatingCouponFormula.rate_index must be a RateIndex"
            )
        object.__setattr__(self, "spread", _as_float(self.spread, label="FloatingCouponFormula.spread"))
        object.__setattr__(self, "gearing", _as_float(self.gearing, label="FloatingCouponFormula.gearing"))


@dataclass(frozen=True)
class QuotedCouponFormula:
    observable: CurveQuote | SurfaceQuote
    spread: float = 0.0
    gearing: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.observable, (CurveQuote, SurfaceQuote)):
            raise StaticLegIRWellFormednessError(
                "QuotedCouponFormula.observable must be a CurveQuote or SurfaceQuote"
            )
        object.__setattr__(self, "spread", _as_float(self.spread, label="QuotedCouponFormula.spread"))
        object.__setattr__(self, "gearing", _as_float(self.gearing, label="QuotedCouponFormula.gearing"))


CouponFormula = FixedCouponFormula | FloatingCouponFormula | QuotedCouponFormula


@dataclass(frozen=True)
class CouponPeriod:
    accrual_start: date
    accrual_end: date
    payment_date: date
    fixing_date: date | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(item, date) for item in (self.accrual_start, self.accrual_end, self.payment_date)):
            raise StaticLegIRWellFormednessError("CouponPeriod requires date boundaries")
        if self.accrual_start >= self.accrual_end:
            raise StaticLegIRWellFormednessError(
                "CouponPeriod requires accrual_start < accrual_end"
            )
        if self.payment_date < self.accrual_end:
            raise StaticLegIRWellFormednessError(
                "CouponPeriod payment_date must be on or after accrual_end"
            )
        if self.fixing_date is not None and not isinstance(self.fixing_date, date):
            raise StaticLegIRWellFormednessError(
                "CouponPeriod.fixing_date must be a date when present"
            )


@dataclass(frozen=True)
class PeriodRateOptionPeriod:
    accrual_start: date
    accrual_end: date
    fixing_date: date
    payment_date: date

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, date)
            for item in (
                self.accrual_start,
                self.accrual_end,
                self.fixing_date,
                self.payment_date,
            )
        ):
            raise StaticLegIRWellFormednessError(
                "PeriodRateOptionPeriod requires date boundaries"
            )
        if self.accrual_start >= self.accrual_end:
            raise StaticLegIRWellFormednessError(
                "PeriodRateOptionPeriod requires accrual_start < accrual_end"
            )
        if self.payment_date < self.accrual_end:
            raise StaticLegIRWellFormednessError(
                "PeriodRateOptionPeriod payment_date must be on or after accrual_end"
            )
        if self.fixing_date > self.payment_date:
            raise StaticLegIRWellFormednessError(
                "PeriodRateOptionPeriod fixing_date must be on or before payment_date"
            )


@dataclass(frozen=True)
class CouponLeg:
    currency: str
    notional_schedule: NotionalSchedule
    coupon_periods: tuple[CouponPeriod, ...]
    coupon_formula: CouponFormula
    day_count: str
    payment_frequency: str
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", _require_text(self.currency, label="CouponLeg.currency").upper())
        if not isinstance(self.notional_schedule, NotionalSchedule):
            raise StaticLegIRWellFormednessError(
                "CouponLeg.notional_schedule must be a NotionalSchedule"
            )
        if not isinstance(self.coupon_periods, tuple):
            object.__setattr__(self, "coupon_periods", tuple(self.coupon_periods))
        if not self.coupon_periods:
            raise StaticLegIRWellFormednessError("CouponLeg.coupon_periods must be non-empty")
        previous_end: date | None = None
        for period in self.coupon_periods:
            if not isinstance(period, CouponPeriod):
                raise StaticLegIRWellFormednessError(
                    "CouponLeg.coupon_periods must contain CouponPeriod values"
                )
            if previous_end is not None and period.accrual_start < previous_end:
                raise StaticLegIRWellFormednessError(
                    "CouponLeg periods must be ordered and non-overlapping"
                )
            previous_end = period.accrual_end
        if not isinstance(
            self.coupon_formula,
            (FixedCouponFormula, FloatingCouponFormula, QuotedCouponFormula),
        ):
            raise StaticLegIRWellFormednessError(
                "CouponLeg.coupon_formula must be a CouponFormula"
            )
        object.__setattr__(self, "day_count", _require_text(self.day_count, label="CouponLeg.day_count"))
        object.__setattr__(
            self,
            "payment_frequency",
            _require_text(self.payment_frequency, label="CouponLeg.payment_frequency").lower(),
        )
        object.__setattr__(self, "label", str(self.label or "").strip())


@dataclass(frozen=True)
class PeriodRateOptionStripLeg:
    currency: str
    notional_schedule: NotionalSchedule
    option_periods: tuple[PeriodRateOptionPeriod, ...]
    rate_index: RateIndex
    strike: float
    option_side: str
    day_count: str
    payment_frequency: str
    label: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "currency",
            _require_text(self.currency, label="PeriodRateOptionStripLeg.currency").upper(),
        )
        if not isinstance(self.notional_schedule, NotionalSchedule):
            raise StaticLegIRWellFormednessError(
                "PeriodRateOptionStripLeg.notional_schedule must be a NotionalSchedule"
            )
        if not isinstance(self.option_periods, tuple):
            object.__setattr__(self, "option_periods", tuple(self.option_periods))
        if not self.option_periods:
            raise StaticLegIRWellFormednessError(
                "PeriodRateOptionStripLeg.option_periods must be non-empty"
            )
        previous_end: date | None = None
        for period in self.option_periods:
            if not isinstance(period, PeriodRateOptionPeriod):
                raise StaticLegIRWellFormednessError(
                    "PeriodRateOptionStripLeg.option_periods must contain PeriodRateOptionPeriod values"
                )
            if previous_end is not None and period.accrual_start < previous_end:
                raise StaticLegIRWellFormednessError(
                    "PeriodRateOptionStripLeg periods must be ordered and non-overlapping"
                )
            previous_end = period.accrual_end
        if not isinstance(
            self.rate_index,
            (TermRateIndex, OvernightRateIndex, CmsRateIndex),
        ):
            raise StaticLegIRWellFormednessError(
                "PeriodRateOptionStripLeg.rate_index must be a RateIndex"
            )
        object.__setattr__(
            self,
            "strike",
            _as_float(self.strike, label="PeriodRateOptionStripLeg.strike"),
        )
        normalized_side = _require_text(
            self.option_side,
            label="PeriodRateOptionStripLeg.option_side",
        ).lower()
        if normalized_side not in {"call", "put"}:
            raise StaticLegIRWellFormednessError(
                "PeriodRateOptionStripLeg.option_side must be 'call' or 'put'"
            )
        object.__setattr__(self, "option_side", normalized_side)
        object.__setattr__(
            self,
            "day_count",
            _require_text(self.day_count, label="PeriodRateOptionStripLeg.day_count"),
        )
        object.__setattr__(
            self,
            "payment_frequency",
            _require_text(
                self.payment_frequency,
                label="PeriodRateOptionStripLeg.payment_frequency",
            ).lower(),
        )
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class KnownCashflow:
    payment_date: date
    amount: float
    currency: str
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.payment_date, date):
            raise StaticLegIRWellFormednessError(
                "KnownCashflow.payment_date must be a date"
            )
        object.__setattr__(self, "amount", _as_float(self.amount, label="KnownCashflow.amount"))
        object.__setattr__(self, "currency", _require_text(self.currency, label="KnownCashflow.currency").upper())
        object.__setattr__(self, "label", str(self.label or "").strip())


@dataclass(frozen=True)
class KnownCashflowLeg:
    currency: str
    cashflows: tuple[KnownCashflow, ...]
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", _require_text(self.currency, label="KnownCashflowLeg.currency").upper())
        if not isinstance(self.cashflows, tuple):
            object.__setattr__(self, "cashflows", tuple(self.cashflows))
        if not self.cashflows:
            raise StaticLegIRWellFormednessError("KnownCashflowLeg.cashflows must be non-empty")
        previous_date: date | None = None
        for cashflow in self.cashflows:
            if not isinstance(cashflow, KnownCashflow):
                raise StaticLegIRWellFormednessError(
                    "KnownCashflowLeg.cashflows must contain KnownCashflow values"
                )
            if previous_date is not None and cashflow.payment_date < previous_date:
                raise StaticLegIRWellFormednessError(
                    "KnownCashflowLeg cashflows must be ordered"
                )
            previous_date = cashflow.payment_date
        object.__setattr__(self, "label", str(self.label or "").strip())


Leg = CouponLeg | PeriodRateOptionStripLeg | KnownCashflowLeg


@dataclass(frozen=True)
class SignedLeg:
    direction: str
    leg: Leg

    def __post_init__(self) -> None:
        normalized = _require_text(self.direction, label="SignedLeg.direction").lower()
        if normalized not in {"receive", "pay"}:
            raise StaticLegIRWellFormednessError(
                "SignedLeg.direction must be 'receive' or 'pay'"
            )
        object.__setattr__(self, "direction", normalized)
        if not isinstance(self.leg, (CouponLeg, PeriodRateOptionStripLeg, KnownCashflowLeg)):
            raise StaticLegIRWellFormednessError(
                "SignedLeg.leg must be a CouponLeg, PeriodRateOptionStripLeg, or KnownCashflowLeg"
            )


@dataclass(frozen=True)
class SettlementRule:
    settlement_kind: str = "cash"
    payout_currency: str = ""
    settlement_lag_days: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "settlement_kind",
            _require_text(self.settlement_kind, label="SettlementRule.settlement_kind").lower(),
        )
        object.__setattr__(self, "payout_currency", str(self.payout_currency or "").strip().upper())
        object.__setattr__(self, "settlement_lag_days", int(self.settlement_lag_days))
        if self.settlement_lag_days < 0:
            raise StaticLegIRWellFormednessError(
                "SettlementRule.settlement_lag_days must be non-negative"
            )


@dataclass(frozen=True)
class StaticLegContractIR:
    legs: tuple[SignedLeg, ...]
    settlement: SettlementRule = field(default_factory=SettlementRule)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.legs, tuple):
            object.__setattr__(self, "legs", tuple(self.legs))
        if not self.legs:
            raise StaticLegIRWellFormednessError("StaticLegContractIR.legs must be non-empty")
        for leg in self.legs:
            if not isinstance(leg, SignedLeg):
                raise StaticLegIRWellFormednessError(
                    "StaticLegContractIR.legs must contain SignedLeg values"
                )
        if not isinstance(self.settlement, SettlementRule):
            raise StaticLegIRWellFormednessError(
                "StaticLegContractIR.settlement must be a SettlementRule"
            )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def currencies(self) -> tuple[str, ...]:
        seen: list[str] = []
        for leg in self.legs:
            currency = getattr(leg.leg, "currency", "")
            if currency and currency not in seen:
                seen.append(currency)
        if self.settlement.payout_currency and self.settlement.payout_currency not in seen:
            seen.append(self.settlement.payout_currency)
        return tuple(seen)


__all__ = [
    "CmsRateIndex",
    "CouponFormula",
    "CouponLeg",
    "CouponPeriod",
    "FixedCouponFormula",
    "FloatingCouponFormula",
    "KnownCashflow",
    "KnownCashflowLeg",
    "Leg",
    "NotionalSchedule",
    "NotionalStep",
    "OvernightRateIndex",
    "PeriodRateOptionPeriod",
    "PeriodRateOptionStripLeg",
    "QuotedCouponFormula",
    "RateIndex",
    "SettlementRule",
    "SignedLeg",
    "StaticLegContractIR",
    "StaticLegIRWellFormednessError",
    "TermRateIndex",
]
