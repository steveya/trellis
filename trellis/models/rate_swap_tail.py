"""Convention-aware swap-tail primitives for physical Bermudan swaptions.

The projection contract in this module is deliberately bounded.  A separate,
named forecast curve supplies each time-zero simple forward and a named
discount curve supplies the corresponding time-zero discount forward.  Their
difference is held as a deterministic additive basis on every short-rate-tree
node.  The basis is therefore static: this module does not model stochastic
basis, reset/payment convexity, compounded overnight coupons, amortization, or
cash settlement.  It also rejects adjusted fixed accrual starts before exercise
because accrued settlement for an already-started fixed coupon is not modeled.

The module returns node values for the generic lattice rollback.  It does not
delegate to the legacy Bermudan helper or to a European/Black fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from collections.abc import Mapping
from typing import Protocol, TypeVar

from trellis.conventions.calendar import (
    BRAZIL,
    SYDNEY,
    TARGET,
    TOKYO,
    TORONTO,
    UK_SETTLEMENT,
    US_SETTLEMENT,
    WEEKEND_ONLY,
    ZURICH,
    BusinessDayAdjustment,
)
from trellis.conventions.day_count import DayCountConvention, year_fraction
from trellis.conventions.schedule import (
    RollConvention,
    StubType,
    generate_schedule,
)
from trellis.core.types import Frequency
from trellis.models.trees.lattice import (
    RecombiningLattice,
    lattice_backward_induction,
    lattice_backward_induction_result,
)


class RateCurveLike(Protocol):
    """Minimal curve surface used by the static-basis swap-tail kernel."""

    def discount(self, time: float) -> float:
        """Return the time-zero discount factor at ``time``."""
        ...


class CalendarLike(Protocol):
    """Calendar behavior required by schedule and model-time resolution."""

    def adjust(
        self,
        target: date,
        convention: BusinessDayAdjustment,
    ) -> date:
        """Adjust ``target`` under one business-day convention."""
        ...

    def add_business_days(self, target: date, count: int) -> date:
        """Shift ``target`` by an exact number of business days."""
        ...

    def business_days_between(self, start: date, end: date) -> int:
        """Count business days for BUS/252 model or coupon time."""
        ...


def _validate_calendar(calendar: object, *, name: str) -> None:
    required = ("adjust", "add_business_days", "business_days_between")
    missing = tuple(method for method in required if not callable(getattr(calendar, method, None)))
    if missing:
        raise TypeError(f"{name} must provide calendar operations: {', '.join(missing)}")


def _validate_exact_non_blank_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be an exact non-blank string without surrounding whitespace")
    return value


@dataclass(frozen=True)
class NamedRateCurve:
    """One exact curve binding; the name is checked against the contract."""

    name: str
    curve: RateCurveLike

    def __post_init__(self) -> None:
        _validate_exact_non_blank_string(self.name, name="named rate curve name")
        if not callable(getattr(self.curve, "discount", None)):
            raise TypeError("named rate curve must provide discount(time)")


@dataclass(frozen=True)
class ExerciseSwapStart:
    """Explicit mapping from one exercise date to its underlying swap start."""

    exercise_date: date
    swap_start_date: date

    def __post_init__(self) -> None:
        if self.swap_start_date < self.exercise_date:
            raise ValueError("swap start date cannot precede its exercise date")


def _validate_business_day_lag(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class FixedLegConvention:
    """Complete convention bundle for the fixed leg of every swap tail."""

    frequency: Frequency
    day_count: DayCountConvention
    calendar: CalendarLike
    business_day_adjustment: BusinessDayAdjustment
    stub_type: StubType
    roll_convention: RollConvention
    payment_lag_business_days: int

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, Frequency):
            raise TypeError("fixed frequency must be a Frequency")
        if not isinstance(self.day_count, DayCountConvention):
            raise TypeError("fixed day count must be a DayCountConvention")
        if self.day_count is DayCountConvention.ACT_ACT_ICMA:
            raise ValueError(
                "ACT/ACT ICMA is not supported for fixed coupons without explicit "
                "quasi-coupon reference periods"
            )
        _validate_calendar(self.calendar, name="fixed calendar")
        if not isinstance(self.business_day_adjustment, BusinessDayAdjustment):
            raise TypeError("fixed business-day adjustment must be explicit")
        if not isinstance(self.stub_type, StubType):
            raise TypeError("fixed stub type must be explicit")
        if not isinstance(self.roll_convention, RollConvention):
            raise TypeError("fixed roll convention must be explicit")
        _validate_business_day_lag(
            self.payment_lag_business_days,
            name="fixed payment business-day lag",
        )


@dataclass(frozen=True)
class FloatingLegConvention:
    """Complete simple-coupon convention bundle for the floating leg."""

    frequency: Frequency
    day_count: DayCountConvention
    calendar: CalendarLike
    business_day_adjustment: BusinessDayAdjustment
    stub_type: StubType
    roll_convention: RollConvention
    reset_lag_business_days: int
    fixing_lag_business_days: int
    payment_lag_business_days: int
    rate_index: str
    compounding: str
    gearing: float
    spread: float

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, Frequency):
            raise TypeError("floating frequency must be a Frequency")
        if not isinstance(self.day_count, DayCountConvention):
            raise TypeError("floating day count must be a DayCountConvention")
        if self.day_count is DayCountConvention.ACT_ACT_ICMA:
            raise ValueError(
                "ACT/ACT ICMA is not supported for floating coupons without explicit "
                "quasi-coupon reference periods"
            )
        _validate_calendar(self.calendar, name="floating calendar")
        if not isinstance(self.business_day_adjustment, BusinessDayAdjustment):
            raise TypeError("floating business-day adjustment must be explicit")
        if not isinstance(self.stub_type, StubType):
            raise TypeError("floating stub type must be explicit")
        if not isinstance(self.roll_convention, RollConvention):
            raise TypeError("floating roll convention must be explicit")
        _validate_business_day_lag(
            self.reset_lag_business_days,
            name="floating reset business-day lag",
        )
        _validate_business_day_lag(
            self.fixing_lag_business_days,
            name="floating fixing business-day lag",
        )
        _validate_business_day_lag(
            self.payment_lag_business_days,
            name="floating payment business-day lag",
        )
        _validate_exact_non_blank_string(
            self.rate_index,
            name="floating rate index",
        )
        if self.compounding != "simple":
            raise ValueError("swap-tail floating compounding must be 'simple'")
        if not isfinite(float(self.gearing)):
            raise ValueError("floating gearing must be finite")
        if not isfinite(float(self.spread)):
            raise ValueError("floating spread must be finite")


@dataclass(frozen=True)
class PhysicalBermudanSwapTailSpec:
    """Strict physical Bermudan contract consumed by the swap-tail kernel."""

    valuation_date: date
    exercise_swap_starts: tuple[ExerciseSwapStart, ...]
    common_maturity_date: date
    notional: float
    fixed_rate: float
    currency: str
    option_side: str
    settlement_style: str
    fixed_convention: FixedLegConvention
    floating_convention: FloatingLegConvention
    discount_curve_name: str
    forecast_curve_name: str
    projection_policy: str
    model_day_count: DayCountConvention
    model_time_calendar: CalendarLike
    max_lattice_date_error_days: float

    def __post_init__(self) -> None:
        exercises = tuple(self.exercise_swap_starts)
        object.__setattr__(self, "exercise_swap_starts", exercises)
        if not exercises:
            raise ValueError("physical Bermudan swaption requires at least one exercise")
        exercise_dates = tuple(item.exercise_date for item in exercises)
        if exercise_dates != tuple(sorted(set(exercise_dates))):
            raise ValueError("exercise dates must be unique and strictly increasing")
        for exercise in exercises:
            if exercise.exercise_date <= self.valuation_date:
                raise ValueError("exercise dates must follow the valuation date")
            if exercise.swap_start_date >= self.common_maturity_date:
                raise ValueError("every swap start must precede the common maturity")
            fixed_accrual_start = self.fixed_convention.calendar.adjust(
                exercise.swap_start_date,
                self.fixed_convention.business_day_adjustment,
            )
            if fixed_accrual_start < exercise.exercise_date:
                raise ValueError(
                    "fixed accrual start cannot precede exercise; seasoned or "
                    "pre-started fixed swap tails are unsupported"
                )
        if not isfinite(float(self.notional)) or float(self.notional) <= 0.0:
            raise ValueError("swap-tail notional must be finite and positive")
        if not isfinite(float(self.fixed_rate)):
            raise ValueError("swap-tail fixed rate must be finite")
        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isascii()
            or not self.currency.isalpha()
            or self.currency != self.currency.upper()
        ):
            raise ValueError("currency must be exactly three uppercase ASCII letters")
        if self.option_side not in {"payer", "receiver"}:
            raise ValueError("option side must be 'payer' or 'receiver'")
        if self.settlement_style != "physical":
            raise ValueError("swap-tail kernel supports physical settlement only")
        _validate_exact_non_blank_string(
            self.discount_curve_name,
            name="discount curve name",
        )
        _validate_exact_non_blank_string(
            self.forecast_curve_name,
            name="forecast curve name",
        )
        if self.discount_curve_name == self.forecast_curve_name:
            raise ValueError("discount and forecast curve names must be separate")
        if self.projection_policy != "static_additive_forward_basis":
            raise ValueError(
                "projection policy must be 'static_additive_forward_basis'"
            )
        if not isinstance(self.model_day_count, DayCountConvention):
            raise TypeError("model-time day count must be explicit")
        if self.model_day_count is not DayCountConvention.ACT_365:
            raise ValueError(
                "model-time day count must be ACT/365F so distinct dates have an "
                "injective uniform-lattice time coordinate"
            )
        _validate_calendar(self.model_time_calendar, name="model-time calendar")
        max_error = float(self.max_lattice_date_error_days)
        if not isfinite(max_error) or max_error < 0.0:
            raise ValueError("maximum lattice date error must be finite and non-negative")


_FREQUENCIES = {
    "annual": Frequency.ANNUAL,
    "semiannual": Frequency.SEMI_ANNUAL,
    "semi_annual": Frequency.SEMI_ANNUAL,
    "quarterly": Frequency.QUARTERLY,
    "monthly": Frequency.MONTHLY,
}

_DAY_COUNTS = {
    "ACT/360": DayCountConvention.ACT_360,
    "ACT/365": DayCountConvention.ACT_365,
    "ACT/365F": DayCountConvention.ACT_365,
    "ACT/ACT": DayCountConvention.ACT_ACT,
    "ACT/ACT ISDA": DayCountConvention.ACT_ACT_ISDA,
    "ACT/ACT ICMA": DayCountConvention.ACT_ACT_ICMA,
    "30/360": DayCountConvention.THIRTY_360,
    "30E/360": DayCountConvention.THIRTY_E_360,
    "30E/360 ISDA": DayCountConvention.THIRTY_E_360_ISDA,
    "ACT/365.25": DayCountConvention.ACT_365_25,
    "BUS/252": DayCountConvention.BUS_252,
    "1/1": DayCountConvention.ONE_ONE,
}

_MODEL_TIME_DAY_COUNTS = {
    "ACT/365F": DayCountConvention.ACT_365,
}

_BUSINESS_DAY_ADJUSTMENTS = {
    member.value: member for member in BusinessDayAdjustment
}

_STUB_TYPES = {
    "short_first": StubType.SHORT_FIRST,
    "short_initial": StubType.SHORT_FIRST,
    "short_last": StubType.SHORT_LAST,
    "short_final": StubType.SHORT_LAST,
    "long_first": StubType.LONG_FIRST,
    "long_initial": StubType.LONG_FIRST,
    "long_last": StubType.LONG_LAST,
    "long_final": StubType.LONG_LAST,
}

_ROLL_CONVENTIONS = {member.value: member for member in RollConvention}

_BUILT_IN_CALENDARS: dict[str, CalendarLike] = {
    "WeekendOnly": WEEKEND_ONLY,
    "weekend_only": WEEKEND_ONLY,
    "USSettlement": US_SETTLEMENT,
    "UKSettlement": UK_SETTLEMENT,
    "TARGET": TARGET,
    "Tokyo": TOKYO,
    "Sydney": SYDNEY,
    "Toronto": TORONTO,
    "Zurich": ZURICH,
    "Brazil": BRAZIL,
}


def _required_semantic_field(spec: object, name: str):
    if isinstance(spec, Mapping):
        if name not in spec:
            raise ValueError(f"{name} is required")
        return spec[name]
    try:
        return getattr(spec, name)
    except AttributeError:
        raise ValueError(f"{name} is required") from None


_Token = TypeVar("_Token")


def _parse_semantic_token(
    spec: object,
    name: str,
    choices: Mapping[str, _Token],
) -> _Token:
    value = _required_semantic_field(spec, name)
    if not isinstance(value, str) or value not in choices:
        supported = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of: {supported}")
    return choices[value]


def _parse_semantic_date(value: object, *, name: str) -> date:
    if type(value) is date:
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"{name} must be an ISO YYYY-MM-DD date")


def _parse_non_negative_semantic_int(spec: object, name: str) -> int:
    value = _required_semantic_field(spec, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _parse_finite_semantic_float(spec: object, name: str) -> float:
    value = _required_semantic_field(spec, name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _parse_non_blank_semantic_string(spec: object, name: str) -> str:
    value = _required_semantic_field(spec, name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return value


def compile_physical_bermudan_swap_tail_spec(
    spec: object,
    *,
    valuation_date: date,
) -> PhysicalBermudanSwapTailSpec:
    """Compile an authored semantic spec into the strict typed pricing contract.

    Token matching is deliberately exact.  Only the aliases listed by this
    function are accepted, so an unfamiliar calendar or convention cannot be
    silently replaced with a nearby library default.
    """

    if type(valuation_date) is not date:
        raise ValueError("valuation_date must be a date")

    authored_exercises_value = _required_semantic_field(spec, "exercise_dates")
    if not isinstance(authored_exercises_value, (tuple, list)):
        raise ValueError("exercise_dates must be an explicit ordered sequence")
    authored_exercises = tuple(
        _parse_semantic_date(value, name=f"exercise_dates[{index}]")
        for index, value in enumerate(authored_exercises_value)
    )

    mapping_value = _required_semantic_field(spec, "exercise_to_swap_start")
    if not isinstance(mapping_value, (tuple, list)):
        raise ValueError(
            "exercise_to_swap_start must be an explicit ordered sequence"
        )
    exercise_swap_starts: list[ExerciseSwapStart] = []
    for index, pair in enumerate(mapping_value):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                "exercise_to_swap_start must contain explicit two-date pairs"
            )
        exercise_swap_starts.append(
            ExerciseSwapStart(
                exercise_date=_parse_semantic_date(
                    pair[0], name=f"exercise_to_swap_start[{index}][0]"
                ),
                swap_start_date=_parse_semantic_date(
                    pair[1], name=f"exercise_to_swap_start[{index}][1]"
                ),
            )
        )
    mapped_exercises = tuple(item.exercise_date for item in exercise_swap_starts)
    if mapped_exercises != authored_exercises:
        if sorted(mapped_exercises) == sorted(authored_exercises):
            raise ValueError(
                "exercise_to_swap_start must map authored exercise_dates in order"
            )
        raise ValueError(
            "exercise_to_swap_start must map every authored exercise_date exactly"
        )

    fixed_convention = FixedLegConvention(
        frequency=_parse_semantic_token(
            spec, "fixed_frequency", _FREQUENCIES
        ),
        day_count=_parse_semantic_token(
            spec, "fixed_day_count", _DAY_COUNTS
        ),
        calendar=_parse_semantic_token(
            spec, "fixed_calendar_name", _BUILT_IN_CALENDARS
        ),
        business_day_adjustment=_parse_semantic_token(
            spec,
            "fixed_business_day_adjustment",
            _BUSINESS_DAY_ADJUSTMENTS,
        ),
        stub_type=_parse_semantic_token(
            spec, "fixed_stub_rule", _STUB_TYPES
        ),
        roll_convention=_parse_semantic_token(
            spec, "fixed_roll_convention", _ROLL_CONVENTIONS
        ),
        payment_lag_business_days=_parse_non_negative_semantic_int(
            spec, "fixed_payment_lag_business_days"
        ),
    )
    floating_convention = FloatingLegConvention(
        frequency=_parse_semantic_token(
            spec, "floating_frequency", _FREQUENCIES
        ),
        day_count=_parse_semantic_token(
            spec, "floating_day_count", _DAY_COUNTS
        ),
        calendar=_parse_semantic_token(
            spec, "floating_calendar_name", _BUILT_IN_CALENDARS
        ),
        business_day_adjustment=_parse_semantic_token(
            spec,
            "floating_business_day_adjustment",
            _BUSINESS_DAY_ADJUSTMENTS,
        ),
        stub_type=_parse_semantic_token(
            spec, "floating_stub_rule", _STUB_TYPES
        ),
        roll_convention=_parse_semantic_token(
            spec, "floating_roll_convention", _ROLL_CONVENTIONS
        ),
        reset_lag_business_days=_parse_non_negative_semantic_int(
            spec, "floating_reset_lag_business_days"
        ),
        fixing_lag_business_days=_parse_non_negative_semantic_int(
            spec, "floating_fixing_lag_business_days"
        ),
        payment_lag_business_days=_parse_non_negative_semantic_int(
            spec, "floating_payment_lag_business_days"
        ),
        rate_index=_parse_non_blank_semantic_string(spec, "floating_rate_index"),
        compounding=_parse_non_blank_semantic_string(
            spec, "floating_compounding"
        ),
        gearing=_parse_finite_semantic_float(spec, "floating_gearing"),
        spread=_parse_finite_semantic_float(spec, "floating_spread"),
    )

    tolerance = _parse_finite_semantic_float(
        spec, "lattice_date_tolerance_days"
    )
    if tolerance < 0.0:
        raise ValueError(
            "lattice_date_tolerance_days must be finite and non-negative"
        )

    return PhysicalBermudanSwapTailSpec(
        valuation_date=valuation_date,
        exercise_swap_starts=tuple(exercise_swap_starts),
        common_maturity_date=_parse_semantic_date(
            _required_semantic_field(spec, "swap_maturity"),
            name="swap_maturity",
        ),
        notional=_parse_finite_semantic_float(spec, "notional"),
        fixed_rate=_parse_finite_semantic_float(spec, "fixed_rate"),
        currency=_parse_non_blank_semantic_string(spec, "currency"),
        option_side=_parse_semantic_token(
            spec, "payer_receiver", {"payer": "payer", "receiver": "receiver"}
        ),
        settlement_style=_parse_semantic_token(
            spec, "settlement_type", {"physical": "physical"}
        ),
        fixed_convention=fixed_convention,
        floating_convention=floating_convention,
        discount_curve_name=_parse_non_blank_semantic_string(
            spec, "discount_curve_id"
        ),
        forecast_curve_name=_parse_non_blank_semantic_string(
            spec, "forecast_curve_id"
        ),
        projection_policy=_parse_semantic_token(
            spec,
            "projection_policy",
            {"static_additive_forward_basis": "static_additive_forward_basis"},
        ),
        model_day_count=_parse_semantic_token(
            spec, "model_time_day_count", _MODEL_TIME_DAY_COUNTS
        ),
        model_time_calendar=_parse_semantic_token(
            spec, "model_time_calendar_name", _BUILT_IN_CALENDARS
        ),
        max_lattice_date_error_days=tolerance,
    )


@dataclass(frozen=True)
class FixedSwapTailPeriod:
    """One fixed coupon period with authored and adjusted dates."""

    authored_start_date: date
    authored_end_date: date
    accrual_start_date: date
    accrual_end_date: date
    payment_date: date
    accrual_fraction: float


@dataclass(frozen=True)
class FloatingSwapTailPeriod:
    """One floating coupon period plus its deterministic time-zero basis."""

    authored_start_date: date
    authored_end_date: date
    accrual_start_date: date
    accrual_end_date: date
    reset_date: date
    fixing_date: date
    payment_date: date
    accrual_fraction: float
    discount_forward_rate: float
    forecast_forward_rate: float
    additive_basis: float


@dataclass(frozen=True)
class ResolvedCoTerminalSwapTail:
    """Convention-resolved fixed and floating schedules for one exercise."""

    exercise_date: date
    swap_start_date: date
    common_maturity_date: date
    fixed_periods: tuple[FixedSwapTailPeriod, ...]
    floating_periods: tuple[FloatingSwapTailPeriod, ...]


@dataclass(frozen=True)
class ResolvedCoTerminalSwapTails:
    """All co-terminal swap tails under one strict Bermudan contract."""

    spec: PhysicalBermudanSwapTailSpec
    discount_curve: NamedRateCurve
    forecast_curve: NamedRateCurve
    tails: tuple[ResolvedCoTerminalSwapTail, ...]


def _model_time(spec: PhysicalBermudanSwapTailSpec, target: date) -> float:
    return float(
        year_fraction(
            spec.valuation_date,
            target,
            spec.model_day_count,
            calendar=spec.model_time_calendar,
        )
    )


def _curve_discount(
    binding: NamedRateCurve,
    *,
    time: float,
) -> float:
    value = float(binding.curve.discount(float(time)))
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"curve {binding.name!r} returned a non-positive discount factor")
    return value


def _simple_forward(
    binding: NamedRateCurve,
    *,
    start_time: float,
    end_time: float,
    accrual_fraction: float,
) -> float:
    if accrual_fraction <= 0.0:
        raise ValueError("floating accrual fractions must be positive")
    start_discount = _curve_discount(binding, time=start_time)
    end_discount = _curve_discount(binding, time=end_time)
    return (start_discount / end_discount - 1.0) / accrual_fraction


def _authored_periods(
    start: date,
    end: date,
    *,
    frequency: Frequency,
    stub_type: StubType,
    roll_convention: RollConvention,
) -> tuple[tuple[date, date], ...]:
    ends = tuple(
        generate_schedule(
            start,
            end,
            frequency,
            calendar=None,
            bda=BusinessDayAdjustment.UNADJUSTED,
            stub=stub_type,
            roll_convention=roll_convention,
        )
    )
    starts = (start,) + ends[:-1]
    return tuple(zip(starts, ends))


def _fixed_periods(
    spec: PhysicalBermudanSwapTailSpec,
    *,
    start: date,
) -> tuple[FixedSwapTailPeriod, ...]:
    convention = spec.fixed_convention
    periods: list[FixedSwapTailPeriod] = []
    for authored_start, authored_end in _authored_periods(
        start,
        spec.common_maturity_date,
        frequency=convention.frequency,
        stub_type=convention.stub_type,
        roll_convention=convention.roll_convention,
    ):
        adjusted_start = convention.calendar.adjust(
            authored_start, convention.business_day_adjustment
        )
        adjusted_end = convention.calendar.adjust(
            authored_end, convention.business_day_adjustment
        )
        if adjusted_end <= adjusted_start:
            raise ValueError("fixed-leg date adjustment produced a non-positive period")
        payment_date = convention.calendar.add_business_days(
            adjusted_end, convention.payment_lag_business_days
        )
        accrual = float(
            year_fraction(
                adjusted_start,
                adjusted_end,
                convention.day_count,
                ref_start=adjusted_start,
                ref_end=adjusted_end,
                frequency=convention.frequency,
                calendar=convention.calendar,
            )
        )
        if not isfinite(accrual) or accrual <= 0.0:
            raise ValueError("fixed-leg accrual fraction must be finite and positive")
        periods.append(
            FixedSwapTailPeriod(
                authored_start_date=authored_start,
                authored_end_date=authored_end,
                accrual_start_date=adjusted_start,
                accrual_end_date=adjusted_end,
                payment_date=payment_date,
                accrual_fraction=accrual,
            )
        )
    return tuple(periods)


def _floating_periods(
    spec: PhysicalBermudanSwapTailSpec,
    *,
    start: date,
    discount_curve: NamedRateCurve,
    forecast_curve: NamedRateCurve,
) -> tuple[FloatingSwapTailPeriod, ...]:
    convention = spec.floating_convention
    periods: list[FloatingSwapTailPeriod] = []
    for authored_start, authored_end in _authored_periods(
        start,
        spec.common_maturity_date,
        frequency=convention.frequency,
        stub_type=convention.stub_type,
        roll_convention=convention.roll_convention,
    ):
        adjusted_start = convention.calendar.adjust(
            authored_start, convention.business_day_adjustment
        )
        adjusted_end = convention.calendar.adjust(
            authored_end, convention.business_day_adjustment
        )
        if adjusted_end <= adjusted_start:
            raise ValueError("floating-leg date adjustment produced a non-positive period")
        reset_date = convention.calendar.add_business_days(
            adjusted_start, -convention.reset_lag_business_days
        )
        fixing_date = convention.calendar.add_business_days(
            adjusted_start, -convention.fixing_lag_business_days
        )
        payment_date = convention.calendar.add_business_days(
            adjusted_end, convention.payment_lag_business_days
        )
        accrual = float(
            year_fraction(
                adjusted_start,
                adjusted_end,
                convention.day_count,
                ref_start=adjusted_start,
                ref_end=adjusted_end,
                frequency=convention.frequency,
                calendar=convention.calendar,
            )
        )
        start_time = _model_time(spec, adjusted_start)
        end_time = _model_time(spec, adjusted_end)
        discount_forward = _simple_forward(
            discount_curve,
            start_time=start_time,
            end_time=end_time,
            accrual_fraction=accrual,
        )
        forecast_forward = _simple_forward(
            forecast_curve,
            start_time=start_time,
            end_time=end_time,
            accrual_fraction=accrual,
        )
        periods.append(
            FloatingSwapTailPeriod(
                authored_start_date=authored_start,
                authored_end_date=authored_end,
                accrual_start_date=adjusted_start,
                accrual_end_date=adjusted_end,
                reset_date=reset_date,
                fixing_date=fixing_date,
                payment_date=payment_date,
                accrual_fraction=accrual,
                discount_forward_rate=discount_forward,
                forecast_forward_rate=forecast_forward,
                additive_basis=forecast_forward - discount_forward,
            )
        )
    return tuple(periods)


def resolve_co_terminal_swap_tails(
    spec: PhysicalBermudanSwapTailSpec,
    *,
    discount_curve: NamedRateCurve,
    forecast_curve: NamedRateCurve,
) -> ResolvedCoTerminalSwapTails:
    """Resolve one convention-complete co-terminal swap tail per exercise."""

    if discount_curve.name != spec.discount_curve_name:
        raise ValueError(
            f"discount curve name {discount_curve.name!r} does not match "
            f"contract name {spec.discount_curve_name!r}"
        )
    if forecast_curve.name != spec.forecast_curve_name:
        raise ValueError(
            f"forecast curve name {forecast_curve.name!r} does not match "
            f"contract name {spec.forecast_curve_name!r}"
        )
    if spec.floating_convention.rate_index != spec.forecast_curve_name:
        raise ValueError(
            f"floating rate index {spec.floating_convention.rate_index!r} must "
            f"exactly match forecast curve name {spec.forecast_curve_name!r}"
        )
    if discount_curve is forecast_curve:
        raise ValueError("discount and forecast curve bindings must be separate")

    tails: list[ResolvedCoTerminalSwapTail] = []
    for exercise in spec.exercise_swap_starts:
        floating_periods = _floating_periods(
            spec,
            start=exercise.swap_start_date,
            discount_curve=discount_curve,
            forecast_curve=forecast_curve,
        )
        for period in floating_periods:
            if period.reset_date < exercise.exercise_date:
                raise ValueError(
                    "floating reset date cannot precede exercise; already-reset swap "
                    "tails are outside the static forward-basis kernel"
                )
            if period.fixing_date < exercise.exercise_date:
                raise ValueError(
                    "floating fixing date cannot precede exercise; historical-fixing "
                    "swap tails are outside the static forward-basis kernel"
                )
        tails.append(
            ResolvedCoTerminalSwapTail(
                exercise_date=exercise.exercise_date,
                swap_start_date=exercise.swap_start_date,
                common_maturity_date=spec.common_maturity_date,
                fixed_periods=_fixed_periods(spec, start=exercise.swap_start_date),
                floating_periods=floating_periods,
            )
        )
    return ResolvedCoTerminalSwapTails(
        spec=spec,
        discount_curve=discount_curve,
        forecast_curve=forecast_curve,
        tails=tuple(tails),
    )


@dataclass(frozen=True)
class LatticeDatePoint:
    """One authored/adjusted event date mapped to a uniform lattice step."""

    tail_index: int
    role: str
    authored_date: date
    adjusted_date: date
    model_time: float
    step: int
    error_days: float


@dataclass(frozen=True)
class MappedSwapTailDates:
    """Validated uniform-lattice mapping for all economically used dates."""

    resolved: ResolvedCoTerminalSwapTails
    date_points: tuple[LatticeDatePoint, ...]

    def step_for(self, adjusted_date: date) -> int:
        """Return the unique mapped step for an adjusted event date."""
        steps = {
            point.step
            for point in self.date_points
            if point.adjusted_date == adjusted_date
        }
        if not steps:
            raise KeyError(f"no mapped lattice step for {adjusted_date.isoformat()}")
        if len(steps) != 1:
            raise ValueError(f"date {adjusted_date.isoformat()} has inconsistent lattice steps")
        return next(iter(steps))


def _event_dates(
    resolved: ResolvedCoTerminalSwapTails,
) -> tuple[tuple[int, str, date, date], ...]:
    events: list[tuple[int, str, date, date]] = []
    for tail_index, tail in enumerate(resolved.tails):
        events.append(
            (tail_index, "exercise", tail.exercise_date, tail.exercise_date)
        )
        for fixed_period in tail.fixed_periods:
            events.extend(
                (
                    (
                        tail_index,
                        "fixed_accrual_start",
                        fixed_period.authored_start_date,
                        fixed_period.accrual_start_date,
                    ),
                    (
                        tail_index,
                        "fixed_accrual_end",
                        fixed_period.authored_end_date,
                        fixed_period.accrual_end_date,
                    ),
                    (
                        tail_index,
                        "fixed_payment",
                        fixed_period.authored_end_date,
                        fixed_period.payment_date,
                    ),
                )
            )
        for floating_period in tail.floating_periods:
            events.extend(
                (
                    (
                        tail_index,
                        "floating_accrual_start",
                        floating_period.authored_start_date,
                        floating_period.accrual_start_date,
                    ),
                    (
                        tail_index,
                        "floating_accrual_end",
                        floating_period.authored_end_date,
                        floating_period.accrual_end_date,
                    ),
                    (
                        tail_index,
                        "floating_reset",
                        floating_period.authored_start_date,
                        floating_period.reset_date,
                    ),
                    (
                        tail_index,
                        "floating_fixing",
                        floating_period.authored_start_date,
                        floating_period.fixing_date,
                    ),
                    (
                        tail_index,
                        "floating_payment",
                        floating_period.authored_end_date,
                        floating_period.payment_date,
                    ),
                )
            )
    return tuple(events)


def map_swap_tail_dates_to_lattice(
    resolved: ResolvedCoTerminalSwapTails,
    lattice: RecombiningLattice,
) -> MappedSwapTailDates:
    """Map adjusted event dates to a uniform grid and fail on lossy mappings.

    The error is reported as the ACT/365-equivalent distance between the
    adjusted model time and the selected uniform step.  Distinct adjusted dates
    may not alias the same step; shared dates across legs may do so.
    """

    if not isfinite(float(lattice.dt)) or float(lattice.dt) <= 0.0:
        raise ValueError("lattice dt must be finite and positive")
    max_error = float(resolved.spec.max_lattice_date_error_days)
    points: list[LatticeDatePoint] = []
    step_dates: dict[int, date] = {}
    for tail_index, role, authored_date, adjusted_date in _event_dates(resolved):
        model_time = _model_time(resolved.spec, adjusted_date)
        raw_step = model_time / float(lattice.dt)
        step = int(round(raw_step))
        if step < 0 or step > lattice.n_steps:
            raise ValueError(
                f"adjusted date {adjusted_date.isoformat()} lies outside the lattice horizon"
            )
        error_days = abs(model_time - step * float(lattice.dt)) * 365.0
        if error_days > max_error + 1e-12:
            raise ValueError(
                f"adjusted date {adjusted_date.isoformat()} exceeds maximum lattice date "
                f"error {max_error:g} days (actual {error_days:.6g})"
            )
        previous_date = step_dates.get(step)
        if previous_date is not None and previous_date != adjusted_date:
            raise ValueError(
                "distinct adjusted dates map to the same lattice step: "
                f"{previous_date.isoformat()} and {adjusted_date.isoformat()} -> {step}"
            )
        step_dates[step] = adjusted_date
        points.append(
            LatticeDatePoint(
                tail_index=tail_index,
                role=role,
                authored_date=authored_date,
                adjusted_date=adjusted_date,
                model_time=model_time,
                step=step,
                error_days=error_days,
            )
        )
    return MappedSwapTailDates(resolved=resolved, date_points=tuple(points))


@dataclass(frozen=True)
class ConditionalDiscountBondObservation:
    """Nodewise conditional discount-bond values at one exercise step."""

    exercise_date: date
    maturity_date: date
    exercise_step: int
    maturity_step: int
    node_values: tuple[float, ...]


@dataclass(frozen=True)
class ConditionalDiscountBondObservations:
    """Immutable lookup surface for all swap-tail discount-bond claims."""

    observations: tuple[ConditionalDiscountBondObservation, ...]

    def values(self, exercise_date: date, maturity_date: date) -> tuple[float, ...]:
        """Return node values for one exercise/maturity pair."""
        for observation in self.observations:
            if (
                observation.exercise_date == exercise_date
                and observation.maturity_date == maturity_date
            ):
                return observation.node_values
        raise KeyError(
            f"no conditional bond observation for {exercise_date} -> {maturity_date}"
        )


def observe_conditional_discount_bonds(
    lattice: RecombiningLattice,
    mapped: MappedSwapTailDates,
) -> ConditionalDiscountBondObservations:
    """Roll unit claims back to each exercise for all required tail dates."""

    observations: list[ConditionalDiscountBondObservation] = []
    seen: set[tuple[date, date]] = set()
    for tail in mapped.resolved.tails:
        maturities = {
            *(period.payment_date for period in tail.fixed_periods),
            *(period.accrual_start_date for period in tail.floating_periods),
            *(period.accrual_end_date for period in tail.floating_periods),
            *(period.payment_date for period in tail.floating_periods),
        }
        exercise_step = mapped.step_for(tail.exercise_date)
        for maturity_date in sorted(maturities):
            key = (tail.exercise_date, maturity_date)
            if key in seen:
                continue
            seen.add(key)
            maturity_step = mapped.step_for(maturity_date)
            if maturity_step < exercise_step:
                raise ValueError("swap-tail cashflow date cannot precede its exercise step")
            if maturity_step == exercise_step:
                node_values = tuple(1.0 for _ in range(lattice.n_nodes(exercise_step)))
            else:
                result = lattice_backward_induction_result(
                    lattice,
                    terminal_value=1.0,
                    terminal_step=maturity_step,
                    observation_steps=(exercise_step,),
                )
                node_values = result.observation_at(exercise_step).post_control_values
            observations.append(
                ConditionalDiscountBondObservation(
                    exercise_date=tail.exercise_date,
                    maturity_date=maturity_date,
                    exercise_step=exercise_step,
                    maturity_step=maturity_step,
                    node_values=tuple(float(value) for value in node_values),
                )
            )
    return ConditionalDiscountBondObservations(observations=tuple(observations))


@dataclass(frozen=True)
class SwapTailExerciseNodeValues:
    """Fixed, floating, signed-swap, and holder exercise values at one date."""

    exercise_date: date
    exercise_step: int
    fixed_leg_values: tuple[float, ...]
    floating_leg_values: tuple[float, ...]
    signed_swap_values: tuple[float, ...]
    exercise_values: tuple[float, ...]


@dataclass(frozen=True)
class BermudanSwaptionExerciseValues:
    """Nodewise intrinsic values suitable for generic Bermudan rollback."""

    by_exercise: tuple[SwapTailExerciseNodeValues, ...]

    def at_step(self, step: int) -> SwapTailExerciseNodeValues:
        """Return the unique exercise record at ``step``."""
        matches = tuple(item for item in self.by_exercise if item.exercise_step == step)
        if not matches:
            raise KeyError(f"no Bermudan exercise values at lattice step {step}")
        if len(matches) != 1:
            raise ValueError(f"multiple Bermudan exercises map to lattice step {step}")
        return matches[0]


def _node_count(
    observations: ConditionalDiscountBondObservations,
    *,
    exercise_date: date,
    tail: ResolvedCoTerminalSwapTail,
) -> int:
    sample_date = (
        tail.fixed_periods[0].payment_date
        if tail.fixed_periods
        else tail.floating_periods[0].payment_date
    )
    return len(observations.values(exercise_date, sample_date))


def build_bermudan_swaption_exercise_values(
    resolved: ResolvedCoTerminalSwapTails,
    mapped: MappedSwapTailDates,
    observations: ConditionalDiscountBondObservations,
) -> BermudanSwaptionExerciseValues:
    """Build convention-aware physical swap exercise values at every node."""

    if mapped.resolved is not resolved:
        raise ValueError("mapped dates do not belong to the supplied resolved tails")
    records: list[SwapTailExerciseNodeValues] = []
    spec = resolved.spec
    for tail in resolved.tails:
        node_count = _node_count(
            observations,
            exercise_date=tail.exercise_date,
            tail=tail,
        )
        fixed_values = [0.0] * node_count
        floating_values = [0.0] * node_count
        for fixed_period in tail.fixed_periods:
            payment_bonds = observations.values(
                tail.exercise_date, fixed_period.payment_date
            )
            for node in range(node_count):
                fixed_values[node] += (
                    float(spec.notional)
                    * float(spec.fixed_rate)
                    * fixed_period.accrual_fraction
                    * payment_bonds[node]
                )
        for floating_period in tail.floating_periods:
            start_bonds = observations.values(
                tail.exercise_date, floating_period.accrual_start_date
            )
            end_bonds = observations.values(
                tail.exercise_date, floating_period.accrual_end_date
            )
            payment_bonds = observations.values(
                tail.exercise_date, floating_period.payment_date
            )
            for node in range(node_count):
                discount_forward = (
                    start_bonds[node] / end_bonds[node] - 1.0
                ) / floating_period.accrual_fraction
                projected_forward = discount_forward + floating_period.additive_basis
                coupon_rate = (
                    float(spec.floating_convention.gearing) * projected_forward
                    + float(spec.floating_convention.spread)
                )
                floating_values[node] += (
                    float(spec.notional)
                    * floating_period.accrual_fraction
                    * coupon_rate
                    * payment_bonds[node]
                )
        direction = 1.0 if spec.option_side == "payer" else -1.0
        signed_values = tuple(
            direction * (floating - fixed)
            for fixed, floating in zip(fixed_values, floating_values)
        )
        records.append(
            SwapTailExerciseNodeValues(
                exercise_date=tail.exercise_date,
                exercise_step=mapped.step_for(tail.exercise_date),
                fixed_leg_values=tuple(fixed_values),
                floating_leg_values=tuple(floating_values),
                signed_swap_values=signed_values,
                exercise_values=tuple(max(value, 0.0) for value in signed_values),
            )
        )
    steps = tuple(record.exercise_step for record in records)
    if len(set(steps)) != len(steps):
        raise ValueError("distinct exercise dates map to the same lattice step")
    return BermudanSwaptionExerciseValues(by_exercise=tuple(records))


def price_physical_bermudan_swaption_lattice(
    lattice: RecombiningLattice,
    spec: PhysicalBermudanSwapTailSpec,
    *,
    discount_curve: NamedRateCurve,
    forecast_curve: NamedRateCurve,
) -> float:
    """Price the strict physical contract on an already calibrated lattice."""

    resolved = resolve_co_terminal_swap_tails(
        spec,
        discount_curve=discount_curve,
        forecast_curve=forecast_curve,
    )
    mapped = map_swap_tail_dates_to_lattice(resolved, lattice)
    observations = observe_conditional_discount_bonds(lattice, mapped)
    exercise_values = build_bermudan_swaption_exercise_values(
        resolved,
        mapped,
        observations,
    )
    latest = exercise_values.by_exercise[-1]
    earlier_steps = [item.exercise_step for item in exercise_values.by_exercise[:-1]]
    return float(
        lattice_backward_induction(
            lattice,
            terminal_payoff=lambda step, node, lattice_: latest.exercise_values[node],
            terminal_step=latest.exercise_step,
            exercise_type="bermudan",
            exercise_steps=earlier_steps,
            exercise_value=lambda step, node, lattice_, continuation: (
                exercise_values.at_step(step).exercise_values[node]
            ),
        )
    )


__all__ = [
    "BermudanSwaptionExerciseValues",
    "CalendarLike",
    "ConditionalDiscountBondObservation",
    "ConditionalDiscountBondObservations",
    "ExerciseSwapStart",
    "FixedLegConvention",
    "FixedSwapTailPeriod",
    "FloatingLegConvention",
    "FloatingSwapTailPeriod",
    "LatticeDatePoint",
    "MappedSwapTailDates",
    "NamedRateCurve",
    "PhysicalBermudanSwapTailSpec",
    "RateCurveLike",
    "ResolvedCoTerminalSwapTail",
    "ResolvedCoTerminalSwapTails",
    "SwapTailExerciseNodeValues",
    "build_bermudan_swaption_exercise_values",
    "compile_physical_bermudan_swap_tail_spec",
    "map_swap_tail_dates_to_lattice",
    "observe_conditional_discount_bonds",
    "price_physical_bermudan_swaption_lattice",
    "resolve_co_terminal_swap_tails",
]
