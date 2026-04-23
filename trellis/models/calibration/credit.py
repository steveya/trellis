"""Typed single-name reduced-form credit calibration workflow."""

from __future__ import annotations

from datetime import date
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as raw_np
from scipy.optimize import brentq

from trellis.conventions.calendar import BusinessDayAdjustment, Calendar
from trellis.conventions.schedule import RollConvention, StubType
from trellis.core.date_utils import add_months
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.credit_curve import CreditCurve
from trellis.models.calibration.materialization import materialize_credit_curve
from trellis.models.calibration.quote_maps import (
    CalibrationQuoteMap,
    QuoteAxisSpec,
    QuoteMapSpec,
    QuoteSemanticsSpec,
    QuoteSettlementSpec,
    QuoteUnitSpec,
)
from trellis.models.calibration.solve_request import (
    ObjectiveBundle,
    SolveBounds,
    SolveProvenance,
    SolveReplayArtifact,
    SolveRequest,
    SolveResult,
    WarmStart,
    build_solve_provenance,
    build_solve_replay_artifact,
    execute_solve_request,
)
from trellis.models.credit_default_swap import (
    build_cds_schedule,
    normalize_cds_upfront_quote,
    normalize_cds_running_spread,
    price_cds_analytical,
    solve_cds_par_spread_analytical,
)

_CDS_CALIBRATION_NOTIONAL = 1.0
_CDS_CALIBRATION_FREQUENCY = Frequency.QUARTERLY
_CDS_CALIBRATION_DAY_COUNT = DayCountConvention.ACT_360


def _require_finite(value: float, *, field_name: str) -> float:
    """Return one finite float value or raise ``ValueError``."""
    normalized = float(value)
    if not raw_np.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _require_finite_positive(value: float, *, field_name: str) -> float:
    """Return one finite positive float value or raise ``ValueError``."""
    normalized = float(value)
    if not raw_np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return normalized


def _coerce_enum(value, enum_type, *, field_name: str):
    """Return ``value`` as ``enum_type`` while accepting stable string names."""
    if isinstance(value, enum_type):
        return value
    text = str(value).strip()
    for candidate in enum_type:
        if text.upper() == candidate.name or text.lower() == str(candidate.value).lower():
            return candidate
    raise ValueError(f"{field_name} must be one of {[candidate.name for candidate in enum_type]}")


def _calendar_name(calendar: Calendar | None) -> str:
    """Return a stable calendar label for diagnostics."""
    return str(getattr(calendar, "name", "") or "WeekendOnly")


def _quote_schedule_aware(quote: "CreditHazardCalibrationQuote") -> bool:
    """Return whether a quote carries non-canonical CDS schedule metadata."""
    return (
        quote.start_date is not None
        or quote.maturity_date is not None
        or quote.frequency != _CDS_CALIBRATION_FREQUENCY
        or quote.day_count != _CDS_CALIBRATION_DAY_COUNT
        or quote.calendar is not None
        or quote.business_day_adjustment != BusinessDayAdjustment.FOLLOWING
        or quote.roll_convention != RollConvention.NONE
        or quote.stub != StubType.SHORT_LAST
        or quote.payment_lag_days != 0
    )


def _build_quote_cds_schedule(
    *,
    settlement: date,
    quote: "CreditHazardCalibrationQuote",
):
    """Return the explicit schedule, maturity, and setup metadata for one quote."""
    start_date = quote.start_date or settlement
    if quote.maturity_date is None:
        months = max(int(round(float(quote.maturity_years) * 12.0)), 1)
        maturity_date = add_months(start_date, months)
    else:
        maturity_date = quote.maturity_date
    if maturity_date <= start_date:
        raise ValueError("credit calibration quote maturity_date must be after start_date")
    schedule = build_cds_schedule(
        start_date,
        maturity_date,
        quote.frequency,
        quote.day_count,
        time_origin=settlement,
        calendar=quote.calendar,
        business_day_adjustment=quote.business_day_adjustment,
        roll_convention=quote.roll_convention,
        stub=quote.stub,
        payment_lag_days=quote.payment_lag_days,
    )
    setup = {
        "start_date": start_date.isoformat(),
        "maturity_date": maturity_date.isoformat(),
        "frequency": quote.frequency.name,
        "day_count": quote.day_count.name,
        "calendar": _calendar_name(quote.calendar),
        "business_day_adjustment": quote.business_day_adjustment.name,
        "roll_convention": quote.roll_convention.name,
        "stub": quote.stub.name,
        "payment_lag_days": int(quote.payment_lag_days),
        "period_count": len(schedule.periods),
        "first_payment_date": schedule.payment_dates[0].isoformat() if schedule.periods else "",
        "last_payment_date": schedule.payment_dates[-1].isoformat() if schedule.periods else "",
    }
    return schedule, maturity_date, setup


def _par_spread_from_curve(
    *,
    schedule,
    credit_curve: CreditCurve,
    discount_curve,
    recovery: float,
) -> float:
    """Return the bounded canonical par spread implied by one credit curve."""
    return solve_cds_par_spread_analytical(
        notional=_CDS_CALIBRATION_NOTIONAL,
        recovery=recovery,
        schedule=schedule,
        credit_curve=credit_curve,
        discount_curve=discount_curve,
    )


def _flat_hazard_curve(
    *,
    maturity_years: float,
    hazard_rate: float,
) -> CreditCurve:
    """Return a flat credit curve for bounded quote normalization."""
    return CreditCurve.flat(float(hazard_rate), max_tenor=max(float(maturity_years), 30.0))


def _hazard_from_par_spread(
    *,
    running_spread: float,
    schedule,
    maturity_years: float,
    discount_curve,
    recovery: float,
    max_hazard: float,
) -> float:
    """Return the flat hazard consistent with one bounded CDS par spread."""
    normalized_spread = normalize_cds_running_spread(float(running_spread))
    tolerance = 1e-12

    def objective(hazard: float) -> float:
        return price_cds_analytical(
            notional=_CDS_CALIBRATION_NOTIONAL,
            spread_quote=normalized_spread,
            recovery=recovery,
            schedule=schedule,
            credit_curve=_flat_hazard_curve(
                maturity_years=maturity_years,
                hazard_rate=float(hazard),
            ),
            discount_curve=discount_curve,
        )

    lower = 1e-8
    upper = max(float(max_hazard), lower * 10.0)
    lower_value = objective(lower)
    if abs(lower_value) <= tolerance:
        return float(lower)
    if lower_value > 0.0:
        raise ValueError(
            "bounded CDS hazard normalization expected non-positive PV at near-zero hazard"
        )
    upper_value = objective(upper)
    while upper_value < 0.0 and upper < 100.0:
        upper *= 2.0
        upper_value = objective(upper)
    if upper_value < 0.0:
        raise ValueError("bounded CDS hazard normalization could not bracket a zero PV hazard")
    return float(brentq(objective, lower, upper, xtol=1e-10))


def _credit_quote_assumptions(
    market_state: MarketState,
    *,
    recovery: float,
    schedule_aware: bool,
) -> tuple[str, ...]:
    """Return normalized assumptions for the single-name credit quote maps."""
    selected = dict(market_state.selected_curve_names or {})
    schedule_text = (
        "CDS repricing uses quote-level CDS schedules, day-count, calendar, "
        "business-day adjustment, roll, stub, and payment-lag metadata when supplied."
        if schedule_aware
        else (
            "CDS repricing uses bounded quarterly ACT/360 schedules built from "
            "market_state.settlement with tenor maturities rounded to calendar months."
        )
    )
    return (
        (
            "Reduced-form single-name calibration binds potential terms as "
            "risky_discount(t)=discount(t)*survival_probability(t)."
        ),
        schedule_text,
        f"Recovery assumption: {float(recovery):.6f}.",
        f"Discount curve role: {selected.get('discount_curve') or '<unbound>'}.",
    )


def _credit_potential_binding(
    market_state: MarketState,
    *,
    curve_name: str,
    recovery: float,
) -> dict[str, object]:
    """Return explicit discount/default potential binding metadata."""
    selected = dict(market_state.selected_curve_names or {})
    return {
        "discount_curve_role": "discount_curve",
        "discount_curve_name": selected.get("discount_curve"),
        "default_curve_role": "credit_curve",
        "default_curve_name": curve_name,
        "recovery": float(recovery),
        "risky_discount_formula": "discount(t) * survival_probability(t)",
    }


def _spread_quote_map(
    *,
    market_quote: float,
    recovery: float,
    source_ref: str,
    assumptions: tuple[str, ...],
    potential_binding: dict[str, object],
) -> CalibrationQuoteMap:
    """Return the explicit spread-to-running-spread quote map."""
    input_unit = "basis_points" if float(market_quote) > 1.0 else "decimal_running_spread"
    return CalibrationQuoteMap(
        spec=QuoteMapSpec(
            quote_family="spread",
            semantics=QuoteSemanticsSpec(
                quote_family="spread",
                quote_subject="single_name_cds",
                axes=(QuoteAxisSpec("maturity", axis_kind="tenor", unit="years"),),
                unit=QuoteUnitSpec(
                    unit_name="decimal_running_spread",
                    value_domain="credit_spread",
                    scaling="absolute",
                ),
                settlement=QuoteSettlementSpec(
                    numeraire="discount_curve",
                    discount_curve_role="discount_curve",
                ),
            ),
        ),
        quote_to_price_fn=lambda quote: normalize_cds_running_spread(float(quote)),
        price_to_quote_fn=(
            (lambda running_spread: float(running_spread) * 1e4)
            if float(market_quote) > 1.0
            else (lambda running_spread: float(running_spread))
        ),
        source_ref=source_ref,
        assumptions=assumptions,
        metadata={
            "quote_kind": "spread",
            "quote_unit": input_unit,
            "normalization_method": "running_spread_decimal",
            "potential_binding": dict(potential_binding),
            "recovery": float(recovery),
        },
    )


def _hazard_quote_map(
    *,
    schedule,
    maturity_years: float,
    discount_curve,
    recovery: float,
    max_hazard: float,
    assumptions: tuple[str, ...],
    potential_binding: dict[str, object],
) -> CalibrationQuoteMap:
    """Return the explicit hazard-to-running-spread quote map."""
    return CalibrationQuoteMap(
        spec=QuoteMapSpec(
            quote_family="hazard",
            semantics=QuoteSemanticsSpec(
                quote_family="hazard",
                quote_subject="single_name_cds",
                axes=(QuoteAxisSpec("maturity", axis_kind="tenor", unit="years"),),
                unit=QuoteUnitSpec(
                    unit_name="hazard_rate",
                    value_domain="default_intensity",
                    scaling="absolute",
                ),
                settlement=QuoteSettlementSpec(
                    numeraire="discount_curve",
                    discount_curve_role="discount_curve",
                ),
            ),
        ),
        quote_to_price_fn=lambda hazard: _par_spread_from_curve(
            schedule=schedule,
            credit_curve=_flat_hazard_curve(
                maturity_years=maturity_years,
                hazard_rate=float(hazard),
            ),
            discount_curve=discount_curve,
            recovery=recovery,
        ),
        price_to_quote_fn=lambda running_spread: _hazard_from_par_spread(
            running_spread=float(running_spread),
            schedule=schedule,
            maturity_years=maturity_years,
            discount_curve=discount_curve,
            recovery=recovery,
            max_hazard=max_hazard,
        ),
        source_ref="_hazard_quote_map",
        assumptions=assumptions,
        metadata={
            "quote_kind": "hazard",
            "quote_unit": "hazard_rate",
            "normalization_method": "cds_pricer_flat_hazard",
            "maturity_years": float(maturity_years),
            "potential_binding": dict(potential_binding),
            "recovery": float(recovery),
        },
    )


def _hazard_from_upfront_quote(
    *,
    upfront_quote: float,
    standard_running_spread: float,
    schedule,
    maturity_years: float,
    discount_curve,
    recovery: float,
    max_hazard: float,
) -> float:
    """Return the flat hazard consistent with a standard-coupon upfront quote."""
    upfront = normalize_cds_upfront_quote(float(upfront_quote))
    standard_spread = normalize_cds_running_spread(float(standard_running_spread))
    tolerance = 1e-12

    def objective(hazard: float) -> float:
        return (
            price_cds_analytical(
                notional=_CDS_CALIBRATION_NOTIONAL,
                spread_quote=standard_spread,
                recovery=recovery,
                schedule=schedule,
                credit_curve=_flat_hazard_curve(
                    maturity_years=maturity_years,
                    hazard_rate=float(hazard),
                ),
                discount_curve=discount_curve,
            )
            - upfront
        )

    lower = 1e-8
    upper = max(float(max_hazard), lower * 10.0)
    lower_value = objective(lower)
    if abs(lower_value) <= tolerance:
        return float(lower)
    upper_value = objective(upper)
    while lower_value * upper_value > 0.0 and upper < 100.0:
        upper *= 2.0
        upper_value = objective(upper)
    if lower_value * upper_value > 0.0:
        raise ValueError("bounded CDS upfront normalization could not bracket a zero PV hazard")
    return float(brentq(objective, lower, upper, xtol=1e-10))


def _upfront_from_par_spread(
    *,
    running_spread: float,
    standard_running_spread: float,
    schedule,
    maturity_years: float,
    discount_curve,
    recovery: float,
    max_hazard: float,
) -> float:
    """Return the flat-hazard upfront quote implied by one par running spread."""
    hazard = _hazard_from_par_spread(
        running_spread=running_spread,
        schedule=schedule,
        maturity_years=maturity_years,
        discount_curve=discount_curve,
        recovery=recovery,
        max_hazard=max_hazard,
    )
    return float(
        price_cds_analytical(
            notional=_CDS_CALIBRATION_NOTIONAL,
            spread_quote=normalize_cds_running_spread(float(standard_running_spread)),
            recovery=recovery,
            schedule=schedule,
            credit_curve=_flat_hazard_curve(
                maturity_years=maturity_years,
                hazard_rate=hazard,
            ),
            discount_curve=discount_curve,
        )
    )


def _upfront_quote_map(
    *,
    schedule,
    maturity_years: float,
    discount_curve,
    recovery: float,
    standard_running_spread: float,
    max_hazard: float,
    market_quote: float,
    assumptions: tuple[str, ...],
    potential_binding: dict[str, object],
) -> CalibrationQuoteMap:
    """Return the explicit standard-coupon-upfront to par-spread quote map."""
    input_unit = "upfront_points" if abs(float(market_quote)) > 1.0 else "decimal_upfront"
    standard_spread = normalize_cds_running_spread(float(standard_running_spread))
    return CalibrationQuoteMap(
        spec=QuoteMapSpec(
            quote_family="upfront",
            semantics=QuoteSemanticsSpec(
                quote_family="upfront",
                quote_subject="single_name_cds",
                axes=(QuoteAxisSpec("maturity", axis_kind="tenor", unit="years"),),
                unit=QuoteUnitSpec(
                    unit_name="decimal_upfront",
                    value_domain="notional_fraction",
                    scaling="absolute",
                ),
                settlement=QuoteSettlementSpec(
                    numeraire="discount_curve",
                    settlement_style="standard_coupon_upfront",
                    discount_curve_role="discount_curve",
                    metadata={"standard_running_spread": standard_spread},
                ),
            ),
        ),
        quote_to_price_fn=lambda upfront: _par_spread_from_curve(
            schedule=schedule,
            credit_curve=_flat_hazard_curve(
                maturity_years=maturity_years,
                hazard_rate=_hazard_from_upfront_quote(
                    upfront_quote=float(upfront),
                    standard_running_spread=standard_spread,
                    schedule=schedule,
                    maturity_years=maturity_years,
                    discount_curve=discount_curve,
                    recovery=recovery,
                    max_hazard=max_hazard,
                ),
            ),
            discount_curve=discount_curve,
            recovery=recovery,
        ),
        price_to_quote_fn=lambda running_spread: _upfront_from_par_spread(
            running_spread=float(running_spread),
            standard_running_spread=standard_spread,
            schedule=schedule,
            maturity_years=maturity_years,
            discount_curve=discount_curve,
            recovery=recovery,
            max_hazard=max_hazard,
        ),
        source_ref="_upfront_quote_map",
        assumptions=assumptions,
        metadata={
            "quote_kind": "upfront",
            "quote_unit": input_unit,
            "normalization_method": "standard_coupon_upfront_to_par_spread",
            "standard_running_spread": standard_spread,
            "potential_binding": dict(potential_binding),
            "recovery": float(recovery),
        },
    )


@dataclass(frozen=True)
class CreditHazardCalibrationQuote:
    """One supported single-name credit calibration quote."""

    maturity_years: float
    quote: float
    quote_kind: Literal["spread", "running_spread", "upfront", "hazard"] = "spread"
    label: str = ""
    weight: float = 1.0
    start_date: date | None = None
    maturity_date: date | None = None
    frequency: Frequency = _CDS_CALIBRATION_FREQUENCY
    day_count: DayCountConvention = _CDS_CALIBRATION_DAY_COUNT
    calendar: Calendar | None = None
    business_day_adjustment: BusinessDayAdjustment = BusinessDayAdjustment.FOLLOWING
    roll_convention: RollConvention = RollConvention.NONE
    stub: StubType = StubType.SHORT_LAST
    payment_lag_days: int = 0
    standard_running_spread: float | None = None

    def __post_init__(self) -> None:
        maturity_years = _require_finite_positive(self.maturity_years, field_name="maturity_years")
        quote_kind = str(self.quote_kind).strip().lower()
        if quote_kind == "running_spread":
            quote_kind = "spread"
        if quote_kind not in {"spread", "upfront", "hazard"}:
            raise ValueError("quote_kind must be 'spread', 'running_spread', 'upfront', or 'hazard'")
        quote = (
            _require_finite(self.quote, field_name="quote")
            if quote_kind == "upfront"
            else _require_finite_positive(self.quote, field_name="quote")
        )
        weight = _require_finite_positive(self.weight, field_name="weight")
        frequency = _coerce_enum(self.frequency, Frequency, field_name="frequency")
        day_count = _coerce_enum(self.day_count, DayCountConvention, field_name="day_count")
        business_day_adjustment = _coerce_enum(
            self.business_day_adjustment,
            BusinessDayAdjustment,
            field_name="business_day_adjustment",
        )
        roll_convention = _coerce_enum(
            self.roll_convention,
            RollConvention,
            field_name="roll_convention",
        )
        stub = _coerce_enum(self.stub, StubType, field_name="stub")
        payment_lag_days = int(self.payment_lag_days)
        if payment_lag_days < 0:
            raise ValueError("payment_lag_days must be non-negative")
        standard_running_spread = self.standard_running_spread
        if standard_running_spread is not None:
            standard_running_spread = _require_finite_positive(
                standard_running_spread,
                field_name="standard_running_spread",
            )
        if quote_kind == "upfront" and standard_running_spread is None:
            raise ValueError("upfront CDS quotes require standard_running_spread")
        if self.start_date is not None and self.maturity_date is not None and self.maturity_date <= self.start_date:
            raise ValueError("maturity_date must be after start_date")
        object.__setattr__(self, "maturity_years", maturity_years)
        object.__setattr__(self, "quote", quote)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "quote_kind", quote_kind)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "day_count", day_count)
        object.__setattr__(self, "business_day_adjustment", business_day_adjustment)
        object.__setattr__(self, "roll_convention", roll_convention)
        object.__setattr__(self, "stub", stub)
        object.__setattr__(self, "payment_lag_days", payment_lag_days)
        object.__setattr__(self, "standard_running_spread", standard_running_spread)

    def resolved_label(self, index: int) -> str:
        """Return a stable per-quote label."""
        if self.label.strip():
            return self.label.strip()
        tenor_label = str(float(self.maturity_years)).replace(".", "_")
        return f"{self.quote_kind}_{tenor_label}y_{index}"

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly quote payload."""
        return {
            "maturity_years": float(self.maturity_years),
            "quote": float(self.quote),
            "quote_kind": self.quote_kind,
            "label": self.label,
            "weight": float(self.weight),
            "start_date": self.start_date.isoformat() if self.start_date is not None else None,
            "maturity_date": self.maturity_date.isoformat() if self.maturity_date is not None else None,
            "frequency": self.frequency.name,
            "day_count": self.day_count.name,
            "calendar": _calendar_name(self.calendar),
            "business_day_adjustment": self.business_day_adjustment.name,
            "roll_convention": self.roll_convention.name,
            "stub": self.stub.name,
            "payment_lag_days": int(self.payment_lag_days),
            "standard_running_spread": (
                None
                if self.standard_running_spread is None
                else float(self.standard_running_spread)
            ),
        }


@dataclass(frozen=True)
class CreditHazardCalibrationResult:
    """Structured result for the supported single-name credit calibration workflow."""

    quotes: tuple[CreditHazardCalibrationQuote, ...]
    credit_curve: CreditCurve
    solve_request: SolveRequest
    solve_result: SolveResult
    solver_provenance: SolveProvenance
    solver_replay_artifact: SolveReplayArtifact
    tenors: tuple[float, ...]
    target_running_spreads: tuple[float, ...]
    model_running_spreads: tuple[float, ...]
    target_hazards: tuple[float, ...]
    model_hazards: tuple[float, ...]
    target_quotes: tuple[float, ...]
    model_quotes: tuple[float, ...]
    repricing_errors: tuple[float, ...]
    survival_probabilities: tuple[float, ...]
    forward_hazards: tuple[float, ...]
    hazard_residuals: tuple[float, ...]
    quote_residuals: tuple[float, ...]
    max_abs_repricing_error: float
    max_abs_hazard_residual: float
    max_abs_quote_residual: float
    curve_name: str = "single_name_credit"
    recovery: float = 0.4
    potential_binding: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "quotes", tuple(self.quotes))
        object.__setattr__(self, "tenors", tuple(float(value) for value in self.tenors))
        object.__setattr__(self, "target_running_spreads", tuple(float(value) for value in self.target_running_spreads))
        object.__setattr__(self, "model_running_spreads", tuple(float(value) for value in self.model_running_spreads))
        object.__setattr__(self, "target_hazards", tuple(float(value) for value in self.target_hazards))
        object.__setattr__(self, "model_hazards", tuple(float(value) for value in self.model_hazards))
        object.__setattr__(self, "target_quotes", tuple(float(value) for value in self.target_quotes))
        object.__setattr__(self, "model_quotes", tuple(float(value) for value in self.model_quotes))
        object.__setattr__(self, "repricing_errors", tuple(float(value) for value in self.repricing_errors))
        object.__setattr__(
            self,
            "survival_probabilities",
            tuple(float(value) for value in self.survival_probabilities),
        )
        object.__setattr__(self, "forward_hazards", tuple(float(value) for value in self.forward_hazards))
        object.__setattr__(self, "hazard_residuals", tuple(float(value) for value in self.hazard_residuals))
        object.__setattr__(self, "quote_residuals", tuple(float(value) for value in self.quote_residuals))
        object.__setattr__(self, "max_abs_repricing_error", float(self.max_abs_repricing_error))
        object.__setattr__(self, "max_abs_hazard_residual", float(self.max_abs_hazard_residual))
        object.__setattr__(self, "max_abs_quote_residual", float(self.max_abs_quote_residual))
        object.__setattr__(self, "recovery", float(self.recovery))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "assumptions", tuple(str(assumption) for assumption in self.assumptions))

    def apply_to_market_state(self, market_state: MarketState) -> MarketState:
        """Return ``market_state`` enriched with the calibrated single-name credit curve."""
        selected_curve_roles = {
            "discount_curve": str(dict(market_state.selected_curve_names or {}).get("discount_curve") or ""),
            "credit_curve": str(self.curve_name),
        }
        return materialize_credit_curve(
            market_state,
            curve_name=self.curve_name,
            credit_curve=self.credit_curve,
            source_kind=str(self.provenance.get("source_kind", "calibrated_surface")),
            source_ref=str(
                self.provenance.get("source_ref", "calibrate_single_name_credit_curve_workflow")
            ),
            selected_curve_roles=selected_curve_roles,
            metadata={
                "instrument_family": "credit",
                "instrument_kind": "single_name_cds",
                "curve_name": self.curve_name,
                "recovery": float(self.recovery),
                "potential_binding": dict(self.potential_binding),
                "hazard_governance": dict(self.summary.get("hazard_governance") or {}),
                "quote_styles": sorted({quote.quote_kind for quote in self.quotes}),
                "max_abs_repricing_error": float(self.max_abs_repricing_error),
            },
        )

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "quotes": [quote.to_payload() for quote in self.quotes],
            "credit_curve": {
                "tenors": [float(value) for value in self.credit_curve.tenors],
                "hazard_rates": [float(value) for value in self.credit_curve.hazard_rates],
            },
            "solve_request": self.solve_request.to_payload(),
            "solve_result": self.solve_result.to_payload(),
            "solver_provenance": self.solver_provenance.to_payload(),
            "solver_replay_artifact": self.solver_replay_artifact.to_payload(),
            "tenors": list(self.tenors),
            "target_running_spreads": list(self.target_running_spreads),
            "model_running_spreads": list(self.model_running_spreads),
            "target_hazards": list(self.target_hazards),
            "model_hazards": list(self.model_hazards),
            "target_quotes": list(self.target_quotes),
            "model_quotes": list(self.model_quotes),
            "repricing_errors": list(self.repricing_errors),
            "survival_probabilities": list(self.survival_probabilities),
            "forward_hazards": list(self.forward_hazards),
            "hazard_residuals": list(self.hazard_residuals),
            "quote_residuals": list(self.quote_residuals),
            "max_abs_repricing_error": self.max_abs_repricing_error,
            "max_abs_hazard_residual": self.max_abs_hazard_residual,
            "max_abs_quote_residual": self.max_abs_quote_residual,
            "curve_name": self.curve_name,
            "recovery": self.recovery,
            "potential_binding": dict(self.potential_binding),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
            "warnings": list(self.warnings),
            "assumptions": list(self.assumptions),
        }


def _normalize_quotes(
    quotes: Sequence[CreditHazardCalibrationQuote],
) -> tuple[CreditHazardCalibrationQuote, ...]:
    """Return sorted quotes and validate maturity uniqueness."""
    resolved = tuple(quotes)
    if not resolved:
        raise ValueError("at least one credit calibration quote is required")
    sorted_quotes = tuple(
        sorted(
            resolved,
            key=lambda quote: (float(quote.maturity_years), quote.quote_kind, float(quote.quote)),
        )
    )
    maturities = [float(quote.maturity_years) for quote in sorted_quotes]
    for left, right in zip(maturities, maturities[1:]):
        if abs(float(right) - float(left)) <= 1e-12:
            raise ValueError("credit calibration quote maturities must be strictly increasing")
    return sorted_quotes


def _survival_and_forward_hazards(
    credit_curve: CreditCurve,
    tenors: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return survival probabilities and implied forward hazards across the tenor grid."""
    survival_probabilities = tuple(float(credit_curve.survival_probability(float(tenor))) for tenor in tenors)
    forward_hazards: list[float] = []
    previous_tenor = 0.0
    previous_survival = 1.0
    for tenor, survival in zip(tenors, survival_probabilities):
        interval = max(float(tenor) - float(previous_tenor), 1e-12)
        survival_ratio = max(float(survival) / max(float(previous_survival), 1e-12), 1e-12)
        forward_hazards.append(float(-raw_np.log(survival_ratio) / interval))
        previous_tenor = float(tenor)
        previous_survival = float(survival)
    return survival_probabilities, tuple(forward_hazards)


def calibrate_single_name_credit_curve_workflow(
    quotes: Sequence[CreditHazardCalibrationQuote],
    market_state: MarketState,
    *,
    recovery: float = 0.4,
    curve_name: str = "single_name_credit",
    max_hazard: float = 5.0,
) -> CreditHazardCalibrationResult:
    """Calibrate one reduced-form single-name credit curve from spread/hazard quotes."""
    if market_state.discount is None:
        raise ValueError("single-name credit calibration requires market_state.discount")
    if market_state.settlement is None:
        raise ValueError("single-name credit calibration requires market_state.settlement")
    recovery = float(recovery)
    if not raw_np.isfinite(recovery) or recovery <= 0.0 or recovery >= 1.0:
        raise ValueError("recovery must be strictly between 0 and 1")
    if not raw_np.isfinite(max_hazard) or float(max_hazard) <= 0.0:
        raise ValueError("max_hazard must be finite and positive")
    normalized_quotes = _normalize_quotes(quotes)
    labels = tuple(quote.resolved_label(index) for index, quote in enumerate(normalized_quotes))
    tenors = tuple(float(quote.maturity_years) for quote in normalized_quotes)
    schedule_aware = any(_quote_schedule_aware(quote) for quote in normalized_quotes)
    quote_normalization_method = (
        "cds_pricer_schedule_aware"
        if schedule_aware or any(quote.quote_kind == "upfront" for quote in normalized_quotes)
        else "cds_pricer"
    )
    potential_binding = _credit_potential_binding(
        market_state,
        curve_name=curve_name,
        recovery=recovery,
    )
    assumptions = _credit_quote_assumptions(
        market_state,
        recovery=recovery,
        schedule_aware=schedule_aware,
    )
    schedules = []
    instrument_setups = []
    quote_maps: list[CalibrationQuoteMap] = []
    for quote in normalized_quotes:
        schedule, _maturity_date, instrument_setup = _build_quote_cds_schedule(
            settlement=market_state.settlement,
            quote=quote,
        )
        schedules.append(schedule)
        instrument_setups.append(instrument_setup)
        if quote.quote_kind == "spread":
            quote_maps.append(
                _spread_quote_map(
                    market_quote=float(quote.quote),
                    recovery=recovery,
                    source_ref="_spread_quote_map",
                    assumptions=assumptions,
                    potential_binding=potential_binding,
                )
            )
        elif quote.quote_kind == "upfront":
            quote_maps.append(
                _upfront_quote_map(
                    schedule=schedule,
                    maturity_years=float(quote.maturity_years),
                    discount_curve=market_state.discount,
                    recovery=recovery,
                    standard_running_spread=float(quote.standard_running_spread),
                    max_hazard=float(max_hazard),
                    market_quote=float(quote.quote),
                    assumptions=assumptions,
                    potential_binding=potential_binding,
                )
            )
        else:
            quote_maps.append(
                _hazard_quote_map(
                    schedule=schedule,
                    maturity_years=float(quote.maturity_years),
                    discount_curve=market_state.discount,
                    recovery=recovery,
                    max_hazard=float(max_hazard),
                    assumptions=assumptions,
                    potential_binding=potential_binding,
                )
            )

    target_running_spreads_values: list[float] = []
    target_hazards_values: list[float] = []
    quote_transform_warnings: list[str] = []
    target_quotes = tuple(float(quote.quote) for quote in normalized_quotes)
    for label, quote, quote_map, schedule in zip(labels, normalized_quotes, quote_maps, schedules):
        target_transform = quote_map.target_price(float(quote.quote))
        if target_transform.failure is not None:
            raise ValueError(
                f"credit quote_to_running_spread failed for `{label}`: {target_transform.failure}"
            )
        for warning in target_transform.warnings:
            quote_transform_warnings.append(f"{label}: {warning}")
        target_running_spread = float(target_transform.value)
        if target_running_spread <= 0.0 or not raw_np.isfinite(target_running_spread):
            raise ValueError(
                f"credit quote `{label}` mapped to non-positive running spread `{target_running_spread}`"
            )
        target_running_spreads_values.append(target_running_spread)
        try:
            target_hazard = _hazard_from_par_spread(
                running_spread=target_running_spread,
                schedule=schedule,
                maturity_years=float(quote.maturity_years),
                discount_curve=market_state.discount,
                recovery=recovery,
                max_hazard=float(max_hazard),
            )
        except Exception as exc:
            raise ValueError(
                f"credit running_spread_to_hazard failed for `{label}` while normalizing targets: {exc}"
            ) from exc
        target_hazards_values.append(float(target_hazard))
    target_running_spreads = tuple(target_running_spreads_values)
    target_hazards = tuple(target_hazards_values)
    weights = tuple(float(quote.weight) for quote in normalized_quotes)
    upper_bound = max(float(max_hazard), max(target_hazards) * 2.0, 1.0)
    parameter_names = tuple(f"hazard_{index + 1}" for index in range(len(normalized_quotes)))
    fit_value_kinds = tuple(
        "upfront" if quote.quote_kind == "upfront" else "par_spread"
        for quote in normalized_quotes
    )
    target_fit_values = tuple(
        normalize_cds_upfront_quote(float(quote.quote))
        if quote.quote_kind == "upfront"
        else float(target_running_spread)
        for quote, target_running_spread in zip(normalized_quotes, target_running_spreads)
    )

    def _model_running_spreads_for_params(params) -> raw_np.ndarray:
        try:
            curve = CreditCurve(tenors, tuple(float(value) for value in params))
        except Exception:
            return raw_np.full(len(tenors), 10.0, dtype=float)
        model_spreads: list[float] = []
        for schedule in schedules:
            try:
                model_spreads.append(
                    _par_spread_from_curve(
                        schedule=schedule,
                        credit_curve=curve,
                        discount_curve=market_state.discount,
                        recovery=recovery,
                    )
                )
            except Exception:
                model_spreads.append(10.0)
        return raw_np.asarray(model_spreads, dtype=float)

    def _model_fit_values_for_params(params) -> raw_np.ndarray:
        try:
            curve = CreditCurve(tenors, tuple(float(value) for value in params))
        except Exception:
            return raw_np.full(len(tenors), 10.0, dtype=float)
        model_values: list[float] = []
        for quote, schedule in zip(normalized_quotes, schedules):
            try:
                if quote.quote_kind == "upfront":
                    model_values.append(
                        price_cds_analytical(
                            notional=_CDS_CALIBRATION_NOTIONAL,
                            spread_quote=float(quote.standard_running_spread),
                            recovery=recovery,
                            schedule=schedule,
                            credit_curve=curve,
                            discount_curve=market_state.discount,
                        )
                    )
                else:
                    model_values.append(
                        _par_spread_from_curve(
                            schedule=schedule,
                            credit_curve=curve,
                            discount_curve=market_state.discount,
                            recovery=recovery,
                        )
                    )
            except Exception:
                model_values.append(10.0)
        return raw_np.asarray(model_values, dtype=float)

    fit_space = (
        "cds_quote_style_value"
        if any(kind == "upfront" for kind in fit_value_kinds)
        else "cds_par_spread"
    )
    request_id = (
        "single_name_credit_schedule_aware_cds_least_squares"
        if quote_normalization_method == "cds_pricer_schedule_aware"
        else "single_name_credit_cds_par_spread_least_squares"
    )
    solve_request = SolveRequest(
        request_id=request_id,
        problem_kind="least_squares",
        parameter_names=parameter_names,
        initial_guess=target_hazards,
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=labels,
            target_values=target_fit_values,
            weights=weights,
            vector_objective_fn=_model_fit_values_for_params,
            metadata={
                "model_family": "reduced_form_credit",
                "curve_name": curve_name,
                "fit_space": fit_space,
                "fit_value_kinds": list(fit_value_kinds),
                "quote_maps": [quote_map.to_payload() for quote_map in quote_maps],
                "potential_binding": dict(potential_binding),
            },
        ),
        bounds=SolveBounds(
            lower=tuple(0.0 for _ in parameter_names),
            upper=tuple(upper_bound for _ in parameter_names),
        ),
        solver_hint="trf",
        warm_start=WarmStart(parameter_values=target_hazards, source="quote_map_seed"),
        metadata={
            "curve_name": curve_name,
            "model_family": "reduced_form_credit",
            "fit_space": fit_space,
            "fit_value_kinds": list(fit_value_kinds),
            "selected_curve_names": dict(market_state.selected_curve_names or {}),
            "potential_binding": dict(potential_binding),
        },
        options={"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12, "maxiter": 80},
    )
    solve_result = execute_solve_request(solve_request)
    if not solve_result.success:
        raise ValueError(
            "single-name credit calibration failed: "
            f"{solve_result.metadata.get('message', 'unknown failure')}"
        )
    model_hazards = tuple(float(value) for value in solve_result.solution)
    credit_curve = CreditCurve(tenors, model_hazards)
    model_running_spreads = tuple(float(value) for value in _model_running_spreads_for_params(model_hazards))
    model_fit_values = tuple(float(value) for value in _model_fit_values_for_params(model_hazards))

    model_quotes_values: list[float] = []
    quote_inverse_failures: list[str] = []
    for label, quote, model_running_spread, quote_map, schedule in zip(
        labels,
        normalized_quotes,
        model_running_spreads,
        quote_maps,
        schedules,
    ):
        if quote.quote_kind == "upfront":
            model_upfront = price_cds_analytical(
                notional=_CDS_CALIBRATION_NOTIONAL,
                spread_quote=float(quote.standard_running_spread),
                recovery=recovery,
                schedule=schedule,
                credit_curve=credit_curve,
                discount_curve=market_state.discount,
            )
            model_quotes_values.append(
                float(model_upfront * 100.0)
                if abs(float(quote.quote)) > 1.0
                else float(model_upfront)
            )
            continue
        model_transform = quote_map.model_quote(float(model_running_spread))
        if model_transform.failure is not None:
            quote_inverse_failures.append(f"{label}: {model_transform.failure}")
            model_quotes_values.append(float("nan"))
            continue
        for warning in model_transform.warnings:
            quote_transform_warnings.append(f"{label}: {warning}")
        model_quotes_values.append(float(model_transform.value))
    model_quotes = tuple(model_quotes_values)
    repricing_errors_values: list[float] = []
    upfront_repricing_errors_values: list[float] = []
    for quote, target_running_spread, schedule in zip(normalized_quotes, target_running_spreads, schedules):
        if quote.quote_kind == "upfront":
            upfront_error = float(
                price_cds_analytical(
                    notional=_CDS_CALIBRATION_NOTIONAL,
                    spread_quote=float(quote.standard_running_spread),
                    recovery=recovery,
                    schedule=schedule,
                    credit_curve=credit_curve,
                    discount_curve=market_state.discount,
                )
                - normalize_cds_upfront_quote(float(quote.quote))
            )
            repricing_errors_values.append(upfront_error)
            upfront_repricing_errors_values.append(upfront_error)
            continue
        par_error = float(
            price_cds_analytical(
                notional=_CDS_CALIBRATION_NOTIONAL,
                spread_quote=float(target_running_spread),
                recovery=recovery,
                schedule=schedule,
                credit_curve=credit_curve,
                discount_curve=market_state.discount,
            )
        )
        repricing_errors_values.append(par_error)
        upfront_repricing_errors_values.append(0.0)
    repricing_errors = tuple(repricing_errors_values)
    upfront_repricing_errors = tuple(upfront_repricing_errors_values)
    survival_probabilities, forward_hazards = _survival_and_forward_hazards(credit_curve, tenors)
    hazard_residuals = tuple(
        float(model_hazard - target_hazard)
        for model_hazard, target_hazard in zip(model_hazards, target_hazards)
    )
    quote_residuals = tuple(
        float(model_quote - target_quote)
        for model_quote, target_quote in zip(model_quotes, target_quotes)
    )
    max_abs_repricing_error = max((abs(value) for value in repricing_errors), default=0.0)
    max_abs_hazard_residual = max((abs(value) for value in hazard_residuals), default=0.0)
    finite_quote_residuals = tuple(abs(value) for value in quote_residuals if value == value)
    max_abs_quote_residual = max(finite_quote_residuals, default=0.0)
    survival_monotone = all(
        survival_probabilities[index + 1] <= survival_probabilities[index] + 1e-10
        for index in range(len(survival_probabilities) - 1)
    )
    min_forward_hazard = min(forward_hazards, default=0.0)
    max_forward_hazard = max(forward_hazards, default=0.0)
    hazard_governance = {
        "survival_monotone": bool(survival_monotone),
        "forward_hazards_non_negative": bool(min_forward_hazard >= -1e-10),
        "forward_hazards_within_max_hazard": bool(max_forward_hazard <= float(max_hazard) + 1e-10),
        "min_forward_hazard": float(min_forward_hazard),
        "max_forward_hazard": float(max_forward_hazard),
        "max_allowed_hazard": float(max_hazard),
        "recovery": float(recovery),
    }

    solver_provenance = build_solve_provenance(solve_request, solve_result)
    solver_replay_artifact = build_solve_replay_artifact(solve_request, solve_result)
    warnings: list[str] = []
    warnings.extend(quote_transform_warnings)
    warnings.extend(quote_inverse_failures)
    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
    if "source_kind" not in market_provenance:
        warnings.append(
            "market_state.market_provenance did not include source_kind; "
            "calibration preserved selected curve names only."
        )
    if any(forward_hazard < -1e-10 for forward_hazard in forward_hazards):
        warnings.append("credit fit produced negative forward hazards on the bounded tenor grid.")
    if not survival_monotone:
        warnings.append("credit fit produced non-monotone survival probabilities on the bounded tenor grid.")
    if max_forward_hazard > float(max_hazard) + 1e-10:
        warnings.append("credit fit produced a forward hazard above max_hazard governance.")

    fit_diagnostics = {
        "fit_space": fit_space,
        "fit_value_kinds": list(fit_value_kinds),
        "target_fit_values": list(target_fit_values),
        "model_fit_values": list(model_fit_values),
        "target_running_spreads": list(target_running_spreads),
        "model_running_spreads": list(model_running_spreads),
        "repricing_errors": list(repricing_errors),
        "upfront_repricing_errors": list(upfront_repricing_errors),
        "survival_probabilities": list(survival_probabilities),
        "forward_hazards": list(forward_hazards),
        "max_abs_repricing_error": float(max_abs_repricing_error),
        "max_abs_hazard_residual": float(max_abs_hazard_residual),
        "max_abs_quote_residual": float(max_abs_quote_residual),
        "hazard_governance": dict(hazard_governance),
    }

    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "calibrate_single_name_credit_curve_workflow",
        "selected_curve_names": dict(market_state.selected_curve_names or {}),
        "market_provenance": market_provenance,
        "potential_binding": dict(potential_binding),
        "calibration_target": {
            "labels": list(labels),
            "tenors": list(tenors),
            "target_running_spreads": list(target_running_spreads),
            "target_hazards": list(target_hazards),
            "target_quotes": list(target_quotes),
            "quote_kinds": [quote.quote_kind for quote in normalized_quotes],
            "quote_maps": [quote_map.to_payload() for quote_map in quote_maps],
            "instrument_setup": [
                {
                    **dict(instrument_setup),
                    "label": label,
                    "maturity_years": float(quote.maturity_years),
                    "standard_running_spread": (
                        None
                        if quote.standard_running_spread is None
                        else normalize_cds_running_spread(float(quote.standard_running_spread))
                    ),
                }
                for label, quote, instrument_setup in zip(labels, normalized_quotes, instrument_setups)
            ],
            "quote_inverse_failures": list(quote_inverse_failures),
            "curve_name": curve_name,
            "recovery": recovery,
        },
        "solve_request": solve_request.to_payload(),
        "solve_result": solve_result.to_payload(),
        "solver_provenance": solver_provenance.to_payload(),
        "solver_replay_artifact": solver_replay_artifact.to_payload(),
        "fit_diagnostics": fit_diagnostics,
        "warnings": list(warnings),
        "assumptions": list(assumptions),
    }
    summary = {
        "quote_count": len(normalized_quotes),
        "curve_name": curve_name,
        "recovery": recovery,
        "quote_normalization_method": quote_normalization_method,
        "max_abs_repricing_error": float(max_abs_repricing_error),
        "max_abs_hazard_residual": float(max_abs_hazard_residual),
        "max_abs_quote_residual": float(max_abs_quote_residual),
        "hazard_governance": dict(hazard_governance),
        "quote_families": [quote_map.spec.quote_family for quote_map in quote_maps],
        "quote_conventions": [quote_map.spec.convention for quote_map in quote_maps],
    }
    return CreditHazardCalibrationResult(
        quotes=normalized_quotes,
        credit_curve=credit_curve,
        solve_request=solve_request,
        solve_result=solve_result,
        solver_provenance=solver_provenance,
        solver_replay_artifact=solver_replay_artifact,
        tenors=tenors,
        target_running_spreads=target_running_spreads,
        model_running_spreads=model_running_spreads,
        target_hazards=target_hazards,
        model_hazards=model_hazards,
        target_quotes=target_quotes,
        model_quotes=model_quotes,
        repricing_errors=repricing_errors,
        survival_probabilities=survival_probabilities,
        forward_hazards=forward_hazards,
        hazard_residuals=hazard_residuals,
        quote_residuals=quote_residuals,
        max_abs_repricing_error=max_abs_repricing_error,
        max_abs_hazard_residual=max_abs_hazard_residual,
        max_abs_quote_residual=max_abs_quote_residual,
        curve_name=curve_name,
        recovery=recovery,
        potential_binding=potential_binding,
        provenance=provenance,
        summary=summary,
        warnings=tuple(warnings),
        assumptions=assumptions,
    )


__all__ = [
    "CreditHazardCalibrationQuote",
    "CreditHazardCalibrationResult",
    "calibrate_single_name_credit_curve_workflow",
]
