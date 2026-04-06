"""Rates calibration helpers for cap/floor and swaption Black-vol fits.

These helpers solve the most common desk-style rates calibration problems in a
multi-curve setting:

* cap/floor quotes calibrated to a flat Black volatility
* European swaption quotes calibrated to a flat Black volatility

The helpers keep the curve-selection provenance from ``MarketState`` in the
result object so downstream traces and replay tools can explain which discount
and forecast curves were used in the calibration run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal, Protocol, Sequence

from trellis.models.bermudan_swaption_tree import BermudanSwaptionTreeSpec, price_bermudan_swaption_tree
from trellis.core.date_utils import build_payment_timeline, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.instruments.cap import CapFloorSpec
from trellis.models.black import black76_call, black76_put
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
from trellis.models.calibration.quote_maps import (
    CalibrationQuoteMap,
    QuoteMapSpec,
    build_identity_quote_map,
    build_implied_vol_quote_map,
)
from trellis.models.calibration.materialization import materialize_model_parameter_set
from trellis.models.hull_white_parameters import build_hull_white_parameter_payload


class SwaptionLike(Protocol):
    """Protocol for the swaption-like inputs used by the calibration helpers."""

    notional: float
    strike: float
    expiry_date: date
    swap_start: date
    swap_end: date
    swap_frequency: Frequency
    day_count: DayCountConvention
    rate_index: str | None
    is_payer: bool


@dataclass(frozen=True)
class _CapFloorTerm:
    """One surviving cap/floor period resolved onto pricing terms."""

    accrual_fraction: float
    fixing_years: float
    payment_years: float
    discount_factor: float
    forward_rate: float


@dataclass(frozen=True)
class RatesCalibrationResult:
    """Structured result for a rates Black-vol calibration."""

    instrument_family: str
    instrument_kind: str
    target_price: float
    calibrated_vol: float
    repriced_price: float
    residual: float
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HullWhiteCalibrationInstrument:
    """One supported swaption-style quote for the Hull-White calibration workflow."""

    notional: float
    strike: float
    exercise_date: date
    swap_end: date
    quote: float
    quote_kind: Literal["price", "black_vol"] = "black_vol"
    label: str = ""
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.quote_kind not in {"price", "black_vol"}:
            raise ValueError("quote_kind must be 'price' or 'black_vol'")
        if self.swap_end <= self.exercise_date:
            raise ValueError("swap_end must be after exercise_date")
        if float(self.notional) <= 0.0:
            raise ValueError("notional must be positive")
        if float(self.weight) <= 0.0:
            raise ValueError("weight must be positive")
        object.__setattr__(self, "notional", float(self.notional))
        object.__setattr__(self, "strike", float(self.strike))
        object.__setattr__(self, "quote", float(self.quote))
        object.__setattr__(self, "weight", float(self.weight))

    def resolved_label(self, index: int) -> str:
        """Return a stable label for the quote inside a solve request."""
        if self.label:
            return self.label
        option_side = "payer" if self.is_payer else "receiver"
        return f"{option_side}_{self.exercise_date.isoformat()}_{self.swap_end.isoformat()}_{index}"

    def tree_spec(self) -> BermudanSwaptionTreeSpec:
        """Return the one-exercise tree spec used by the calibration workflow."""
        return BermudanSwaptionTreeSpec(
            notional=float(self.notional),
            strike=float(self.strike),
            exercise_dates=(self.exercise_date,),
            swap_end=self.swap_end,
            swap_frequency=self.swap_frequency,
            day_count=self.day_count,
            rate_index=self.rate_index,
            is_payer=bool(self.is_payer),
        )

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "label": self.label,
            "notional": float(self.notional),
            "strike": float(self.strike),
            "exercise_date": self.exercise_date.isoformat(),
            "swap_end": self.swap_end.isoformat(),
            "quote": float(self.quote),
            "quote_kind": self.quote_kind,
            "swap_frequency": self.swap_frequency.name,
            "day_count": self.day_count.name,
            "rate_index": self.rate_index,
            "is_payer": bool(self.is_payer),
            "weight": float(self.weight),
        }


@dataclass(frozen=True)
class HullWhiteCalibrationResult:
    """Structured result for the supported Hull-White calibration workflow."""

    instruments: tuple[HullWhiteCalibrationInstrument, ...]
    mean_reversion: float
    sigma: float
    solve_request: SolveRequest
    solve_result: SolveResult
    solver_provenance: SolveProvenance
    solver_replay_artifact: SolveReplayArtifact
    target_prices: tuple[float, ...]
    model_prices: tuple[float, ...]
    target_quotes: tuple[float, ...]
    model_quotes: tuple[float, ...]
    price_residuals: tuple[float, ...]
    quote_residuals: tuple[float, ...]
    max_abs_price_residual: float
    max_abs_quote_residual: float
    parameter_set_name: str = "hull_white"
    model_parameters: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(self, "mean_reversion", float(self.mean_reversion))
        object.__setattr__(self, "sigma", float(self.sigma))
        object.__setattr__(self, "target_prices", tuple(float(value) for value in self.target_prices))
        object.__setattr__(self, "model_prices", tuple(float(value) for value in self.model_prices))
        object.__setattr__(self, "target_quotes", tuple(float(value) for value in self.target_quotes))
        object.__setattr__(self, "model_quotes", tuple(float(value) for value in self.model_quotes))
        object.__setattr__(self, "price_residuals", tuple(float(value) for value in self.price_residuals))
        object.__setattr__(self, "quote_residuals", tuple(float(value) for value in self.quote_residuals))
        object.__setattr__(self, "max_abs_price_residual", float(self.max_abs_price_residual))
        object.__setattr__(self, "max_abs_quote_residual", float(self.max_abs_quote_residual))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    def apply_to_market_state(self, market_state: MarketState) -> MarketState:
        """Return ``market_state`` enriched with the calibrated Hull-White parameters."""
        selected_curve_roles: dict[str, str] = {}
        calibration_target = self.provenance.get("calibration_target", {})
        if isinstance(calibration_target, dict):
            quote_maps = calibration_target.get("quote_maps", ())
            if isinstance(quote_maps, Sequence) and quote_maps:
                first_quote_map = quote_maps[0]
                if isinstance(first_quote_map, dict):
                    selected_curve_roles = {
                        str(key): str(value)
                        for key, value in dict(first_quote_map.get("multi_curve_roles", {})).items()
                        if str(key).strip() and str(value).strip()
                    }
        if not selected_curve_roles:
            selected_curve_roles = {
                str(key): str(value)
                for key, value in dict(market_state.selected_curve_names or {}).items()
                if str(key).strip() and str(value).strip()
            }
        return materialize_model_parameter_set(
            market_state,
            parameter_set_name=self.parameter_set_name,
            model_parameters=dict(self.model_parameters),
            source_kind=str(self.model_parameters.get("source_kind", "calibrated")),
            source_ref="calibrate_hull_white",
            selected_curve_roles=selected_curve_roles,
            metadata={
                "instrument_family": "rates",
                "instrument_kind": "hull_white",
                "parameter_set_name": self.parameter_set_name,
            },
        )

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "instruments": [instrument.to_payload() for instrument in self.instruments],
            "mean_reversion": self.mean_reversion,
            "sigma": self.sigma,
            "solve_request": self.solve_request.to_payload(),
            "solve_result": self.solve_result.to_payload(),
            "solver_provenance": self.solver_provenance.to_payload(),
            "solver_replay_artifact": self.solver_replay_artifact.to_payload(),
            "target_prices": list(self.target_prices),
            "model_prices": list(self.model_prices),
            "target_quotes": list(self.target_quotes),
            "model_quotes": list(self.model_quotes),
            "price_residuals": list(self.price_residuals),
            "quote_residuals": list(self.quote_residuals),
            "max_abs_price_residual": self.max_abs_price_residual,
            "max_abs_quote_residual": self.max_abs_quote_residual,
            "parameter_set_name": self.parameter_set_name,
            "model_parameters": dict(self.model_parameters),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
            "warnings": list(self.warnings),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class _HullWhiteSwaptionLike:
    """Internal swaption-like contract shared by the workflow helpers."""

    notional: float
    strike: float
    expiry_date: date
    swap_start: date
    swap_end: date
    swap_frequency: Frequency
    day_count: DayCountConvention
    rate_index: str | None
    is_payer: bool


def _build_provenance(
    market_state: MarketState,
    *,
    rate_index: str | None = None,
    vol_surface_name: str | None = None,
    correlation_source: str | None = None,
) -> dict[str, object]:
    """Return a JSON-serializable provenance payload for calibration runs."""
    provenance: dict[str, object] = {
        "selected_curve_names": dict(market_state.selected_curve_names or {}),
    }
    market_provenance = getattr(market_state, "market_provenance", None)
    if market_provenance:
        provenance["market_provenance"] = dict(market_provenance)
    if rate_index is not None:
        provenance["rate_index"] = rate_index
    if vol_surface_name is not None:
        provenance["vol_surface_name"] = vol_surface_name
    if correlation_source is not None:
        provenance["correlation_source"] = correlation_source
    return provenance


def _implied_flat_vol(
    price_fn: Callable[[float], float],
    target_price: float,
    *,
    lower: float = 0.0,
    upper: float = 5.0,
    tol: float = 1e-8,
) -> tuple[float, SolveRequest, SolveResult]:
    """Solve for the flat Black volatility that reproduces ``target_price``."""
    if lower < 0.0:
        raise ValueError("lower volatility bound must be non-negative")
    if upper <= lower:
        raise ValueError("upper volatility bound must be greater than lower")

    target_price = float(target_price)
    midpoint = 0.5 * (lower + upper)
    request = SolveRequest(
        request_id="rates_flat_black_vol_root",
        problem_kind="root_scalar",
        parameter_names=("flat_black_vol",),
        initial_guess=(midpoint,),
        objective=ObjectiveBundle(
            objective_kind="root_scalar",
            labels=("price_residual",),
            target_values=(0.0,),
            scalar_objective_fn=lambda vol: float(price_fn(float(vol))) - target_price,
            metadata={"target_price": float(target_price)},
        ),
        bounds=SolveBounds(lower=(lower,), upper=(upper,)),
        solver_hint="brentq",
        warm_start=WarmStart(parameter_values=(midpoint,), source="interval_midpoint"),
        metadata={"problem_family": "rates_flat_black_vol", "solver_family": "scipy"},
        options={"tol": float(tol)},
    )
    low_price = float(price_fn(lower))
    low_residual = low_price - target_price
    if abs(low_residual) <= tol:
        return (
            float(lower),
            request,
            SolveResult(
                solution=(float(lower),),
                objective_value=abs(low_residual),
                residual_vector=(low_residual,),
                success=True,
                method="brentq",
                metadata={
                    "solver_family": "scipy",
                    "backend_id": "scipy",
                    "requested_backend": "scipy",
                    "message": "lower bracket matched target price",
                },
            ),
        )

    high_price = float(price_fn(upper))
    high_residual = high_price - target_price
    if abs(high_residual) <= tol:
        return (
            float(upper),
            request,
            SolveResult(
                solution=(float(upper),),
                objective_value=abs(high_residual),
                residual_vector=(high_residual,),
                success=True,
                method="brentq",
                metadata={
                    "solver_family": "scipy",
                    "backend_id": "scipy",
                    "requested_backend": "scipy",
                    "message": "upper bracket matched target price",
                },
            ),
        )

    if low_residual * high_residual > 0:
        raise ValueError(
            "target price is not bracketed by the supplied volatility range; "
            f"price(lower={lower})={low_price:.10g}, price(upper={upper})={high_price:.10g}, "
            f"target={target_price:.10g}"
        )
    solve_result = execute_solve_request(request)
    return float(solve_result.solution[0]), request, solve_result


def _solver_artifacts(
    solve_request: SolveRequest,
    solve_result: SolveResult,
) -> tuple[SolveProvenance, SolveReplayArtifact]:
    """Build the standardized solver provenance and replay artifacts."""
    return (
        build_solve_provenance(solve_request, solve_result),
        build_solve_replay_artifact(solve_request, solve_result),
    )


def _rates_residual_tolerance_abs(
    target_price: float,
    *,
    solve_tol: float,
) -> float:
    """Return the shared absolute residual tolerance for rates-vol calibration checks."""
    target_scale = max(abs(float(target_price)), 1.0)
    return max(1e-5, target_scale * max(float(solve_tol), 0.0))


def _attach_residual_tolerance_policy(
    summary: dict[str, object],
    *,
    target_price: float,
    residual: float,
    solve_tol: float,
) -> dict[str, object]:
    """Attach residual/tolerance diagnostics using the shared rates policy."""
    tolerance_abs = _rates_residual_tolerance_abs(target_price, solve_tol=solve_tol)
    summary["residual_tolerance_abs"] = float(tolerance_abs)
    summary["residual_within_tolerance"] = abs(float(residual)) <= float(tolerance_abs)
    return summary


def _rates_implied_vol_quote_map(
    market_state: MarketState,
    *,
    rate_index: str | None,
    quote_to_price_fn: Callable[[float], float],
    price_to_quote_fn: Callable[[float], float] | None,
    source_ref: str,
) -> CalibrationQuoteMap:
    """Return the shared implied-vol quote map for rates workflows."""
    return build_implied_vol_quote_map(
        convention="black",
        quote_to_price_fn=quote_to_price_fn,
        price_to_quote_fn=price_to_quote_fn,
        source_ref=source_ref,
        assumptions=_rates_quote_assumptions(market_state, rate_index=rate_index),
        metadata={"multi_curve_roles": _rates_multi_curve_roles(market_state, rate_index=rate_index)},
    )


def _hull_white_swaption_like(instrument: HullWhiteCalibrationInstrument) -> _HullWhiteSwaptionLike:
    """Return the Black76-style swaption view used for quote normalization."""
    return _HullWhiteSwaptionLike(
        notional=float(instrument.notional),
        strike=float(instrument.strike),
        expiry_date=instrument.exercise_date,
        swap_start=instrument.exercise_date,
        swap_end=instrument.swap_end,
        swap_frequency=instrument.swap_frequency,
        day_count=instrument.day_count,
        rate_index=instrument.rate_index,
        is_payer=bool(instrument.is_payer),
    )


def _rates_multi_curve_roles(
    market_state: MarketState,
    *,
    rate_index: str | None,
) -> dict[str, object]:
    """Return explicit multi-curve role bindings used by rates quote maps."""
    selected = dict(market_state.selected_curve_names or {})
    return {
        "discount_curve": selected.get("discount_curve"),
        "forecast_curve": selected.get("forecast_curve"),
        "rate_index": rate_index,
    }


def _rates_quote_assumptions(
    market_state: MarketState,
    *,
    rate_index: str | None,
) -> tuple[str, ...]:
    """Return standardized rates quote-assumption labels."""
    roles = _rates_multi_curve_roles(market_state, rate_index=rate_index)
    return (
        "Rates quote transforms use explicit multi-curve discount/forecast roles.",
        (
            "Discount curve role: "
            f"{roles.get('discount_curve') or '<unbound>'}; "
            f"forecast curve role: {roles.get('forecast_curve') or '<unbound>'}."
        ),
        f"Rate-index binding: {roles.get('rate_index') or '<unbound>'}.",
    )


def _hull_white_quote_map(
    instrument: HullWhiteCalibrationInstrument,
    market_state: MarketState,
) -> CalibrationQuoteMap:
    """Return the explicit quote map for one Hull-White calibration instrument."""
    roles = _rates_multi_curve_roles(market_state, rate_index=instrument.rate_index)
    assumptions = _rates_quote_assumptions(market_state, rate_index=instrument.rate_index)
    metadata = {
        "multi_curve_roles": roles,
        "quote_kind": instrument.quote_kind,
    }
    if instrument.quote_kind == "price":
        return build_identity_quote_map(
            QuoteMapSpec(quote_family="price"),
            source_ref="_hull_white_quote_map",
            assumptions=assumptions,
            metadata=metadata,
        )
    swaption_like = _hull_white_swaption_like(instrument)
    return build_implied_vol_quote_map(
        convention="black",
        quote_to_price_fn=lambda quote: _swaption_black76_price(swaption_like, market_state, float(quote))[0],
        price_to_quote_fn=lambda price: calibrate_swaption_black_vol(
            swaption_like,
            market_state,
            float(price),
            tol=1e-10,
        ).calibrated_vol,
        source_ref="_hull_white_quote_map",
        assumptions=assumptions,
        metadata=metadata,
    )


def _initial_hull_white_guess(
    instruments: Sequence[HullWhiteCalibrationInstrument],
    market_state: MarketState,
    *,
    initial_guess: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return the starting point for the Hull-White least-squares solve."""
    if initial_guess is not None:
        return float(initial_guess[0]), float(initial_guess[1])

    first = instruments[0]
    settlement = market_state.settlement
    horizon = year_fraction(settlement, first.swap_end, first.day_count)
    r0 = float(market_state.discount.zero_rate(max(horizon / 2.0, 1e-6)))
    sigma_seed = 0.01
    if first.quote_kind == "black_vol":
        sigma_seed = float(first.quote) * max(abs(r0), 1e-6)
    elif market_state.vol_surface is not None:
        option_horizon = year_fraction(settlement, first.exercise_date, first.day_count)
        black_vol = float(
            market_state.vol_surface.black_vol(max(option_horizon, 1e-6), max(abs(float(first.strike)), 1e-6))
        )
        sigma_seed = black_vol * max(abs(r0), 1e-6)
    return 0.1, max(float(sigma_seed), 1e-6)


def calibrate_hull_white(
    instruments: Sequence[HullWhiteCalibrationInstrument],
    market_state: MarketState,
    *,
    mean_reversion_bounds: tuple[float | None, float | None] = (1e-4, 1.0),
    sigma_bounds: tuple[float | None, float | None] = (1e-6, 1.0),
    initial_guess: tuple[float, float] | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    n_steps: int | None = None,
    parameter_set_name: str = "hull_white",
) -> HullWhiteCalibrationResult:
    """Calibrate Hull-White mean reversion and sigma to a supported swaption strip."""
    if market_state.discount is None:
        raise ValueError("Hull-White calibration requires market_state.discount")

    resolved_instruments = tuple(instruments)
    if len(resolved_instruments) < 2:
        raise ValueError("Hull-White calibration requires at least two instruments")

    labels = tuple(instrument.resolved_label(index) for index, instrument in enumerate(resolved_instruments))
    quote_maps = tuple(_hull_white_quote_map(instrument, market_state) for instrument in resolved_instruments)
    target_price_values: list[float] = []
    quote_transform_warnings: list[str] = []
    for label, instrument, quote_map in zip(labels, resolved_instruments, quote_maps):
        target_transform = quote_map.target_price(float(instrument.quote))
        if target_transform.failure is not None:
            raise ValueError(f"Hull-White quote_to_price failed for `{label}`: {target_transform.failure}")
        for warning in target_transform.warnings:
            quote_transform_warnings.append(f"{label}: {warning}")
        target_price_values.append(float(target_transform.value))
    target_prices = tuple(target_price_values)
    weights = tuple(float(instrument.weight) for instrument in resolved_instruments)
    start = _initial_hull_white_guess(
        resolved_instruments,
        market_state,
        initial_guess=initial_guess,
    )

    def vector_objective(params):
        mean_reversion = float(params[0])
        sigma = float(params[1])
        return [
            price_bermudan_swaption_tree(
                market_state,
                instrument.tree_spec(),
                model="hull_white",
                mean_reversion=mean_reversion,
                sigma=sigma,
                n_steps=n_steps,
            )
            for instrument in resolved_instruments
        ]

    solve_request = SolveRequest(
        request_id="hull_white_swaption_least_squares",
        problem_kind="least_squares",
        parameter_names=("mean_reversion", "sigma"),
        initial_guess=start,
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=labels,
            target_values=target_prices,
            weights=weights,
            vector_objective_fn=vector_objective,
            metadata={
                "instrument_count": len(resolved_instruments),
                "parameter_set_name": parameter_set_name,
                "selected_curve_names": dict(market_state.selected_curve_names or {}),
            },
        ),
        bounds=SolveBounds(lower=mean_reversion_bounds[:1] + sigma_bounds[:1], upper=mean_reversion_bounds[1:] + sigma_bounds[1:]),
        solver_hint="trf",
        warm_start=WarmStart(parameter_values=start, source="heuristic_seed"),
        metadata={
            "problem_family": "hull_white_swaption_calibration",
            "parameter_set_name": parameter_set_name,
            "selected_curve_names": dict(market_state.selected_curve_names or {}),
        },
        options={"maxiter": int(max_iter), "ftol": float(tol), "xtol": float(tol), "gtol": float(tol)},
    )
    solve_result = execute_solve_request(solve_request)
    if not solve_result.success:
        raise ValueError(f"Hull-White calibration failed: {solve_result.metadata.get('message', 'unknown failure')}")

    mean_reversion, sigma = solve_result.solution
    model_prices = tuple(
        float(
            price_bermudan_swaption_tree(
                market_state,
                instrument.tree_spec(),
                model="hull_white",
                mean_reversion=mean_reversion,
                sigma=sigma,
                n_steps=n_steps,
            )
        )
        for instrument in resolved_instruments
    )
    target_quotes = tuple(float(instrument.quote) for instrument in resolved_instruments)
    model_quote_values: list[float] = []
    quote_inverse_failures: list[str] = []
    for label, quote_map, model_price in zip(labels, quote_maps, model_prices):
        model_transform = quote_map.model_quote(float(model_price))
        if model_transform.failure is not None:
            quote_inverse_failures.append(f"{label}: {model_transform.failure}")
            model_quote_values.append(float("nan"))
            continue
        for warning in model_transform.warnings:
            quote_transform_warnings.append(f"{label}: {warning}")
        model_quote_values.append(float(model_transform.value))
    model_quotes = tuple(model_quote_values)
    price_residuals = tuple(model_price - target_price for model_price, target_price in zip(model_prices, target_prices))
    quote_residuals = tuple(model_quote - target_quote for model_quote, target_quote in zip(model_quotes, target_quotes))
    max_abs_price_residual = max((abs(value) for value in price_residuals), default=0.0)
    finite_quote_residuals = tuple(abs(value) for value in quote_residuals if value == value)
    max_abs_quote_residual = max(finite_quote_residuals, default=0.0)

    solver_provenance, solver_replay_artifact = _solver_artifacts(solve_request, solve_result)
    assumptions = (
        "Supported Hull-White calibration quotes assume the underlying swap starts on the exercise date.",
    )
    warnings: list[str] = []
    warnings.extend(quote_transform_warnings)
    warnings.extend(quote_inverse_failures)
    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
    if "bootstrap_runs" not in market_provenance:
        warnings.append(
            "market_state.market_provenance did not include bootstrap_runs; only selected curve names were preserved."
        )

    model_parameters = build_hull_white_parameter_payload(
        mean_reversion,
        sigma,
        parameter_set_name=parameter_set_name,
        source_kind="calibrated",
        metadata={
            "selected_curve_names": dict(market_state.selected_curve_names or {}),
            "instrument_labels": labels,
        },
    )
    provenance = _build_provenance(market_state, rate_index=resolved_instruments[0].rate_index)
    provenance["solve_request"] = solve_request.to_payload()
    provenance["solve_result"] = solve_result.to_payload()
    provenance["solver_provenance"] = solver_provenance.to_payload()
    provenance["solver_replay_artifact"] = solver_replay_artifact.to_payload()
    provenance["calibration_target"] = {
        "labels": list(labels),
        "parameter_names": ["mean_reversion", "sigma"],
        "target_prices": list(target_prices),
        "target_quotes": list(target_quotes),
        "quote_kinds": [instrument.quote_kind for instrument in resolved_instruments],
        "quote_maps": [quote_map.to_payload() for quote_map in quote_maps],
        "quote_inverse_failures": list(quote_inverse_failures),
        "parameter_set_name": parameter_set_name,
        "n_steps": n_steps,
    }
    provenance["model_parameters"] = dict(model_parameters)
    provenance["assumptions"] = list(assumptions)
    provenance["warnings"] = list(warnings)

    summary = {
        "instrument_count": len(resolved_instruments),
        "parameter_set_name": parameter_set_name,
        "selected_curve_names": dict(market_state.selected_curve_names or {}),
        "max_abs_price_residual": float(max_abs_price_residual),
        "max_abs_quote_residual": float(max_abs_quote_residual),
        "quote_kinds": [instrument.quote_kind for instrument in resolved_instruments],
        "quote_families": [quote_map.spec.quote_family for quote_map in quote_maps],
        "quote_conventions": [quote_map.spec.convention for quote_map in quote_maps],
    }
    return HullWhiteCalibrationResult(
        instruments=resolved_instruments,
        mean_reversion=mean_reversion,
        sigma=sigma,
        solve_request=solve_request,
        solve_result=solve_result,
        solver_provenance=solver_provenance,
        solver_replay_artifact=solver_replay_artifact,
        target_prices=target_prices,
        model_prices=model_prices,
        target_quotes=target_quotes,
        model_quotes=model_quotes,
        price_residuals=price_residuals,
        quote_residuals=quote_residuals,
        max_abs_price_residual=max_abs_price_residual,
        max_abs_quote_residual=max_abs_quote_residual,
        parameter_set_name=parameter_set_name,
        model_parameters=model_parameters,
        provenance=provenance,
        summary=summary,
        warnings=tuple(warnings),
        assumptions=assumptions,
    )


def _cap_floor_terms(
    spec: CapFloorSpec,
    market_state: MarketState,
) -> tuple[_CapFloorTerm, ...]:
    """Resolve surviving cap/floor periods onto forward/discount pricing terms."""
    timeline = build_payment_timeline(
        spec.start_date,
        spec.end_date,
        spec.frequency,
        day_count=spec.day_count,
        time_origin=market_state.settlement,
        label="cap_floor_calibration_timeline",
    )
    fwd_curve = market_state.forecast_forward_curve(spec.rate_index)
    terms: list[_CapFloorTerm] = []
    for period in timeline:
        if period.payment_date <= market_state.settlement:
            continue
        accrual_fraction = float(period.accrual_fraction or 0.0)
        fixing_years = float(period.t_start or 0.0)
        payment_years = float(period.t_payment or 0.0)
        if fixing_years <= 0.0:
            continue
        fixing_years = max(fixing_years, 1e-6)
        payment_years = max(payment_years, fixing_years)
        discount_factor = float(market_state.discount.discount(payment_years))
        forward_rate = float(fwd_curve.forward_rate(fixing_years, payment_years))
        terms.append(
            _CapFloorTerm(
                accrual_fraction=accrual_fraction,
                fixing_years=fixing_years,
                payment_years=payment_years,
                discount_factor=discount_factor,
                forward_rate=forward_rate,
            )
        )
    return tuple(terms)


def _cap_floor_summary(
    spec: CapFloorSpec,
    market_state: MarketState,
    terms: Sequence[_CapFloorTerm] | None = None,
) -> dict[str, object]:
    """Return a compact summary of cap/floor calibration terms."""
    resolved_terms = tuple(terms) if terms is not None else _cap_floor_terms(spec, market_state)
    first_fix = resolved_terms[0].fixing_years if resolved_terms else None
    first_pay = resolved_terms[0].payment_years if resolved_terms else None
    last_pay = resolved_terms[-1].payment_years if resolved_terms else None
    annuity = sum(term.accrual_fraction * term.discount_factor for term in resolved_terms)
    float_leg_pv = sum(term.forward_rate * term.accrual_fraction * term.discount_factor for term in resolved_terms)
    forward_rate = float_leg_pv / annuity if annuity > 0.0 else 0.0
    return {
        "period_count": len(resolved_terms),
        "payment_count": len(resolved_terms),
        "frequency": spec.frequency.name,
        "day_count": getattr(spec.day_count, "name", str(spec.day_count)),
        "rate_index": spec.rate_index,
        "expiry_years": float(first_fix) if first_fix is not None else None,
        "annuity": float(annuity),
        "forward_rate": float(forward_rate),
        "first_fix_years": float(first_fix) if first_fix is not None else None,
        "first_pay_years": float(first_pay) if first_pay is not None else None,
        "last_pay_years": float(last_pay) if last_pay is not None else None,
    }


def _cap_floor_black76_price(
    spec: CapFloorSpec,
    market_state: MarketState,
    vol: float,
    *,
    kind: Literal["cap", "floor"],
) -> tuple[float, dict[str, object]]:
    """Return Black76 cap/floor PV and term summary from shared cap/floor terms."""
    if kind not in {"cap", "floor"}:
        raise ValueError(f"kind must be 'cap' or 'floor', got {kind!r}")
    terms = _cap_floor_terms(spec, market_state)
    if not terms:
        summary = _cap_floor_summary(spec, market_state, terms=terms)
        summary["option_type"] = kind
        return 0.0, summary

    option_kernel = black76_call if kind == "cap" else black76_put
    pv = 0.0
    for term in terms:
        option_value = float(option_kernel(term.forward_rate, spec.strike, vol, term.fixing_years))
        pv += spec.notional * term.accrual_fraction * term.discount_factor * option_value
    summary = _cap_floor_summary(spec, market_state, terms=terms)
    summary["option_type"] = kind
    return float(pv), summary


def swaption_terms(
    spec: SwaptionLike,
    market_state: MarketState,
) -> tuple[float, float, float, int]:
    """Return expiry, annuity, forward swap rate, and payment count."""
    timeline = build_payment_timeline(
        spec.swap_start,
        spec.swap_end,
        spec.swap_frequency,
        day_count=spec.day_count,
        time_origin=market_state.settlement,
        label="swaption_underlier_timeline",
    )
    if not timeline:
        return 0.0, 0.0, 0.0, 0

    fwd_curve = market_state.forecast_forward_curve(spec.rate_index)
    annuity = 0.0
    float_pv = 0.0
    payment_count = 0

    for period in timeline:
        if period.end_date <= market_state.settlement:
            continue
        tau = float(period.accrual_fraction or 0.0)
        t_start = float(period.t_start or 0.0)
        t_end = float(period.t_end or 0.0)
        t_start = max(t_start, 1e-6)
        df = float(market_state.discount.discount(t_end))
        fwd = float(fwd_curve.forward_rate(t_start, t_end))
        annuity += tau * df
        float_pv += fwd * tau * df
        payment_count += 1

    expiry = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
    swap_rate = float_pv / annuity if annuity > 0.0 else 0.0
    return float(expiry), float(annuity), float(swap_rate), payment_count


def _swaption_black76_price(
    spec: SwaptionLike,
    market_state: MarketState,
    vol: float,
) -> tuple[float, dict[str, object]]:
    """Return the Black76 swaption PV and a summary of the assembled terms."""
    T, annuity, swap_rate, payment_count = swaption_terms(spec, market_state)
    if T <= 0.0 or annuity <= 0.0:
        return 0.0, {
            "expiry_years": float(T),
            "annuity": float(annuity),
            "forward_swap_rate": float(swap_rate),
            "payment_count": payment_count,
            "option_type": "payer" if spec.is_payer else "receiver",
        }

    if spec.is_payer:
        option_value = black76_call(swap_rate, spec.strike, vol, T)
    else:
        option_value = black76_put(swap_rate, spec.strike, vol, T)

    pv = spec.notional * annuity * float(option_value)
    return float(pv), {
        "expiry_years": float(T),
        "annuity": float(annuity),
        "forward_swap_rate": float(swap_rate),
        "payment_count": payment_count,
        "option_type": "payer" if spec.is_payer else "receiver",
    }


def calibrate_cap_floor_black_vol(
    spec: CapFloorSpec,
    market_state: MarketState,
    target_price: float,
    *,
    kind: Literal["cap", "floor"] = "cap",
    vol_lower: float = 0.0,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
    vol_surface_name: str | None = None,
    correlation_source: str | None = None,
) -> RatesCalibrationResult:
    """Calibrate a flat Black volatility to a cap or floor price quote.

    The helper preserves the selected curve provenance from ``market_state``
    and adds optional source labels for the volatility and correlation inputs.
    """
    if kind not in {"cap", "floor"}:
        raise ValueError(f"kind must be 'cap' or 'floor', got {kind!r}")

    def price_at(vol: float) -> float:
        pv, _summary = _cap_floor_black76_price(spec, market_state, float(vol), kind=kind)
        return float(pv)

    quote_map = _rates_implied_vol_quote_map(
        market_state,
        rate_index=spec.rate_index,
        quote_to_price_fn=price_at,
        price_to_quote_fn=lambda price: _implied_flat_vol(
            price_at,
            float(price),
            lower=vol_lower,
            upper=vol_upper,
            tol=tol,
        )[0],
        source_ref="calibrate_cap_floor_black_vol",
    )
    calibrated_vol, solve_request, solve_result = _implied_flat_vol(
        price_at,
        target_price,
        lower=vol_lower,
        upper=vol_upper,
        tol=tol,
    )
    repriced_price, summary = _cap_floor_black76_price(spec, market_state, calibrated_vol, kind=kind)
    residual = float(repriced_price - target_price)
    summary["rate_index"] = spec.rate_index
    summary["strike"] = float(spec.strike)
    summary["notional"] = float(spec.notional)
    summary = _attach_residual_tolerance_policy(
        summary,
        target_price=target_price,
        residual=residual,
        solve_tol=tol,
    )
    provenance = _build_provenance(
        market_state,
        rate_index=spec.rate_index,
        vol_surface_name=vol_surface_name,
        correlation_source=correlation_source,
    )
    solver_provenance, solver_replay_artifact = _solver_artifacts(solve_request, solve_result)
    provenance["solve_request"] = solve_request.to_payload()
    provenance["solve_result"] = solve_result.to_payload()
    provenance["solver_provenance"] = solver_provenance.to_payload()
    provenance["solver_replay_artifact"] = solver_replay_artifact.to_payload()
    provenance["quote_map"] = quote_map.to_payload()
    summary["quote_family"] = quote_map.spec.quote_family
    summary["quote_convention"] = quote_map.spec.convention
    return RatesCalibrationResult(
        instrument_family="cap_floor",
        instrument_kind=kind,
        target_price=float(target_price),
        calibrated_vol=float(calibrated_vol),
        repriced_price=float(repriced_price),
        residual=float(residual),
        provenance=provenance,
        summary=summary,
    )


def calibrate_swaption_black_vol(
    spec: SwaptionLike,
    market_state: MarketState,
    target_price: float,
    *,
    vol_lower: float = 0.0,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
    vol_surface_name: str | None = None,
    correlation_source: str | None = None,
) -> RatesCalibrationResult:
    """Calibrate a flat Black volatility to a European swaption price quote."""

    def price_at(vol: float) -> float:
        pv, _summary = _swaption_black76_price(spec, market_state, float(vol))
        return pv

    quote_map = _rates_implied_vol_quote_map(
        market_state,
        rate_index=spec.rate_index,
        quote_to_price_fn=price_at,
        price_to_quote_fn=lambda price: _implied_flat_vol(
            price_at,
            float(price),
            lower=vol_lower,
            upper=vol_upper,
            tol=tol,
        )[0],
        source_ref="calibrate_swaption_black_vol",
    )
    calibrated_vol, solve_request, solve_result = _implied_flat_vol(
        price_at,
        target_price,
        lower=vol_lower,
        upper=vol_upper,
        tol=tol,
    )
    repriced_price, summary = _swaption_black76_price(spec, market_state, calibrated_vol)
    residual = float(repriced_price - target_price)
    provenance = _build_provenance(
        market_state,
        rate_index=spec.rate_index,
        vol_surface_name=vol_surface_name,
        correlation_source=correlation_source,
    )
    solver_provenance, solver_replay_artifact = _solver_artifacts(solve_request, solve_result)
    provenance["solve_request"] = solve_request.to_payload()
    provenance["solve_result"] = solve_result.to_payload()
    provenance["solver_provenance"] = solver_provenance.to_payload()
    provenance["solver_replay_artifact"] = solver_replay_artifact.to_payload()
    provenance["quote_map"] = quote_map.to_payload()
    summary["rate_index"] = spec.rate_index
    summary["strike"] = float(spec.strike)
    summary["notional"] = float(spec.notional)
    summary["quote_family"] = quote_map.spec.quote_family
    summary["quote_convention"] = quote_map.spec.convention
    summary = _attach_residual_tolerance_policy(
        summary,
        target_price=target_price,
        residual=residual,
        solve_tol=tol,
    )
    return RatesCalibrationResult(
        instrument_family="swaption",
        instrument_kind="payer" if spec.is_payer else "receiver",
        target_price=float(target_price),
        calibrated_vol=float(calibrated_vol),
        repriced_price=float(repriced_price),
        residual=float(residual),
        provenance=provenance,
        summary=summary,
    )


__all__ = [
    "HullWhiteCalibrationInstrument",
    "HullWhiteCalibrationResult",
    "RatesCalibrationResult",
    "calibrate_hull_white",
    "SwaptionLike",
    "calibrate_cap_floor_black_vol",
    "calibrate_swaption_black_vol",
    "swaption_terms",
]
