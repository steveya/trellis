"""Supported Heston smile calibration workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as raw_np

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.calibration.implied_vol import implied_vol
from trellis.models.calibration.materialization import materialize_model_parameter_set
from trellis.models.calibration.quote_maps import (
    QuoteAxisSpec,
    QuoteMapSpec,
    QuoteSemanticsSpec,
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
from trellis.models.processes.heston import (
    Heston,
    HestonRuntimeBinding,
    build_heston_parameter_payload,
    resolve_heston_runtime_binding,
)
from trellis.models.transforms.fft_pricer import fft_price


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy."""
    return MappingProxyType(dict(mapping or {}))


def _normalize_float_tuple(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize numeric sequences onto immutable float tuples."""
    return tuple(float(value) for value in values)


def _default_implied_vol_quote_map_spec() -> QuoteMapSpec:
    """Return the shared implied-vol quote-map spec used by vol workflows."""
    return QuoteMapSpec(
        quote_family="implied_vol",
        convention="black",
        semantics=QuoteSemanticsSpec(
            quote_family="implied_vol",
            convention="black",
            quote_subject="equity_option",
            axes=(
                QuoteAxisSpec("expiry", axis_kind="time_to_expiry", unit="years"),
                QuoteAxisSpec("strike", axis_kind="option_strike", unit="price"),
            ),
            unit=QuoteUnitSpec(
                unit_name="decimal_volatility",
                value_domain="volatility",
                scaling="absolute",
            ),
        ),
    )


_HESTON_PARAMETER_NAMES = ("kappa", "theta", "xi", "rho", "v0")
_HESTON_LOWER_BOUNDS = raw_np.asarray((0.05, 0.005, 0.05, -0.999, 0.005), dtype=float)
_HESTON_UPPER_BOUNDS = raw_np.asarray((5.0, 0.5, 2.0, 0.999, 0.5), dtype=float)


@dataclass(frozen=True)
class HestonSmilePoint:
    """One supported equity-vol smile point for Heston calibration."""

    strike: float
    market_vol: float
    weight: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if not raw_np.isfinite(self.strike) or float(self.strike) <= 0.0:
            raise ValueError("strike must be finite and positive")
        if not raw_np.isfinite(self.market_vol) or float(self.market_vol) <= 0.0:
            raise ValueError("market_vol must be finite and positive")
        if not raw_np.isfinite(self.weight) or float(self.weight) <= 0.0:
            raise ValueError("weight must be finite and positive")
        object.__setattr__(self, "strike", float(self.strike))
        object.__setattr__(self, "market_vol", float(self.market_vol))
        object.__setattr__(self, "weight", float(self.weight))

    def resolved_label(self) -> str:
        """Return a stable point label."""
        if self.label:
            return self.label
        return f"strike_{float(self.strike):g}"

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "label": self.resolved_label(),
            "strike": float(self.strike),
            "market_vol": float(self.market_vol),
            "weight": float(self.weight),
        }


@dataclass(frozen=True)
class HestonSmileSurface:
    """Reusable single-expiry equity-vol smile for Heston fits."""

    spot: float
    rate: float
    expiry_years: float
    points: tuple[HestonSmilePoint, ...]
    dividend_yield: float = 0.0
    surface_name: str = ""
    source_kind: str = "option_surface"
    source_ref: str = "build_heston_smile_surface"
    warnings: tuple[str, ...] = ()
    quote_map_spec: QuoteMapSpec = field(default_factory=_default_implied_vol_quote_map_spec)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not raw_np.isfinite(self.spot) or float(self.spot) <= 0.0:
            raise ValueError("spot must be finite and positive")
        if not raw_np.isfinite(self.rate):
            raise ValueError("rate must be finite")
        if not raw_np.isfinite(self.expiry_years) or float(self.expiry_years) <= 0.0:
            raise ValueError("expiry_years must be finite and positive")
        if len(self.points) < 5:
            raise ValueError("Heston smile surfaces require at least five points")
        if self.quote_map_spec.quote_family != "implied_vol":
            raise ValueError("Heston smile surfaces require an implied-vol quote map")
        object.__setattr__(self, "spot", float(self.spot))
        object.__setattr__(self, "rate", float(self.rate))
        object.__setattr__(self, "expiry_years", float(self.expiry_years))
        object.__setattr__(self, "dividend_yield", float(self.dividend_yield))
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def strikes(self) -> tuple[float, ...]:
        return tuple(point.strike for point in self.points)

    @property
    def market_vols(self) -> tuple[float, ...]:
        return tuple(point.market_vol for point in self.points)

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(point.weight for point in self.points)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(point.resolved_label() for point in self.points)

    @property
    def atm_index(self) -> int:
        strikes = raw_np.asarray(self.strikes, dtype=float)
        return int(raw_np.argmin(raw_np.abs(strikes - self.forward)))

    @property
    def forward(self) -> float:
        """Return the forward implied by spot, carry, and expiry."""
        return float(self.spot) * float(raw_np.exp((self.rate - self.dividend_yield) * self.expiry_years))

    @property
    def payload(self) -> dict[str, object]:
        return self.to_payload()

    def to_payload(self) -> dict[str, object]:
        atm_index = self.atm_index
        return {
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "surface_name": self.surface_name,
            "spot": float(self.spot),
            "rate": float(self.rate),
            "dividend_yield": float(self.dividend_yield),
            "expiry_years": float(self.expiry_years),
            "strike_count": len(self.points),
            "labels": list(self.labels),
            "strikes": list(self.strikes),
            "market_vols": list(self.market_vols),
            "weights": list(self.weights),
            "atm_strike": float(self.strikes[atm_index]),
            "atm_market_vol": float(self.market_vols[atm_index]),
            "warnings": list(self.warnings),
            "quote_map": self.quote_map_spec.to_payload(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HestonSmileFitDiagnostics:
    """Fit diagnostics for a Heston smile calibration result."""

    labels: tuple[str, ...]
    market_vols: tuple[float, ...]
    model_vols: tuple[float, ...]
    residuals: tuple[float, ...]
    max_abs_vol_error: float
    rms_vol_error: float
    weighted_rms_vol_error: float
    atm_abs_vol_error: float
    point_count: int
    warning_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "market_vols", _normalize_float_tuple(self.market_vols))
        object.__setattr__(self, "model_vols", _normalize_float_tuple(self.model_vols))
        object.__setattr__(self, "residuals", _normalize_float_tuple(self.residuals))
        object.__setattr__(self, "max_abs_vol_error", float(self.max_abs_vol_error))
        object.__setattr__(self, "rms_vol_error", float(self.rms_vol_error))
        object.__setattr__(self, "weighted_rms_vol_error", float(self.weighted_rms_vol_error))
        object.__setattr__(self, "atm_abs_vol_error", float(self.atm_abs_vol_error))
        object.__setattr__(self, "point_count", int(self.point_count))
        object.__setattr__(self, "warning_count", int(self.warning_count))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly diagnostics payload."""
        return {
            "labels": list(self.labels),
            "market_vols": list(self.market_vols),
            "model_vols": list(self.model_vols),
            "residuals": list(self.residuals),
            "max_abs_vol_error": float(self.max_abs_vol_error),
            "rms_vol_error": float(self.rms_vol_error),
            "weighted_rms_vol_error": float(self.weighted_rms_vol_error),
            "atm_abs_vol_error": float(self.atm_abs_vol_error),
            "point_count": int(self.point_count),
            "warning_count": int(self.warning_count),
        }


@dataclass(frozen=True)
class HestonSmileCalibrationResult:
    """Structured result for the supported Heston smile workflow."""

    surface: HestonSmileSurface
    runtime_binding: HestonRuntimeBinding
    solve_request: SolveRequest
    solve_result: SolveResult
    solver_provenance: SolveProvenance
    solver_replay_artifact: SolveReplayArtifact
    diagnostics: HestonSmileFitDiagnostics
    parameter_set_name: str = "heston"
    model_parameters: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "assumptions", tuple(str(assumption) for assumption in self.assumptions))

    def apply_to_market_state(self, market_state: MarketState) -> MarketState:
        """Return ``market_state`` enriched with the calibrated Heston parameters."""
        return materialize_model_parameter_set(
            market_state,
            parameter_set_name=self.parameter_set_name,
            model_parameters=dict(self.model_parameters),
            source_kind=str(self.provenance.get("source_kind", "calibrated_surface")),
            source_ref=str(self.provenance.get("source_ref", "fit_heston_smile_surface")),
            selected_curve_roles=dict(self.provenance.get("selected_curve_names", {})),
            metadata={
                "instrument_family": "equity_vol",
                "model_family": "heston",
                "parameter_set_name": self.parameter_set_name,
                "runtime_binding": self.runtime_binding.to_payload(),
            },
        )

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly calibration payload."""
        return {
            "surface": self.surface.to_payload(),
            "runtime_binding": self.runtime_binding.to_payload(),
            "solve_request": self.solve_request.to_payload(),
            "solve_result": self.solve_result.to_payload(),
            "solver_provenance": self.solver_provenance.to_payload(),
            "solver_replay_artifact": self.solver_replay_artifact.to_payload(),
            "fit_diagnostics": self.diagnostics.to_payload(),
            "parameter_set_name": self.parameter_set_name,
            "model_parameters": dict(self.model_parameters),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
            "assumptions": list(self.assumptions),
        }


def _validate_heston_inputs(
    spot: float,
    expiry_years: float,
    strikes: raw_np.ndarray,
    market_vols: raw_np.ndarray,
    *,
    rate: float,
    dividend_yield: float,
) -> None:
    """Reject ill-posed Heston smile inputs."""
    if not raw_np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive")
    if not raw_np.isfinite(rate):
        raise ValueError("rate must be finite")
    if not raw_np.isfinite(dividend_yield):
        raise ValueError("dividend_yield must be finite")
    if not raw_np.isfinite(expiry_years) or expiry_years <= 0.0:
        raise ValueError("expiry_years must be finite and positive")
    if strikes.ndim != 1 or market_vols.ndim != 1:
        raise ValueError("strikes and market_vols must be one-dimensional sequences")
    if strikes.size != market_vols.size:
        raise ValueError("strikes and market_vols must have the same length")
    if strikes.size < 5:
        raise ValueError("at least five smile points are required for Heston calibration")
    if not raw_np.all(raw_np.isfinite(strikes)) or not raw_np.all(raw_np.isfinite(market_vols)):
        raise ValueError("strikes and market_vols must be finite")
    if raw_np.any(strikes <= 0.0):
        raise ValueError("strikes must be positive")
    if raw_np.any(raw_np.diff(strikes) <= 0.0):
        raise ValueError("strikes must be strictly increasing")
    if raw_np.any(market_vols <= 0.0):
        raise ValueError("market_vols must be positive")


def build_heston_smile_surface(
    spot: float,
    expiry_years: float,
    strikes: Sequence[float],
    market_vols: Sequence[float],
    *,
    rate: float,
    dividend_yield: float = 0.0,
    labels: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    surface_name: str = "",
    metadata: Mapping[str, object] | None = None,
) -> HestonSmileSurface:
    """Normalize one supported Heston smile input surface."""
    strike_array = raw_np.asarray(strikes, dtype=float)
    vol_array = raw_np.asarray(market_vols, dtype=float)
    _validate_heston_inputs(
        float(spot),
        float(expiry_years),
        strike_array,
        vol_array,
        rate=float(rate),
        dividend_yield=float(dividend_yield),
    )

    labels = tuple(labels or ())
    weights = tuple(float(weight) for weight in (weights or ()))
    if labels and len(labels) != strike_array.size:
        raise ValueError("labels must align with strikes")
    if weights and len(weights) != strike_array.size:
        raise ValueError("weights must align with strikes")

    warnings: list[str] = []
    forward = float(spot) * float(raw_np.exp((float(rate) - float(dividend_yield)) * float(expiry_years)))
    below = raw_np.any(strike_array < forward)
    above = raw_np.any(strike_array > forward)
    if not (below and above):
        warnings.append("Heston smile input does not bracket the forward with strikes on both sides.")
    atm_index = int(raw_np.argmin(raw_np.abs(strike_array - forward)))
    if strike_array[atm_index] != forward:
        warnings.append(
            f"Heston smile ATM quote uses nearest strike {float(strike_array[atm_index]):g} rather than exact forward {forward:g}."
        )

    points = tuple(
        HestonSmilePoint(
            strike=float(strike),
            market_vol=float(market_vol),
            weight=weights[index] if weights else 1.0,
            label=labels[index] if labels else "",
        )
        for index, (strike, market_vol) in enumerate(zip(strike_array, vol_array))
    )
    return HestonSmileSurface(
        spot=float(spot),
        rate=float(rate),
        expiry_years=float(expiry_years),
        points=points,
        dividend_yield=float(dividend_yield),
        surface_name=surface_name,
        warnings=tuple(warnings),
        metadata=metadata or {},
    )


def _default_initial_guess(surface: HestonSmileSurface) -> tuple[float, ...]:
    """Return a stable default parameter guess from the smile shape."""
    atm_vol = float(surface.market_vols[surface.atm_index])
    slope = float(surface.market_vols[-1] - surface.market_vols[0])
    rho_guess = -0.3 if slope < -1e-6 else 0.3 if slope > 1e-6 else 0.0
    variance_guess = max(atm_vol ** 2, 0.01)
    return (1.5, variance_guess, max(0.25, atm_vol), rho_guess, variance_guess)


def _normalize_warm_start(
    warm_start: Sequence[float] | WarmStart | None,
) -> WarmStart | None:
    """Normalize user warm-start input onto the typed record."""
    if warm_start is None:
        return None
    if isinstance(warm_start, WarmStart):
        return warm_start
    return WarmStart(parameter_values=tuple(float(value) for value in warm_start), source="explicit")


def _heston_model_vols(
    surface: HestonSmileSurface,
    params: Sequence[float],
    *,
    fft_points: int,
    fft_eta: float,
    fft_alpha: float,
) -> raw_np.ndarray:
    """Return model vols for one Heston parameter vector."""
    kappa, theta, xi, rho, v0 = (float(value) for value in params)
    if theta <= 0.0 or xi <= 0.0 or v0 <= 0.0 or kappa <= 0.0 or abs(rho) >= 1.0:
        return raw_np.full(len(surface.points), 10.0, dtype=float)

    process = Heston(
        mu=surface.rate - surface.dividend_yield,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        v0=v0,
    )
    model_vols: list[float] = []
    for strike in surface.strikes:
        try:
            price = fft_price(
                lambda u: process.characteristic_function(u, surface.expiry_years, log_spot=raw_np.log(surface.spot)),
                surface.spot,
                float(strike),
                surface.expiry_years,
                surface.rate,
                alpha=fft_alpha,
                N=int(fft_points),
                eta=float(fft_eta),
            )
            model_vol = implied_vol(
                price,
                surface.spot,
                float(strike),
                surface.expiry_years,
                surface.rate,
                option_type="call",
            )
        except Exception:
            model_vol = 10.0
        if not raw_np.isfinite(model_vol) or model_vol <= 0.0:
            model_vol = 10.0
        model_vols.append(float(model_vol))
    return raw_np.asarray(model_vols, dtype=float)


def _fit_diagnostics(
    surface: HestonSmileSurface,
    model_vols: raw_np.ndarray,
    *,
    warning_count: int,
) -> HestonSmileFitDiagnostics:
    """Return fit diagnostics for one solved Heston smile."""
    market_vols = raw_np.asarray(surface.market_vols, dtype=float)
    weights = raw_np.asarray(surface.weights, dtype=float)
    residuals = model_vols - market_vols
    weighted = weights * residuals ** 2 if weights.size else residuals ** 2
    return HestonSmileFitDiagnostics(
        labels=surface.labels,
        market_vols=tuple(float(value) for value in market_vols),
        model_vols=tuple(float(value) for value in model_vols),
        residuals=tuple(float(value) for value in residuals),
        max_abs_vol_error=float(raw_np.max(raw_np.abs(residuals))),
        rms_vol_error=float(raw_np.sqrt(raw_np.mean(residuals ** 2))),
        weighted_rms_vol_error=float(raw_np.sqrt(raw_np.mean(weighted))),
        atm_abs_vol_error=float(abs(residuals[surface.atm_index])),
        point_count=len(surface.points),
        warning_count=warning_count,
    )


def _heston_parameter_bump(
    params: raw_np.ndarray,
    index: int,
    *,
    relative_step: float = 1e-3,
    absolute_floor: float = 1e-4,
) -> tuple[raw_np.ndarray, float]:
    """Return one bounded finite-difference perturbation for a Heston parameter."""
    base_value = float(params[index])
    raw_step = max(abs(base_value) * relative_step, absolute_floor)
    upper_room = float(_HESTON_UPPER_BOUNDS[index] - base_value)
    lower_room = float(base_value - _HESTON_LOWER_BOUNDS[index])
    if upper_room > 1e-12:
        step = min(raw_step, upper_room)
        shocked = raw_np.array(params, dtype=float, copy=True)
        shocked[index] = base_value + step
        return shocked, float(step)
    if lower_room > 1e-12:
        step = min(raw_step, lower_room)
        shocked = raw_np.array(params, dtype=float, copy=True)
        shocked[index] = base_value - step
        return shocked, float(-step)
    return raw_np.array(params, dtype=float, copy=True), 0.0


def _heston_vol_jacobian(
    surface: HestonSmileSurface,
    params: Sequence[float],
    *,
    fft_points: int,
    fft_eta: float,
    fft_alpha: float,
) -> raw_np.ndarray:
    """Return a bounded finite-difference Jacobian of smile vols to Heston parameters."""
    base_params = raw_np.asarray(params, dtype=float)
    base_vols = _heston_model_vols(
        surface,
        base_params,
        fft_points=fft_points,
        fft_eta=fft_eta,
        fft_alpha=fft_alpha,
    )
    jacobian_matrix = raw_np.zeros((len(surface.points), base_params.size), dtype=float)
    for index in range(base_params.size):
        shocked_params, signed_step = _heston_parameter_bump(base_params, index)
        if abs(signed_step) <= 1e-12:
            continue
        shocked_vols = _heston_model_vols(
            surface,
            shocked_params,
            fft_points=fft_points,
            fft_eta=fft_eta,
            fft_alpha=fft_alpha,
        )
        jacobian_matrix[:, index] = (shocked_vols - base_vols) / signed_step
    return jacobian_matrix


def fit_heston_smile_surface(
    surface: HestonSmileSurface,
    *,
    initial_guess: Sequence[float] | None = None,
    warm_start: Sequence[float] | WarmStart | None = None,
    parameter_set_name: str = "heston",
    fft_points: int = 1024,
    fft_eta: float = 0.1,
    fft_alpha: float = 1.5,
) -> HestonSmileCalibrationResult:
    """Fit a supported Heston smile surface onto the shared solve substrate."""
    normalized_warm_start = _normalize_warm_start(warm_start)
    starting_point = tuple(float(value) for value in initial_guess) if initial_guess is not None else None
    if starting_point is None and normalized_warm_start is not None:
        starting_point = normalized_warm_start.parameter_values
    if starting_point is None:
        starting_point = _default_initial_guess(surface)

    objective = ObjectiveBundle(
        objective_kind="least_squares",
        labels=surface.labels,
        target_values=surface.market_vols,
        weights=surface.weights,
        vector_objective_fn=lambda params: _heston_model_vols(
            surface,
            params,
            fft_points=fft_points,
            fft_eta=fft_eta,
            fft_alpha=fft_alpha,
        ),
        jacobian_fn=lambda params: _heston_vol_jacobian(
            surface,
            params,
            fft_points=fft_points,
            fft_eta=fft_eta,
            fft_alpha=fft_alpha,
        ),
        metadata={
            "model_family": "heston",
            "surface_name": surface.surface_name,
            "fit_space": "implied_vol",
            "quote_map": surface.quote_map_spec.to_payload(),
            "derivative_method": "finite_difference_vector_jacobian",
        },
    )
    request = SolveRequest(
        request_id="heston_smile_least_squares",
        problem_kind="least_squares",
        parameter_names=_HESTON_PARAMETER_NAMES,
        initial_guess=starting_point,
        objective=objective,
        bounds=SolveBounds(
            lower=tuple(float(value) for value in _HESTON_LOWER_BOUNDS),
            upper=tuple(float(value) for value in _HESTON_UPPER_BOUNDS),
        ),
        solver_hint="trf",
        warm_start=normalized_warm_start,
        metadata={
            "surface": surface.to_payload(),
            "parameter_set_name": parameter_set_name,
            "model_family": "heston",
        },
        options={"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8, "maxiter": 80},
    )
    solve_result = execute_solve_request(request)
    solver_provenance = build_solve_provenance(request, solve_result)
    solver_replay_artifact = build_solve_replay_artifact(request, solve_result)

    model_vols = _heston_model_vols(
        surface,
        solve_result.solution,
        fft_points=fft_points,
        fft_eta=fft_eta,
        fft_alpha=fft_alpha,
    )
    warnings = list(surface.warnings)
    diagnostics = _fit_diagnostics(surface, model_vols, warning_count=len(warnings))

    model_parameters = build_heston_parameter_payload(
        mu=surface.rate - surface.dividend_yield,
        kappa=solve_result.solution[0],
        theta=solve_result.solution[1],
        xi=solve_result.solution[2],
        rho=solve_result.solution[3],
        v0=solve_result.solution[4],
        parameter_set_name=parameter_set_name,
        source_kind="calibration_workflow",
        metadata={
            "surface_name": surface.surface_name,
            "fit_diagnostics": diagnostics.to_payload(),
        },
    )
    binding_state = MarketState(
        as_of=date(2000, 1, 1),
        settlement=date(2000, 1, 1),
        discount=YieldCurve.flat(surface.rate),
        spot=surface.spot,
        model_parameters=dict(model_parameters),
        model_parameter_sets={parameter_set_name: dict(model_parameters)},
    )
    runtime_binding = resolve_heston_runtime_binding(binding_state, parameter_set_name=parameter_set_name)
    warnings.extend(runtime_binding.warnings)

    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "fit_heston_smile_surface",
        "calibration_target": surface.to_payload(),
        "solve_request": request.to_payload(),
        "solve_result": solve_result.to_payload(),
        "solver_provenance": solver_provenance.to_payload(),
        "solver_replay_artifact": solver_replay_artifact.to_payload(),
        "fit_diagnostics": diagnostics.to_payload(),
        "runtime_binding": runtime_binding.to_payload(),
        "warnings": list(warnings),
    }
    summary = {
        "surface_name": surface.surface_name,
        "parameter_set_name": parameter_set_name,
        "spot": surface.spot,
        "rate": surface.rate,
        "dividend_yield": surface.dividend_yield,
        "point_count": len(surface.points),
        "optimizer_success": bool(solve_result.success),
        "max_abs_vol_error": diagnostics.max_abs_vol_error,
        "quote_family": surface.quote_map_spec.quote_family,
        "quote_convention": surface.quote_map_spec.convention,
    }
    return HestonSmileCalibrationResult(
        surface=surface,
        runtime_binding=runtime_binding,
        solve_request=request,
        solve_result=solve_result,
        solver_provenance=solver_provenance,
        solver_replay_artifact=solver_replay_artifact,
        diagnostics=diagnostics,
        parameter_set_name=parameter_set_name,
        model_parameters=dict(model_parameters),
        warnings=tuple(warnings),
        provenance=provenance,
        summary=summary,
        assumptions=tuple(runtime_binding.assumptions),
    )


def calibrate_heston_smile_workflow(
    spot: float,
    expiry_years: float,
    strikes: Sequence[float],
    market_vols: Sequence[float],
    *,
    rate: float,
    dividend_yield: float = 0.0,
    labels: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    surface_name: str = "",
    parameter_set_name: str = "heston",
    initial_guess: Sequence[float] | None = None,
    warm_start: Sequence[float] | WarmStart | None = None,
    metadata: Mapping[str, object] | None = None,
) -> HestonSmileCalibrationResult:
    """Run the supported Heston smile workflow from raw smile inputs."""
    surface = build_heston_smile_surface(
        spot,
        expiry_years,
        strikes,
        market_vols,
        rate=rate,
        dividend_yield=dividend_yield,
        labels=labels,
        weights=weights,
        surface_name=surface_name,
        metadata=metadata,
    )
    result = fit_heston_smile_surface(
        surface,
        initial_guess=initial_guess,
        warm_start=warm_start,
        parameter_set_name=parameter_set_name,
    )
    provenance = dict(result.provenance)
    provenance["source_ref"] = "calibrate_heston_smile_workflow"
    return replace(result, provenance=provenance)


__all__ = [
    "HestonSmilePoint",
    "HestonSmileSurface",
    "HestonSmileFitDiagnostics",
    "HestonSmileCalibrationResult",
    "build_heston_smile_surface",
    "fit_heston_smile_surface",
    "calibrate_heston_smile_workflow",
]
