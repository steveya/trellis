"""SABR model calibration with reusable smile-surface assembly and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as raw_np

from trellis.core.differentiable import get_numpy, gradient
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
from trellis.models.processes.sabr import SABRProcess

np = get_numpy()


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy."""
    return MappingProxyType(dict(mapping or {}))


def _normalize_float_tuple(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize numeric sequences onto immutable float tuples."""
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class SABRSmilePoint:
    """One smile point on a supported SABR calibration surface."""

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
        """Return a stable smile-point label."""
        if self.label:
            return self.label
        return f"strike_{float(self.strike):g}"

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly smile-point payload."""
        return {
            "label": self.resolved_label(),
            "strike": float(self.strike),
            "market_vol": float(self.market_vol),
            "weight": float(self.weight),
        }


@dataclass(frozen=True)
class SABRSmileSurface:
    """Reusable smile-surface input for SABR fit workflows."""

    forward: float
    expiry_years: float
    beta: float
    points: tuple[SABRSmilePoint, ...]
    surface_name: str = ""
    source_kind: str = "option_surface"
    source_ref: str = "build_sabr_smile_surface"
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not raw_np.isfinite(self.forward) or float(self.forward) <= 0.0:
            raise ValueError("forward must be finite and positive")
        if not raw_np.isfinite(self.expiry_years) or float(self.expiry_years) <= 0.0:
            raise ValueError("expiry_years must be finite and positive")
        if len(self.points) < 3:
            raise ValueError("SABR smile surfaces require at least three points")
        object.__setattr__(self, "forward", float(self.forward))
        object.__setattr__(self, "expiry_years", float(self.expiry_years))
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def strikes(self) -> tuple[float, ...]:
        """Return the ordered strike grid."""
        return tuple(point.strike for point in self.points)

    @property
    def market_vols(self) -> tuple[float, ...]:
        """Return the ordered market-vol grid."""
        return tuple(point.market_vol for point in self.points)

    @property
    def weights(self) -> tuple[float, ...]:
        """Return the ordered point weights."""
        return tuple(point.weight for point in self.points)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the stable point labels."""
        return tuple(point.resolved_label() for point in self.points)

    @property
    def atm_index(self) -> int:
        """Return the index of the closest strike to the forward."""
        strikes = raw_np.asarray(self.strikes, dtype=float)
        return int(raw_np.argmin(raw_np.abs(strikes - self.forward)))

    @property
    def payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return self.to_payload()

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly surface payload."""
        atm_idx = self.atm_index
        return {
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "surface_name": self.surface_name,
            "forward": float(self.forward),
            "expiry_years": float(self.expiry_years),
            "beta": float(self.beta),
            "strike_count": len(self.points),
            "labels": list(self.labels),
            "strikes": list(self.strikes),
            "market_vols": list(self.market_vols),
            "weights": list(self.weights),
            "atm_strike": float(self.strikes[atm_idx]),
            "atm_market_vol": float(self.market_vols[atm_idx]),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SABRSmileFitDiagnostics:
    """Fit diagnostics for one SABR smile calibration result."""

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
class SABRSmileCalibrationResult:
    """Reusable result surface for a SABR smile fit."""

    surface: SABRSmileSurface
    sabr: SABRProcess
    solve_request: SolveRequest
    solve_result: SolveResult
    solver_provenance: SolveProvenance
    solver_replay_artifact: SolveReplayArtifact
    diagnostics: SABRSmileFitDiagnostics
    warnings: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly calibration payload."""
        return {
            "surface": self.surface.to_payload(),
            "sabr_parameters": {
                "alpha": float(self.sabr.alpha),
                "beta": float(self.sabr.beta),
                "rho": float(self.sabr.rho),
                "nu": float(self.sabr.nu),
            },
            "solve_request": self.solve_request.to_payload(),
            "solve_result": self.solve_result.to_payload(),
            "solver_provenance": self.solver_provenance.to_payload(),
            "solver_replay_artifact": self.solver_replay_artifact.to_payload(),
            "fit_diagnostics": self.diagnostics.to_payload(),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
        }


def _validate_sabr_inputs(
    F: float,
    T: float,
    strikes: raw_np.ndarray,
    market_vols: raw_np.ndarray,
) -> None:
    """Reject ill-posed SABR calibration inputs before optimization starts."""
    if not raw_np.isfinite(F) or F <= 0:
        raise ValueError("forward price F must be finite and positive")
    if not raw_np.isfinite(T) or T <= 0:
        raise ValueError("time to expiry T must be finite and positive")
    if strikes.ndim != 1 or market_vols.ndim != 1:
        raise ValueError("strikes and market_vols must be one-dimensional sequences")
    if strikes.size != market_vols.size:
        raise ValueError("strikes and market_vols must have the same length")
    if strikes.size < 3:
        raise ValueError("at least three strike points are required for SABR calibration")
    if not raw_np.all(raw_np.isfinite(strikes)) or not raw_np.all(raw_np.isfinite(market_vols)):
        raise ValueError("strikes and market_vols must be finite")
    if raw_np.any(strikes <= 0):
        raise ValueError("strikes must be positive")
    if raw_np.any(raw_np.diff(strikes) <= 0):
        raise ValueError("strikes must be strictly increasing")
    if raw_np.any(market_vols <= 0):
        raise ValueError("market_vols must be positive")


def build_sabr_smile_surface(
    F: float,
    T: float,
    strikes: Sequence[float],
    market_vols: Sequence[float],
    *,
    beta: float = 0.5,
    labels: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    surface_name: str = "",
    source_kind: str = "option_surface",
    source_ref: str = "build_sabr_smile_surface",
    metadata: Mapping[str, object] | None = None,
) -> SABRSmileSurface:
    """Build a reusable SABR smile surface from raw market inputs."""
    strikes_array = raw_np.asarray(strikes, dtype=float)
    market_vols_array = raw_np.asarray(market_vols, dtype=float)
    _validate_sabr_inputs(F, T, strikes_array, market_vols_array)

    if labels is not None and len(labels) != strikes_array.size:
        raise ValueError("labels must have the same length as strikes")

    if weights is None:
        weights_array = raw_np.ones(strikes_array.size, dtype=float)
    else:
        weights_array = raw_np.asarray(weights, dtype=float)
        if weights_array.ndim != 1 or weights_array.size != strikes_array.size:
            raise ValueError("weights must have the same length as strikes")
        if not raw_np.all(raw_np.isfinite(weights_array)) or raw_np.any(weights_array <= 0.0):
            raise ValueError("weights must be finite and positive")

    resolved_labels = (
        tuple(str(label) for label in labels)
        if labels is not None
        else tuple(f"strike_{float(strike):g}" for strike in strikes_array)
    )

    warnings: list[str] = []
    atm_idx = int(raw_np.argmin(raw_np.abs(strikes_array - float(F))))
    atm_strike = float(strikes_array[atm_idx])
    if abs(atm_strike - float(F)) > 1e-12:
        warnings.append(
            f"ATM forward {float(F):g} not observed exactly; nearest strike {atm_strike:g} is used for diagnostics."
        )
    if raw_np.all(strikes_array < float(F)) or raw_np.all(strikes_array > float(F)):
        warnings.append(
            "Smile surface does not bracket the forward; SABR fit diagnostics are extrapolative around ATM."
        )

    points = tuple(
        SABRSmilePoint(
            strike=float(strike),
            market_vol=float(vol),
            weight=float(weight),
            label=resolved_labels[index],
        )
        for index, (strike, vol, weight) in enumerate(zip(strikes_array, market_vols_array, weights_array))
    )
    return SABRSmileSurface(
        forward=float(F),
        expiry_years=float(T),
        beta=float(beta),
        points=points,
        surface_name=surface_name,
        source_kind=source_kind,
        source_ref=source_ref,
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _fit_diagnostics(
    surface: SABRSmileSurface,
    model_vols: Sequence[float],
) -> SABRSmileFitDiagnostics:
    """Return stable fit diagnostics for one calibrated smile."""
    model = raw_np.asarray(model_vols, dtype=float)
    market = raw_np.asarray(surface.market_vols, dtype=float)
    weights = raw_np.asarray(surface.weights, dtype=float)
    residuals = model - market
    atm_idx = surface.atm_index
    return SABRSmileFitDiagnostics(
        labels=surface.labels,
        market_vols=tuple(float(value) for value in market),
        model_vols=tuple(float(value) for value in model),
        residuals=tuple(float(value) for value in residuals),
        max_abs_vol_error=float(raw_np.max(raw_np.abs(residuals))),
        rms_vol_error=float(raw_np.sqrt(raw_np.mean(residuals ** 2))),
        weighted_rms_vol_error=float(raw_np.sqrt(raw_np.sum(weights * residuals ** 2) / raw_np.sum(weights))),
        atm_abs_vol_error=float(abs(residuals[atm_idx])),
        point_count=int(residuals.size),
        warning_count=len(surface.warnings),
    )


def fit_sabr_smile_surface(
    surface: SABRSmileSurface,
    *,
    initial_guess: tuple[float, float, float] | None = None,
) -> SABRSmileCalibrationResult:
    """Fit SABR parameters to one reusable smile surface."""
    strikes = raw_np.asarray(surface.strikes, dtype=float)
    market_vols = raw_np.asarray(surface.market_vols, dtype=float)
    weights = raw_np.asarray(surface.weights, dtype=float)

    if initial_guess is None:
        alpha0 = float(surface.market_vols[surface.atm_index]) * float(surface.forward) ** (1 - float(surface.beta))
        resolved_initial_guess = (float(alpha0), 0.0, 0.3)
        warm_start_source = "atm_seed"
    else:
        resolved_initial_guess = tuple(float(value) for value in initial_guess)
        warm_start_source = "explicit_seed"

    def objective(params):
        """Return the weighted squared-error objective for one SABR parameter vector."""
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        sabr = SABRProcess(alpha, surface.beta, rho, nu)
        model_vols = np.array([sabr.implied_vol(surface.forward, K, surface.expiry_years) for K in strikes])
        residuals = model_vols - market_vols
        return np.sum(weights * residuals ** 2)

    objective_grad = gradient(objective)
    solve_request = SolveRequest(
        request_id="sabr_smile_least_squares",
        problem_kind="least_squares",
        parameter_names=("alpha", "rho", "nu"),
        initial_guess=resolved_initial_guess,
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=surface.labels,
            target_values=tuple(float(vol) for vol in market_vols),
            weights=surface.weights,
            vector_objective_fn=lambda params: raw_np.array(
                [
                    SABRProcess(float(params[0]), surface.beta, float(params[1]), float(params[2])).implied_vol(
                        surface.forward,
                        K,
                        surface.expiry_years,
                    )
                    for K in strikes
                ],
                dtype=float,
            ),
            scalar_objective_fn=objective,
            jacobian_fn=objective_grad,
            metadata={
                "forward": float(surface.forward),
                "expiry_years": float(surface.expiry_years),
                "beta": float(surface.beta),
                "surface_name": surface.surface_name,
            },
        ),
        bounds=SolveBounds(lower=(1e-6, -0.999, 1e-6), upper=(None, 0.999, None)),
        solver_hint="L-BFGS-B",
        warm_start=WarmStart(parameter_values=resolved_initial_guess, source=warm_start_source),
        metadata={
            "problem_family": "sabr_smile",
            "solver_family": "scipy",
            "surface_name": surface.surface_name,
        },
    )
    solve_result = execute_solve_request(solve_request)
    if not solve_result.success:
        raise ValueError(f"SABR calibration failed: {solve_result.metadata.get('message', 'unknown failure')}")

    alpha, rho, nu = solve_result.solution
    sabr = SABRProcess(alpha, surface.beta, rho, nu)
    solver_provenance = build_solve_provenance(solve_request, solve_result)
    solver_replay_artifact = build_solve_replay_artifact(solve_request, solve_result)
    model_vols = tuple(
        float(sabr.implied_vol(surface.forward, strike, surface.expiry_years))
        for strike in surface.strikes
    )
    diagnostics = _fit_diagnostics(surface, model_vols)
    solved_parameters = {
        "alpha": float(sabr.alpha),
        "beta": float(sabr.beta),
        "rho": float(sabr.rho),
        "nu": float(sabr.nu),
    }
    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "fit_sabr_smile_surface",
        "calibration_target": surface.to_payload(),
        "solved_parameters": solved_parameters,
        "objective_value": float(solve_result.objective_value),
        "optimizer": {
            "method": solve_result.method,
            "success": bool(solve_result.success),
            "message": str(solve_result.metadata.get("message", "")),
        },
        "solve_request": solve_request.to_payload(),
        "solve_result": solve_result.to_payload(),
        "solver_provenance": solver_provenance.to_payload(),
        "solver_replay_artifact": solver_replay_artifact.to_payload(),
        "fit_diagnostics": diagnostics.to_payload(),
        "warnings": list(surface.warnings),
    }
    summary = {
        "target_kind": "option_surface",
        "point_count": diagnostics.point_count,
        "atm_strike": float(surface.strikes[surface.atm_index]),
        "atm_market_vol": float(surface.market_vols[surface.atm_index]),
        "objective_value": float(solve_result.objective_value),
        "optimizer_success": bool(solve_result.success),
        "surface_name": surface.surface_name,
    }
    return SABRSmileCalibrationResult(
        surface=surface,
        sabr=sabr,
        solve_request=solve_request,
        solve_result=solve_result,
        solver_provenance=solver_provenance,
        solver_replay_artifact=solver_replay_artifact,
        diagnostics=diagnostics,
        warnings=surface.warnings,
        provenance=provenance,
        summary=summary,
    )


def calibrate_sabr_smile_workflow(
    F: float,
    T: float,
    strikes: Sequence[float],
    market_vols: Sequence[float],
    *,
    beta: float = 0.5,
    labels: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    surface_name: str = "",
    metadata: Mapping[str, object] | None = None,
    initial_guess: tuple[float, float, float] | None = None,
) -> SABRSmileCalibrationResult:
    """Run the supported raw-input SABR smile workflow."""
    result = fit_sabr_smile_surface(
        build_sabr_smile_surface(
            F,
            T,
            strikes,
            market_vols,
            beta=beta,
            labels=labels,
            weights=weights,
            surface_name=surface_name,
            source_ref="calibrate_sabr_smile_workflow",
            metadata=metadata,
        ),
        initial_guess=initial_guess,
    )
    provenance = dict(result.provenance)
    provenance["source_ref"] = "calibrate_sabr_smile_workflow"
    summary = dict(result.summary)
    summary["surface_name"] = surface_name
    return SABRSmileCalibrationResult(
        surface=result.surface,
        sabr=result.sabr,
        solve_request=result.solve_request,
        solve_result=result.solve_result,
        solver_provenance=result.solver_provenance,
        solver_replay_artifact=result.solver_replay_artifact,
        diagnostics=result.diagnostics,
        warnings=result.warnings,
        provenance=provenance,
        summary=summary,
    )


def calibrate_sabr(
    F: float,
    T: float,
    strikes: Sequence[float],
    market_vols: Sequence[float],
    beta: float = 0.5,
) -> SABRProcess:
    """Calibrate SABR parameters (alpha, rho, nu) to market implied vols."""
    result = calibrate_sabr_smile_workflow(
        F,
        T,
        strikes,
        market_vols,
        beta=beta,
    )
    sabr = result.sabr
    provenance = dict(result.provenance)
    provenance["source_ref"] = "calibrate_sabr"
    sabr.calibration_provenance = provenance
    sabr.calibration_target = result.surface.to_payload()
    sabr.calibration_summary = dict(result.summary)
    sabr.calibration_fit_diagnostics = result.diagnostics.to_payload()
    sabr.calibration_warnings = list(result.warnings)
    return sabr


__all__ = [
    "SABRSmilePoint",
    "SABRSmileSurface",
    "SABRSmileFitDiagnostics",
    "SABRSmileCalibrationResult",
    "build_sabr_smile_surface",
    "fit_sabr_smile_surface",
    "calibrate_sabr_smile_workflow",
    "calibrate_sabr",
]
