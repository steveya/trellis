"""Dupire local volatility surface construction and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as raw_np

from trellis.models.calibration.quote_maps import QuoteMapSpec


def _default_implied_vol_quote_map_spec() -> QuoteMapSpec:
    """Return the shared implied-vol quote-map spec used by vol workflows."""
    return QuoteMapSpec(quote_family="implied_vol", convention="black")


@dataclass(frozen=True)
class LocalVolSurfaceDiagnostics:
    """Stability diagnostics for a Dupire local-vol fit."""

    unstable_point_count: int
    total_point_count: int
    min_denominator: float
    min_numerator: float
    max_abs_local_vol: float
    sample_unstable_points: tuple[dict[str, float], ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly diagnostics payload."""
        return {
            "unstable_point_count": int(self.unstable_point_count),
            "total_point_count": int(self.total_point_count),
            "min_denominator": float(self.min_denominator),
            "min_numerator": float(self.min_numerator),
            "max_abs_local_vol": float(self.max_abs_local_vol),
            "sample_unstable_points": [dict(point) for point in self.sample_unstable_points],
        }


@dataclass(frozen=True)
class LocalVolCalibrationResult:
    """Structured result for the hardened Dupire local-vol workflow."""

    local_vol_surface: object
    surface_name: str
    calibration_target: dict[str, object]
    diagnostics: LocalVolSurfaceDiagnostics
    warnings: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    def apply_to_market_state(self, market_state):
        """Return ``market_state`` enriched with the calibrated local-vol surface."""
        surface_map = dict(market_state.local_vol_surfaces or {})
        surface_map[self.surface_name] = self.local_vol_surface
        return replace(
            market_state,
            local_vol_surface=self.local_vol_surface,
            local_vol_surfaces=surface_map,
        )

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly result payload."""
        return {
            "surface_name": self.surface_name,
            "calibration_target": dict(self.calibration_target),
            "fit_diagnostics": self.diagnostics.to_payload(),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
        }


def _validate_local_vol_inputs(
    strikes: raw_np.ndarray,
    expiries: raw_np.ndarray,
    implied_vols: raw_np.ndarray,
    S0: float,
    r: float,
) -> None:
    """Reject malformed local-vol surface inputs with a clear error."""
    if strikes.ndim != 1 or expiries.ndim != 1:
        raise ValueError("strikes and expiries must be one-dimensional sequences")
    if implied_vols.ndim != 2:
        raise ValueError("implied_vols must be a two-dimensional surface")
    if implied_vols.shape != (expiries.size, strikes.size):
        raise ValueError("implied_vols must have shape (len(expiries), len(strikes))")
    if strikes.size < 4 or expiries.size < 4:
        raise ValueError("at least four strikes and four expiries are required for Dupire fitting")
    if not raw_np.all(raw_np.isfinite(strikes)) or not raw_np.all(raw_np.isfinite(expiries)):
        raise ValueError("strikes and expiries must be finite")
    if not raw_np.all(raw_np.isfinite(implied_vols)):
        raise ValueError("implied_vols must be finite")
    if raw_np.any(strikes <= 0) or raw_np.any(expiries <= 0):
        raise ValueError("strikes and expiries must be positive")
    if raw_np.any(raw_np.diff(strikes) <= 0):
        raise ValueError("strikes must be strictly increasing")
    if raw_np.any(raw_np.diff(expiries) <= 0):
        raise ValueError("expiries must be strictly increasing")
    if not raw_np.isfinite(S0) or S0 <= 0:
        raise ValueError("S0 must be finite and positive")
    if not raw_np.isfinite(r):
        raise ValueError("r must be finite")


def _local_vol_calibration_target(
    strikes: raw_np.ndarray,
    expiries: raw_np.ndarray,
    S0: float,
    r: float,
) -> dict[str, object]:
    """Return a compact description of the Dupire calibration target."""
    quote_map = _default_implied_vol_quote_map_spec()
    return {
        "source_kind": "option_surface",
        "spot": float(S0),
        "rate": float(r),
        "quote_map": quote_map.to_payload(),
        "strike_count": int(strikes.size),
        "expiry_count": int(expiries.size),
        "surface_shape": (int(expiries.size), int(strikes.size)),
        "strikes": tuple(float(v) for v in strikes),
        "expiries": tuple(float(v) for v in expiries),
    }


def _dupire_terms(spline, *, S0: float, r: float, strike: float, expiry: float) -> tuple[float, float, float]:
    """Return local-vol candidate plus numerator/denominator diagnostics."""
    sigma = float(spline(expiry, strike, grid=False))
    dsigma_dT = float(spline(expiry, strike, dx=1, grid=False))
    dsigma_dK = float(spline(expiry, strike, dy=1, grid=False))
    d2sigma_dK2 = float(spline(expiry, strike, dy=2, grid=False))

    d1 = (raw_np.log(S0 / strike) + (r + 0.5 * sigma ** 2) * expiry) / (sigma * raw_np.sqrt(expiry))
    numer = sigma ** 2 + 2 * sigma * expiry * (dsigma_dT + r * strike * dsigma_dK)
    denom = (1 + strike * d1 * raw_np.sqrt(expiry) * dsigma_dK) ** 2 + (
        strike ** 2 * expiry * sigma * (d2sigma_dK2 - d1 * raw_np.sqrt(expiry) * dsigma_dK ** 2)
    )
    if not raw_np.isfinite(numer) or not raw_np.isfinite(denom) or denom <= 0.0 or numer < 0.0:
        return float(sigma), float(numer), float(denom)
    return float(raw_np.sqrt(max(numer / denom, 0.0))), float(numer), float(denom)


def _local_vol_diagnostics(
    spline,
    *,
    strikes: raw_np.ndarray,
    expiries: raw_np.ndarray,
    S0: float,
    r: float,
) -> LocalVolSurfaceDiagnostics:
    """Return diagnostics for the fitted Dupire surface over the calibration grid."""
    unstable_points: list[dict[str, float]] = []
    unstable_point_count = 0
    min_denominator = raw_np.inf
    min_numerator = raw_np.inf
    max_abs_local_vol = 0.0

    for expiry in expiries:
        for strike in strikes:
            local_vol, numer, denom = _dupire_terms(spline, S0=S0, r=r, strike=float(strike), expiry=float(expiry))
            min_denominator = min(min_denominator, float(denom))
            min_numerator = min(min_numerator, float(numer))
            max_abs_local_vol = max(max_abs_local_vol, abs(float(local_vol)))
            if not raw_np.isfinite(numer) or not raw_np.isfinite(denom) or denom <= 0.0 or numer < 0.0:
                unstable_point_count += 1
                if len(unstable_points) < 8:
                    unstable_points.append(
                        {
                            "expiry": float(expiry),
                            "strike": float(strike),
                            "numerator": float(numer),
                            "denominator": float(denom),
                        }
                    )

    return LocalVolSurfaceDiagnostics(
        unstable_point_count=unstable_point_count,
        total_point_count=int(len(strikes) * len(expiries)),
        min_denominator=float(min_denominator),
        min_numerator=float(min_numerator),
        max_abs_local_vol=float(max_abs_local_vol),
        sample_unstable_points=tuple(unstable_points),
    )


def dupire_local_vol_result(
    strikes: raw_np.ndarray,
    expiries: raw_np.ndarray,
    implied_vols: raw_np.ndarray,
    S0: float,
    r: float,
    *,
    surface_name: str = "local_vol",
    source_kind: str = "calibrated_surface",
    source_ref: str = "dupire_local_vol_result",
    metadata: dict[str, object] | None = None,
) -> LocalVolCalibrationResult:
    """Construct a Dupire local-vol surface together with explicit diagnostics."""
    from scipy.interpolate import RectBivariateSpline

    strikes = raw_np.asarray(strikes, dtype=float)
    expiries = raw_np.asarray(expiries, dtype=float)
    implied_vols = raw_np.asarray(implied_vols, dtype=float)
    _validate_local_vol_inputs(strikes, expiries, implied_vols, S0, r)
    calibration_target = _local_vol_calibration_target(strikes, expiries, S0, r)
    calibration_target["surface_name"] = surface_name
    spline = RectBivariateSpline(expiries, strikes, implied_vols)
    diagnostics = _local_vol_diagnostics(spline, strikes=strikes, expiries=expiries, S0=S0, r=r)

    warnings: list[str] = []
    if diagnostics.unstable_point_count:
        warnings.append(
            "Unstable local-vol regions were detected on the calibration grid; Dupire fallback to implied vol will be used there."
        )

    def local_vol(S, t):
        """Evaluate Dupire local volatility at spot ``S`` and time ``t``."""
        expiry = max(float(t), 1e-6)
        strike = float(S)
        local_vol_value, numer, denom = _dupire_terms(spline, S0=S0, r=r, strike=strike, expiry=expiry)
        if not raw_np.isfinite(numer) or not raw_np.isfinite(denom) or denom <= 0.0 or numer < 0.0:
            return float(spline(expiry, strike, grid=False))
        return float(local_vol_value)

    provenance = {
        "source_kind": source_kind,
        "source_ref": source_ref,
        "surface_name": surface_name,
        "calibration_target": calibration_target,
        "parameterization": {
            "spot": float(S0),
            "rate": float(r),
            "surface_shape": calibration_target["surface_shape"],
        },
        "fit_diagnostics": diagnostics.to_payload(),
        "warnings": list(warnings),
    }
    if metadata:
        provenance["metadata"] = dict(metadata)
    summary = {
        "target_kind": "option_surface",
        "surface_name": surface_name,
        "surface_shape": calibration_target["surface_shape"],
        "strike_count": int(strikes.size),
        "expiry_count": int(expiries.size),
        "spot": float(S0),
        "rate": float(r),
        "unstable_point_count": diagnostics.unstable_point_count,
    }
    return LocalVolCalibrationResult(
        local_vol_surface=local_vol,
        surface_name=surface_name,
        calibration_target=calibration_target,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        provenance=provenance,
        summary=summary,
    )


def dupire_local_vol(
    strikes: raw_np.ndarray,
    expiries: raw_np.ndarray,
    implied_vols: raw_np.ndarray,
    S0: float,
    r: float,
) -> callable:
    """Construct Dupire local vol function from an implied vol surface."""
    result = dupire_local_vol_result(strikes, expiries, implied_vols, S0, r)
    local_vol = result.local_vol_surface
    local_vol.calibration_provenance = dict(result.provenance)
    local_vol.calibration_target = dict(result.calibration_target)
    local_vol.calibration_summary = dict(result.summary)
    local_vol.calibration_diagnostics = result.diagnostics.to_payload()
    local_vol.calibration_warnings = list(result.warnings)
    return local_vol


def calibrate_local_vol_surface_workflow(
    strikes: raw_np.ndarray,
    expiries: raw_np.ndarray,
    implied_vols: raw_np.ndarray,
    S0: float,
    r: float,
    *,
    surface_name: str = "local_vol",
    metadata: dict[str, object] | None = None,
) -> LocalVolCalibrationResult:
    """Return the supported local-vol workflow result surface."""
    return dupire_local_vol_result(
        strikes,
        expiries,
        implied_vols,
        S0,
        r,
        surface_name=surface_name,
        source_kind="calibrated_surface",
        source_ref="calibrate_local_vol_surface_workflow",
        metadata=metadata,
    )


__all__ = [
    "LocalVolSurfaceDiagnostics",
    "LocalVolCalibrationResult",
    "calibrate_local_vol_surface_workflow",
    "dupire_local_vol_result",
    "dupire_local_vol",
]
