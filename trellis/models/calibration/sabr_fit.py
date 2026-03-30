"""SABR model calibration with gradient-assisted L-BFGS-B."""

from __future__ import annotations

import numpy as raw_np
from scipy.optimize import minimize

from trellis.core.differentiable import get_numpy, gradient
from trellis.models.processes.sabr import SABRProcess

np = get_numpy()


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


def _sabr_calibration_target(
    F: float,
    T: float,
    strikes: raw_np.ndarray,
    market_vols: raw_np.ndarray,
    beta: float,
) -> dict[str, object]:
    """Return a compact, JSON-friendly description of the calibration target."""
    atm_idx = int(raw_np.argmin(raw_np.abs(strikes - F)))
    return {
        "source_kind": "option_surface",
        "forward": float(F),
        "expiry_years": float(T),
        "beta": float(beta),
        "strike_count": int(strikes.size),
        "strikes": tuple(float(v) for v in strikes),
        "market_vols": tuple(float(v) for v in market_vols),
        "atm_strike": float(strikes[atm_idx]),
        "atm_market_vol": float(market_vols[atm_idx]),
    }


def _attach_calibration_metadata(
    sabr: SABRProcess,
    *,
    target: dict[str, object],
    objective_value: float,
    optimizer_success: bool,
    optimizer_message: str,
) -> SABRProcess:
    """Attach provenance and summary payloads to the calibrated SABR process."""
    solved_parameters = {
        "alpha": float(sabr.alpha),
        "beta": float(sabr.beta),
        "rho": float(sabr.rho),
        "nu": float(sabr.nu),
    }
    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "calibrate_sabr",
        "calibration_target": target,
        "solved_parameters": solved_parameters,
        "objective_value": float(objective_value),
        "optimizer": {
            "method": "L-BFGS-B",
            "success": bool(optimizer_success),
            "message": optimizer_message,
        },
    }
    summary = {
        "target_kind": "option_surface",
        "point_count": int(target["strike_count"]),
        "atm_strike": float(target["atm_strike"]),
        "atm_market_vol": float(target["atm_market_vol"]),
        "objective_value": float(objective_value),
        "optimizer_success": bool(optimizer_success),
    }
    sabr.calibration_provenance = provenance
    sabr.calibration_target = target
    sabr.calibration_summary = summary
    return sabr


def calibrate_sabr(
    F: float,
    T: float,
    strikes: list[float],
    market_vols: list[float],
    beta: float = 0.5,
) -> SABRProcess:
    """Calibrate SABR parameters (alpha, rho, nu) to market implied vols.

    Parameters
    ----------
    F : float
        Forward price.
    T : float
        Time to expiry.
    strikes : list[float]
        Strike prices.
    market_vols : list[float]
        Market implied volatilities at each strike.
    beta : float
        CEV exponent (typically fixed, not calibrated).

    Returns
    -------
    SABRProcess with calibrated parameters.
    """
    strikes = raw_np.asarray(strikes, dtype=float)
    market_vols = raw_np.asarray(market_vols, dtype=float)
    _validate_sabr_inputs(F, T, strikes, market_vols)
    target = _sabr_calibration_target(F, T, strikes, market_vols, beta)

    def objective(params):
        """Return the squared-error objective for one SABR parameter vector."""
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        sabr = SABRProcess(alpha, beta, rho, nu)
        model_vols = np.array([sabr.implied_vol(F, K, T) for K in strikes])
        return np.sum((model_vols - market_vols) ** 2)

    # Initial guess: ATM vol for alpha
    atm_idx = raw_np.argmin(raw_np.abs(strikes - F))
    alpha0 = market_vols[atm_idx] * F ** (1 - beta)

    objective_grad = gradient(objective)
    result = minimize(
        objective,
        x0=raw_np.array([alpha0, 0.0, 0.3]),
        jac=objective_grad,
        bounds=[(1e-6, None), (-0.999, 0.999), (1e-6, None)],
        method="L-BFGS-B",
    )
    if not result.success:
        raise ValueError(f"SABR calibration failed: {result.message}")

    alpha, rho, nu = result.x
    sabr = SABRProcess(alpha, beta, rho, nu)
    return _attach_calibration_metadata(
        sabr,
        target=target,
        objective_value=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )
