"""Dupire local volatility surface construction."""

from __future__ import annotations

import numpy as raw_np


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
        raise ValueError(
            "implied_vols must have shape (len(expiries), len(strikes))"
        )
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
    return {
        "source_kind": "option_surface",
        "spot": float(S0),
        "rate": float(r),
        "strike_count": int(strikes.size),
        "expiry_count": int(expiries.size),
        "surface_shape": (int(expiries.size), int(strikes.size)),
        "strikes": tuple(float(v) for v in strikes),
        "expiries": tuple(float(v) for v in expiries),
    }


def dupire_local_vol(
    strikes: raw_np.ndarray,
    expiries: raw_np.ndarray,
    implied_vols: raw_np.ndarray,
    S0: float,
    r: float,
) -> callable:
    """Construct Dupire local vol function from an implied vol surface.

    Parameters
    ----------
    strikes : ndarray of shape (n_K,)
    expiries : ndarray of shape (n_T,)
    implied_vols : ndarray of shape (n_T, n_K)
        Market implied vols.
    S0 : float
        Spot price.
    r : float
        Risk-free rate.

    Returns
    -------
    callable(S, t) -> float
        Local volatility function.
    """
    from scipy.interpolate import RectBivariateSpline

    strikes = raw_np.asarray(strikes, dtype=float)
    expiries = raw_np.asarray(expiries, dtype=float)
    implied_vols = raw_np.asarray(implied_vols, dtype=float)
    _validate_local_vol_inputs(strikes, expiries, implied_vols, S0, r)
    calibration_target = _local_vol_calibration_target(strikes, expiries, S0, r)

    # Fit a smooth surface to implied vols
    spline = RectBivariateSpline(expiries, strikes, implied_vols)

    def local_vol(S, t):
        """Evaluate Dupire local volatility at spot ``S`` and time ``t``."""
        t = max(t, 1e-6)
        K = S  # local vol evaluated at S=K

        sigma = float(spline(t, K, grid=False))
        dsigma_dT = float(spline(t, K, dx=1, grid=False))
        dsigma_dK = float(spline(t, K, dy=1, grid=False))
        d2sigma_dK2 = float(spline(t, K, dy=2, grid=False))

        d1 = (raw_np.log(S0 / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * raw_np.sqrt(t))

        # Dupire formula
        numer = sigma ** 2 + 2 * sigma * t * (dsigma_dT + r * K * dsigma_dK)
        denom = (1 + K * d1 * raw_np.sqrt(t) * dsigma_dK) ** 2 + \
                K ** 2 * t * sigma * (d2sigma_dK2 - d1 * raw_np.sqrt(t) * dsigma_dK ** 2)

        if denom <= 0:
            return sigma  # fallback
        return raw_np.sqrt(max(numer / denom, 0))

    local_vol.calibration_provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "dupire_local_vol",
        "calibration_target": calibration_target,
        "parameterization": {
            "spot": float(S0),
            "rate": float(r),
            "surface_shape": calibration_target["surface_shape"],
        },
    }
    local_vol.calibration_target = calibration_target
    local_vol.calibration_summary = {
        "target_kind": "option_surface",
        "surface_shape": calibration_target["surface_shape"],
        "strike_count": int(strikes.size),
        "expiry_count": int(expiries.size),
        "spot": float(S0),
        "rate": float(r),
    }
    return local_vol
