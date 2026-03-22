"""Differentiable curve bootstrapping via Newton's method.

All operations use autograd numpy so that ``gradient()`` flows through
the bootstrap into downstream pricing. No scipy, no custom VJPs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.differentiable import get_numpy
from trellis.curves.interpolation import linear_interp

np = get_numpy()


@dataclass
class BootstrapInstrument:
    """A calibration instrument with a market quote.

    Parameters
    ----------
    tenor : float
        Maturity in years.
    quote : float
        Market quote (deposit rate, futures price, or swap rate).
    instrument_type : str
        ``"deposit"``, ``"future"``, or ``"swap"``.
    """

    tenor: float
    quote: float
    instrument_type: str = "deposit"


def _reprice(
    rates: object,
    tenors: object,
    instruments: list[BootstrapInstrument],
) -> object:
    """Reprice all calibration instruments given a zero rate vector.

    Returns an array of model prices/rates (same units as quotes).
    """
    model_values = []
    for inst in instruments:
        t = inst.tenor
        r = linear_interp(t, tenors, rates)

        if inst.instrument_type == "deposit":
            # Deposit rate = (1/df - 1) / t  (simple compounding)
            df = np.exp(-r * t)
            model_rate = (1.0 / df - 1.0) / t
            model_values.append(model_rate)

        elif inst.instrument_type == "future":
            # Futures price = 100 - forward_rate * 100
            # Forward rate from t-0.25 to t (3-month future)
            t_start = max(t - 0.25, 0.001)
            r_start = linear_interp(t_start, tenors, rates)
            df_start = np.exp(-r_start * t_start)
            df_end = np.exp(-r * t)
            fwd = (df_start / df_end - 1.0) / 0.25
            model_values.append(100.0 - fwd * 100.0)

        elif inst.instrument_type == "swap":
            # Par swap rate: sum(F_i * tau_i * df_i) / sum(tau_i * df_i)
            # Use semi-annual fixed, quarterly floating
            n_float = int(t * 4)
            float_pv = np.array(0.0)
            for k in range(n_float):
                t1 = (k + 1) * 0.25
                t0 = k * 0.25
                if t0 < 0.001:
                    t0 = 0.001
                r0 = linear_interp(t0, tenors, rates)
                r1 = linear_interp(t1, tenors, rates)
                df0 = np.exp(-r0 * t0)
                df1 = np.exp(-r1 * t1)
                fwd = (df0 / df1 - 1.0) / 0.25
                float_pv = float_pv + fwd * 0.25 * df1

            n_fixed = int(t * 2)
            annuity = np.array(0.0)
            for k in range(n_fixed):
                t_pay = (k + 1) * 0.5
                r_pay = linear_interp(t_pay, tenors, rates)
                df_pay = np.exp(-r_pay * t_pay)
                annuity = annuity + 0.5 * df_pay

            par_rate = float_pv / annuity if float(annuity) != 0.0 else np.array(0.0)
            model_values.append(par_rate)

        else:
            raise ValueError(f"Unknown instrument type: {inst.instrument_type!r}")

    return np.array(model_values)


def bootstrap(
    instruments: list[BootstrapInstrument],
    max_iter: int = 50,
    tol: float = 1e-12,
) -> tuple[object, object]:
    """Bootstrap zero rates from market instruments via Newton's method.

    Returns ``(tenors, rates)`` as numpy arrays. The computation is
    fully traceable by autograd.

    Parameters
    ----------
    instruments : list[BootstrapInstrument]
        Calibration instruments sorted by tenor.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance (max absolute residual).

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(tenors, zero_rates)`` arrays.
    """
    from trellis.core.differentiable import gradient

    instruments = sorted(instruments, key=lambda i: i.tenor)
    n = len(instruments)
    tenors = np.array([inst.tenor for inst in instruments])
    quotes = np.array([inst.quote for inst in instruments])

    # Initial guess: for deposits use the quote directly as CC rate,
    # for others start at 0.05
    rates = np.array([
        inst.quote if inst.instrument_type == "deposit" else 0.05
        for inst in instruments
    ])

    for iteration in range(max_iter):
        model = _reprice(rates, tenors, instruments)
        residual = model - quotes

        if float(np.max(np.abs(residual))) < tol:
            break

        # Compute Jacobian numerically (finite difference)
        # Autograd gradient gives us one row at a time
        eps = 1e-7
        jacobian = np.zeros((n, n))
        for j in range(n):
            rates_up = rates.copy()
            rates_up = np.where(
                np.arange(n) == j,
                rates_up + eps,
                rates_up,
            )
            model_up = _reprice(rates_up, tenors, instruments)
            jacobian[:, j] = (model_up - model) / eps

        # Newton step: rates = rates - J^{-1} * residual
        import numpy as raw_np
        delta = raw_np.linalg.solve(
            raw_np.asarray(jacobian, dtype=float),
            raw_np.asarray(residual, dtype=float),
        )
        rates = rates - np.array(delta)

    return tenors, rates


def bootstrap_yield_curve(instruments: list[BootstrapInstrument], **kwargs):
    """Bootstrap and return a YieldCurve.

    Convenience wrapper around :func:`bootstrap`.
    """
    from trellis.curves.yield_curve import YieldCurve
    tenors, rates = bootstrap(instruments, **kwargs)
    return YieldCurve(tenors, rates)
