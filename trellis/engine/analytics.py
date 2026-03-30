"""Risk sensitivity (Greek) computation via automatic differentiation.

Greeks measure how a bond's price changes when interest rates move.
This module uses autograd to differentiate the pricing function with
respect to the curve rates, giving exact sensitivities without
finite-difference approximation.
"""

from __future__ import annotations

from trellis.core.differentiable import get_numpy, gradient

np = get_numpy()


def compute_greeks(
    price_fn,
    curve_rates,
    tenors=None,
    measures: list[str] | None = None,
) -> dict[str, float]:
    """Compute interest-rate Greeks for a pricing function.

    Parameters
    ----------
    price_fn : callable
        ``price_fn(rates_array) -> float`` — the bond price as a function
        of the continuously compounded zero-rate vector.
    curve_rates : array-like
        Current curve rates (1-D).
    tenors : array-like or None
        Corresponding tenors in years (used for DV01 scaling).
    measures : list[str] or None
        Specific Greeks to compute.  ``None`` → compute everything (backward
        compatible).  When provided, only the requested keys are returned.

    Returns
    -------
    dict with keys: dv01, duration, modified_duration, convexity,
    and per-tenor sensitivities (key_rate_durations).
    """
    import numpy as _np
    rates = _np.asarray(curve_rates, dtype=float)
    price = float(price_fn(rates))

    greeks: dict[str, float] = {"price": price}

    need_all = measures is None
    need = set(measures) if measures is not None else set()

    # First-order Greeks need the gradient
    need_grad = need_all or bool(
        need & {"dv01", "duration", "modified_duration", "key_rate_durations"}
    )

    dP_dr = None
    if need_grad:
        grad_fn = gradient(price_fn, 0)
        dP_dr = grad_fn(rates)

        if need_all or "dv01" in need:
            greeks["dv01"] = -float(np.sum(dP_dr)) * 0.0001

        dur = -float(np.sum(dP_dr)) / price
        if need_all or "duration" in need:
            greeks["duration"] = dur
        if need_all or "modified_duration" in need:
            greeks["modified_duration"] = dur

    # Convexity via the second derivative of a parallel rate shift.
    if need_all or "convexity" in need:
        def shifted_price(shift):
            return price_fn(rates + shift)

        second_derivative = gradient(gradient(shifted_price, 0), 0)
        d2p_dy2 = float(second_derivative(0.0))
        greeks["convexity"] = 0.0 if price == 0 else d2p_dy2 / price

    # Key-rate durations
    if tenors is not None and (need_all or "key_rate_durations" in need):
        if dP_dr is None:
            grad_fn = gradient(price_fn, 0)
            dP_dr = grad_fn(rates)
        krd = {}
        for i, t in enumerate(tenors):
            krd[f"KRD_{t}y"] = -float(dP_dr[i]) / price
        greeks["key_rate_durations"] = krd

    return greeks
