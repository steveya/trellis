"""Jamshidian closed-form formula for European options on zero-coupon bonds
under the Hull-White one-factor model.

Reference: Hull, *Options, Futures, and Other Derivatives*, Table 28.3;
Brigo & Mercurio (2006), Section 3.3.

The formula prices a European call (put) on a zero-coupon bond with
maturity *T_bond*, where the option expires at *T_exp* with strike *K*
(quoted on unit notional).

Under Hull-White::

    dr = [theta(t) - a*r] dt + sigma dW

the call price is::

    Call = P(0, T_bond) * N(d1) - K * P(0, T_exp) * N(d2)

with::

    B(a, tau) = (1 - exp(-a * tau)) / a          (tau = T_bond - T_exp)
    sigma_p   = sigma * B * sqrt((1 - exp(-2*a*T_exp)) / (2*a))
    d1        = ln(P(0,T_bond) / (K * P(0,T_exp))) / sigma_p + sigma_p / 2
    d2        = d1 - sigma_p

Put via put-call parity on ZCBs::

    Put = K * P(0, T_exp) * N(-d2) - P(0, T_bond) * N(-d1)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _B(a: float, tau: float) -> float:
    """Hull-White B(a, tau) function, safe for a -> 0."""
    if abs(a) < 1e-10:
        return tau
    return (1.0 - np.exp(-a * tau)) / a


def _sigma_p(sigma: float, a: float, T_exp: float, T_bond: float) -> float:
    """Bond option volatility (sigma_P) under Hull-White."""
    tau = T_bond - T_exp
    B_val = _B(a, tau)
    if abs(a) < 1e-10:
        # Limit: (1 - exp(-2aT))/(2a) -> T  as a -> 0
        return sigma * B_val * np.sqrt(T_exp)
    return sigma * B_val * np.sqrt((1.0 - np.exp(-2.0 * a * T_exp)) / (2.0 * a))


def zcb_option_hw(
    discount_curve,
    K: float,
    T_exp: float,
    T_bond: float,
    sigma: float,
    a: float,
) -> dict[str, float]:
    """Jamshidian formula for a European option on a zero-coupon bond.

    Parameters
    ----------
    discount_curve : DiscountCurve
        Must support ``discount_curve.discount(t) -> float``.
    K : float
        Strike price on *unit face* (e.g., 0.63 for $63 on $100 face).
    T_exp : float
        Option expiry in years.
    T_bond : float
        ZCB maturity in years (must be > T_exp).
    sigma : float
        Hull-White absolute rate volatility.
    a : float
        Hull-White mean-reversion speed.

    Returns
    -------
    dict with keys ``"call"`` and ``"put"`` (prices on unit face).
    """
    if T_bond <= T_exp:
        raise ValueError(f"T_bond ({T_bond}) must be > T_exp ({T_exp})")

    P_exp = float(discount_curve.discount(T_exp))
    P_bond = float(discount_curve.discount(T_bond))
    sp = _sigma_p(sigma, a, T_exp, T_bond)

    if sp < 1e-15:
        # Zero vol: intrinsic value
        call = max(P_bond - K * P_exp, 0.0)
        put = max(K * P_exp - P_bond, 0.0)
        return {"call": call, "put": put}

    d1 = np.log(P_bond / (K * P_exp)) / sp + sp / 2.0
    d2 = d1 - sp

    call = P_bond * norm.cdf(d1) - K * P_exp * norm.cdf(d2)
    put = K * P_exp * norm.cdf(-d2) - P_bond * norm.cdf(-d1)

    return {"call": float(call), "put": float(put)}
