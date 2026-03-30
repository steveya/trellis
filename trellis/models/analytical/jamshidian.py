"""Jamshidian closed-form formula for European options on zero-coupon bonds.

Reference: Hull, *Options, Futures, and Other Derivatives*, Table 28.3;
Brigo & Mercurio (2006), Section 3.3.

The raw kernel is split from the curve-resolving adapter so autograd-enabled
code can differentiate the closed form without going through discount-curve
lookups or float casting.
"""

from __future__ import annotations

from dataclasses import dataclass

from autograd.scipy.stats import norm

from trellis.core.differentiable import get_numpy

np = get_numpy()

_EPS = 1e-12


@dataclass(frozen=True)
class ResolvedJamshidianInputs:
    """Resolved discount factors and contract terms for Jamshidian pricing."""

    discount_factor_expiry: float
    discount_factor_bond: float
    strike: float
    T_exp: float
    T_bond: float
    sigma: float
    a: float


def _B(a: float, tau: float) -> float:
    """Hull-White B(a, tau) function, safe for a -> 0."""
    a_abs = np.abs(a)
    a_safe = np.where(a_abs < _EPS, 1.0, a)
    raw = (1.0 - np.exp(-a_safe * tau)) / a_safe
    return np.where(a_abs < _EPS, tau, raw)


def _sigma_p(sigma: float, a: float, T_exp: float, T_bond: float) -> float:
    """Bond option volatility (sigma_P) under Hull-White."""
    tau = T_bond - T_exp
    B_val = _B(a, tau)
    a_abs = np.abs(a)
    a_safe = np.where(a_abs < _EPS, 1.0, a)
    reversion_term = (1.0 - np.exp(-2.0 * a_safe * T_exp)) / (2.0 * a_safe)
    reversion_term = np.where(a_abs < _EPS, T_exp, reversion_term)
    return sigma * B_val * np.sqrt(np.maximum(reversion_term, 0.0))


def zcb_option_hw_raw(
    resolved: ResolvedJamshidianInputs,
) -> dict[str, float]:
    """Raw Jamshidian kernel over resolved inputs.

    This is the traced entrypoint used by autograd-aware code. It consumes
    resolved discount factors directly and leaves market-data resolution to the
    outer adapter.
    """
    if resolved.T_bond <= resolved.T_exp:
        raise ValueError(
            f"T_bond ({resolved.T_bond}) must be > T_exp ({resolved.T_exp})"
        )

    sp = _sigma_p(
        resolved.sigma,
        resolved.a,
        resolved.T_exp,
        resolved.T_bond,
    )
    zero_vol = sp < _EPS
    safe_sp = np.where(zero_vol, 1.0, sp)

    log_moneyness = np.log(
        resolved.discount_factor_bond / (
            resolved.strike * resolved.discount_factor_expiry
        )
    )
    d1 = log_moneyness / safe_sp + 0.5 * safe_sp
    d2 = d1 - safe_sp

    call = np.where(
        zero_vol,
        np.maximum(
            resolved.discount_factor_bond
            - resolved.strike * resolved.discount_factor_expiry,
            0.0,
        ),
        resolved.discount_factor_bond * norm.cdf(d1)
        - resolved.strike * resolved.discount_factor_expiry * norm.cdf(d2),
    )
    put = np.where(
        zero_vol,
        np.maximum(
            resolved.strike * resolved.discount_factor_expiry
            - resolved.discount_factor_bond,
            0.0,
        ),
        resolved.strike * resolved.discount_factor_expiry * norm.cdf(-d2)
        - resolved.discount_factor_bond * norm.cdf(-d1),
    )

    return {"call": call, "put": put}


def zcb_option_hw(
    discount_curve,
    K: float,
    T_exp: float,
    T_bond: float,
    sigma: float,
    a: float,
) -> dict[str, float]:
    """Jamshidian formula for a European option on a zero-coupon bond."""
    resolved = ResolvedJamshidianInputs(
        discount_factor_expiry=float(discount_curve.discount(T_exp)),
        discount_factor_bond=float(discount_curve.discount(T_bond)),
        strike=K,
        T_exp=T_exp,
        T_bond=T_bond,
        sigma=sigma,
        a=a,
    )
    raw = zcb_option_hw_raw(resolved)
    return {"call": float(raw["call"]), "put": float(raw["put"])}


__all__ = [
    "ResolvedJamshidianInputs",
    "zcb_option_hw",
    "zcb_option_hw_raw",
]
