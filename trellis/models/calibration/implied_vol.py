"""Implied volatility solvers."""

from __future__ import annotations

import numpy as raw_np
from scipy.optimize import brentq
from scipy.stats import norm


class ImpliedVolError(ValueError):
    """Structured implied-vol inversion failure."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        market_price: float,
        lower_bound: float,
        upper_bound: float,
        carry_rate: float,
    ) -> None:
        super().__init__(message)
        self.reason = str(reason)
        self.market_price = float(market_price)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.carry_rate = float(carry_rate)


def _resolve_carry_rate(
    r: float,
    *,
    dividend_yield: float = 0.0,
    carry_rate: float | None = None,
) -> float:
    """Return one consistent carry rate for Black-style pricing."""
    if carry_rate is not None and abs(float(dividend_yield)) > 1e-12:
        raise ValueError("specify either dividend_yield or carry_rate, not both")
    if not raw_np.isfinite(r):
        raise ValueError("r must be finite")
    if not raw_np.isfinite(dividend_yield):
        raise ValueError("dividend_yield must be finite")
    if carry_rate is None:
        return float(r) - float(dividend_yield)
    if not raw_np.isfinite(carry_rate):
        raise ValueError("carry_rate must be finite")
    return float(carry_rate)


def _black_inputs(S, K, T, r, *, carry_rate: float) -> tuple[float, float, float]:
    """Return discounted-forward inputs for Black-style option pricing."""
    if not raw_np.isfinite(S) or S <= 0.0:
        raise ValueError("S must be finite and positive")
    if not raw_np.isfinite(K) or K <= 0.0:
        raise ValueError("K must be finite and positive")
    if not raw_np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and positive")
    forward = float(S) * float(raw_np.exp(float(carry_rate) * float(T)))
    discount = float(raw_np.exp(-float(r) * float(T)))
    return float(forward), float(discount), float(raw_np.sqrt(float(T)))


def _black_price_bounds(
    S,
    K,
    T,
    r,
    *,
    option_type: str,
    carry_rate: float,
) -> tuple[float, float]:
    """Return Black-style arbitrage bounds under one carry convention."""
    forward, discount, _sqrt_t = _black_inputs(S, K, T, r, carry_rate=carry_rate)
    discounted_forward = discount * forward
    discounted_strike = discount * float(K)
    if option_type == "call":
        return max(discounted_forward - discounted_strike, 0.0), discounted_forward
    if option_type == "put":
        return max(discounted_strike - discounted_forward, 0.0), discounted_strike
    raise ValueError("option_type must be `call` or `put`")


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    *,
    dividend_yield: float = 0.0,
    carry_rate: float | None = None,
) -> float:
    """Find Black-Scholes implied vol via Brent's method.

    Parameters
    ----------
    market_price : float
        Observed option price.
    S, K, T, r : float
        Spot, strike, time, rate.
    option_type : str
        ``"call"`` or ``"put"``.
    """
    resolved_carry_rate = _resolve_carry_rate(
        r,
        dividend_yield=dividend_yield,
        carry_rate=carry_rate,
    )
    lower_bound, upper_bound = _black_price_bounds(
        S,
        K,
        T,
        r,
        option_type=option_type,
        carry_rate=resolved_carry_rate,
    )
    if market_price < lower_bound - tol or market_price > upper_bound + tol:
        raise ImpliedVolError(
            (
                f"market price {float(market_price):g} is outside Black implied-vol bounds "
                f"[{lower_bound:g}, {upper_bound:g}] for the selected carry convention"
            ),
            reason="quote_convention_mismatch",
            market_price=market_price,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            carry_rate=resolved_carry_rate,
        )
    if market_price <= lower_bound + tol:
        return 0.0

    def objective(sigma):
        """Return the Black-Scholes pricing error at volatility ``sigma``."""
        return _bs_price(
            S,
            K,
            T,
            r,
            sigma,
            option_type,
            carry_rate=resolved_carry_rate,
        ) - market_price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=tol)
    except ValueError as exc:
        raise ImpliedVolError(
            (
                f"Black implied-vol inversion failed numerically for price {float(market_price):g} "
                f"under carry rate {resolved_carry_rate:g}: {exc}"
            ),
            reason="numerical_failure",
            market_price=market_price,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            carry_rate=resolved_carry_rate,
        ) from exc


def implied_vol_jaeckel(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    *,
    dividend_yield: float = 0.0,
    carry_rate: float | None = None,
) -> float:
    """Implied vol via rational approximation (Jaeckel-inspired).

    Uses Brent as fallback for edge cases.
    """
    resolved_carry_rate = _resolve_carry_rate(
        r,
        dividend_yield=dividend_yield,
        carry_rate=carry_rate,
    )
    # Normalized price
    F, df, _sqrt_t = _black_inputs(S, K, T, r, carry_rate=resolved_carry_rate)

    if option_type == "call":
        intrinsic = max(F - K, 0) * df
    else:
        intrinsic = max(K - F, 0) * df

    lower_bound, upper_bound = _black_price_bounds(
        S,
        K,
        T,
        r,
        option_type=option_type,
        carry_rate=resolved_carry_rate,
    )
    if market_price < lower_bound - 1e-12 or market_price > upper_bound + 1e-12:
        raise ImpliedVolError(
            (
                f"market price {float(market_price):g} is outside Black implied-vol bounds "
                f"[{lower_bound:g}, {upper_bound:g}] for the selected carry convention"
            ),
            reason="quote_convention_mismatch",
            market_price=market_price,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            carry_rate=resolved_carry_rate,
        )
    if market_price <= intrinsic + 1e-12:
        return 0.0

    # Initial guess from Brenner-Subrahmanyam
    sigma_init = raw_np.sqrt(2 * raw_np.pi / T) * market_price / max(df * F, 1e-12)

    # Newton refinement
    sigma = max(sigma_init, 0.01)
    for _ in range(20):
        price = _bs_price(
            S,
            K,
            T,
            r,
            sigma,
            option_type,
            carry_rate=resolved_carry_rate,
        )
        vega = _bs_vega(S, K, T, r, sigma, carry_rate=resolved_carry_rate)
        if vega < 1e-12:
            break
        sigma -= (price - market_price) / vega
        sigma = max(sigma, 1e-6)
        if abs(price - market_price) < 1e-10:
            return sigma

    # Fallback
    return implied_vol(
        market_price,
        S,
        K,
        T,
        r,
        option_type,
        dividend_yield=dividend_yield,
        carry_rate=carry_rate,
    )


def _bs_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type,
    *,
    dividend_yield: float = 0.0,
    carry_rate: float | None = None,
):
    """Return the Black-Scholes call or put price used by the IV solvers."""
    resolved_carry_rate = _resolve_carry_rate(
        r,
        dividend_yield=dividend_yield,
        carry_rate=carry_rate,
    )
    forward, discount, sqrt_t = _black_inputs(S, K, T, r, carry_rate=resolved_carry_rate)
    d1 = (raw_np.log(forward / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if option_type == "call":
        return discount * (forward * norm.cdf(d1) - K * norm.cdf(d2))
    if option_type == "put":
        return discount * (K * norm.cdf(-d2) - forward * norm.cdf(-d1))
    raise ValueError("option_type must be `call` or `put`")


def _bs_vega(
    S,
    K,
    T,
    r,
    sigma,
    *,
    dividend_yield: float = 0.0,
    carry_rate: float | None = None,
):
    """Return the Black-Scholes vega used by the Newton-style IV refinement."""
    resolved_carry_rate = _resolve_carry_rate(
        r,
        dividend_yield=dividend_yield,
        carry_rate=carry_rate,
    )
    forward, discount, sqrt_t = _black_inputs(S, K, T, r, carry_rate=resolved_carry_rate)
    d1 = (raw_np.log(forward / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_t)
    return discount * forward * sqrt_t * norm.pdf(d1)
