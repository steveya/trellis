"""Native benchmark outputs for Black-Scholes equity vanilla options (QUA-862).

Returns the canonical ``{price, delta, gamma, vega, theta}`` dict directly
from closed-form Black-Scholes formulae so the FinancePy parity harness can
report Greeks without post-hoc bump-and-reprice (see QUA-863 for the
fallback path that this replaces for exact-binding analytical routes).

Assumes the same contract the deterministic Black-Scholes ``evaluate`` body
assumes: continuous dividend yield ``q == 0``.  The effective rate ``r`` is
recovered from ``market_state.discount``; ``sigma`` is queried from the
Black vol surface at ``(T, K)``.  Notional scales ``price`` only; per-unit
Greeks are returned to match FinancePy's ``EquityVanillaOption.delta``
etc., which themselves do not multiply by ``num_options``.
"""

from __future__ import annotations

from typing import Any

from autograd.scipy.stats import norm

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention

np = get_numpy()


def _zero_time_outputs(
    spot: float,
    strike: float,
    option_type: str,
    notional: float,
) -> dict[str, float]:
    """Return outputs at or past expiry: intrinsic price, zero Greeks."""
    intrinsic = (
        max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    )
    return {
        "price": notional * intrinsic,
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
    }


def equity_vanilla_bs_outputs(
    market_state: MarketState,
    spec: Any,
) -> dict[str, float]:
    """Return ``{price, delta, gamma, vega, theta}`` for a BS equity vanilla."""
    if market_state.discount is None:
        raise ValueError("market_state.discount is required for Black-Scholes outputs")
    if market_state.vol_surface is None:
        raise ValueError("market_state.vol_surface is required for Black-Scholes outputs")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    T = max(
        float(year_fraction(market_state.settlement, spec.expiry_date, day_count)),
        0.0,
    )
    spot = float(spec.spot)
    strike = float(spec.strike)
    option_type = str(spec.option_type or "call").strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError(f"Unsupported option_type {spec.option_type!r}; expected 'call' or 'put'")
    notional = float(spec.notional)

    if T <= 0.0:
        return _zero_time_outputs(spot, strike, option_type, notional)

    df = float(market_state.discount.discount(T))
    sigma = float(market_state.vol_surface.black_vol(max(T, 1e-6), strike))
    sqrt_T = float(np.sqrt(T))
    sigma_sqrt_T = sigma * sqrt_T
    if sigma_sqrt_T <= 0.0:
        return _zero_time_outputs(spot, strike, option_type, notional)

    df_safe = max(df, 1e-12)
    forward = spot / df_safe
    # Continuous-compounding rate implied by ``df``; consistent with the
    # deterministic Black-Scholes evaluate body in executor._deterministic
    # _exact_binding_evaluate_body that uses ``forward = spot / df``.
    r = -float(np.log(df_safe)) / max(T, 1e-12)

    d1 = (float(np.log(forward / strike)) + 0.5 * sigma * sigma * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    nd1 = float(norm.cdf(d1))
    nd2 = float(norm.cdf(d2))
    nd1_neg = float(norm.cdf(-d1))
    nd2_neg = float(norm.cdf(-d2))
    pdf_d1 = float(norm.pdf(d1))

    if option_type == "call":
        price_per_unit = df * (forward * nd1 - strike * nd2)
        delta = nd1
        theta = -(spot * pdf_d1 * sigma) / (2.0 * sqrt_T) - r * strike * df * nd2
    else:
        price_per_unit = df * (strike * nd2_neg - forward * nd1_neg)
        delta = nd1 - 1.0
        theta = -(spot * pdf_d1 * sigma) / (2.0 * sqrt_T) + r * strike * df * nd2_neg

    gamma = pdf_d1 / (spot * sigma_sqrt_T)
    vega = spot * pdf_d1 * sqrt_T

    return {
        "price": notional * price_per_unit,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


__all__ = ("equity_vanilla_bs_outputs",)
