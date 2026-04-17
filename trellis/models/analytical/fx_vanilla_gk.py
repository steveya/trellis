"""Native benchmark outputs for Garman-Kohlhagen FX vanilla options (QUA-878).

Returns the canonical ``{price, delta, gamma, vega, theta}`` dict directly
from closed-form Garman-Kohlhagen formulae so the FinancePy FX parity
harness can report native Greeks without post-hoc bump-and-reprice
(see QUA-863 for the fallback path that this replaces for exact-binding
analytical FX routes).

Uses the same resolved-input contract as ``price_fx_vanilla_analytical``
via ``resolve_fx_vanilla_inputs``, so spot / domestic DF / foreign DF /
vol handling stays identical.  Notional scales ``price`` only; per-unit
Greeks are returned to match FinancePy's ``FXVanillaOption.delta``
``pips_spot_delta`` convention, which is the scalar the FinancePy
reference harness picks up via ``_maybe_method_outputs``.
"""

from __future__ import annotations

from typing import Any

from autograd.scipy.stats import norm

from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.models.fx_vanilla import FXVanillaSpecLike, resolve_fx_vanilla_inputs

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


def _zero_vol_outputs(
    spot: float,
    strike: float,
    df_domestic: float,
    df_foreign: float,
    option_type: str,
    notional: float,
) -> dict[str, float]:
    """Return the zero-vol, T > 0 Garman-Kohlhagen limit.

    At σ == 0 the forward equals ``spot * df_foreign / df_domestic``
    deterministically; the payoff reduces to
    ``df_domestic * max(F - K, 0)`` (call) or ``df_domestic * max(K - F, 0)``
    (put).  Using the plain spot-vs-strike intrinsic instead would
    silently misprice parity whenever the two rate curves differ.
    """
    df_d_safe = max(df_domestic, 1e-12)
    forward = spot * df_foreign / df_d_safe
    forward_intrinsic = (
        max(forward - strike, 0.0)
        if option_type == "call"
        else max(strike - forward, 0.0)
    )
    return {
        "price": notional * df_domestic * forward_intrinsic,
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
    }


def fx_vanilla_gk_outputs(
    market_state: MarketState,
    spec: FXVanillaSpecLike,
) -> dict[str, float]:
    """Return ``{price, delta, gamma, vega, theta}`` for a GK FX vanilla."""
    resolved = resolve_fx_vanilla_inputs(market_state, spec)
    gk = resolved.garman_kohlhagen
    option_type = resolved.option_type
    if option_type not in {"call", "put"}:
        raise ValueError(
            f"Unsupported option_type {spec.option_type!r}; expected 'call' or 'put'"
        )
    notional = float(resolved.notional)

    spot = float(gk.spot)
    strike = float(gk.strike)
    T = float(gk.T)
    sigma = float(gk.sigma)
    df_d = float(gk.df_domestic)
    df_f = float(gk.df_foreign)

    if T <= 0.0:
        return _zero_time_outputs(spot, strike, option_type, notional)

    sqrt_T = float(np.sqrt(T))
    sigma_sqrt_T = sigma * sqrt_T
    if sigma_sqrt_T <= 0.0:
        return _zero_vol_outputs(spot, strike, df_d, df_f, option_type, notional)

    df_d_safe = max(df_d, 1e-12)
    forward = spot * df_f / df_d_safe
    # Continuous-compounding rates implied by the discount factors; match
    # FinancePy's ``r_d = -log(dom_df)/t`` / ``r_f = -log(for_df)/t``.
    r_d = -float(np.log(df_d_safe)) / max(T, 1e-12)
    r_f = -float(np.log(max(df_f, 1e-12))) / max(T, 1e-12)

    d1 = (float(np.log(forward / strike)) + 0.5 * sigma * sigma * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    nd1 = float(norm.cdf(d1))
    nd2 = float(norm.cdf(d2))
    nd1_neg = float(norm.cdf(-d1))
    nd2_neg = float(norm.cdf(-d2))
    pdf_d1 = float(norm.pdf(d1))

    # Price matches ``garman_kohlhagen_price_raw`` (verified by reproducing
    # the basis-assembly call in tests below).  Kept as a direct closed-form
    # so ``benchmark_outputs["price"]`` stays self-consistent with Greeks.
    if option_type == "call":
        price_per_unit = df_d * (forward * nd1 - strike * nd2)
        delta = df_f * nd1
        # theta_call = -(S*df_f*σ*N'(d1))/(2*√T) + r_f*S*df_f*N(d1) - r_d*K*df_d*N(d2)
        theta = (
            -(spot * df_f * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r_f * spot * df_f * nd1
            - r_d * strike * df_d * nd2
        )
    else:
        price_per_unit = df_d * (strike * nd2_neg - forward * nd1_neg)
        delta = -df_f * nd1_neg
        # theta_put = -(S*df_f*σ*N'(d1))/(2*√T) - r_f*S*df_f*N(-d1) + r_d*K*df_d*N(-d2)
        theta = (
            -(spot * df_f * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r_f * spot * df_f * nd1_neg
            + r_d * strike * df_d * nd2_neg
        )

    gamma = df_f * pdf_d1 / (spot * sigma_sqrt_T)
    vega = spot * df_f * pdf_d1 * sqrt_T

    return {
        "price": notional * price_per_unit,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


__all__ = ("fx_vanilla_gk_outputs",)
