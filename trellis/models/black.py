"""Black76 option pricing formulas — autograd-compatible."""

from __future__ import annotations

from autograd.scipy.stats import norm

from trellis.core.differentiable import get_numpy
from trellis.models.analytical.support import terminal_vanilla_from_basis

np = get_numpy()


def _d1d2(F: float, K: float, sigma: float, T: float) -> tuple[float, float]:
    """Compute d1 and d2 for Black76."""
    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


def _black76_intrinsic(F: float, K: float, *, call: bool) -> float:
    """Return the zero-vol intrinsic value for a call or put."""
    if call:
        return np.maximum(F - K, 0.0)
    return np.maximum(K - F, 0.0)


def _black76_cash_or_nothing_intrinsic(F: float, K: float, *, call: bool) -> float:
    """Return the zero-vol intrinsic value for a cash-or-nothing digital."""
    if call:
        return np.where(F > K, 1.0, 0.0)
    return np.where(F < K, 1.0, 0.0)


def _black76_asset_or_nothing_intrinsic(F: float, K: float, *, call: bool) -> float:
    """Return the zero-vol intrinsic value for an asset-or-nothing digital."""
    if call:
        return np.where(F > K, F, 0.0)
    return np.where(F < K, F, 0.0)


def _black76_terms(F: float, K: float, sigma: float, T: float) -> tuple[float, float, bool]:
    """Return the shared Black76 terms for basis pricing."""
    sigma_safe = np.where(sigma > 0.0, sigma, 1.0)
    T_safe = np.where(T > 0.0, T, 1.0)
    d1, d2 = _d1d2(F, K, sigma_safe, T_safe)
    valid = (sigma > 0.0) & (T > 0.0)
    return d1, d2, valid


def _black76_asset_or_nothing_price(
    F: float,
    K: float,
    sigma: float,
    T: float,
    *,
    call: bool,
) -> float:
    """Return a differentiable Black76 asset-or-nothing price with fallback."""
    d1, _, valid = _black76_terms(F, K, sigma, T)
    if call:
        price = F * norm.cdf(d1)
    else:
        price = F * norm.cdf(-d1)
    return np.where(valid, price, _black76_asset_or_nothing_intrinsic(F, K, call=call))


def _black76_basis_prices(
    F: float,
    K: float,
    sigma: float,
    T: float,
) -> tuple[float, float, float, float]:
    """Return the four terminal basis claims implied by Black76."""
    d1, d2, valid = _black76_terms(F, K, sigma, T)
    asset_call = np.where(
        valid,
        F * norm.cdf(d1),
        _black76_asset_or_nothing_intrinsic(F, K, call=True),
    )
    asset_put = np.where(
        valid,
        F * norm.cdf(-d1),
        _black76_asset_or_nothing_intrinsic(F, K, call=False),
    )
    cash_call = np.where(
        valid,
        norm.cdf(d2),
        _black76_cash_or_nothing_intrinsic(F, K, call=True),
    )
    cash_put = np.where(
        valid,
        norm.cdf(-d2),
        _black76_cash_or_nothing_intrinsic(F, K, call=False),
    )
    return asset_call, asset_put, cash_call, cash_put


def _black76_cash_or_nothing_price(
    F: float,
    K: float,
    sigma: float,
    T: float,
    *,
    call: bool,
) -> float:
    """Return a differentiable Black76 digital price with an intrinsic fallback."""
    _, d2, valid = _black76_terms(F, K, sigma, T)
    if call:
        price = norm.cdf(d2)
    else:
        price = norm.cdf(-d2)
    return np.where(valid, price, _black76_cash_or_nothing_intrinsic(F, K, call=call))


def black76_call(F: float, K: float, sigma: float, T: float) -> float:
    """Undiscounted Black76 call price assembled from basis claims.

    Parameters
    ----------
    F : float
        Forward rate.
    K : float
        Strike rate.
    sigma : float
        Black (lognormal) volatility.
    T : float
        Time to expiry in years.

    Returns
    -------
    float
        Undiscounted call value: F*N(d1) - K*N(d2).
    """
    asset_call, _, cash_call, _ = _black76_basis_prices(F, K, sigma, T)
    return asset_call - K * cash_call


def black76_put(F: float, K: float, sigma: float, T: float) -> float:
    """Undiscounted Black76 put price assembled from basis claims.

    Returns
    -------
    float
        Undiscounted put value: K*N(-d2) - F*N(-d1).
    """
    _, asset_put, _, cash_put = _black76_basis_prices(F, K, sigma, T)
    return K * cash_put - asset_put


def black76_asset_or_nothing_call(F: float, K: float, sigma: float, T: float) -> float:
    """Undiscounted Black76 asset-or-nothing call price."""
    return _black76_asset_or_nothing_price(F, K, sigma, T, call=True)


def black76_asset_or_nothing_put(F: float, K: float, sigma: float, T: float) -> float:
    """Undiscounted Black76 asset-or-nothing put price."""
    return _black76_asset_or_nothing_price(F, K, sigma, T, call=False)


def black76_cash_or_nothing_call(F: float, K: float, sigma: float, T: float) -> float:
    """Undiscounted Black76 cash-or-nothing call price."""
    return _black76_cash_or_nothing_price(F, K, sigma, T, call=True)


def black76_cash_or_nothing_put(F: float, K: float, sigma: float, T: float) -> float:
    """Undiscounted Black76 cash-or-nothing put price."""
    return _black76_cash_or_nothing_price(F, K, sigma, T, call=False)


def garman_kohlhagen_call(
    spot: float,
    strike: float,
    sigma: float,
    T: float,
    df_domestic: float,
    df_foreign: float,
) -> float:
    """Domestic-currency FX vanilla call under Garman-Kohlhagen basis assembly.

    Parameters
    ----------
    spot
        Spot FX quote using the same convention as :class:`trellis.instruments.fx.FXRate`.
    strike
        Strike in domestic currency per unit of foreign currency.
    sigma
        Lognormal FX volatility.
    T
        Time to expiry in years.
    df_domestic
        Domestic discount factor to expiry.
    df_foreign
        Foreign discount factor to expiry.
    """
    forward = spot * df_foreign / df_domestic
    asset_call, _, cash_call, _ = _black76_basis_prices(forward, strike, sigma, T)
    return df_domestic * terminal_vanilla_from_basis(
        "call",
        asset_value=asset_call,
        cash_value=cash_call,
        strike=strike,
    )


def garman_kohlhagen_put(
    spot: float,
    strike: float,
    sigma: float,
    T: float,
    df_domestic: float,
    df_foreign: float,
) -> float:
    """Domestic-currency FX vanilla put under Garman-Kohlhagen basis assembly."""
    forward = spot * df_foreign / df_domestic
    _, asset_put, _, cash_put = _black76_basis_prices(forward, strike, sigma, T)
    return df_domestic * terminal_vanilla_from_basis(
        "put",
        asset_value=asset_put,
        cash_value=cash_put,
        strike=strike,
    )
