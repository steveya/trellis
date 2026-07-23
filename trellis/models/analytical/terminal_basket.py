"""Raw analytical kernels for two-asset terminal basket options."""

from __future__ import annotations

import math

from scipy.special import roots_hermitenorm

from trellis.core.differentiable import get_numpy
from trellis.models.analytical.support import (
    bivariate_standard_normal_cdf,
    implied_zero_rate,
    standard_normal_cdf,
)
from trellis.models.payoffs import terminal_basket_option_payoff

np = get_numpy()


def two_asset_extremum_option_stulz(
    *,
    spots: tuple[float, float],
    strike: float,
    T: float,
    discount_factor: float,
    dividend_yields: tuple[float, float],
    volatilities: tuple[float, float],
    correlation: float,
    basket_style: str,
    option_type: str,
) -> float:
    """Price a call or put on the maximum or minimum of two assets.

    This is the Stulz two-asset closed form. Inputs are explicit scalar market
    coordinates; the function performs no market-state or product resolution.
    """
    s1, s2 = _positive_pair(spots, name="spots")
    q1, q2 = _finite_pair(dividend_yields, name="dividend_yields")
    v1, v2 = _nonnegative_pair(volatilities, name="volatilities")
    normalized_style = _extremum_style(basket_style)
    normalized_option_type = _option_type(option_type)
    normalized_strike = float(strike)
    horizon = float(T)
    df = _positive_float(discount_factor, name="discount_factor")
    rho = _correlation(correlation)

    if normalized_strike < 0.0:
        raise ValueError("strike must be non-negative")
    if horizon <= 0.0:
        payoff = terminal_basket_option_payoff(
            np.asarray([[s1, s2]], dtype=float),
            weights=(0.5, 0.5),
            basket_style=normalized_style,
            strike=normalized_strike,
            option_type=normalized_option_type,
        )
        return float(payoff[0])
    if normalized_strike == 0.0 or v1 == 0.0 or v2 == 0.0:
        return two_asset_terminal_basket_gauss_hermite(
            spots=(s1, s2),
            weights=(0.5, 0.5),
            strike=normalized_strike,
            T=horizon,
            discount_factor=df,
            dividend_yields=(q1, q2),
            volatilities=(v1, v2),
            correlation=rho,
            basket_style=normalized_style,
            option_type=normalized_option_type,
            n_points=64,
        )

    rate = implied_zero_rate(df, horizon)
    b1 = rate - q1
    b2 = rate - q2
    relative_vol = math.sqrt(
        max(v1 * v1 + v2 * v2 - 2.0 * rho * v1 * v2, 0.0)
    )
    if relative_vol <= 1e-14:
        return two_asset_terminal_basket_gauss_hermite(
            spots=(s1, s2),
            weights=(0.5, 0.5),
            strike=normalized_strike,
            T=horizon,
            discount_factor=df,
            dividend_yields=(q1, q2),
            volatilities=(v1, v2),
            correlation=rho,
            basket_style=normalized_style,
            option_type=normalized_option_type,
            n_points=64,
        )

    sqrt_t = math.sqrt(horizon)
    d = (
        math.log(s1 / s2)
        + (b1 - b2 + 0.5 * relative_vol * relative_vol) * horizon
    ) / (relative_vol * sqrt_t)
    y1 = (
        math.log(s1 / normalized_strike) + (b1 + 0.5 * v1 * v1) * horizon
    ) / (v1 * sqrt_t)
    y2 = (
        math.log(s2 / normalized_strike) + (b2 + 0.5 * v2 * v2) * horizon
    ) / (v2 * sqrt_t)
    rho1 = (v1 - rho * v2) / relative_vol
    rho2 = (v2 - rho * v1) / relative_vol
    discounted_s1 = s1 * math.exp(-q1 * horizon)
    discounted_s2 = s2 * math.exp(-q2 * horizon)

    call_max = (
        discounted_s1 * bivariate_standard_normal_cdf(y1, d, rho1)
        + discounted_s2
        * bivariate_standard_normal_cdf(
            y2,
            -d + relative_vol * sqrt_t,
            rho2,
        )
        - normalized_strike
        * df
        * (
            1.0
            - bivariate_standard_normal_cdf(
                -y1 + v1 * sqrt_t,
                -y2 + v2 * sqrt_t,
                rho,
            )
        )
    )
    call_min = (
        discounted_s1 * bivariate_standard_normal_cdf(y1, -d, -rho1)
        + discounted_s2
        * bivariate_standard_normal_cdf(
            y2,
            d - relative_vol * sqrt_t,
            -rho2,
        )
        - normalized_strike
        * df
        * bivariate_standard_normal_cdf(
            y1 - v1 * sqrt_t,
            y2 - v2 * sqrt_t,
            rho,
        )
    )

    if normalized_option_type == "call":
        return float(call_max if normalized_style == "best_of" else call_min)

    expected_max = (
        discounted_s2
        + discounted_s1 * standard_normal_cdf(d)
        - discounted_s2
        * standard_normal_cdf(d - relative_vol * sqrt_t)
    )
    expected_min = (
        discounted_s1
        - discounted_s1 * standard_normal_cdf(d)
        + discounted_s2
        * standard_normal_cdf(d - relative_vol * sqrt_t)
    )
    if normalized_style == "best_of":
        return float(normalized_strike * df - expected_max + call_max)
    return float(normalized_strike * df - expected_min + call_min)


def two_asset_spread_option_kirk(
    *,
    forwards: tuple[float, float],
    strike: float,
    T: float,
    discount_factor: float,
    volatilities: tuple[float, float],
    correlation: float,
    weights: tuple[float, float] = (1.0, -1.0),
    option_type: str = "call",
) -> float:
    """Price a weighted two-asset spread option with Kirk's approximation."""
    forward_pair = _positive_pair(forwards, name="forwards")
    volatility_pair = _nonnegative_pair(volatilities, name="volatilities")
    normalized_weights = _finite_pair(weights, name="weights")
    long_idx, short_idx = _spread_leg_indices(normalized_weights)
    normalized_option_type = _option_type(option_type)
    normalized_strike = float(strike)
    horizon = float(T)
    df = _positive_float(discount_factor, name="discount_factor")
    rho = _correlation(correlation)
    if normalized_strike < 0.0:
        raise ValueError("strike must be non-negative for Kirk pricing")

    long_forward = abs(normalized_weights[long_idx]) * forward_pair[long_idx]
    short_forward = abs(normalized_weights[short_idx]) * forward_pair[short_idx]
    denominator = short_forward + normalized_strike
    if denominator <= 0.0:
        raise ValueError("weighted short forward plus strike must be positive")
    if horizon <= 0.0:
        call = max(long_forward - denominator, 0.0)
        if normalized_option_type == "call":
            return float(call)
        return float(call - long_forward + denominator)

    long_vol = volatility_pair[long_idx]
    short_vol = volatility_pair[short_idx]
    short_ratio = short_forward / denominator
    effective_variance = (
        long_vol * long_vol
        - 2.0 * rho * long_vol * short_vol * short_ratio
        + short_vol * short_vol * short_ratio * short_ratio
    )
    effective_vol = math.sqrt(max(effective_variance, 0.0))
    if effective_vol <= 0.0:
        call = df * max(long_forward - denominator, 0.0)
    else:
        sqrt_t = math.sqrt(horizon)
        d1 = (
            math.log(long_forward / denominator)
            + 0.5 * effective_vol * effective_vol * horizon
        ) / (effective_vol * sqrt_t)
        d2 = d1 - effective_vol * sqrt_t
        call = df * (
            long_forward * standard_normal_cdf(d1)
            - denominator * standard_normal_cdf(d2)
        )
    if normalized_option_type == "call":
        return float(call)
    return float(call - df * (long_forward - denominator))


def two_asset_terminal_basket_gauss_hermite(
    *,
    spots: tuple[float, float],
    weights: tuple[float, float],
    strike: float,
    T: float,
    discount_factor: float,
    dividend_yields: tuple[float, float],
    volatilities: tuple[float, float],
    correlation: float,
    basket_style: str,
    option_type: str,
    n_points: int = 32,
) -> float:
    """Return a two-dimensional Gauss-Hermite reference price.

    This numerical reference is intentionally separate from the Stulz, Kirk,
    and Fourier algorithms so callers can cross-check method-specific kernels.
    """
    s1, s2 = _positive_pair(spots, name="spots")
    normalized_weights = _finite_pair(weights, name="weights")
    q1, q2 = _finite_pair(dividend_yields, name="dividend_yields")
    v1, v2 = _nonnegative_pair(volatilities, name="volatilities")
    normalized_style = _basket_style(basket_style)
    normalized_option_type = _option_type(option_type)
    horizon = float(T)
    df = _positive_float(discount_factor, name="discount_factor")
    rho = _correlation(correlation)
    point_count = int(n_points)
    if point_count < 4:
        raise ValueError("n_points must be at least 4")
    if horizon <= 0.0:
        payoff = terminal_basket_option_payoff(
            np.asarray([[s1, s2]], dtype=float),
            weights=normalized_weights,
            basket_style=normalized_style,
            strike=float(strike),
            option_type=normalized_option_type,
        )
        return float(payoff[0])

    nodes, quadrature_weights = roots_hermitenorm(point_count)
    rate = implied_zero_rate(df, horizon)
    sqrt_t = math.sqrt(horizon)
    sqrt_one_minus_rho_sq = math.sqrt(max(1.0 - rho * rho, 0.0))
    drift1 = (rate - q1 - 0.5 * v1 * v1) * horizon
    drift2 = (rate - q2 - 0.5 * v2 * v2) * horizon
    scale1 = v1 * sqrt_t
    scale2 = v2 * sqrt_t
    expectation = 0.0
    for i, first_normal in enumerate(nodes):
        for j, independent_normal in enumerate(nodes):
            second_normal = (
                rho * first_normal
                + sqrt_one_minus_rho_sq * independent_normal
            )
            terminal = np.asarray(
                [
                    [
                        s1 * math.exp(drift1 + scale1 * first_normal),
                        s2 * math.exp(drift2 + scale2 * second_normal),
                    ]
                ],
                dtype=float,
            )
            payoff = terminal_basket_option_payoff(
                terminal,
                weights=normalized_weights,
                basket_style=normalized_style,
                strike=float(strike),
                option_type=normalized_option_type,
            )
            expectation += (
                float(quadrature_weights[i])
                * float(quadrature_weights[j])
                * float(payoff[0])
            )
    return float(df * expectation / (2.0 * math.pi))


def _finite_pair(values, *, name: str) -> tuple[float, float]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} values must be finite")
    return normalized


def _positive_pair(values, *, name: str) -> tuple[float, float]:
    normalized = _finite_pair(values, name=name)
    if any(value <= 0.0 for value in normalized):
        raise ValueError(f"{name} values must be positive")
    return normalized


def _nonnegative_pair(values, *, name: str) -> tuple[float, float]:
    normalized = _finite_pair(values, name=name)
    if any(value < 0.0 for value in normalized):
        raise ValueError(f"{name} values must be non-negative")
    return normalized


def _positive_float(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return normalized


def _correlation(value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not -1.0 <= normalized <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    return normalized


def _basket_style(value: object) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "best": "best_of",
        "bestof": "best_of",
        "best_of_two": "best_of",
        "worst": "worst_of",
        "worstof": "worst_of",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"weighted_sum", "spread", "best_of", "worst_of"}:
        raise ValueError(
            "basket_style must be weighted_sum, spread, best_of, or worst_of"
        )
    return normalized


def _extremum_style(value: object) -> str:
    normalized = _basket_style(value)
    if normalized not in {"best_of", "worst_of"}:
        raise ValueError("Stulz pricing requires basket_style best_of or worst_of")
    return normalized


def _option_type(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return normalized


def _spread_leg_indices(weights: tuple[float, float]) -> tuple[int, int]:
    positive = tuple(index for index, value in enumerate(weights) if value > 0.0)
    negative = tuple(index for index, value in enumerate(weights) if value < 0.0)
    if len(positive) != 1 or len(negative) != 1:
        raise ValueError("spread weights require one positive and one negative value")
    return positive[0], negative[0]


__all__ = [
    "two_asset_extremum_option_stulz",
    "two_asset_spread_option_kirk",
    "two_asset_terminal_basket_gauss_hermite",
]
