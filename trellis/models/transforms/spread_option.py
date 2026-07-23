"""Two-dimensional Fourier primitives for lognormal spread options."""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.fft import ifft2
from scipy.special import loggamma

from trellis.core.differentiable import get_numpy

np = get_numpy()


def correlated_gbm_log_return_characteristic_function(
    u1,
    u2,
    *,
    T: float,
    rate: float,
    dividend_yields: tuple[float, float],
    volatilities: tuple[float, float],
    correlation: float,
):
    """Return the joint characteristic function of two GBM log returns."""
    horizon = float(T)
    if horizon < 0.0:
        raise ValueError("T must be non-negative")
    q1, q2 = _finite_pair(dividend_yields, name="dividend_yields")
    v1, v2 = _nonnegative_pair(volatilities, name="volatilities")
    rho = _correlation(correlation)
    normalized_rate = float(rate)
    if not math.isfinite(normalized_rate):
        raise ValueError("rate must be finite")

    first_frequency = np.asarray(u1, dtype=complex)
    second_frequency = np.asarray(u2, dtype=complex)
    first_drift = (normalized_rate - q1 - 0.5 * v1 * v1) * horizon
    second_drift = (normalized_rate - q2 - 0.5 * v2 * v2) * horizon
    covariance_term = (
        v1 * v1 * first_frequency * first_frequency
        + 2.0 * rho * v1 * v2 * first_frequency * second_frequency
        + v2 * v2 * second_frequency * second_frequency
    )
    return np.exp(
        1j
        * (
            first_frequency * first_drift
            + second_frequency * second_drift
        )
        - 0.5 * horizon * covariance_term
    )


def hurd_zhou_spread_option_2d_fft(
    characteristic_function: Callable[[object, object], object],
    *,
    spots: tuple[float, float],
    weights: tuple[float, float],
    strike: float,
    discount_factor: float,
    option_type: str = "call",
    grid_size: int = 256,
    frequency_step: float = 0.25,
    damping: tuple[float, float] = (-3.0, 1.0),
) -> float:
    """Price ``(a*S1 - b*S2 - K)+`` using the Hurd-Zhou 2D FFT.

    ``characteristic_function`` is the joint characteristic function of the
    two log returns over the pricing horizon. The positive and negative basket
    weights identify the long and short legs. Put values follow from discounted
    spread parity after computing the Fourier call.
    """
    spot_pair = _positive_pair(spots, name="spots")
    normalized_weights = _finite_pair(weights, name="weights")
    long_idx, short_idx = _spread_leg_indices(normalized_weights)
    normalized_strike = float(strike)
    if not math.isfinite(normalized_strike) or normalized_strike <= 0.0:
        raise ValueError("strike must be positive for Hurd-Zhou pricing")
    df = float(discount_factor)
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("discount_factor must be positive and finite")
    normalized_option_type = str(option_type or "").strip().lower()
    if normalized_option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    n = int(grid_size)
    if n < 32 or n & (n - 1):
        raise ValueError("grid_size must be a power of two and at least 32")
    eta = float(frequency_step)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("frequency_step must be positive and finite")
    eps1, eps2 = _finite_pair(damping, name="damping")
    if eps2 <= 0.0 or eps1 + eps2 >= -1.0:
        raise ValueError("damping requires eps2 > 0 and eps1 + eps2 < -1")

    long_spot = (
        abs(normalized_weights[long_idx]) * spot_pair[long_idx]
    )
    short_spot = (
        abs(normalized_weights[short_idx]) * spot_pair[short_idx]
    )
    x_target = np.asarray(
        [
            math.log(long_spot / normalized_strike),
            math.log(short_spot / normalized_strike),
        ],
        dtype=float,
    )

    frequency = (np.arange(n, dtype=float) - n / 2.0) * eta
    u1_real, u2_real = np.meshgrid(frequency, frequency, indexing="ij")
    u1 = u1_real + 1j * eps1
    u2 = u2_real + 1j * eps2
    transformed_payoff = np.exp(
        loggamma(1j * (u1 + u2) - 1.0)
        + loggamma(-1j * u2)
        - loggamma(1j * u1 + 1.0)
    )

    if long_idx == 0:
        model_cf = characteristic_function(u1, u2)
    else:
        model_cf = characteristic_function(u2, u1)
    alternating = (-1.0) ** (
        np.arange(n, dtype=int)[:, None] + np.arange(n, dtype=int)[None, :]
    )
    shifted_transform = (
        np.exp(
            1j
            * (
                u1_real * float(x_target[0])
                + u2_real * float(x_target[1])
            )
        )
        * model_cf
        * transformed_payoff
    )
    transformed_price = alternating * shifted_transform

    reciprocal_step = 2.0 * math.pi / (n * eta)
    log_offsets = (np.arange(n, dtype=float) - n / 2.0) * reciprocal_step
    inverse = ifft2(transformed_price)
    price_panel = (
        alternating
        * np.exp(
            -eps1 * log_offsets[:, None]
            - eps2 * log_offsets[None, :]
        )
        * (eta * n / (2.0 * math.pi)) ** 2
        * inverse
    )
    centered_value = price_panel[n // 2, n // 2]
    normalized_call = float(
        np.real(
            centered_value
            * math.exp(
                -eps1 * float(x_target[0])
                - eps2 * float(x_target[1])
            )
        )
    )
    call = max(normalized_strike * df * normalized_call, 0.0)
    if normalized_option_type == "call":
        return call
    if long_idx == 0:
        long_growth = characteristic_function(-1j, 0.0)
        short_growth = characteristic_function(0.0, -1j)
    else:
        long_growth = characteristic_function(0.0, -1j)
        short_growth = characteristic_function(-1j, 0.0)
    discounted_forward_spread = df * (
        long_spot * float(np.real(long_growth))
        - short_spot * float(np.real(short_growth))
        - normalized_strike
    )
    return float(max(call - discounted_forward_spread, 0.0))


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


def _correlation(value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not -1.0 <= normalized <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    return normalized


def _spread_leg_indices(weights: tuple[float, float]) -> tuple[int, int]:
    positive = tuple(index for index, value in enumerate(weights) if value > 0.0)
    negative = tuple(index for index, value in enumerate(weights) if value < 0.0)
    if len(positive) != 1 or len(negative) != 1:
        raise ValueError("spread weights require one positive and one negative value")
    return positive[0], negative[0]


__all__ = [
    "correlated_gbm_log_return_characteristic_function",
    "hurd_zhou_spread_option_2d_fft",
]
