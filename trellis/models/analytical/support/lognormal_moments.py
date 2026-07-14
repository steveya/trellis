"""Moments and bounded lognormal matching for weighted lognormal sums."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as raw_np

from trellis.core.differentiable import get_numpy


_VALIDATION_TOLERANCE = 1e-12
np = get_numpy()


def _float_tuple(values) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _value_tuple(values):
    return tuple(values)


def _value_matrix(values):
    return tuple(tuple(row) for row in values)


def _primal_float(value) -> float:
    primal = value
    while hasattr(primal, "_value"):
        primal = primal._value
    array = raw_np.asarray(primal)
    if array.size != 1:
        raise ValueError("weighted lognormal inputs must be scalar values")
    return float(array.reshape(-1)[0])


def _scale_tolerance(*values: float) -> float:
    return _VALIDATION_TOLERANCE * max(1.0, *(abs(float(value)) for value in values))


@dataclass(frozen=True)
class WeightedLognormalSumContract:
    """Explicit marginals and log covariance for a weighted lognormal sum.

    ``carries`` are expected-growth rates, so observation ``i`` has mean
    ``initial_levels[i] * exp(carries[i] * observation_times[i])``.
    ``log_covariance`` is the covariance matrix of the observations' log
    returns over their declared horizons.
    """

    observation_times: tuple[float, ...]
    weights: tuple[float, ...]
    initial_levels: tuple[float, ...]
    carries: tuple[float, ...]
    log_covariance: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        times = _float_tuple(self.observation_times)
        weights = _value_tuple(self.weights)
        levels = _value_tuple(self.initial_levels)
        carries = _value_tuple(self.carries)
        covariance = _value_matrix(self.log_covariance)
        dimension = len(times)

        if dimension == 0:
            raise ValueError("weighted lognormal sum requires at least one observation")
        if not (
            len(weights) == dimension
            and len(levels) == dimension
            and len(carries) == dimension
        ):
            raise ValueError(
                "observation_times, weights, initial_levels, and carries must have the same length"
            )
        input_values = tuple(_primal_float(value) for value in (*weights, *levels, *carries))
        if any(not isfinite(value) for value in (*times, *input_values)):
            raise ValueError("weighted lognormal sum inputs must contain finite values")
        if any(time < 0.0 for time in times):
            raise ValueError("observation_times must be non-negative")
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("observation_times must be non-decreasing")
        level_values = tuple(_primal_float(level) for level in levels)
        weight_values = tuple(_primal_float(weight) for weight in weights)
        if any(level <= 0.0 for level in level_values):
            raise ValueError("initial_levels must be strictly positive")
        if all(weight == 0.0 for weight in weight_values):
            raise ValueError("weights must contain at least one non-zero value")
        if len(covariance) != dimension or any(
            len(row) != dimension for row in covariance
        ):
            raise ValueError(
                "log_covariance must be a square matrix matching the observation count"
            )

        covariance_array = raw_np.asarray(
            tuple(
                tuple(_primal_float(value) for value in row)
                for row in covariance
            ),
            dtype=float,
        )
        if raw_np.any(~raw_np.isfinite(covariance_array)):
            raise ValueError("log_covariance must contain finite values")
        if not raw_np.allclose(
            covariance_array,
            covariance_array.T,
            rtol=0.0,
            atol=_VALIDATION_TOLERANCE,
        ):
            raise ValueError("log_covariance must be symmetric")
        if raw_np.any(raw_np.diag(covariance_array) < -_VALIDATION_TOLERANCE):
            raise ValueError("log_covariance diagonal must be non-negative")
        try:
            minimum_eigenvalue = float(
                raw_np.min(raw_np.linalg.eigvalsh(covariance_array))
            )
        except raw_np.linalg.LinAlgError as exc:
            raise ValueError("log_covariance must be positive semidefinite") from exc
        if minimum_eigenvalue < -_VALIDATION_TOLERANCE:
            raise ValueError("log_covariance must be positive semidefinite")

        object.__setattr__(self, "observation_times", times)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "initial_levels", levels)
        object.__setattr__(self, "carries", carries)
        object.__setattr__(self, "log_covariance", covariance)


@dataclass(frozen=True)
class WeightedLognormalSumMoments:
    """First two raw moments and variance of a weighted lognormal sum."""

    mean: float
    second_moment: float
    variance: float
    nonnegative_support: bool = True

    def __post_init__(self) -> None:
        mean = _primal_float(self.mean)
        second_moment = _primal_float(self.second_moment)
        variance = _primal_float(self.variance)
        if not all(isfinite(value) for value in (mean, second_moment, variance)):
            raise ValueError("weighted lognormal moments must be finite")

        expected_variance = second_moment - mean * mean
        tolerance = _scale_tolerance(mean * mean, second_moment, variance)
        if expected_variance < -tolerance:
            raise ValueError("second moment cannot be below mean squared")
        if abs(variance - expected_variance) > tolerance:
            raise ValueError("variance must equal second_moment minus mean squared")
        normalized_variance = (
            0.0 if abs(expected_variance) <= tolerance else self.variance
        )

        object.__setattr__(self, "mean", self.mean)
        object.__setattr__(self, "second_moment", self.second_moment)
        object.__setattr__(self, "variance", normalized_variance)
        object.__setattr__(self, "nonnegative_support", bool(self.nonnegative_support))


@dataclass(frozen=True)
class LognormalMomentMatch:
    """Moment-matched lognormal parameters for a positive weighted sum."""

    mean: float
    second_moment: float
    variance: float
    total_log_variance: float

    def __post_init__(self) -> None:
        mean = _primal_float(self.mean)
        second_moment = _primal_float(self.second_moment)
        variance = _primal_float(self.variance)
        total_log_variance = _primal_float(self.total_log_variance)
        if not all(
            isfinite(value)
            for value in (mean, second_moment, variance, total_log_variance)
        ):
            raise ValueError("matched lognormal moments must be finite")
        if mean <= 0.0:
            raise ValueError("matched lognormal mean must be strictly positive")

        expected_variance = second_moment - mean * mean
        moment_tolerance = _scale_tolerance(
            mean * mean,
            second_moment,
            variance,
        )
        if expected_variance < -moment_tolerance:
            raise ValueError("second moment cannot be below mean squared")
        if abs(variance - expected_variance) > moment_tolerance:
            raise ValueError("variance must equal second_moment minus mean squared")
        normalized_variance = (
            0.0 if abs(expected_variance) <= moment_tolerance else self.variance
        )
        ratio = self.second_moment / (self.mean * self.mean)
        ratio_value = _primal_float(ratio)
        log_tolerance = _scale_tolerance(ratio_value, total_log_variance)
        expected_log_variance = (
            0.0 if ratio_value <= 1.0 + log_tolerance else np.log(ratio)
        )
        if total_log_variance < -log_tolerance:
            raise ValueError("total_log_variance must be non-negative")
        if (
            abs(total_log_variance - _primal_float(expected_log_variance))
            > log_tolerance
        ):
            raise ValueError("total_log_variance is inconsistent with the declared moments")

        object.__setattr__(self, "mean", self.mean)
        object.__setattr__(self, "second_moment", self.second_moment)
        object.__setattr__(self, "variance", normalized_variance)
        object.__setattr__(self, "total_log_variance", expected_log_variance)

    def effective_volatility(self, *, maturity: float) -> float:
        """Return the Black-style volatility over a positive maturity."""
        horizon = _primal_float(maturity)
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("maturity must be finite and positive")
        return np.sqrt(self.total_log_variance / maturity)


def single_factor_lognormal_sum_contract(
    *,
    spot: float,
    observation_times: tuple[float, ...],
    weights: tuple[float, ...],
    carry: float,
    volatility: float,
) -> WeightedLognormalSumContract:
    """Build the exact log-covariance contract for one constant-parameter GBM."""
    initial_level = spot
    drift = carry
    sigma = volatility
    times = _float_tuple(observation_times)
    initial_level_value = _primal_float(initial_level)
    drift_value = _primal_float(drift)
    sigma_value = _primal_float(sigma)
    if not all(
        isfinite(value) for value in (initial_level_value, drift_value, sigma_value)
    ):
        raise ValueError("spot, carry, and volatility must be finite")
    if initial_level_value <= 0.0:
        raise ValueError("spot must be strictly positive")
    if sigma_value < 0.0:
        raise ValueError("volatility must be non-negative")

    covariance = tuple(
        tuple(sigma * sigma * min(left_time, right_time) for right_time in times)
        for left_time in times
    )
    return WeightedLognormalSumContract(
        observation_times=times,
        weights=weights,
        initial_levels=(initial_level,) * len(times),
        carries=(drift,) * len(times),
        log_covariance=covariance,
    )


def weighted_lognormal_sum_moments(
    contract: WeightedLognormalSumContract,
) -> WeightedLognormalSumMoments:
    """Return exact first and second moments under the declared log covariance."""
    marginal_means = tuple(
        level * np.exp(carry * time)
        for level, carry, time in zip(
            contract.initial_levels,
            contract.carries,
            contract.observation_times,
            strict=True,
        )
    )
    weighted_means = tuple(
        weight * marginal_mean
        for weight, marginal_mean in zip(
            contract.weights,
            marginal_means,
            strict=True,
        )
    )
    mean = sum(weighted_means)
    second_moment = sum(
        left_mean * right_mean * np.exp(contract.log_covariance[left_idx][right_idx])
        for left_idx, left_mean in enumerate(weighted_means)
        for right_idx, right_mean in enumerate(weighted_means)
    )
    variance = second_moment - mean * mean
    mean_value = _primal_float(mean)
    second_moment_value = _primal_float(second_moment)
    variance_value = _primal_float(variance)
    tolerance = _scale_tolerance(mean_value * mean_value, second_moment_value)
    if variance_value < 0.0 and variance_value >= -tolerance:
        variance = np.maximum(variance, 0.0)
    return WeightedLognormalSumMoments(
        mean=mean,
        second_moment=second_moment,
        variance=variance,
        nonnegative_support=all(
            _primal_float(weight) >= 0.0 for weight in contract.weights
        ),
    )


def match_lognormal_moments(
    moments: WeightedLognormalSumMoments,
) -> LognormalMomentMatch:
    """Fit a lognormal distribution to valid moments of a positive sum."""
    if not moments.nonnegative_support:
        raise ValueError("lognormal matching requires non-negative weights")
    mean = _primal_float(moments.mean)
    if mean <= 0.0:
        raise ValueError("lognormal matching requires a strictly positive mean")

    ratio = moments.second_moment / (moments.mean * moments.mean)
    ratio_value = _primal_float(ratio)
    tolerance = _scale_tolerance(ratio_value)
    if ratio_value < 1.0 - tolerance:
        raise ValueError("second moment cannot be below mean squared")
    total_log_variance = (
        0.0 if ratio_value <= 1.0 + tolerance else np.log(ratio)
    )
    return LognormalMomentMatch(
        mean=moments.mean,
        second_moment=moments.second_moment,
        variance=moments.variance,
        total_log_variance=total_log_variance,
    )


__all__ = [
    "LognormalMomentMatch",
    "WeightedLognormalSumContract",
    "WeightedLognormalSumMoments",
    "match_lognormal_moments",
    "single_factor_lognormal_sum_contract",
    "weighted_lognormal_sum_moments",
]
