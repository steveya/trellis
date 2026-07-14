"""Moments and bounded lognormal matching for weighted lognormal sums."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt

import numpy as raw_np

from trellis.core.differentiable import get_numpy


_VALIDATION_TOLERANCE = 1e-12
np = get_numpy()


def _float_tuple(values) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _float_matrix(values) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


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
        weights = _float_tuple(self.weights)
        levels = _float_tuple(self.initial_levels)
        carries = _float_tuple(self.carries)
        covariance = _float_matrix(self.log_covariance)
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
        if any(not isfinite(value) for value in (*times, *weights, *levels, *carries)):
            raise ValueError("weighted lognormal sum inputs must contain finite values")
        if any(time < 0.0 for time in times):
            raise ValueError("observation_times must be non-negative")
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("observation_times must be non-decreasing")
        if any(level <= 0.0 for level in levels):
            raise ValueError("initial_levels must be strictly positive")
        if all(weight == 0.0 for weight in weights):
            raise ValueError("weights must contain at least one non-zero value")
        if len(covariance) != dimension or any(
            len(row) != dimension for row in covariance
        ):
            raise ValueError(
                "log_covariance must be a square matrix matching the observation count"
            )

        covariance_array = raw_np.asarray(covariance, dtype=float)
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
        minimum_eigenvalue = float(raw_np.min(raw_np.linalg.eigvalsh(covariance_array)))
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
        mean = float(self.mean)
        second_moment = float(self.second_moment)
        variance = float(self.variance)
        if not all(isfinite(value) for value in (mean, second_moment, variance)):
            raise ValueError("weighted lognormal moments must be finite")

        expected_variance = second_moment - mean * mean
        tolerance = _scale_tolerance(mean * mean, second_moment, variance)
        if expected_variance < -tolerance:
            raise ValueError("second moment cannot be below mean squared")
        if abs(variance - expected_variance) > tolerance:
            raise ValueError("variance must equal second_moment minus mean squared")
        normalized_variance = 0.0 if abs(expected_variance) <= tolerance else expected_variance

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "second_moment", second_moment)
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
        mean = float(self.mean)
        second_moment = float(self.second_moment)
        variance = float(self.variance)
        total_log_variance = float(self.total_log_variance)
        if not all(
            isfinite(value)
            for value in (mean, second_moment, variance, total_log_variance)
        ):
            raise ValueError("matched lognormal moments must be finite")
        if mean <= 0.0:
            raise ValueError("matched lognormal mean must be strictly positive")

        expected_variance = second_moment - mean * mean
        tolerance = _scale_tolerance(
            mean * mean,
            second_moment,
            variance,
            total_log_variance,
        )
        if expected_variance < -tolerance:
            raise ValueError("second moment cannot be below mean squared")
        if abs(variance - expected_variance) > tolerance:
            raise ValueError("variance must equal second_moment minus mean squared")
        normalized_variance = 0.0 if abs(expected_variance) <= tolerance else expected_variance
        ratio = second_moment / (mean * mean)
        expected_log_variance = 0.0 if ratio <= 1.0 + tolerance else log(ratio)
        if total_log_variance < -tolerance:
            raise ValueError("total_log_variance must be non-negative")
        if abs(total_log_variance - expected_log_variance) > tolerance:
            raise ValueError("total_log_variance is inconsistent with the declared moments")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "second_moment", second_moment)
        object.__setattr__(self, "variance", normalized_variance)
        object.__setattr__(self, "total_log_variance", expected_log_variance)

    def effective_volatility(self, *, maturity: float) -> float:
        """Return the Black-style volatility over a positive maturity."""
        horizon = float(maturity)
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("maturity must be finite and positive")
        return sqrt(self.total_log_variance / horizon)


def single_factor_lognormal_sum_contract(
    *,
    spot: float,
    observation_times: tuple[float, ...],
    weights: tuple[float, ...],
    carry: float,
    volatility: float,
) -> WeightedLognormalSumContract:
    """Build the exact log-covariance contract for one constant-parameter GBM."""
    initial_level = float(spot)
    drift = float(carry)
    sigma = float(volatility)
    times = _float_tuple(observation_times)
    if not all(isfinite(value) for value in (initial_level, drift, sigma)):
        raise ValueError("spot, carry, and volatility must be finite")
    if initial_level <= 0.0:
        raise ValueError("spot must be strictly positive")
    if sigma < 0.0:
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
    times = np.asarray(contract.observation_times)
    weights = np.asarray(contract.weights)
    levels = np.asarray(contract.initial_levels)
    carries = np.asarray(contract.carries)
    covariance = np.asarray(contract.log_covariance)

    marginal_means = levels * np.exp(carries * times)
    weighted_means = weights * marginal_means
    mean = float(np.sum(weighted_means))
    second_moment = float(
        np.sum(np.outer(weighted_means, weighted_means) * np.exp(covariance))
    )
    variance = second_moment - mean * mean
    tolerance = _scale_tolerance(mean * mean, second_moment)
    if variance < 0.0 and variance >= -tolerance:
        variance = 0.0
        second_moment = mean * mean
    return WeightedLognormalSumMoments(
        mean=mean,
        second_moment=second_moment,
        variance=variance,
        nonnegative_support=all(weight >= 0.0 for weight in contract.weights),
    )


def match_lognormal_moments(
    moments: WeightedLognormalSumMoments,
) -> LognormalMomentMatch:
    """Fit a lognormal distribution to valid moments of a positive sum."""
    if not moments.nonnegative_support:
        raise ValueError("lognormal matching requires non-negative weights")
    if moments.mean <= 0.0:
        raise ValueError("lognormal matching requires a strictly positive mean")

    ratio = moments.second_moment / (moments.mean * moments.mean)
    tolerance = _scale_tolerance(ratio)
    if ratio < 1.0 - tolerance:
        raise ValueError("second moment cannot be below mean squared")
    total_log_variance = 0.0 if ratio <= 1.0 + tolerance else log(ratio)
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
