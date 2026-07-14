from __future__ import annotations

import math

import numpy as raw_np
import pytest


def test_single_factor_lognormal_sum_moments_match_direct_gbm_formula():
    from trellis.models.analytical.support.lognormal_moments import (
        match_lognormal_moments,
        single_factor_lognormal_sum_contract,
        weighted_lognormal_sum_moments,
    )

    spot = 100.0
    carry = 0.03
    volatility = 0.20
    times = (0.0, 0.5, 1.0)
    weights = (0.2, 0.3, 0.5)
    contract = single_factor_lognormal_sum_contract(
        spot=spot,
        observation_times=times,
        weights=weights,
        carry=carry,
        volatility=volatility,
    )

    moments = weighted_lognormal_sum_moments(contract)
    expected_mean = sum(
        weight * spot * math.exp(carry * time)
        for weight, time in zip(weights, times, strict=True)
    )
    expected_second = sum(
        left_weight
        * right_weight
        * spot
        * spot
        * math.exp(
            carry * (left_time + right_time)
            + volatility * volatility * min(left_time, right_time)
        )
        for left_weight, left_time in zip(weights, times, strict=True)
        for right_weight, right_time in zip(weights, times, strict=True)
    )

    assert moments.mean == pytest.approx(expected_mean)
    assert moments.second_moment == pytest.approx(expected_second)
    assert moments.variance == pytest.approx(expected_second - expected_mean**2)

    matched = match_lognormal_moments(moments)
    expected_log_variance = math.log(expected_second / expected_mean**2)
    assert matched.mean == pytest.approx(expected_mean)
    assert matched.total_log_variance == pytest.approx(expected_log_variance)
    assert matched.effective_volatility(maturity=1.0) == pytest.approx(
        math.sqrt(expected_log_variance)
    )


def test_general_lognormal_sum_supports_signed_weights_and_scales_moments():
    from trellis.models.analytical.support.lognormal_moments import (
        WeightedLognormalSumContract,
        match_lognormal_moments,
        weighted_lognormal_sum_moments,
    )

    contract = WeightedLognormalSumContract(
        observation_times=(0.5, 1.0),
        weights=(1.0, -0.25),
        initial_levels=(100.0, 80.0),
        carries=(0.02, 0.01),
        log_covariance=((0.02, 0.01), (0.01, 0.09)),
    )
    moments = weighted_lognormal_sum_moments(contract)
    means = raw_np.array(
        [100.0 * math.exp(0.02 * 0.5), 80.0 * math.exp(0.01)],
        dtype=float,
    )
    weights = raw_np.array([1.0, -0.25], dtype=float)
    covariance = raw_np.array(contract.log_covariance, dtype=float)
    expected_second = float(
        raw_np.sum(
            raw_np.outer(weights * means, weights * means) * raw_np.exp(covariance)
        )
    )

    assert moments.mean == pytest.approx(float(weights @ means))
    assert moments.second_moment == pytest.approx(expected_second)

    scaled = weighted_lognormal_sum_moments(
        WeightedLognormalSumContract(
            observation_times=contract.observation_times,
            weights=tuple(3.0 * weight for weight in contract.weights),
            initial_levels=contract.initial_levels,
            carries=contract.carries,
            log_covariance=contract.log_covariance,
        )
    )
    assert scaled.mean == pytest.approx(3.0 * moments.mean)
    assert scaled.second_moment == pytest.approx(9.0 * moments.second_moment)
    assert scaled.variance == pytest.approx(9.0 * moments.variance)
    with pytest.raises(ValueError, match="non-negative weights"):
        match_lognormal_moments(moments)


def test_zero_volatility_match_is_deterministic_and_black_ready():
    from trellis.models.analytical.support.lognormal_moments import (
        match_lognormal_moments,
        single_factor_lognormal_sum_contract,
        weighted_lognormal_sum_moments,
    )
    from trellis.models.black import black76_call

    moments = weighted_lognormal_sum_moments(
        single_factor_lognormal_sum_contract(
            spot=100.0,
            observation_times=(0.0, 0.5, 1.0),
            weights=(1.0 / 3.0,) * 3,
            carry=0.0,
            volatility=0.0,
        )
    )
    matched = match_lognormal_moments(moments)

    assert moments.mean == pytest.approx(100.0)
    assert moments.second_moment == pytest.approx(10_000.0)
    assert moments.variance == pytest.approx(0.0, abs=1e-12)
    assert matched.total_log_variance == pytest.approx(0.0)
    assert matched.effective_volatility(maturity=1.0) == pytest.approx(0.0)
    assert black76_call(
        matched.mean,
        95.0,
        matched.effective_volatility(maturity=1.0),
        1.0,
    ) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "observation_times": (),
                "weights": (),
                "initial_levels": (),
                "carries": (),
                "log_covariance": (),
            },
            "at least one",
        ),
        (
            {
                "observation_times": (0.5, 1.0),
                "weights": (1.0,),
                "initial_levels": (100.0, 100.0),
                "carries": (0.0, 0.0),
                "log_covariance": ((0.01, 0.01), (0.01, 0.02)),
            },
            "same length",
        ),
        (
            {
                "observation_times": (1.0, 0.5),
                "weights": (0.5, 0.5),
                "initial_levels": (100.0, 100.0),
                "carries": (0.0, 0.0),
                "log_covariance": ((0.01, 0.01), (0.01, 0.02)),
            },
            "non-decreasing",
        ),
        (
            {
                "observation_times": (0.5, 1.0),
                "weights": (0.5, 0.5),
                "initial_levels": (100.0, 0.0),
                "carries": (0.0, 0.0),
                "log_covariance": ((0.01, 0.01), (0.01, 0.02)),
            },
            "strictly positive",
        ),
        (
            {
                "observation_times": (0.5, 1.0),
                "weights": (0.5, 0.5),
                "initial_levels": (100.0, 100.0),
                "carries": (0.0, 0.0),
                "log_covariance": ((0.01, 0.00), (0.01, 0.02)),
            },
            "symmetric",
        ),
        (
            {
                "observation_times": (0.5, 1.0),
                "weights": (0.5, 0.5),
                "initial_levels": (100.0, 100.0),
                "carries": (0.0, 0.0),
                "log_covariance": ((0.01, 0.02), (0.02, 0.01)),
            },
            "positive semidefinite",
        ),
    ],
)
def test_lognormal_sum_contract_fails_closed_on_invalid_inputs(kwargs, message):
    from trellis.models.analytical.support.lognormal_moments import (
        WeightedLognormalSumContract,
    )

    with pytest.raises(ValueError, match=message):
        WeightedLognormalSumContract(**kwargs)


def test_lognormal_match_rejects_nonpositive_mean_and_material_negative_variance():
    from trellis.models.analytical.support.lognormal_moments import (
        WeightedLognormalSumMoments,
        match_lognormal_moments,
    )

    with pytest.raises(ValueError, match="strictly positive mean"):
        match_lognormal_moments(
            WeightedLognormalSumMoments(
                mean=0.0,
                second_moment=1.0,
                variance=1.0,
            )
        )
    with pytest.raises(ValueError, match="second moment cannot be below mean squared"):
        WeightedLognormalSumMoments(
            mean=2.0,
            second_moment=3.0,
            variance=-1.0,
        )


def test_lognormal_match_result_validates_fields_and_maturity():
    from trellis.models.analytical.support.lognormal_moments import (
        LognormalMomentMatch,
    )

    with pytest.raises(ValueError, match="inconsistent"):
        LognormalMomentMatch(
            mean=100.0,
            second_moment=10_100.0,
            variance=100.0,
            total_log_variance=0.5,
        )

    matched = LognormalMomentMatch(
        mean=100.0,
        second_moment=10_100.0,
        variance=100.0,
        total_log_variance=math.log(1.01),
    )
    with pytest.raises(ValueError, match="maturity must be finite and positive"):
        matched.effective_volatility(maturity=0.0)
