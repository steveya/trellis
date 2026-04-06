"""Tests for shared early-exercise Monte Carlo contracts."""

from __future__ import annotations

import numpy as raw_np
import pytest


def _american_put_paths(*, seed: int = 123, n_steps: int = 24, n_paths: int = 4000):
    from trellis.models.monte_carlo.discretization import exact_simulation
    from trellis.models.processes.gbm import GBM

    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    paths = exact_simulation(
        GBM(mu=r, sigma=sigma),
        S0,
        T,
        n_steps,
        n_paths,
        rng=raw_np.random.default_rng(seed),
    )

    def put_payoff(S):
        return raw_np.maximum(K - S, 0.0)

    return paths, list(range(1, n_steps + 1)), put_payoff, r, T / n_steps


def test_longstaff_schwartz_result_returns_shared_policy_result():
    from trellis.models.monte_carlo.early_exercise import EarlyExercisePolicyResult
    from trellis.models.monte_carlo.lsm import longstaff_schwartz_result

    paths, exercise_dates, put_payoff, r, dt = _american_put_paths()

    result = longstaff_schwartz_result(
        paths,
        exercise_dates,
        put_payoff,
        r,
        dt,
    )

    assert isinstance(result, EarlyExercisePolicyResult)
    assert result.policy_class == "longstaff_schwartz"
    assert result.price_upper is None
    assert result.price_lower > 0.0
    assert result.diagnostics is not None
    assert result.diagnostics.policy_class == "longstaff_schwartz"
    assert result.diagnostics.exercise_dates_count == len(exercise_dates)


@pytest.mark.legacy_compat
def test_longstaff_schwartz_legacy_function_matches_policy_result():
    from trellis.models.monte_carlo.lsm import longstaff_schwartz, longstaff_schwartz_result

    paths = raw_np.array(
        [
            [1.0, 0.9, 0.8],
            [1.0, 1.1, 1.2],
            [1.0, 0.95, 0.7],
            [1.0, 1.05, 0.9],
        ]
    )

    def payoff_fn(S):
        return raw_np.maximum(1.0 - S, 0.0)

    result = longstaff_schwartz_result(paths, [1, 2], payoff_fn, 0.01, 0.5)
    legacy = longstaff_schwartz(paths, [1, 2], payoff_fn, 0.01, 0.5)

    assert legacy == result.price_lower


def test_least_squares_continuation_estimator_is_vectorized():
    from trellis.models.monte_carlo.early_exercise import LeastSquaresContinuationEstimator

    estimator = LeastSquaresContinuationEstimator()
    S = raw_np.array([80.0, 90.0, 100.0, 110.0])
    discounted_cf = raw_np.array([20.0, 10.0, 5.0, 0.0])

    continuation, failed = estimator.fit_predict(S, discounted_cf)

    assert continuation.shape == S.shape
    assert failed is False


def test_default_continuation_estimator_prefers_fast_polynomial_basis():
    from trellis.models.monte_carlo.early_exercise import (
        FastPolynomialContinuationEstimator,
        default_continuation_estimator,
    )

    estimator = default_continuation_estimator()
    assert isinstance(estimator, FastPolynomialContinuationEstimator)


def test_fast_polynomial_estimator_matches_generic_least_squares():
    from trellis.models.monte_carlo.early_exercise import (
        FastPolynomialContinuationEstimator,
        LeastSquaresContinuationEstimator,
    )

    S = raw_np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0])
    discounted_cf = raw_np.array([21.0, 14.0, 10.0, 7.0, 4.0, 2.0])

    fast, fast_failed = FastPolynomialContinuationEstimator().fit_predict(S, discounted_cf)
    generic, generic_failed = LeastSquaresContinuationEstimator().fit_predict(S, discounted_cf)

    assert fast_failed is False
    assert generic_failed is False
    raw_np.testing.assert_allclose(fast, generic, rtol=1e-10, atol=1e-10)


def test_fast_laguerre_estimator_matches_generic_least_squares():
    from trellis.models.monte_carlo.early_exercise import (
        FastLaguerreContinuationEstimator,
        LeastSquaresContinuationEstimator,
    )
    from trellis.models.monte_carlo.lsm import laguerre_basis

    S = raw_np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    discounted_cf = raw_np.array([0.25, 0.18, 0.12, 0.08, 0.05, 0.03])

    fast, fast_failed = FastLaguerreContinuationEstimator().fit_predict(S, discounted_cf)
    generic, generic_failed = LeastSquaresContinuationEstimator(
        basis_fn=laguerre_basis,
    ).fit_predict(S, discounted_cf)

    assert fast_failed is False
    assert generic_failed is False
    raw_np.testing.assert_allclose(fast, generic, rtol=1e-10, atol=1e-10)


def test_longstaff_schwartz_fast_estimator_matches_generic_estimator():
    from trellis.models.monte_carlo.early_exercise import LeastSquaresContinuationEstimator
    from trellis.models.monte_carlo.lsm import longstaff_schwartz_result

    paths, exercise_dates, put_payoff, r, dt = _american_put_paths(n_paths=2500)

    fast = longstaff_schwartz_result(
        paths, exercise_dates, put_payoff, r, dt,
    )
    generic = longstaff_schwartz_result(
        paths,
        exercise_dates,
        put_payoff,
        r,
        dt,
        continuation_estimator=LeastSquaresContinuationEstimator(),
    )

    assert fast.diagnostics is not None
    assert generic.diagnostics is not None
    assert fast.diagnostics.estimator_name == "least_squares_regression_polynomial_fast"
    assert generic.diagnostics.estimator_name == "least_squares_regression"
    assert fast.price_lower == pytest.approx(generic.price_lower, rel=1e-10, abs=1e-10)


def test_early_exercise_policies_treat_maturity_as_an_implicit_exercise_date():
    from trellis.models.monte_carlo.lsm import longstaff_schwartz_result
    from trellis.models.monte_carlo.primal_dual import primal_dual_mc_result
    from trellis.models.monte_carlo.stochastic_mesh import stochastic_mesh_result
    from trellis.models.monte_carlo.tv_regression import tsitsiklis_van_roy_result

    paths = raw_np.array(
        [
            [1.0, 0.9, 0.8],
            [1.0, 1.0, 1.2],
            [1.0, 1.1, 1.3],
        ]
    )

    def call_payoff(S):
        return raw_np.maximum(S - 1.0, 0.0)

    for result_fn in (
        longstaff_schwartz_result,
        primal_dual_mc_result,
        stochastic_mesh_result,
        tsitsiklis_van_roy_result,
    ):
        implicit = result_fn(paths, [1], call_payoff, 0.0, 0.5)
        explicit = result_fn(paths, [1, 2], call_payoff, 0.0, 0.5)

        assert implicit.price_lower == pytest.approx(explicit.price_lower, rel=1e-12, abs=1e-12)
        assert implicit.diagnostics is not None
        assert explicit.diagnostics is not None
        assert implicit.diagnostics.exercise_dates_count == explicit.diagnostics.exercise_dates_count
        if implicit.price_upper is not None or explicit.price_upper is not None:
            assert implicit.price_upper == pytest.approx(explicit.price_upper, rel=1e-12, abs=1e-12)


def test_primal_dual_mc_result_returns_ordered_bounds():
    from trellis.models.monte_carlo.early_exercise import EarlyExercisePolicyResult
    from trellis.models.monte_carlo.primal_dual import primal_dual_mc, primal_dual_mc_result

    paths, exercise_dates, put_payoff, r, dt = _american_put_paths()

    result = primal_dual_mc_result(
        paths,
        exercise_dates,
        put_payoff,
        r,
        dt,
    )
    legacy = primal_dual_mc(
        paths,
        exercise_dates,
        put_payoff,
        r,
        dt,
    )

    assert isinstance(result, EarlyExercisePolicyResult)
    assert result.policy_class == "primal_dual_mc"
    assert result.price_upper is not None
    assert result.price_lower > 0.0
    assert result.price_upper >= result.price_lower
    assert legacy == result.price_lower
    assert result.diagnostics is not None
    assert result.diagnostics.policy_class == "primal_dual_mc"


def test_tsitsiklis_van_roy_result_returns_shared_policy_result():
    from trellis.models.monte_carlo.early_exercise import EarlyExercisePolicyResult
    from trellis.models.monte_carlo.tv_regression import (
        tsitsiklis_van_roy,
        tsitsiklis_van_roy_result,
    )

    paths, exercise_dates, put_payoff, r, dt = _american_put_paths()

    result = tsitsiklis_van_roy_result(
        paths,
        exercise_dates,
        put_payoff,
        r,
        dt,
    )
    legacy = tsitsiklis_van_roy(
        paths,
        exercise_dates,
        put_payoff,
        r,
        dt,
    )

    assert isinstance(result, EarlyExercisePolicyResult)
    assert result.policy_class == "tsitsiklis_van_roy"
    assert result.price_upper is None
    assert result.price_lower > 0.0
    assert legacy == result.price_lower
    assert result.diagnostics is not None
    assert result.diagnostics.policy_class == "tsitsiklis_van_roy"


def test_tsitsiklis_van_roy_is_reasonably_close_to_longstaff_schwartz():
    from trellis.models.monte_carlo.lsm import longstaff_schwartz_result
    from trellis.models.monte_carlo.tv_regression import tsitsiklis_van_roy_result

    paths, exercise_dates, put_payoff, r, dt = _american_put_paths()

    lsm = longstaff_schwartz_result(paths, exercise_dates, put_payoff, r, dt)
    tvr = tsitsiklis_van_roy_result(paths, exercise_dates, put_payoff, r, dt)

    assert tvr.price_lower > 0.0
    assert abs(tvr.price_lower - lsm.price_lower) / lsm.price_lower < 0.35


def test_stochastic_mesh_result_returns_shared_policy_result():
    from trellis.models.monte_carlo.early_exercise import EarlyExercisePolicyResult
    from trellis.models.monte_carlo.stochastic_mesh import (
        stochastic_mesh,
        stochastic_mesh_result,
    )

    paths, exercise_dates, put_payoff, r, dt = _american_put_paths(n_paths=1500)

    result = stochastic_mesh_result(
        paths,
        exercise_dates,
        put_payoff,
        r,
        dt,
    )
    legacy = stochastic_mesh(
        paths,
        exercise_dates,
        put_payoff,
        r,
        dt,
    )

    assert isinstance(result, EarlyExercisePolicyResult)
    assert result.policy_class == "stochastic_mesh"
    assert result.price_upper is None
    assert result.price_lower > 0.0
    assert legacy == result.price_lower
    assert result.diagnostics is not None
    assert result.diagnostics.policy_class == "stochastic_mesh"
