"""Tests for product-neutral analytical probability primitives."""

from __future__ import annotations

from math import asin, pi

import pytest

from trellis.models.analytical.support.probability import (
    bivariate_standard_normal_cdf,
    standard_normal_cdf,
)
from trellis.models.calibration import (
    ObjectiveBundle,
    SolveBounds,
    SolveRequest,
    execute_solve_request,
)


def test_standard_normal_cdf_matches_reference_values():
    assert standard_normal_cdf(0.0) == pytest.approx(0.5)
    assert standard_normal_cdf(1.0) == pytest.approx(0.8413447460685429)
    assert standard_normal_cdf(-1.0) == pytest.approx(1.0 - standard_normal_cdf(1.0))


def test_bivariate_standard_normal_cdf_matches_independent_and_zero_threshold_cases():
    x = 0.35
    y = -0.2

    assert bivariate_standard_normal_cdf(x, y, 0.0) == pytest.approx(
        standard_normal_cdf(x) * standard_normal_cdf(y)
    )
    for correlation in (-0.75, -0.25, 0.25, 0.75):
        expected = 0.25 + asin(correlation) / (2.0 * pi)
        assert bivariate_standard_normal_cdf(0.0, 0.0, correlation) == pytest.approx(
            expected,
            abs=1e-10,
        )


def test_bivariate_standard_normal_cdf_is_symmetric_and_handles_singular_boundaries():
    x = -0.4
    y = 0.7

    observed = bivariate_standard_normal_cdf(x, y, 0.35)
    assert observed == pytest.approx(
        bivariate_standard_normal_cdf(y, x, 0.35),
        abs=1e-12,
    )
    assert bivariate_standard_normal_cdf(x, y, 1.0) == pytest.approx(
        standard_normal_cdf(min(x, y))
    )
    assert bivariate_standard_normal_cdf(x, y, -1.0) == pytest.approx(
        max(standard_normal_cdf(x) - standard_normal_cdf(-y), 0.0)
    )


def test_standard_normal_cdf_composes_with_typed_bounded_scalar_root():
    target_probability = 0.75
    request = SolveRequest(
        request_id="standard_normal_quantile",
        problem_kind="root_scalar",
        parameter_names=("critical_state",),
        initial_guess=(0.0,),
        bounds=SolveBounds(lower=(-5.0,), upper=(5.0,)),
        objective=ObjectiveBundle(
            objective_kind="root_scalar",
            labels=("probability_residual",),
            target_values=(target_probability,),
            scalar_objective_fn=lambda value: standard_normal_cdf(float(value))
            - target_probability,
        ),
        solver_hint="brentq",
    )

    result = execute_solve_request(request)

    assert result.success is True
    assert result.solution == pytest.approx((0.6744897501960817,), abs=1e-10)
    assert result.objective_value <= 1e-12


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_gaussian_probability_primitives_reject_non_finite_inputs(value):
    with pytest.raises(ValueError, match="finite"):
        standard_normal_cdf(value)
    with pytest.raises(ValueError, match="finite"):
        bivariate_standard_normal_cdf(value, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        bivariate_standard_normal_cdf(0.0, value, 0.0)
    with pytest.raises(ValueError, match="finite"):
        bivariate_standard_normal_cdf(0.0, 0.0, value)


@pytest.mark.parametrize("correlation", [-1.000001, 1.000001])
def test_bivariate_standard_normal_cdf_rejects_invalid_correlation(correlation):
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        bivariate_standard_normal_cdf(0.0, 0.0, correlation)
