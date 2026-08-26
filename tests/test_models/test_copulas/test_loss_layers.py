"""Tests for reusable homogeneous loss-layer composition primitives."""

from __future__ import annotations

import numpy as raw_np
import pytest

from trellis.core.differentiable import gradient
from trellis.models.copulas.correlation import equicorrelation_matrix
from trellis.models.copulas.factor import FactorCopula
from trellis.models.copulas.student_t import StudentTCopula
from trellis.models.loss_layers import (
    bounded_layer_loss_fraction,
    homogeneous_pool_loss_fraction,
)


def test_equicorrelation_matrix_constructs_valid_symmetric_matrix():
    correlation = equicorrelation_matrix(4, 0.25)

    assert correlation.shape == (4, 4)
    assert raw_np.diag(correlation) == pytest.approx(raw_np.ones(4))
    assert correlation[raw_np.triu_indices(4, 1)] == pytest.approx(
        raw_np.full(6, 0.25)
    )
    assert correlation == pytest.approx(correlation.T)


@pytest.mark.parametrize("dimension", (True, 0, -1, 2.5))
def test_equicorrelation_matrix_rejects_invalid_dimension(dimension):
    with pytest.raises(ValueError, match="dimension must be a positive integer"):
        equicorrelation_matrix(dimension, 0.25)


@pytest.mark.parametrize("correlation", (-0.01, 1.0, float("nan"), float("inf")))
def test_equicorrelation_matrix_rejects_invalid_correlation(correlation):
    with pytest.raises(ValueError, match=r"0 <= correlation < 1"):
        equicorrelation_matrix(4, correlation)


def test_homogeneous_pool_loss_fraction_preserves_scalar_and_array_algebra():
    scalar = homogeneous_pool_loss_fraction(2, pool_size=10, recovery=0.4)
    vector = homogeneous_pool_loss_fraction(
        raw_np.array([0.0, 2.0, 10.0]),
        pool_size=10,
        recovery=0.4,
    )

    assert scalar == pytest.approx(0.12)
    assert vector == pytest.approx(raw_np.array([0.0, 0.12, 0.6]))


def test_homogeneous_pool_loss_fraction_preserves_traced_count_gradient():
    derivative = gradient(
        lambda count: homogeneous_pool_loss_fraction(
            count,
            pool_size=10,
            recovery=0.4,
        )
    )

    assert derivative(2.0) == pytest.approx(0.06)


@pytest.mark.parametrize("pool_size", (True, 0, -1, 2.5))
def test_homogeneous_pool_loss_fraction_rejects_invalid_pool_size(pool_size):
    with pytest.raises(ValueError, match="pool_size must be a positive integer"):
        homogeneous_pool_loss_fraction(1, pool_size=pool_size, recovery=0.4)


@pytest.mark.parametrize("recovery", (-0.01, 1.0, float("nan"), float("inf")))
def test_homogeneous_pool_loss_fraction_rejects_invalid_recovery(recovery):
    with pytest.raises(ValueError, match=r"0 <= recovery < 1"):
        homogeneous_pool_loss_fraction(1, pool_size=10, recovery=recovery)


@pytest.mark.parametrize(
    "default_counts",
    (
        raw_np.array([]),
        raw_np.array([-1.0, 0.0]),
        raw_np.array([0.0, 11.0]),
        raw_np.array([0.0, float("nan")]),
        raw_np.array([0.0, float("inf")]),
    ),
)
def test_homogeneous_pool_loss_fraction_rejects_invalid_counts(default_counts):
    with pytest.raises(ValueError, match="default_counts"):
        homogeneous_pool_loss_fraction(default_counts, pool_size=10, recovery=0.4)


def test_bounded_layer_loss_fraction_projects_attachment_and_detachment():
    portfolio_loss = raw_np.array([0.0, 0.02, 0.03, 0.05, 0.07, 0.09, 1.0])

    layer_loss = bounded_layer_loss_fraction(
        portfolio_loss,
        attachment=0.03,
        detachment=0.07,
    )

    assert layer_loss == pytest.approx(
        raw_np.array([0.0, 0.0, 0.0, 0.02, 0.04, 0.04, 0.04])
    )


def test_bounded_layer_loss_fraction_preserves_traced_interior_gradient():
    derivative = gradient(
        lambda loss: bounded_layer_loss_fraction(
            loss,
            attachment=0.03,
            detachment=0.07,
        )
    )

    assert derivative(0.05) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("attachment", "detachment"),
    (
        (-0.01, 0.07),
        (0.03, 0.03),
        (0.08, 0.07),
        (0.03, 1.01),
        (float("nan"), 0.07),
        (0.03, float("inf")),
    ),
)
def test_bounded_layer_loss_fraction_rejects_invalid_bounds(
    attachment,
    detachment,
):
    with pytest.raises(ValueError, match=r"0 <= attachment < detachment <= 1"):
        bounded_layer_loss_fraction(
            0.05,
            attachment=attachment,
            detachment=detachment,
        )


@pytest.mark.parametrize(
    "portfolio_loss",
    (
        raw_np.array([]),
        raw_np.array([-0.01, 0.0]),
        raw_np.array([0.0, 1.01]),
        raw_np.array([0.0, float("nan")]),
        raw_np.array([0.0, float("inf")]),
    ),
)
def test_bounded_layer_loss_fraction_rejects_invalid_portfolio_loss(portfolio_loss):
    with pytest.raises(ValueError, match="portfolio_loss_fraction"):
        bounded_layer_loss_fraction(
            portfolio_loss,
            attachment=0.03,
            detachment=0.07,
        )


def test_gaussian_distribution_and_seeded_student_t_samples_share_loss_projection():
    pool_size = 20
    recovery = 0.4
    attachment = 0.03
    detachment = 0.07

    gaussian_counts, probabilities = FactorCopula(
        n_names=pool_size,
        correlation=0.25,
    ).loss_distribution(marginal_prob=0.08)
    gaussian_layer_losses = bounded_layer_loss_fraction(
        homogeneous_pool_loss_fraction(
            gaussian_counts,
            pool_size=pool_size,
            recovery=recovery,
        ),
        attachment=attachment,
        detachment=detachment,
    )

    student_t = StudentTCopula(
        equicorrelation_matrix(pool_size, 0.25),
        df=5.0,
    )
    default_times = student_t.sample_default_times(
        raw_np.full(pool_size, 0.04),
        5_000,
        rng=raw_np.random.default_rng(42),
    )
    sampled_counts = raw_np.sum(default_times <= 5.0, axis=1)
    student_t_layer_losses = bounded_layer_loss_fraction(
        homogeneous_pool_loss_fraction(
            sampled_counts,
            pool_size=pool_size,
            recovery=recovery,
        ),
        attachment=attachment,
        detachment=detachment,
    )

    gaussian_expected_loss = float(raw_np.sum(gaussian_layer_losses * probabilities))
    student_t_expected_loss = float(raw_np.mean(student_t_layer_losses))

    assert 0.0 < gaussian_expected_loss <= detachment - attachment
    assert 0.0 < student_t_expected_loss <= detachment - attachment
