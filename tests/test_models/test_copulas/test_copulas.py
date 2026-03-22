"""Tests for copulas: Gaussian, Student-t, Factor."""

import numpy as raw_np
import pytest

from trellis.models.copulas.gaussian import GaussianCopula
from trellis.models.copulas.student_t import StudentTCopula
from trellis.models.copulas.factor import FactorCopula


# ---------------------------------------------------------------------------
# GaussianCopula
# ---------------------------------------------------------------------------


class TestGaussianCopula:
    def test_sample_uniforms_in_unit_interval(self):
        corr = raw_np.array([[1.0, 0.5], [0.5, 1.0]])
        gc = GaussianCopula(corr)
        U = gc.sample_uniforms(5000, rng=raw_np.random.default_rng(42))
        assert U.shape == (5000, 2)
        assert raw_np.all(U >= 0.0)
        assert raw_np.all(U <= 1.0)

    def test_identity_correlation_approx_independent(self):
        """With identity correlation, the copula produces nearly independent uniforms."""
        n = 3
        corr = raw_np.eye(n)
        gc = GaussianCopula(corr)
        U = gc.sample_uniforms(50000, rng=raw_np.random.default_rng(7))
        # Empirical correlation between columns should be near zero
        for i in range(n):
            for j in range(i + 1, n):
                emp_corr = raw_np.corrcoef(U[:, i], U[:, j])[0, 1]
                assert abs(emp_corr) < 0.03

    def test_marginals_uniform(self):
        """Each marginal should be approximately U(0,1)."""
        corr = raw_np.array([[1.0, 0.8], [0.8, 1.0]])
        gc = GaussianCopula(corr)
        U = gc.sample_uniforms(50000, rng=raw_np.random.default_rng(1))
        for j in range(2):
            assert raw_np.mean(U[:, j]) == pytest.approx(0.5, abs=0.02)
            assert raw_np.std(U[:, j]) == pytest.approx(1.0 / raw_np.sqrt(12), abs=0.02)


# ---------------------------------------------------------------------------
# StudentTCopula
# ---------------------------------------------------------------------------


class TestStudentTCopula:
    def test_fatter_tails_than_gaussian(self):
        """Student-t copula should produce more joint extremes than Gaussian."""
        corr = raw_np.array([[1.0, 0.5], [0.5, 1.0]])
        n_paths = 100000
        rng_g = raw_np.random.default_rng(42)
        rng_t = raw_np.random.default_rng(42)

        gc = GaussianCopula(corr)
        tc = StudentTCopula(corr, df=3.0)

        U_gauss = gc.sample_uniforms(n_paths, rng=rng_g)
        U_t = tc.sample_uniforms(n_paths, rng=rng_t)

        # Count joint extreme events: both variables below 5th percentile
        threshold = 0.05
        joint_gauss = raw_np.mean((U_gauss[:, 0] < threshold) & (U_gauss[:, 1] < threshold))
        joint_t = raw_np.mean((U_t[:, 0] < threshold) & (U_t[:, 1] < threshold))

        # t-copula should have more joint extremes
        assert joint_t > joint_gauss

    def test_sample_uniforms_in_unit_interval(self):
        corr = raw_np.array([[1.0, 0.3], [0.3, 1.0]])
        tc = StudentTCopula(corr, df=5.0)
        U = tc.sample_uniforms(5000, rng=raw_np.random.default_rng(10))
        assert U.shape == (5000, 2)
        assert raw_np.all(U >= 0.0)
        assert raw_np.all(U <= 1.0)


# ---------------------------------------------------------------------------
# FactorCopula
# ---------------------------------------------------------------------------


class TestFactorCopula:
    def test_loss_distribution_probabilities_sum_to_one(self):
        fc = FactorCopula(n_names=50, correlation=0.3)
        losses, probs = fc.loss_distribution(marginal_prob=0.02, n_factor_points=50)
        assert raw_np.sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert len(losses) == 51  # 0 to 50

    def test_conditional_default_prob_between_zero_and_one(self):
        fc = FactorCopula(n_names=100, correlation=0.3)
        for m in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            p = fc.conditional_default_prob(0.05, m)
            assert 0.0 <= p <= 1.0

    def test_higher_correlation_heavier_tails(self):
        """Higher correlation should produce a fatter-tailed loss distribution."""
        marginal_prob = 0.05
        n_names = 100

        fc_low = FactorCopula(n_names, correlation=0.05)
        fc_high = FactorCopula(n_names, correlation=0.50)

        _, probs_low = fc_low.loss_distribution(marginal_prob, n_factor_points=50)
        _, probs_high = fc_high.loss_distribution(marginal_prob, n_factor_points=50)

        # Variance of loss count should be higher with more correlation
        losses = raw_np.arange(n_names + 1)
        mean_low = raw_np.sum(losses * probs_low)
        mean_high = raw_np.sum(losses * probs_high)
        var_low = raw_np.sum((losses - mean_low)**2 * probs_low)
        var_high = raw_np.sum((losses - mean_high)**2 * probs_high)

        assert var_high > var_low

    def test_conditional_prob_increases_with_negative_factor(self):
        """Negative factor realization should increase default probability."""
        fc = FactorCopula(n_names=50, correlation=0.3)
        p_neg = fc.conditional_default_prob(0.05, -2.0)
        p_pos = fc.conditional_default_prob(0.05, 2.0)
        assert p_neg > p_pos
