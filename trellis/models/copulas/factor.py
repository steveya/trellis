"""Factor copula models (one-factor Gaussian, multi-factor)."""

from __future__ import annotations

import numpy as raw_np
from scipy.stats import norm


class FactorCopula:
    """One-factor Gaussian copula (Li's model for CDO pricing).

    X_i = sqrt(rho) * M + sqrt(1 - rho) * Z_i
    where M is the common factor and Z_i are idiosyncratic.

    Parameters
    ----------
    n_names : int
        Number of names in the portfolio.
    correlation : float
        Pairwise correlation (single-factor: all pairs equal).
    """

    def __init__(self, n_names: int, correlation: float):
        self.n_names = n_names
        self.rho = correlation

    def conditional_default_prob(
        self,
        marginal_prob: float,
        factor_value: float,
    ) -> float:
        """P(default | M = factor_value) under one-factor Gaussian.

        P(tau < T | M) = Phi((Phi^{-1}(p) - sqrt(rho)*M) / sqrt(1 - rho))
        """
        threshold = norm.ppf(marginal_prob)
        return norm.cdf(
            (threshold - raw_np.sqrt(self.rho) * factor_value)
            / raw_np.sqrt(1 - self.rho)
        )

    def loss_distribution(
        self,
        marginal_prob: float,
        n_factor_points: int = 50,
    ) -> tuple[raw_np.ndarray, raw_np.ndarray]:
        """Compute portfolio loss distribution via numerical integration.

        Returns (losses, probabilities) where losses = 0, 1, ..., n_names.
        """
        from scipy.stats import binom

        # Gauss-Hermite quadrature over the factor
        points, weights = raw_np.polynomial.hermite.hermgauss(n_factor_points)
        # Transform: M = sqrt(2) * point, weight *= 1/sqrt(pi)
        M_values = raw_np.sqrt(2) * points
        adj_weights = weights / raw_np.sqrt(raw_np.pi)

        losses = raw_np.arange(self.n_names + 1)
        loss_probs = raw_np.zeros(self.n_names + 1)

        for m_val, w in zip(M_values, adj_weights):
            p_cond = self.conditional_default_prob(marginal_prob, m_val)
            p_cond = raw_np.clip(p_cond, 0, 1)
            # Binomial distribution for homogeneous portfolio
            binom_probs = binom.pmf(losses, self.n_names, p_cond)
            loss_probs += w * binom_probs

        return losses, loss_probs

    def sample_defaults(
        self, marginal_prob: float, n_paths: int, rng=None,
    ) -> raw_np.ndarray:
        """Monte Carlo simulation of defaults.

        Returns (n_paths, n_names) boolean array of defaults.
        """
        if rng is None:
            rng = raw_np.random.default_rng()

        M = rng.standard_normal(n_paths)
        Z = rng.standard_normal((n_paths, self.n_names))

        X = (raw_np.sqrt(self.rho) * M[:, raw_np.newaxis]
             + raw_np.sqrt(1 - self.rho) * Z)

        threshold = norm.ppf(marginal_prob)
        return X < threshold
