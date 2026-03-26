"""Gaussian copula for correlated default simulation."""

from __future__ import annotations

import numpy as raw_np
from scipy.stats import norm


class GaussianCopula:
    """Gaussian copula for generating correlated default times.

    Parameters
    ----------
    correlation_matrix : ndarray of shape (n, n)
        Positive-definite correlation matrix.
    """

    def __init__(self, correlation_matrix: raw_np.ndarray):
        """Store the correlation matrix and its Cholesky factor."""
        self._corr = raw_np.asarray(correlation_matrix, dtype=float)
        self._chol = raw_np.linalg.cholesky(self._corr)
        self.n = self._corr.shape[0]

    def sample_uniforms(self, n_paths: int, rng=None) -> raw_np.ndarray:
        """Generate correlated uniform samples.

        Returns (n_paths, n) array of correlated U[0,1] variables.
        """
        if rng is None:
            rng = raw_np.random.default_rng()
        Z = rng.standard_normal((n_paths, self.n))
        correlated = Z @ self._chol.T
        return norm.cdf(correlated)

    def sample_default_times(
        self,
        hazard_rates: raw_np.ndarray,
        n_paths: int,
        rng=None,
    ) -> raw_np.ndarray:
        """Generate correlated default times.

        Parameters
        ----------
        hazard_rates : ndarray of shape (n,)
            Constant hazard rate for each name.
        n_paths : int

        Returns
        -------
        ndarray of shape (n_paths, n)
            Default times for each name on each path.
        """
        U = self.sample_uniforms(n_paths, rng)
        # tau = -ln(U) / lambda (exponential default times)
        return -raw_np.log(U) / hazard_rates[raw_np.newaxis, :]
