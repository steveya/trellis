"""Student-t copula for fatter-tailed default correlation."""

from __future__ import annotations

import numpy as raw_np
from scipy.stats import t as t_dist


class StudentTCopula:
    """Student-t copula with specified degrees of freedom.

    Parameters
    ----------
    correlation_matrix : ndarray of shape (n, n)
    df : float
        Degrees of freedom (lower = fatter tails).
    """

    def __init__(self, correlation_matrix: raw_np.ndarray, df: float = 5.0):
        self._corr = raw_np.asarray(correlation_matrix, dtype=float)
        self._chol = raw_np.linalg.cholesky(self._corr)
        self.n = self._corr.shape[0]
        self.df = df

    def sample_uniforms(self, n_paths: int, rng=None) -> raw_np.ndarray:
        """Generate correlated uniform samples via t-copula."""
        if rng is None:
            rng = raw_np.random.default_rng()

        Z = rng.standard_normal((n_paths, self.n))
        correlated = Z @ self._chol.T

        # Scale by chi-squared to get t-distributed
        chi2 = rng.chisquare(self.df, size=(n_paths, 1))
        T = correlated * raw_np.sqrt(self.df / chi2)

        return t_dist.cdf(T, self.df)

    def sample_default_times(
        self, hazard_rates: raw_np.ndarray, n_paths: int, rng=None,
    ) -> raw_np.ndarray:
        U = self.sample_uniforms(n_paths, rng)
        return -raw_np.log(U) / hazard_rates[raw_np.newaxis, :]
