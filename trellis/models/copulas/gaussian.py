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

    def __init__(
        self,
        correlation_matrix: raw_np.ndarray | None = None,
        *,
        correlation: float | None = None,
        n_names: int | None = None,
    ):
        """Store the correlation matrix and its Cholesky factor."""
        if correlation_matrix is None:
            if correlation is None:
                raise TypeError("GaussianCopula requires correlation_matrix or correlation")
            if n_names is None:
                self._scalar_correlation = _validate_scalar_correlation(correlation)
                self._corr = None
                self._chol = None
                self.n = None
                return
            correlation_matrix = _equicorrelation_matrix(n_names, correlation)
        self._scalar_correlation = None
        self._corr = raw_np.asarray(correlation_matrix, dtype=float)
        self._chol = raw_np.linalg.cholesky(self._corr)
        self.n = self._corr.shape[0]

    def sample_uniforms(self, n_paths: int, rng=None) -> raw_np.ndarray:
        """Generate correlated uniform samples.

        Returns (n_paths, n) array of correlated U[0,1] variables.
        """
        if rng is None:
            rng = raw_np.random.default_rng()
        if self._chol is None or self.n is None:
            raise ValueError(
                "GaussianCopula.sample_uniforms requires n_names when initialized "
                "with scalar correlation"
            )
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
        hazard_rates = raw_np.asarray(hazard_rates, dtype=float)
        if self._chol is None:
            n_names = int(hazard_rates.shape[0])
            corr = _equicorrelation_matrix(n_names, self._scalar_correlation)
            self._corr = corr
            self._chol = raw_np.linalg.cholesky(corr)
            self.n = n_names
        U = self.sample_uniforms(n_paths, rng)
        # tau = -ln(U) / lambda (exponential default times)
        return -raw_np.log(U) / hazard_rates[raw_np.newaxis, :]


def _validate_scalar_correlation(correlation: float | None) -> float:
    value = float(correlation if correlation is not None else 0.0)
    if not raw_np.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("correlation must satisfy 0 <= correlation < 1")
    return value


def _equicorrelation_matrix(n_names: int, correlation: float | None) -> raw_np.ndarray:
    count = int(n_names)
    if count <= 0:
        raise ValueError("n_names must be positive")
    value = _validate_scalar_correlation(correlation)
    corr = raw_np.full((count, count), value, dtype=float)
    raw_np.fill_diagonal(corr, 1.0)
    return corr
