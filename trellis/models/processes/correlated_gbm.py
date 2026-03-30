"""Correlated multi-asset geometric Brownian motion."""

from __future__ import annotations

import numpy as raw_np

from trellis.models.processes.base import StochasticProcess


class CorrelatedGBM(StochasticProcess):
    """Correlated geometric Brownian motion under independent factor shocks.

    Parameters
    ----------
    mu : array-like
        Per-asset drift inputs, typically the risk-free rate under
        risk-neutral pricing.
    sigma : array-like
        Per-asset volatilities.
    corr : array-like
        Correlation matrix for the Brownian drivers.
    dividend_yield : array-like or None
        Optional per-asset dividend yields. Effective drift is ``mu - q``.
    """

    def __init__(
        self,
        mu=None,
        sigma=None,
        corr=None,
        dividend_yield=None,
        *,
        mu1=None,
        sigma1=None,
        mu2=None,
        sigma2=None,
        rho=None,
        spot_prices=None,
        spots=None,
        vols=None,
        corr_matrix=None,
        correlation=None,
        rates=None,
        dividends=None,
        div_yields=None,
    ):
        """Initialize the process with canonical or generated alias keywords.

        The canonical signature is ``mu``, ``sigma``, ``corr``, and
        ``dividend_yield``. Generated route code has historically used several
        alias names for the same concepts, so we normalize them here instead of
        forcing every adapter to special-case the constructor call.
        """
        shorthand_args = (mu1, sigma1, mu2, sigma2, rho)
        if mu is None and sigma is None and corr is None and any(arg is not None for arg in shorthand_args):
            if not all(arg is not None for arg in shorthand_args):
                raise ValueError(
                    "CorrelatedGBM two-asset shorthand requires mu1, sigma1, mu2, sigma2, and rho together",
                )
            mu = [mu1, mu2]
            sigma = [sigma1, sigma2]
            corr = [[1.0, rho], [rho, 1.0]]

        if mu is None and rates is not None:
            mu = rates
        if sigma is None and vols is not None:
            sigma = vols
        if corr is None:
            corr = corr_matrix if corr_matrix is not None else correlation
        if dividend_yield is None:
            dividend_yield = div_yields if div_yields is not None else dividends

        # ``spot_prices`` and ``spots`` are intentionally accepted as aliases
        # for generated route code, but they do not affect the process
        # parameters directly. The pricing route uses them to carry the initial
        # state separately.
        _ = spot_prices if spot_prices is not None else spots

        if mu is None or sigma is None or corr is None:
            raise TypeError("CorrelatedGBM requires either mu/sigma/corr or the full two-asset shorthand")

        mu_arr = raw_np.asarray(mu, dtype=float)
        sigma_arr = raw_np.asarray(sigma, dtype=float)

        if mu_arr.ndim != 1 or sigma_arr.ndim != 1:
            raise ValueError("mu and sigma must be one-dimensional arrays")
        if mu_arr.shape != sigma_arr.shape:
            raise ValueError("mu and sigma must have the same shape")

        n_assets = int(mu_arr.shape[0])
        corr_arr = raw_np.asarray(corr, dtype=float)
        if corr_arr.shape != (n_assets, n_assets):
            raise ValueError("corr must be a square matrix aligned with mu/sigma")
        if not raw_np.allclose(corr_arr, corr_arr.T, atol=1e-12, rtol=0.0):
            raise ValueError("corr must be symmetric")
        if not raw_np.allclose(raw_np.diag(corr_arr), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("corr diagonal entries must equal 1")

        if dividend_yield is None:
            dividend_arr = raw_np.zeros(n_assets, dtype=float)
        else:
            dividend_arr = raw_np.asarray(dividend_yield, dtype=float)
            if dividend_arr.shape != mu_arr.shape:
                raise ValueError("dividend_yield must match mu/sigma shape")

        self.mu = mu_arr
        self.sigma = sigma_arr
        self.corr = corr_arr
        self.dividend_yield = dividend_arr
        self._effective_mu = self.mu - self.dividend_yield
        self._chol = raw_np.linalg.cholesky(self.corr)

    @property
    def state_dim(self) -> int:
        """Return the number of simulated assets."""
        return int(self.mu.shape[0])

    @property
    def factor_dim(self) -> int:
        """Return the number of independent Gaussian factors."""
        return self.state_dim

    @property
    def cholesky_factor(self) -> raw_np.ndarray:
        """Return the cached Cholesky factor of the correlation matrix."""
        return self._chol

    def drift(self, x, t):
        """Return the per-asset drift vector ``(mu - q) * S``."""
        return raw_np.asarray(x, dtype=float) * self._effective_mu

    def diffusion(self, x, t):
        """Return factor loadings ``diag(sigma * S) @ chol`` per path."""
        x_arr = raw_np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            diag = self.sigma * x_arr
            return diag[:, None] * self._chol
        if x_arr.ndim != 2 or x_arr.shape[1] != self.state_dim:
            raise ValueError("x must have shape (n_assets,) or (n_paths, n_assets)")
        diag = self.sigma[None, :] * x_arr
        return diag[:, :, None] * self._chol[None, :, :]

    def exact_sample(self, x, t, dt, dw):
        """Return the exact correlated lognormal transition."""
        x_arr = raw_np.asarray(x, dtype=float)
        dw_arr = raw_np.asarray(dw, dtype=float)
        correlated = dw_arr @ self._chol.T
        drift_term = (self._effective_mu - 0.5 * self.sigma ** 2) * dt
        diffusion_term = self.sigma * raw_np.sqrt(dt) * correlated
        return x_arr * raw_np.exp(drift_term + diffusion_term)
