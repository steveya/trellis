"""Local volatility model: dS = mu*S*dt + sigma_local(S,t)*S*dW."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.models.processes.base import StochasticProcess

np = get_numpy()


class LocalVol(StochasticProcess):
    """Local volatility model (Dupire).

    Parameters
    ----------
    mu : float
        Risk-neutral drift.
    vol_fn : callable(S, t) -> float
        Local vol function sigma(S, t).
    """

    def __init__(self, mu: float, vol_fn):
        self.mu = mu
        self._vol_fn = vol_fn

    def drift(self, x, t):
        return self.mu * x

    def diffusion(self, x, t):
        return self._vol_fn(x, t) * x

    @classmethod
    def from_flat(cls, mu: float, sigma: float) -> LocalVol:
        """Create a flat local vol surface (equivalent to GBM)."""
        return cls(mu, lambda s, t: sigma)
