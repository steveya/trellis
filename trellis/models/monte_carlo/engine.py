"""Monte Carlo pricing engine."""

from __future__ import annotations

import numpy as raw_np

from trellis.models.monte_carlo.discretization import euler_maruyama, exact_simulation


class MonteCarloEngine:
    """Generic Monte Carlo pricing engine.

    Parameters
    ----------
    process : StochasticProcess
        The underlying stochastic process.
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps per path.
    seed : int or None
        Random seed for reproducibility.
    method : str
        Discretization: ``"euler"``, ``"milstein"``, or ``"exact"``.
    """

    def __init__(
        self,
        process,
        n_paths: int = 10000,
        n_steps: int = 100,
        seed: int | None = None,
        method: str = "euler",
    ):
        self.process = process
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rng = raw_np.random.default_rng(seed)
        self.method = method

    def simulate(self, x0: float, T: float) -> raw_np.ndarray:
        """Generate paths. Returns (n_paths, n_steps + 1) array."""
        if self.method == "exact":
            return exact_simulation(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
            )
        elif self.method == "milstein":
            from trellis.models.monte_carlo.discretization import milstein
            return milstein(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
            )
        else:
            return euler_maruyama(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
            )

    def price(self, x0: float, T: float, payoff_fn, discount_rate: float = 0.0) -> dict:
        """Price a derivative via Monte Carlo.

        Parameters
        ----------
        x0 : float
            Initial value of the process.
        T : float
            Time to maturity.
        payoff_fn : callable(paths: ndarray) -> ndarray
            Maps (n_paths, n_steps+1) paths to (n_paths,) payoffs.
        discount_rate : float
            Risk-free rate for discounting.

        Returns
        -------
        dict with 'price', 'std_error', 'paths' keys.
        """
        paths = self.simulate(x0, T)
        payoffs = payoff_fn(paths)
        df = raw_np.exp(-discount_rate * T)
        discounted = df * payoffs

        price = float(raw_np.mean(discounted))
        std_error = float(raw_np.std(discounted) / raw_np.sqrt(self.n_paths))

        return {
            "price": price,
            "std_error": std_error,
            "n_paths": self.n_paths,
            "paths": paths,
        }
