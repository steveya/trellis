"""Tests for Monte Carlo methods: discretization, engine, brownian bridge, LSM, quasi-random."""

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.models.processes.gbm import GBM
from trellis.models.monte_carlo.discretization import euler_maruyama, exact_simulation
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.brownian_bridge import brownian_bridge
from trellis.models.monte_carlo.lsm import longstaff_schwartz
from trellis.models.monte_carlo.variance_reduction import sobol_normals


def bs_call(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return S * norm.cdf(d1) - K * raw_np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return K * raw_np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Euler-Maruyama
# ---------------------------------------------------------------------------


class TestEulerMaruyama:
    def test_paths_shape(self):
        gbm = GBM(mu=0.05, sigma=0.20)
        n_paths, n_steps = 100, 50
        paths = euler_maruyama(gbm, x0=100.0, T=1.0, n_steps=n_steps,
                               n_paths=n_paths, rng=raw_np.random.default_rng(42))
        assert paths.shape == (n_paths, n_steps + 1)

    def test_initial_value(self):
        gbm = GBM(mu=0.05, sigma=0.20)
        paths = euler_maruyama(gbm, x0=100.0, T=1.0, n_steps=50,
                               n_paths=100, rng=raw_np.random.default_rng(42))
        assert raw_np.all(paths[:, 0] == 100.0)


# ---------------------------------------------------------------------------
# Exact simulation
# ---------------------------------------------------------------------------


class TestExactSimulation:
    def test_mean_of_terminal_values(self):
        """Mean of terminal GBM values ~ S0*exp(mu*T)."""
        mu, sigma = 0.05, 0.20
        S0, T = 100.0, 1.0
        gbm = GBM(mu, sigma)
        n_paths = 50000
        paths = exact_simulation(gbm, S0, T, n_steps=1, n_paths=n_paths,
                                 rng=raw_np.random.default_rng(123))
        terminal = paths[:, -1]
        expected_mean = S0 * raw_np.exp(mu * T)
        sample_mean = raw_np.mean(terminal)
        std_err = raw_np.std(terminal) / raw_np.sqrt(n_paths)
        assert abs(sample_mean - expected_mean) < 3 * std_err


# ---------------------------------------------------------------------------
# MonteCarloEngine
# ---------------------------------------------------------------------------


class TestMonteCarloEngine:
    def test_price_european_call_converges_to_bs(self):
        """MC European call within 3 standard errors of BS."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        gbm = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(gbm, n_paths=50000, n_steps=100,
                                  seed=42, method="exact")

        def payoff_fn(paths):
            return raw_np.maximum(paths[:, -1] - K, 0.0)

        result = engine.price(S0, T, payoff_fn, discount_rate=r)
        bs_ref = bs_call(S0, K, T, r, sigma)
        assert abs(result["price"] - bs_ref) < 3 * result["std_error"]

    def test_price_returns_correct_keys(self):
        gbm = GBM(mu=0.05, sigma=0.20)
        engine = MonteCarloEngine(gbm, n_paths=100, n_steps=10, seed=1)

        def payoff_fn(paths):
            return raw_np.maximum(paths[:, -1] - 100, 0.0)

        result = engine.price(100.0, 1.0, payoff_fn, discount_rate=0.05)
        assert "price" in result
        assert "std_error" in result
        assert "paths" in result


# ---------------------------------------------------------------------------
# Brownian bridge
# ---------------------------------------------------------------------------


class TestBrownianBridge:
    def test_starts_at_zero(self):
        W = brownian_bridge(T=1.0, n_steps=100, n_paths=500,
                            rng=raw_np.random.default_rng(42))
        assert raw_np.all(W[:, 0] == 0.0)

    def test_variance_at_each_time(self):
        """Var[W(t)] = t for standard BM."""
        T = 1.0
        n_steps = 64  # power of 2 for clean bisection
        n_paths = 50000
        W = brownian_bridge(T, n_steps, n_paths, rng=raw_np.random.default_rng(99))
        dt = T / n_steps
        for step in [n_steps // 4, n_steps // 2, n_steps]:
            t = step * dt
            empirical_var = raw_np.var(W[:, step])
            assert empirical_var == pytest.approx(t, rel=0.1)

    def test_shape(self):
        W = brownian_bridge(T=1.0, n_steps=50, n_paths=200,
                            rng=raw_np.random.default_rng(1))
        assert W.shape == (200, 51)


# ---------------------------------------------------------------------------
# LSM (Longstaff-Schwartz)
# ---------------------------------------------------------------------------


class TestLSM:
    def test_american_put_geq_european_put(self):
        """LSM American put price >= BS European put price."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        n_steps, n_paths = 50, 20000
        gbm = GBM(mu=r, sigma=sigma)
        paths = exact_simulation(gbm, S0, T, n_steps, n_paths,
                                 rng=raw_np.random.default_rng(42))

        def put_payoff(S):
            return raw_np.maximum(K - S, 0.0)

        exercise_dates = list(range(1, n_steps + 1))
        dt = T / n_steps
        amer_price = longstaff_schwartz(paths, exercise_dates, put_payoff, r, dt)
        euro_price = bs_put(S0, K, T, r, sigma)
        assert amer_price >= euro_price - 0.5  # allow small MC noise


# ---------------------------------------------------------------------------
# Sobol normals
# ---------------------------------------------------------------------------


class TestSobolNormals:
    def test_shape(self):
        n_paths, n_steps = 256, 10  # n_paths should be power of 2 for Sobol
        Z = sobol_normals(n_paths, n_steps)
        assert Z.shape == (n_paths, n_steps)

    def test_roughly_standard_normal_marginals(self):
        """Each column should have mean ~ 0 and std ~ 1."""
        n_paths = 1024
        n_steps = 5
        Z = sobol_normals(n_paths, n_steps)
        for j in range(n_steps):
            assert raw_np.mean(Z[:, j]) == pytest.approx(0.0, abs=0.15)
            assert raw_np.std(Z[:, j]) == pytest.approx(1.0, abs=0.15)
