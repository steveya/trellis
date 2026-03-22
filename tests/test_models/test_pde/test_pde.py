"""Tests for PDE solvers: Thomas algorithm, Grid, Crank-Nicolson, implicit FD, PSOR."""

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.models.pde.thomas import thomas_solve
from trellis.models.pde.grid import Grid
from trellis.models.pde.crank_nicolson import crank_nicolson_1d
from trellis.models.pde.implicit_fd import implicit_fd_1d
from trellis.models.pde.psor import psor_1d


def bs_call(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return S * norm.cdf(d1) - K * raw_np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return K * raw_np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Thomas algorithm
# ---------------------------------------------------------------------------


class TestThomasSolve:
    def test_known_3x3_system(self):
        """Solve:
        [2 -1  0] [x0]   [1]
        [-1 2 -1] [x1] = [0]
        [0 -1  2] [x2]   [1]
        Solution: x = [1, 1, 1].
        """
        a = raw_np.array([-1.0, -1.0])
        b = raw_np.array([2.0, 2.0, 2.0])
        c = raw_np.array([-1.0, -1.0])
        d = raw_np.array([1.0, 0.0, 1.0])
        x = thomas_solve(a, b, c, d)
        assert x[0] == pytest.approx(1.0, abs=1e-10)
        assert x[1] == pytest.approx(1.0, abs=1e-10)
        assert x[2] == pytest.approx(1.0, abs=1e-10)

    def test_identity_system(self):
        """Diagonal system: I*x = d => x = d."""
        n = 5
        a = raw_np.zeros(n - 1)
        b = raw_np.ones(n)
        c = raw_np.zeros(n - 1)
        d = raw_np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = thomas_solve(a, b, c, d)
        raw_np.testing.assert_allclose(x, d, atol=1e-12)

    def test_matches_numpy_solve(self):
        """Compare against numpy.linalg.solve for a random tridiagonal system."""
        n = 10
        rng = raw_np.random.default_rng(7)
        a = rng.standard_normal(n - 1)
        b = rng.standard_normal(n) + 5  # diagonally dominant
        c = rng.standard_normal(n - 1)
        d = rng.standard_normal(n)

        # Build full matrix
        A = raw_np.diag(b) + raw_np.diag(a, -1) + raw_np.diag(c, 1)
        x_np = raw_np.linalg.solve(A, d)
        x_thomas = thomas_solve(a, b, c, d)
        raw_np.testing.assert_allclose(x_thomas, x_np, atol=1e-8)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


class TestGrid:
    def test_correct_number_of_points(self):
        g = Grid(x_min=0, x_max=200, n_x=201, T=1.0, n_t=100)
        assert len(g.x) == 201
        assert len(g.t) == 101  # n_t + 1

    def test_x_min_x_max(self):
        g = Grid(x_min=10.0, x_max=300.0, n_x=100, T=1.0, n_t=50)
        assert g.x[0] == pytest.approx(10.0, abs=1e-12)
        assert g.x[-1] == pytest.approx(300.0, abs=1e-12)
        assert g.x_min == 10.0
        assert g.x_max == 300.0

    def test_dt_correct(self):
        g = Grid(x_min=0, x_max=200, n_x=100, T=2.0, n_t=200)
        assert g.dt == pytest.approx(0.01, abs=1e-14)


# ---------------------------------------------------------------------------
# Crank-Nicolson for European call
# ---------------------------------------------------------------------------


class TestCrankNicolson:
    def test_european_call_runs_without_error(self):
        """CN solver runs and produces a non-negative price array."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        S_max = 300.0
        n_x, n_t = 200, 200

        grid = Grid(x_min=0.0, x_max=S_max, n_x=n_x, T=T, n_t=n_t)
        S = grid.x
        terminal = raw_np.maximum(S - K, 0.0)

        V = crank_nicolson_1d(
            grid,
            lambda s, t: sigma,
            lambda t: r,
            terminal,
            lower_bc_fn=lambda t: 0.0,
            upper_bc_fn=lambda t: S_max - K * raw_np.exp(-r * (T - t)),
        )
        assert V.shape == (n_x,)
        # Terminal condition at S=0 should give V~0 (boundary)
        assert V[0] == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# Implicit FD for European call
# ---------------------------------------------------------------------------


class TestImplicitFD:
    def test_european_call_runs_without_error(self):
        """Implicit FD solver runs and produces a result array."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        S_max = 300.0
        n_x, n_t = 200, 200

        grid = Grid(x_min=0.0, x_max=S_max, n_x=n_x, T=T, n_t=n_t)
        S = grid.x
        terminal = raw_np.maximum(S - K, 0.0)

        V = implicit_fd_1d(
            grid,
            lambda s, t: sigma,
            lambda t: r,
            terminal,
            lower_bc_fn=lambda t: 0.0,
            upper_bc_fn=lambda t: S_max - K * raw_np.exp(-r * (T - t)),
        )
        assert V.shape == (n_x,)
        assert V[0] == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# PSOR for American put
# ---------------------------------------------------------------------------


class TestPSOR:
    def test_american_put_geq_european_and_intrinsic(self):
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        S_max = 300.0
        n_x, n_t = 200, 200

        grid = Grid(x_min=0.0, x_max=S_max, n_x=n_x, T=T, n_t=n_t)
        S = grid.x

        terminal = raw_np.maximum(K - S, 0.0)
        exercise = raw_np.maximum(K - S, 0.0)

        V = psor_1d(grid, lambda s, t: sigma, lambda t: r, terminal, exercise,
                    lower_bc_fn=lambda t: K * raw_np.exp(-r * (T - t)),
                    upper_bc_fn=lambda t: 0.0)

        idx = raw_np.searchsorted(S, S0)
        idx = min(idx, n_x - 2)
        w = (S0 - S[idx]) / (S[idx + 1] - S[idx])
        amer_price = V[idx] * (1 - w) + V[idx + 1] * w

        euro_price = bs_put(S0, K, T, r, sigma)
        intrinsic = max(K - S0, 0.0)

        assert amer_price >= euro_price - 0.1
        assert amer_price >= intrinsic - 1e-6
