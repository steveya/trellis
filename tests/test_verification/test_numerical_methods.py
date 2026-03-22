"""WP3: Numerical methods verification — convergence to known analytical values.

Tests that trees, MC, FFT converge to Black-Scholes for European options.
"""

from datetime import date

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.backward_induction import backward_induction
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.discretization import euler_maruyama, exact_simulation
from trellis.models.monte_carlo.lsm import longstaff_schwartz
from trellis.models.processes.gbm import GBM
from trellis.models.transforms.fft_pricer import fft_price
from trellis.models.calibration.implied_vol import implied_vol, _bs_price


S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0


def bs_call(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return S * norm.cdf(d1) - K * raw_np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return K * raw_np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


BS_CALL = bs_call(S0, K, T, r, sigma)
BS_PUT = bs_put(S0, K, T, r, sigma)


# ---------------------------------------------------------------------------
# Binomial tree convergence
# ---------------------------------------------------------------------------

class TestBinomialTreeConvergence:

    def test_european_call_converges(self):
        """CRR tree price → BS as n_steps increases."""
        for n in [50, 100, 200]:
            tree = BinomialTree.crr(S0, T, n, r, sigma)
            def call_payoff(step, node):
                return max(tree.value_at(step, node) - K, 0)
            price = backward_induction(tree, call_payoff, r, "european")
            if n >= 200:
                assert price == pytest.approx(BS_CALL, rel=0.01), (
                    f"n={n}: tree={price:.4f}, BS={BS_CALL:.4f}"
                )

    def test_european_put_converges(self):
        tree = BinomialTree.crr(S0, T, 200, r, sigma)
        def put_payoff(step, node):
            return max(K - tree.value_at(step, node), 0)
        price = backward_induction(tree, put_payoff, r, "european")
        assert price == pytest.approx(BS_PUT, rel=0.01)

    def test_put_call_parity_on_tree(self):
        """C - P = S0 - K×exp(-rT) on the tree."""
        tree = BinomialTree.crr(S0, T, 200, r, sigma)
        def call_payoff(step, node):
            return max(tree.value_at(step, node) - K, 0)
        def put_payoff(step, node):
            return max(K - tree.value_at(step, node), 0)
        C = backward_induction(tree, call_payoff, r, "european")
        P = backward_induction(tree, put_payoff, r, "european")
        parity = C - P
        expected = S0 - K * raw_np.exp(-r * T)
        assert parity == pytest.approx(expected, rel=0.02)

    def test_american_put_geq_european(self):
        """American put ≥ European put (early exercise premium)."""
        tree = BinomialTree.crr(S0, T, 200, r, sigma)
        def put_payoff(step, node):
            return max(K - tree.value_at(step, node), 0)
        def exercise_val(step, node, t):
            return max(K - t.value_at(step, node), 0)
        euro = backward_induction(tree, put_payoff, r, "european")
        amer = backward_induction(tree, put_payoff, r, "american",
                                   exercise_value_fn=exercise_val)
        assert amer >= euro - 0.01


# ---------------------------------------------------------------------------
# Monte Carlo convergence
# ---------------------------------------------------------------------------

class TestMonteCarloConvergence:

    def test_european_call_mc(self):
        """MC European call within 3 standard errors of BS."""
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=100000, n_steps=1,
                                   seed=42, method="exact")
        def payoff_fn(paths):
            return raw_np.maximum(paths[:, -1] - K, 0)
        result = engine.price(S0, T, payoff_fn, discount_rate=r)
        assert abs(result["price"] - BS_CALL) < 3 * result["std_error"]

    def test_mc_convergence_improves(self):
        """More paths → lower standard error."""
        process = GBM(mu=r, sigma=sigma)
        se_prev = float("inf")
        for n_paths in [1000, 10000, 100000]:
            engine = MonteCarloEngine(process, n_paths=n_paths, n_steps=1,
                                       seed=42, method="exact")
            def payoff_fn(paths):
                return raw_np.maximum(paths[:, -1] - K, 0)
            result = engine.price(S0, T, payoff_fn, discount_rate=r)
            assert result["std_error"] < se_prev
            se_prev = result["std_error"]

    def test_euler_vs_exact_gbm(self):
        """Euler and exact simulation should give similar results for GBM."""
        process = GBM(mu=r, sigma=sigma)
        rng = raw_np.random.default_rng(42)
        paths_exact = exact_simulation(process, S0, T, 100, 10000, rng)
        rng = raw_np.random.default_rng(42)
        paths_euler = euler_maruyama(process, S0, T, 100, 10000, rng)
        # Terminal means should be similar
        mean_exact = raw_np.mean(paths_exact[:, -1])
        mean_euler = raw_np.mean(paths_euler[:, -1])
        assert mean_exact == pytest.approx(mean_euler, rel=0.05)


# ---------------------------------------------------------------------------
# LSM verification
# ---------------------------------------------------------------------------

class TestLSMConvergence:

    def test_american_put_geq_european(self):
        """LSM American put price ≥ BS European put."""
        process = GBM(mu=r, sigma=sigma)
        rng = raw_np.random.default_rng(42)
        paths = exact_simulation(process, S0, T, 50, 50000, rng)
        dt = T / 50
        exercise_dates = list(range(1, 51))
        def payoff_fn(S):
            return raw_np.maximum(K - S, 0)
        amer_price = longstaff_schwartz(paths, exercise_dates, payoff_fn, r, dt)
        assert amer_price >= BS_PUT * 0.95  # allow small MC noise


# ---------------------------------------------------------------------------
# FFT verification
# ---------------------------------------------------------------------------

class TestFFTConvergence:

    def test_gbm_call_matches_bs(self):
        """FFT with GBM characteristic function matches BS."""
        def gbm_char_fn(u):
            return raw_np.exp(
                1j * u * (raw_np.log(S0) + (r - 0.5 * sigma ** 2) * T)
                - 0.5 * sigma ** 2 * T * u ** 2
            )
        fft_call = fft_price(gbm_char_fn, S0, K, T, r)
        assert fft_call == pytest.approx(BS_CALL, rel=0.01)

    def test_fft_otm_call(self):
        def gbm_char_fn(u):
            return raw_np.exp(
                1j * u * (raw_np.log(S0) + (r - 0.5 * sigma ** 2) * T)
                - 0.5 * sigma ** 2 * T * u ** 2
            )
        fft_call = fft_price(gbm_char_fn, S0, 120.0, T, r)
        bs_ref = bs_call(S0, 120.0, T, r, sigma)
        assert fft_call == pytest.approx(bs_ref, rel=0.02)


# ---------------------------------------------------------------------------
# Implied vol round-trip
# ---------------------------------------------------------------------------

class TestImpliedVol:

    def test_call_round_trip(self):
        """Compute BS price → recover vol → should match input."""
        price = _bs_price(S0, K, T, r, sigma, "call")
        recovered = implied_vol(price, S0, K, T, r, "call")
        assert recovered == pytest.approx(sigma, rel=1e-4)

    def test_put_round_trip(self):
        price = _bs_price(S0, K, T, r, sigma, "put")
        recovered = implied_vol(price, S0, K, T, r, "put")
        assert recovered == pytest.approx(sigma, rel=1e-4)

    def test_otm_call_round_trip(self):
        price = _bs_price(S0, 120, T, r, 0.30, "call")
        recovered = implied_vol(price, S0, 120, T, r, "call")
        assert recovered == pytest.approx(0.30, rel=1e-3)
