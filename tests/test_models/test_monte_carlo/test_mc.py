"""Tests for Monte Carlo methods: discretization, engine, brownian bridge, LSM, quasi-random."""

import importlib

import numpy as raw_np
import pytest
from scipy.stats import norm

import trellis.models.monte_carlo.discretization as mc_discretization
import trellis.models.monte_carlo.engine as mc_engine_module
from trellis.models.processes.gbm import GBM
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.monte_carlo.discretization import euler_maruyama, exact_simulation, milstein
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.local_vol import (
    local_vol_european_vanilla_price,
    local_vol_european_vanilla_price_result,
)
from trellis.models.monte_carlo.brownian_bridge import brownian_bridge
from trellis.models.monte_carlo.lsm import longstaff_schwartz
from trellis.models.monte_carlo.path_state import (
    MonteCarloPathRequirement,
    PathReducer,
    barrier_payoff,
    terminal_value_payoff,
)
from trellis.models.monte_carlo.schemes import Exact, Milstein
from trellis.curves.yield_curve import YieldCurve
from trellis.models.monte_carlo.variance_reduction import (
    antithetic_normals,
    brownian_bridge_increments,
    sobol_normals,
)

mc_bridge_module = importlib.import_module("trellis.models.monte_carlo.brownian_bridge")


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

    def test_uses_vectorized_drift_and_diffusion_when_supported(self):
        """Euler-Maruyama should call vectorized process methods once per time step."""

        class VectorProcess:
            def __init__(self):
                self.drift_calls = []
                self.diffusion_calls = []

            def drift(self, x, t):
                arr = raw_np.asarray(x)
                self.drift_calls.append(arr)
                return 0.05 * arr

            def diffusion(self, x, t):
                arr = raw_np.asarray(x)
                self.diffusion_calls.append(arr)
                return 0.20 * arr

        process = VectorProcess()
        n_paths, n_steps = 8, 4
        paths = euler_maruyama(
            process,
            x0=100.0,
            T=1.0,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=raw_np.random.default_rng(9),
        )

        assert paths.shape == (n_paths, n_steps + 1)
        assert len(process.drift_calls) == n_steps
        assert len(process.diffusion_calls) == n_steps
        assert all(call.shape == (n_paths,) for call in process.drift_calls)
        assert all(call.shape == (n_paths,) for call in process.diffusion_calls)

    def test_scalar_only_process_fallback(self):
        """Scalar-only process methods should still simulate correctly."""

        class ScalarProcess:
            def drift(self, x, t):
                return 0.05 * float(x)

            def diffusion(self, x, t):
                return 0.20 * float(x)

        paths = euler_maruyama(
            ScalarProcess(),
            x0=100.0,
            T=1.0,
            n_steps=10,
            n_paths=32,
            rng=raw_np.random.default_rng(5),
        )

        assert paths.shape == (32, 11)
        assert raw_np.all(raw_np.isfinite(paths))

    def test_specialized_gbm_path_matches_generic_path(self, monkeypatch):
        """Built-in GBM fast path should match the generic evaluator path."""
        gbm = GBM(mu=0.05, sigma=0.20)

        fast = euler_maruyama(
            gbm, x0=100.0, T=1.0, n_steps=8, n_paths=64,
            rng=raw_np.random.default_rng(17),
        )

        monkeypatch.setattr(
            mc_discretization,
            "_maybe_specialized_simulation",
            lambda *args, **kwargs: None,
        )
        slow = euler_maruyama(
            gbm, x0=100.0, T=1.0, n_steps=8, n_paths=64,
            rng=raw_np.random.default_rng(17),
        )

        raw_np.testing.assert_allclose(fast, slow, atol=0.0, rtol=0.0)


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

    def test_uses_vectorized_exact_sample_when_supported(self):
        """Exact simulation should use one vectorized process call per time step."""

        class VectorProcess:
            def __init__(self):
                self.calls = []

            def exact_sample(self, x, t, dt, dw):
                self.calls.append((raw_np.asarray(x), raw_np.asarray(dw)))
                return x + dt + dw

        process = VectorProcess()
        n_paths, n_steps = 8, 4
        paths = exact_simulation(
            process,
            x0=1.0,
            T=1.0,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=raw_np.random.default_rng(7),
        )

        assert paths.shape == (n_paths, n_steps + 1)
        assert len(process.calls) == n_steps
        assert all(call_x.shape == (n_paths,) for call_x, _ in process.calls)
        assert all(call_dw.shape == (n_paths,) for _, call_dw in process.calls)

    def test_scalar_only_exact_sample_fallback(self):
        """Scalar-only exact samplers should still simulate correctly."""

        class ScalarProcess:
            def exact_sample(self, x, t, dt, dw):
                x = float(x)
                dw = float(dw)
                return x * raw_np.exp((0.05 - 0.5 * 0.20**2) * dt + 0.20 * raw_np.sqrt(dt) * dw)

        paths = exact_simulation(
            ScalarProcess(),
            x0=100.0,
            T=1.0,
            n_steps=10,
            n_paths=32,
            rng=raw_np.random.default_rng(6),
        )

        assert paths.shape == (32, 11)
        assert raw_np.all(paths > 0.0)

    def test_specialized_exact_gbm_path_matches_generic_path(self, monkeypatch):
        """Built-in exact GBM fast path should match the generic evaluator path."""
        gbm = GBM(mu=0.05, sigma=0.20)

        fast = exact_simulation(
            gbm, x0=100.0, T=1.0, n_steps=8, n_paths=64,
            rng=raw_np.random.default_rng(19),
        )

        monkeypatch.setattr(
            mc_discretization,
            "_maybe_specialized_simulation",
            lambda *args, **kwargs: None,
        )
        slow = exact_simulation(
            gbm, x0=100.0, T=1.0, n_steps=8, n_paths=64,
            rng=raw_np.random.default_rng(19),
        )

        raw_np.testing.assert_allclose(fast, slow, atol=0.0, rtol=0.0)


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
        assert "st_err" in result
        assert result["st_err"] == result["std_error"]
        assert "paths" in result
        assert "path_state" in result

    def test_exact_scheme_dispatches_to_exact_simulation(self, monkeypatch):
        """Known scheme names should reuse the optimized discretization functions."""
        called = {}

        def fake_exact(process, x0, T, n_steps, n_paths, rng):
            called["args"] = (process, x0, T, n_steps, n_paths, rng)
            return raw_np.zeros((n_paths, n_steps + 1))

        monkeypatch.setattr(mc_engine_module, "exact_simulation", fake_exact)

        engine = MonteCarloEngine(
            GBM(mu=0.05, sigma=0.20), n_paths=8, n_steps=4, seed=3, scheme=Exact(),
        )
        paths = engine.simulate(100.0, 1.0)

        assert paths.shape == (8, 5)
        assert called["args"][1:5] == (100.0, 1.0, 4, 8)

    def test_milstein_scheme_dispatches_to_milstein_function(self, monkeypatch):
        """Milstein scheme objects should route through the shared Milstein implementation."""
        called = {}

        def fake_milstein(process, x0, T, n_steps, n_paths, rng, *, fd_epsilon=1e-6):
            called["fd_epsilon"] = fd_epsilon
            return raw_np.zeros((n_paths, n_steps + 1))

        monkeypatch.setattr(
            mc_discretization,
            "milstein",
            fake_milstein,
        )

        scheme = Milstein(fd_epsilon=1e-4)
        engine = MonteCarloEngine(
            GBM(mu=0.05, sigma=0.20), n_paths=8, n_steps=4, seed=4, scheme=scheme,
        )
        paths = engine.simulate(100.0, 1.0)

        assert paths.shape == (8, 5)
        assert called["fd_epsilon"] == pytest.approx(1e-4)

    def test_reduced_terminal_storage_matches_full_path_pricing(self):
        gbm = GBM(mu=0.05, sigma=0.20)
        payoff = terminal_value_payoff(lambda terminal: raw_np.maximum(terminal - 100.0, 0.0))

        reduced_engine = MonteCarloEngine(gbm, n_paths=512, n_steps=16, seed=11, method="exact")
        reduced = reduced_engine.price(
            100.0,
            1.0,
            payoff,
            discount_rate=0.05,
            return_paths=False,
        )

        full_engine = MonteCarloEngine(gbm, n_paths=512, n_steps=16, seed=11, method="exact")
        full = full_engine.price(
            100.0,
            1.0,
            payoff,
            discount_rate=0.05,
        )

        assert reduced["paths"] is None
        assert reduced["path_state"] is not None
        assert reduced["path_state"].full_paths is None
        assert reduced["price"] == pytest.approx(full["price"], abs=0.0)
        assert reduced["std_error"] == pytest.approx(full["std_error"], abs=0.0)

    def test_simulate_state_snapshots_match_full_paths(self):
        gbm = GBM(mu=0.05, sigma=0.20)
        requirement = MonteCarloPathRequirement.snapshots((3, 7, 12))

        reduced_engine = MonteCarloEngine(gbm, n_paths=128, n_steps=12, seed=13, method="euler")
        state = reduced_engine.simulate_state(100.0, 1.0, requirement)

        full_engine = MonteCarloEngine(gbm, n_paths=128, n_steps=12, seed=13, method="euler")
        paths = full_engine.simulate(100.0, 1.0)

        assert state.full_paths is None
        raw_np.testing.assert_allclose(state.terminal_values, paths[:, -1], atol=0.0, rtol=0.0)
        for step in (3, 7, 12):
            raw_np.testing.assert_allclose(state.snapshot(step), paths[:, step], atol=0.0, rtol=0.0)

    def test_reduced_barrier_storage_matches_full_path_pricing(self):
        gbm = GBM(mu=0.05, sigma=0.20)
        payoff = barrier_payoff(
            barrier=90.0,
            direction="down",
            knock="out",
            terminal_payoff_fn=lambda terminal: raw_np.maximum(terminal - 100.0, 0.0),
        )

        reduced_engine = MonteCarloEngine(gbm, n_paths=1024, n_steps=24, seed=17, method="exact")
        reduced = reduced_engine.price(
            100.0,
            1.0,
            payoff,
            discount_rate=0.05,
            return_paths=False,
        )

        full_engine = MonteCarloEngine(gbm, n_paths=1024, n_steps=24, seed=17, method="exact")
        full = full_engine.price(
            100.0,
            1.0,
            payoff,
            discount_rate=0.05,
        )

        assert reduced["price"] == pytest.approx(full["price"], abs=0.0)
        assert reduced["std_error"] == pytest.approx(full["std_error"], abs=0.0)
        raw_np.testing.assert_array_equal(
            reduced["path_state"].barrier_hit("barrier"),
            raw_np.any(full["paths"] <= 90.0, axis=1),
        )

    def test_vector_state_terminal_storage_matches_full_path_pricing(self):
        process = CorrelatedGBM(
            mu=[0.05, 0.05],
            sigma=[0.20, 0.25],
            corr=[[1.0, 0.35], [0.35, 1.0]],
            dividend_yield=[0.01, 0.02],
        )
        weights = raw_np.array([0.6, 0.4])
        payoff = terminal_value_payoff(
            lambda terminal: raw_np.maximum(terminal @ weights - 100.0, 0.0)
        )
        x0 = raw_np.array([100.0, 95.0])

        reduced_engine = MonteCarloEngine(process, n_paths=512, n_steps=12, seed=23, method="exact")
        reduced = reduced_engine.price(
            x0,
            1.0,
            payoff,
            discount_rate=0.05,
            return_paths=False,
        )

        full_engine = MonteCarloEngine(process, n_paths=512, n_steps=12, seed=23, method="exact")
        full = full_engine.price(
            x0,
            1.0,
            payoff,
            discount_rate=0.05,
        )

        assert reduced["paths"] is None
        assert reduced["path_state"].terminal_values.shape == (512, 2)
        assert reduced["price"] == pytest.approx(full["price"], abs=0.0)
        assert reduced["std_error"] == pytest.approx(full["std_error"], abs=0.0)

    def test_vector_state_snapshots_and_reducers_match_full_paths(self):
        process = CorrelatedGBM(
            mu=[0.05, 0.05],
            sigma=[0.20, 0.25],
            corr=[[1.0, 0.25], [0.25, 1.0]],
        )
        weights = raw_np.array([0.7, 0.3])
        observation_steps = (2, 4, 6)
        reducer = PathReducer(
            name="running_weighted_sum",
            init_fn=lambda initial, n_steps: raw_np.zeros(initial.shape[0], dtype=float),
            update_fn=lambda acc, values, step: (
                acc + values @ weights if step in observation_steps else acc
            ),
        )
        requirement = MonteCarloPathRequirement(
            snapshot_steps=observation_steps,
            reducers=(reducer,),
        )
        x0 = raw_np.array([100.0, 90.0])

        reduced_engine = MonteCarloEngine(process, n_paths=128, n_steps=6, seed=29, method="exact")
        state = reduced_engine.simulate_state(x0, 1.0, requirement)

        full_engine = MonteCarloEngine(process, n_paths=128, n_steps=6, seed=29, method="exact")
        paths = full_engine.simulate(x0, 1.0)

        for step in observation_steps:
            raw_np.testing.assert_allclose(state.snapshot(step), paths[:, step], atol=0.0, rtol=0.0)

        expected_running_sum = sum(paths[:, step, :] @ weights for step in observation_steps)
        raw_np.testing.assert_allclose(
            state.reduced_value("running_weighted_sum"),
            expected_running_sum,
            atol=0.0,
            rtol=0.0,
        )

    def test_simulate_with_explicit_factor_shocks_is_deterministic(self):
        process = CorrelatedGBM(
            mu=[0.05, 0.05],
            sigma=[0.20, 0.25],
            corr=[[1.0, 0.40], [0.40, 1.0]],
        )
        engine = MonteCarloEngine(process, n_paths=8, n_steps=5, seed=31, method="exact")
        shocks = raw_np.arange(8 * 5 * 2, dtype=float).reshape(8, 5, 2) / 50.0
        x0 = raw_np.array([100.0, 90.0])

        paths_a = engine.simulate_with_shocks(x0, 1.0, shocks)
        paths_b = engine.simulate_with_shocks(x0, 1.0, shocks)

        raw_np.testing.assert_allclose(paths_a, paths_b, atol=0.0, rtol=0.0)


class TestLocalVolMonteCarlo:
    def test_flat_local_vol_matches_black_scholes_reasonably_well(self):
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        discount_curve = YieldCurve.flat(r)

        mc_result = local_vol_european_vanilla_price_result(
            spot=S0,
            strike=K,
            maturity=T,
            discount_curve=discount_curve,
            local_vol_surface=lambda s, t: sigma,
            option_type="call",
            n_paths=40_000,
            n_steps=120,
            seed=7,
        )

        bs_ref = bs_call(S0, K, T, r, sigma)
        assert abs(mc_result.price - bs_ref) < 4 * mc_result.std_error
        assert mc_result.n_paths == 40_000

    def test_scalar_helper_returns_price_only(self):
        result = local_vol_european_vanilla_price(
            spot=100.0,
            strike=95.0,
            maturity=0.75,
            discount_curve=YieldCurve.flat(0.03),
            local_vol_surface=lambda s, t: 0.18,
            option_type="put",
            n_paths=2_000,
            n_steps=48,
            seed=3,
        )

        assert isinstance(result, float)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Brownian bridge
# ---------------------------------------------------------------------------


class TestBrownianBridge:
    def test_matches_recursive_reference_schedule(self, monkeypatch):
        def recursive_reference(T, n_steps, n_paths, rng):
            W = raw_np.zeros((n_paths, n_steps + 1), dtype=float)
            W[:, -1] = rng.normal(0.0, raw_np.sqrt(T), size=n_paths)

            def fill(i_start, i_end):
                if i_end - i_start <= 1:
                    return
                dt = T / n_steps
                i_mid = (i_start + i_end) // 2
                t_start = i_start * dt
                t_mid = i_mid * dt
                t_end = i_end * dt
                tau = t_end - t_start
                alpha = (t_mid - t_start) / tau
                bridge_var = (t_mid - t_start) * (t_end - t_mid) / tau
                W[:, i_mid] = (
                    (1.0 - alpha) * W[:, i_start]
                    + alpha * W[:, i_end]
                    + rng.normal(0.0, raw_np.sqrt(bridge_var), size=n_paths)
                )
                fill(i_start, i_mid)
                fill(i_mid, i_end)

            fill(0, n_steps)
            return W

        monkeypatch.setattr(mc_bridge_module, "NUMBA_AVAILABLE", False)

        actual = brownian_bridge(1.0, 7, 16, rng=raw_np.random.default_rng(123))
        expected = recursive_reference(1.0, 7, 16, rng=raw_np.random.default_rng(123))
        raw_np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

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

    def test_end_values_are_respected(self):
        end_values = raw_np.linspace(-1.0, 1.0, 9)
        W = brownian_bridge(
            T=1.0,
            n_steps=8,
            n_paths=9,
            rng=raw_np.random.default_rng(2),
            end_values=end_values,
        )
        raw_np.testing.assert_allclose(W[:, -1], end_values, atol=0.0, rtol=0.0)

    def test_bridge_shocks_are_reproducible(self):
        bridge_shocks = raw_np.arange(32, dtype=float).reshape(4, 8) / 10.0
        W_a = brownian_bridge(T=1.0, n_steps=8, n_paths=4, bridge_shocks=bridge_shocks)
        W_b = brownian_bridge(T=1.0, n_steps=8, n_paths=4, bridge_shocks=bridge_shocks)
        raw_np.testing.assert_allclose(W_a, W_b, atol=0.0, rtol=0.0)

    def test_bridge_increments_support_multiple_factors(self):
        normals = raw_np.arange(2 * 8 * 3, dtype=float).reshape(2, 8, 3) / 25.0
        increments = brownian_bridge_increments(normals, 1.0)
        assert increments.shape == (2, 8, 3)


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

    def test_multi_factor_shape(self):
        Z = sobol_normals(256, 10, n_factors=3)
        assert Z.shape == (256, 10, 3)

    def test_roughly_standard_normal_marginals(self):
        """Each column should have mean ~ 0 and std ~ 1."""
        n_paths = 1024
        n_steps = 5
        Z = sobol_normals(n_paths, n_steps)
        for j in range(n_steps):
            assert raw_np.mean(Z[:, j]) == pytest.approx(0.0, abs=0.15)
            assert raw_np.std(Z[:, j]) == pytest.approx(1.0, abs=0.15)


class TestAntitheticNormals:
    def test_pairs_are_negatives(self):
        Z = antithetic_normals(8, 5, n_factors=2, rng=raw_np.random.default_rng(7))
        raw_np.testing.assert_allclose(Z[:4], -Z[4:], atol=0.0, rtol=0.0)


class TestMilstein:
    def test_uses_vectorized_drift_and_diffusion_when_supported(self):
        """Milstein should use vectorized drift/diffusion evaluations when available."""

        class VectorProcess:
            def __init__(self):
                self.drift_calls = []
                self.diffusion_calls = []

            def drift(self, x, t):
                arr = raw_np.asarray(x)
                self.drift_calls.append(arr)
                return 0.05 * arr

            def diffusion(self, x, t):
                arr = raw_np.asarray(x)
                self.diffusion_calls.append(arr)
                return 0.20 * arr

        process = VectorProcess()
        n_paths, n_steps = 8, 4
        paths = milstein(
            process,
            x0=100.0,
            T=1.0,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=raw_np.random.default_rng(11),
        )

        assert paths.shape == (n_paths, n_steps + 1)
        assert len(process.drift_calls) == n_steps
        assert len(process.diffusion_calls) == 2 * n_steps
        assert all(call.shape == (n_paths,) for call in process.drift_calls)
        assert all(call.shape == (n_paths,) for call in process.diffusion_calls)
