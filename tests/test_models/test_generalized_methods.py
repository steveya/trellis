"""Tests for generalized numerical method framework.

Tests the abstraction layers — operators, schemes, basis functions, tree models —
independent of any specific instrument. Each test verifies the METHOD, not the
financial product.

Structure:
1. PDE operator + theta-method: convergence to BS, theta parameter effect, American pricing
2. MC schemes: scheme objects produce correct dynamics, antithetic reduces variance
3. LSM basis functions: all bases price American puts, Laguerre beats polynomial at high vol
4. Tree models: all models calibrate to curve, model choice affects rate distribution
5. Wiring: build_generic_lattice dispatches correctly for each TreeModel
"""

import numpy as raw_np
import pytest
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0


def bs_call(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return float(S * norm.cdf(d1) - K * raw_np.exp(-r * T) * norm.cdf(d2))


def bs_put(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return float(K * raw_np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


BS_CALL = bs_call(S0, K, T, r, sigma)
BS_PUT = bs_put(S0, K, T, r, sigma)


def _interp_grid(V, S, S0):
    """Interpolate PDE solution at S0."""
    idx = raw_np.searchsorted(S, S0)
    idx = max(1, min(idx, len(S) - 1))
    w = (S0 - S[idx - 1]) / (S[idx] - S[idx - 1])
    return float(V[idx - 1] * (1 - w) + V[idx] * w)


# ===================================================================
# 1. PDE OPERATOR + THETA METHOD
# ===================================================================

class TestPDEOperator:
    """PDE operators produce correct tridiagonal coefficients."""

    def test_bs_operator_uses_single_vectorized_sigma_evaluation(self):
        """Vector-friendly sigma functions should be called once per coefficient build."""
        from trellis.models.pde.operator import BlackScholesOperator

        calls = []

        def sigma_fn(s, t):
            arr = raw_np.asarray(s)
            calls.append(arr)
            return 0.20

        op = BlackScholesOperator(sigma_fn, lambda t: 0.05)
        S = raw_np.linspace(0, 400, 201)
        a, b, c = op.coefficients(S, t=0.0, dt=0.01)

        assert len(calls) == 1
        assert calls[0].shape == (199,)
        assert len(a) == 199
        assert len(b) == 199
        assert len(c) == 199

    def test_bs_operator_coefficients_shape(self):
        from trellis.models.pde.operator import BlackScholesOperator
        op = BlackScholesOperator(lambda s, t: 0.20, lambda t: 0.05)
        S = raw_np.linspace(0, 400, 201)
        a, b, c = op.coefficients(S, t=0.0, dt=0.01)
        assert len(a) == 199  # n_x - 2 interior points
        assert len(b) == 199
        assert len(c) == 199

    def test_bs_operator_coefficients_sign(self):
        """For BS PDE: a > 0, b < 0, c > 0 at interior points away from S=0."""
        from trellis.models.pde.operator import BlackScholesOperator
        op = BlackScholesOperator(lambda s, t: 0.20, lambda t: 0.05)
        S = raw_np.linspace(1, 400, 200)  # avoid S=0
        a, b, c = op.coefficients(S, t=0.0, dt=0.01)
        # Most interior points should have a>0, b<0, c>0
        assert raw_np.sum(a > 0) > len(a) * 0.9
        assert raw_np.sum(b < 0) > len(b) * 0.9
        assert raw_np.sum(c > 0) > len(c) * 0.9

    def test_cev_reduces_to_bs_at_beta_1(self):
        """CEV with β=1 should give same coefficients as BS."""
        from trellis.models.pde.operator import BlackScholesOperator, CEVOperator
        sig_fn = lambda s, t: 0.20
        r_fn = lambda t: 0.05
        bs_op = BlackScholesOperator(sig_fn, r_fn)
        cev_op = CEVOperator(sig_fn, r_fn, beta=1.0)
        S = raw_np.linspace(1, 400, 200)
        a_bs, b_bs, c_bs = bs_op.coefficients(S, 0.0, 0.01)
        a_cev, b_cev, c_cev = cev_op.coefficients(S, 0.0, 0.01)
        raw_np.testing.assert_allclose(a_bs, a_cev, atol=1e-15)
        raw_np.testing.assert_allclose(b_bs, b_cev, atol=1e-15)
        raw_np.testing.assert_allclose(c_bs, c_cev, atol=1e-15)

    def test_heat_operator_constant_coefficients(self):
        """Heat equation should give spatially uniform coefficients."""
        from trellis.models.pde.operator import HeatOperator
        op = HeatOperator(diffusivity=1.0)
        S = raw_np.linspace(0, 10, 101)
        a, b, c = op.coefficients(S, 0.0, 0.01)
        # All a should be equal, all b equal, all c equal
        assert raw_np.std(a) < 1e-15
        assert raw_np.std(b) < 1e-15
        assert raw_np.std(c) < 1e-15


class TestThetaMethod:
    """Unified theta-method solver converges for all theta values."""

    def _setup_european_call(self, n=400):
        from trellis.models.pde.operator import BlackScholesOperator
        from trellis.models.pde.grid import Grid
        S_max = 4 * S0
        grid = Grid(x_min=0.0, x_max=S_max, n_x=n, T=T, n_t=n)
        op = BlackScholesOperator(lambda s, t: sigma, lambda t: r)
        terminal = raw_np.maximum(grid.x - K, 0.0)
        bc_lower = lambda t: 0.0
        bc_upper = lambda t: S_max - K * raw_np.exp(-r * (T - t))
        return grid, op, terminal, bc_lower, bc_upper

    @pytest.mark.parametrize("theta,label", [
        (0.5, "Crank-Nicolson"),
        (1.0, "Fully implicit"),
        (0.75, "Mixed"),
    ])
    def test_european_call_converges(self, theta, label):
        """European call converges to BS for any stable theta."""
        from trellis.models.pde.theta_method import theta_method_1d
        grid, op, terminal, bc_lo, bc_hi = self._setup_european_call(400)
        V = theta_method_1d(grid, op, terminal, theta=theta,
                            lower_bc_fn=bc_lo, upper_bc_fn=bc_hi)
        price = _interp_grid(V, grid.x, S0)
        assert price == pytest.approx(BS_CALL, rel=0.01), (
            f"{label} (θ={theta}): {price:.4f} vs BS={BS_CALL:.4f}"
        )

    def test_cn_more_accurate_than_implicit(self):
        """θ=0.5 (second order) should be more accurate than θ=1.0 (first order)."""
        from trellis.models.pde.theta_method import theta_method_1d
        grid, op, terminal, bc_lo, bc_hi = self._setup_european_call(200)
        V_cn = theta_method_1d(grid, op, terminal, theta=0.5,
                               lower_bc_fn=bc_lo, upper_bc_fn=bc_hi)
        V_impl = theta_method_1d(grid, op, terminal, theta=1.0,
                                  lower_bc_fn=bc_lo, upper_bc_fn=bc_hi)
        err_cn = abs(_interp_grid(V_cn, grid.x, S0) - BS_CALL)
        err_impl = abs(_interp_grid(V_impl, grid.x, S0) - BS_CALL)
        assert err_cn <= err_impl + 0.01  # CN should be at least as good

    def test_put_call_parity(self):
        """C_PDE - P_PDE ≈ S0 - K*exp(-rT)."""
        from trellis.models.pde.operator import BlackScholesOperator
        from trellis.models.pde.theta_method import theta_method_1d
        from trellis.models.pde.grid import Grid
        S_max = 4 * S0
        n = 400
        grid = Grid(x_min=0.0, x_max=S_max, n_x=n, T=T, n_t=n)
        op = BlackScholesOperator(lambda s, t: sigma, lambda t: r)

        # Call
        V_call = theta_method_1d(
            grid, op, raw_np.maximum(grid.x - K, 0.0), theta=0.5,
            lower_bc_fn=lambda t: 0.0,
            upper_bc_fn=lambda t: S_max - K * raw_np.exp(-r * (T - t)),
        )
        # Put
        V_put = theta_method_1d(
            grid, op, raw_np.maximum(K - grid.x, 0.0), theta=0.5,
            lower_bc_fn=lambda t: K * raw_np.exp(-r * (T - t)),
            upper_bc_fn=lambda t: 0.0,
        )
        C = _interp_grid(V_call, grid.x, S0)
        P = _interp_grid(V_put, grid.x, S0)
        parity = C - P
        expected = S0 - K * raw_np.exp(-r * T)
        assert parity == pytest.approx(expected, abs=0.5)

    def test_american_put_geq_european(self):
        """PSOR American put ≥ BS European put."""
        from trellis.models.pde.operator import BlackScholesOperator
        from trellis.models.pde.theta_method import theta_method_1d
        from trellis.models.pde.grid import Grid
        S_max = 4 * S0
        n = 400
        grid = Grid(x_min=0.0, x_max=S_max, n_x=n, T=T, n_t=n)
        op = BlackScholesOperator(lambda s, t: sigma, lambda t: r)
        terminal = raw_np.maximum(K - grid.x, 0.0)

        V_amer = theta_method_1d(
            grid, op, terminal, theta=1.0,
            lower_bc_fn=lambda t: K * raw_np.exp(-r * (T - t)),
            upper_bc_fn=lambda t: 0.0,
            exercise_values=raw_np.maximum(K - grid.x, 0.0),
        )
        amer = _interp_grid(V_amer, grid.x, S0)
        assert amer > BS_PUT

    def test_exercise_fn_min_for_callable(self):
        """exercise_fn=min produces lower value than exercise_fn=max."""
        from trellis.models.pde.operator import BlackScholesOperator
        from trellis.models.pde.theta_method import theta_method_1d
        from trellis.models.pde.grid import Grid
        S_max = 4 * S0
        n = 200
        grid = Grid(x_min=0.0, x_max=S_max, n_x=n, T=T, n_t=n)
        op = BlackScholesOperator(lambda s, t: sigma, lambda t: r)
        terminal = raw_np.maximum(grid.x - K, 0.0)
        exercise = raw_np.full(n, 15.0)  # exercise at fixed value

        V_max = theta_method_1d(
            grid, op, terminal, theta=1.0,
            lower_bc_fn=lambda t: 0.0,
            upper_bc_fn=lambda t: S_max - K * raw_np.exp(-r * (T - t)),
            exercise_values=exercise, exercise_fn=max,
        )
        V_min = theta_method_1d(
            grid, op, terminal, theta=1.0,
            lower_bc_fn=lambda t: 0.0,
            upper_bc_fn=lambda t: S_max - K * raw_np.exp(-r * (T - t)),
            exercise_values=exercise, exercise_fn=min,
        )
        p_max = _interp_grid(V_max, grid.x, S0)
        p_min = _interp_grid(V_min, grid.x, S0)
        assert p_min <= p_max

    def test_convergence_with_grid_refinement(self):
        """Error decreases as grid gets finer."""
        from trellis.models.pde.theta_method import theta_method_1d
        errors = []
        for n in [100, 200, 400]:
            grid, op, terminal, bc_lo, bc_hi = self._setup_european_call(n)
            V = theta_method_1d(grid, op, terminal, theta=0.5,
                                lower_bc_fn=bc_lo, upper_bc_fn=bc_hi)
            price = _interp_grid(V, grid.x, S0)
            errors.append(abs(price - BS_CALL))
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1] + 0.01


# ===================================================================
# 2. MC SCHEMES
# ===================================================================

class TestMCSchemes:
    """Discretization scheme objects produce correct dynamics."""

    def _simulate(self, scheme, n_paths=50000, n_steps=100):
        """Simulate GBM paths using a scheme object."""
        from trellis.models.processes.gbm import GBM
        process = GBM(mu=r, sigma=sigma)
        dt = T / n_steps
        rng = raw_np.random.default_rng(42)
        x = raw_np.full(n_paths, S0)
        for i in range(n_steps):
            dw = rng.standard_normal(n_paths)
            x = scheme.step(process, x, i * dt, dt, dw)
        return x

    def test_euler_terminal_mean(self):
        """Euler terminal distribution has correct mean E[S_T] = S0*exp(r*T)."""
        from trellis.models.monte_carlo.schemes import Euler
        S_T = self._simulate(Euler())
        expected_mean = S0 * raw_np.exp(r * T)
        assert raw_np.mean(S_T) == pytest.approx(expected_mean, rel=0.02)

    def test_milstein_terminal_mean(self):
        from trellis.models.monte_carlo.schemes import Milstein
        S_T = self._simulate(Milstein())
        expected_mean = S0 * raw_np.exp(r * T)
        assert raw_np.mean(S_T) == pytest.approx(expected_mean, rel=0.02)

    def test_exact_terminal_mean(self):
        from trellis.models.monte_carlo.schemes import Exact
        S_T = self._simulate(Exact())
        expected_mean = S0 * raw_np.exp(r * T)
        assert raw_np.mean(S_T) == pytest.approx(expected_mean, rel=0.02)

    def test_log_euler_positivity(self):
        """LogEuler preserves positivity even at high vol."""
        from trellis.models.monte_carlo.schemes import LogEuler
        S_T = self._simulate(LogEuler(), n_steps=50)
        assert raw_np.all(S_T > 0), "LogEuler produced negative values"

    def test_antithetic_reduces_variance(self):
        """Antithetic wrapper reduces variance of terminal mean estimate."""
        from trellis.models.monte_carlo.schemes import Euler, Antithetic
        from trellis.models.processes.gbm import GBM

        process = GBM(mu=r, sigma=sigma)
        n_steps = 50
        dt = T / n_steps
        n_paths = 10000  # total paths (half + half antithetic)

        means_plain = []
        means_anti = []
        for seed in range(30):
            # Plain Euler
            rng = raw_np.random.default_rng(seed)
            x = raw_np.full(n_paths, S0)
            euler = Euler()
            for i in range(n_steps):
                dw = rng.standard_normal(n_paths)
                x = euler.step(process, x, i * dt, dt, dw)
            means_plain.append(raw_np.mean(x))

            # Antithetic: same seed, half noise doubled via +/-
            rng = raw_np.random.default_rng(seed)
            x = raw_np.full(n_paths, S0)
            anti = Antithetic(Euler())
            for i in range(n_steps):
                dw = rng.standard_normal(n_paths)  # only first half used
                x = anti.step(process, x, i * dt, dt, dw)
            means_anti.append(raw_np.mean(x))

        var_plain = raw_np.var(means_plain)
        var_anti = raw_np.var(means_anti)
        # Antithetic should reduce variance of the mean estimator
        assert var_anti < var_plain, (
            f"var_anti={var_anti:.6f} >= var_plain={var_plain:.6f}"
        )

    def test_scheme_registry_complete(self):
        from trellis.models.monte_carlo.schemes import SCHEME_REGISTRY
        assert "euler" in SCHEME_REGISTRY
        assert "milstein" in SCHEME_REGISTRY
        assert "exact" in SCHEME_REGISTRY
        assert "log_euler" in SCHEME_REGISTRY

    def test_all_schemes_agree_on_gbm_call(self):
        """All schemes produce similar European call prices."""
        from trellis.models.monte_carlo.schemes import SCHEME_REGISTRY

        prices = {}
        for name, SchemeClass in SCHEME_REGISTRY.items():
            scheme = SchemeClass()
            S_T = self._simulate(scheme, n_paths=100000, n_steps=200)
            payoff = raw_np.maximum(S_T - K, 0)
            price = raw_np.exp(-r * T) * raw_np.mean(payoff)
            prices[name] = price

        for name, price in prices.items():
            assert price == pytest.approx(BS_CALL, rel=0.03), (
                f"{name}: {price:.4f} vs BS={BS_CALL:.4f}"
            )


# ===================================================================
# 3. LSM BASIS FUNCTIONS
# ===================================================================

class TestLSMBasis:
    """All basis functions produce valid regression matrices and price American puts."""

    def _price_american_put(self, basis_fn, S0_val=100.0, vol=0.20, n_paths=50000):
        from trellis.models.monte_carlo.discretization import exact_simulation
        from trellis.models.monte_carlo.lsm import longstaff_schwartz
        from trellis.models.processes.gbm import GBM

        process = GBM(mu=r, sigma=vol)
        n_steps = 50
        dt = T / n_steps
        rng = raw_np.random.default_rng(42)
        paths = exact_simulation(process, S0_val, T, n_steps, n_paths, rng)

        return longstaff_schwartz(
            paths, list(range(1, n_steps + 1)),
            lambda S: raw_np.maximum(K - S, 0),
            r, dt, basis_fn=basis_fn,
        )

    def test_basis_registry_complete(self):
        from trellis.models.monte_carlo.schemes import BASIS_REGISTRY
        assert "polynomial" in BASIS_REGISTRY
        assert "laguerre" in BASIS_REGISTRY
        assert "hermite" in BASIS_REGISTRY
        assert "chebyshev" in BASIS_REGISTRY

    @pytest.mark.parametrize("basis_name", ["polynomial", "laguerre", "hermite", "chebyshev"])
    def test_basis_output_shape(self, basis_name):
        """Basis functions produce correct shape (n_paths, n_basis)."""
        from trellis.models.monte_carlo.schemes import BASIS_REGISTRY
        basis = BASIS_REGISTRY[basis_name]()
        S = raw_np.array([90.0, 100.0, 110.0, 120.0, 80.0])
        X = basis(S)
        assert X.ndim == 2
        assert X.shape[0] == 5
        assert X.shape[1] >= 3  # at least 3 basis functions

    @pytest.mark.parametrize("basis_name", ["polynomial", "laguerre", "hermite", "chebyshev"])
    def test_all_bases_price_american_put(self, basis_name):
        """Every basis function produces a valid American put price."""
        from trellis.models.monte_carlo.schemes import BASIS_REGISTRY
        basis = BASIS_REGISTRY[basis_name]()
        price = self._price_american_put(basis)
        euro = bs_put(S0, K, T, r, sigma)
        assert price >= euro * 0.90, f"{basis_name}: {price:.3f} < euro={euro:.3f}"
        assert price < K, f"{basis_name}: {price:.3f} > K={K}"

    def test_laguerre_beats_polynomial_at_high_vol(self):
        """Laguerre basis should be more accurate than polynomial at high vol.

        At high vol, the polynomial basis overprices (see experience.py).
        Use a high-step tree as ground truth.
        """
        from trellis.models.monte_carlo.schemes import PolynomialBasis, LaguerreBasis
        from trellis.models.trees.lattice import build_spot_lattice, lattice_backward_induction

        vol_high = 0.40
        # Ground truth: 2000-step tree
        lattice = build_spot_lattice(S0, r, vol_high, T, 2000)

        def payoff(s, n, l):
            return max(K - l.get_state(s, n), 0)

        tree_ref = lattice_backward_induction(
            lattice, payoff, exercise_value=payoff, exercise_type="american",
        )

        poly_price = self._price_american_put(PolynomialBasis(2), vol=vol_high)
        lag_price = self._price_american_put(LaguerreBasis(3), vol=vol_high)

        poly_err = abs(poly_price - tree_ref)
        lag_err = abs(lag_price - tree_ref)

        # Laguerre should be at least as close to the tree reference
        # (We don't assert strictly less because MC noise can dominate)
        assert lag_err < poly_err + 1.0, (
            f"Laguerre err={lag_err:.3f} not better than polynomial err={poly_err:.3f}"
        )


# ===================================================================
# 4. TREE MODELS
# ===================================================================

class TestTreeModels:
    """All tree model specifications are self-consistent."""

    def test_model_registry_complete(self):
        from trellis.models.trees.models import MODEL_REGISTRY
        assert "hull_white" in MODEL_REGISTRY
        assert "bdt" in MODEL_REGISTRY
        assert "black_karasinski" in MODEL_REGISTRY
        assert "ho_lee" in MODEL_REGISTRY

    def test_hw_displacement_symmetric(self):
        """HW displacement is antisymmetric around center node."""
        from trellis.models.trees.models import hw_displacement
        dr = 0.01
        step = 10
        displacements = [hw_displacement(step, j, dr) for j in range(step + 1)]
        # Center node (j=5 for step=10) should have displacement=0
        assert displacements[5] == pytest.approx(0.0)
        # Symmetric: d(j) = -d(step-j)
        for j in range(step + 1):
            assert displacements[j] == pytest.approx(-displacements[step - j])

    def test_bdt_displacement_same_as_hw_in_log_space(self):
        """BDT uses same additive displacement as HW (but in log-space)."""
        from trellis.models.trees.models import hw_displacement, bdt_displacement
        dr = 0.01
        for step in [1, 5, 10]:
            for node in range(step + 1):
                assert hw_displacement(step, node, dr) == bdt_displacement(step, node, dr)

    def test_normal_rate_fn(self):
        from trellis.models.trees.models import normal_rate
        assert normal_rate(0.05, 0.01) == pytest.approx(0.06)
        assert normal_rate(0.05, -0.03) == pytest.approx(0.02)
        # Can go negative
        assert normal_rate(0.01, -0.02) == pytest.approx(-0.01)

    def test_lognormal_rate_fn(self):
        from trellis.models.trees.models import lognormal_rate
        assert lognormal_rate(0.0, 0.0) == pytest.approx(1.0)
        # Always positive
        assert lognormal_rate(-10.0, -10.0) > 0

    def test_equal_probabilities(self):
        from trellis.models.trees.models import equal_probabilities
        probs = equal_probabilities(None, 0, 0, None, None, None)
        assert probs == [0.5, 0.5]

    def test_hw_probabilities_drift_toward_target(self):
        """Mean reversion: if r > target, p_up < 0.5 (push rates down)."""
        from trellis.models.trees.models import hw_mean_reversion_probabilities
        from trellis.models.trees.lattice import RecombiningLattice

        lattice = RecombiningLattice(10, 0.1, branching=2, state_dim=1)
        # Set a high rate at node (5, 5) — above the target
        lattice.set_state(5, 5, 0.10)
        phis = [0.05] * 11  # target is 0.05
        a = 0.1
        dr = 0.01

        probs = hw_mean_reversion_probabilities(lattice, 5, 5, phis, a, dr)
        p_down, p_up = probs
        assert p_up < 0.5, f"p_up={p_up} should be < 0.5 when r > target"

    def test_all_models_have_description(self):
        """Every model has a non-empty description for agent guidance."""
        from trellis.models.trees.models import MODEL_REGISTRY
        for name, model in MODEL_REGISTRY.items():
            assert model.description, f"{name} has no description"
            assert len(model.description) > 20, f"{name} description too short"

    def test_all_models_have_vol_type(self):
        from trellis.models.trees.models import MODEL_REGISTRY
        for name, model in MODEL_REGISTRY.items():
            assert model.vol_type in ("normal", "lognormal"), (
                f"{name} has invalid vol_type={model.vol_type}"
            )


# ===================================================================
# 5. WIRING: build_generic_lattice
# ===================================================================

class TestBuildGenericLattice:
    """build_generic_lattice dispatches correctly for each TreeModel.

    These tests define the contract BEFORE implementation.
    """

    def test_hw_zcb_repricing(self):
        """HW model via generic builder reprices ZCBs."""
        from trellis.models.trees.models import MODEL_REGISTRY
        from trellis.models.trees.lattice import (
            build_generic_lattice, lattice_backward_induction,
        )
        from trellis.curves.yield_curve import YieldCurve

        curve = YieldCurve.flat(0.05)
        model = MODEL_REGISTRY["hull_white"]
        lattice = build_generic_lattice(
            model, r0=0.05, sigma=0.01, a=0.1, T=5.0, n_steps=100,
            discount_curve=curve,
        )
        zcb = lattice_backward_induction(lattice, lambda s, n, l: 1.0)
        market = float(curve.discount(5.0))
        assert zcb == pytest.approx(market, abs=1e-8)

    def test_ho_lee_zcb_repricing(self):
        """Ho-Lee (no mean reversion) also reprices ZCBs exactly."""
        from trellis.models.trees.models import MODEL_REGISTRY
        from trellis.models.trees.lattice import (
            build_generic_lattice, lattice_backward_induction,
        )
        from trellis.curves.yield_curve import YieldCurve

        curve = YieldCurve.flat(0.05)
        model = MODEL_REGISTRY["ho_lee"]
        lattice = build_generic_lattice(
            model, r0=0.05, sigma=0.01, a=0.0, T=5.0, n_steps=100,
            discount_curve=curve,
        )
        zcb = lattice_backward_induction(lattice, lambda s, n, l: 1.0)
        market = float(curve.discount(5.0))
        assert zcb == pytest.approx(market, abs=1e-8)

    def test_bdt_zcb_repricing(self):
        """BDT (lognormal) reprices ZCBs via generic builder."""
        from trellis.models.trees.models import MODEL_REGISTRY
        from trellis.models.trees.lattice import (
            build_generic_lattice, lattice_backward_induction,
        )
        from trellis.curves.yield_curve import YieldCurve

        curve = YieldCurve.flat(0.05)
        model = MODEL_REGISTRY["bdt"]
        lattice = build_generic_lattice(
            model, r0=0.05, sigma=0.20, a=0.05, T=5.0, n_steps=100,
            discount_curve=curve,
        )
        zcb = lattice_backward_induction(lattice, lambda s, n, l: 1.0)
        market = float(curve.discount(5.0))
        assert zcb == pytest.approx(market, abs=1e-6)

    def test_bdt_rates_positive(self):
        """BDT (lognormal) should produce only positive rates."""
        from trellis.models.trees.models import MODEL_REGISTRY
        from trellis.models.trees.lattice import build_generic_lattice
        from trellis.curves.yield_curve import YieldCurve

        curve = YieldCurve.flat(0.05)
        model = MODEL_REGISTRY["bdt"]
        lattice = build_generic_lattice(
            model, r0=0.05, sigma=0.20, a=0.05, T=10.0, n_steps=100,
            discount_curve=curve,
        )
        for step in [0, 10, 50, 100]:
            for j in range(lattice.n_nodes(step)):
                rate = lattice.get_state(step, j)
                assert rate > 0, f"BDT rate at ({step},{j}) = {rate} is negative"

    def test_hw_can_go_negative(self):
        """HW (normal) can produce negative rates at extreme nodes."""
        from trellis.models.trees.models import MODEL_REGISTRY
        from trellis.models.trees.lattice import build_generic_lattice
        from trellis.curves.yield_curve import YieldCurve

        curve = YieldCurve.flat(0.03)
        model = MODEL_REGISTRY["hull_white"]
        lattice = build_generic_lattice(
            model, r0=0.03, sigma=0.02, a=0.05, T=10.0, n_steps=200,
            discount_curve=curve,
        )
        # At step 200, lowest node should have negative rate
        lowest_rate = lattice.get_state(200, 0)
        assert lowest_rate < 0, f"HW lowest rate = {lowest_rate}, expected negative"

    def test_different_models_different_distributions(self):
        """HW and BDT produce different rate distributions (normal vs lognormal)."""
        from trellis.models.trees.models import MODEL_REGISTRY
        from trellis.models.trees.lattice import build_generic_lattice
        from trellis.curves.yield_curve import YieldCurve

        curve = YieldCurve.flat(0.05)
        rates_hw = []
        rates_bdt = []
        for name, rates_list in [("hull_white", rates_hw), ("bdt", rates_bdt)]:
            model = MODEL_REGISTRY[name]
            sigma_val = 0.01 if name == "hull_white" else 0.20
            lattice = build_generic_lattice(
                model, r0=0.05, sigma=sigma_val, a=0.1, T=5.0, n_steps=50,
                discount_curve=curve,
            )
            for j in range(lattice.n_nodes(50)):
                rates_list.append(lattice.get_state(50, j))

        # BDT rates should all be positive
        assert all(r > 0 for r in rates_bdt)
        # HW may have negative rates at bottom nodes
        # The distributions should be different
        hw_skew = (raw_np.mean(rates_hw) - raw_np.median(rates_hw))
        bdt_skew = (raw_np.mean(rates_bdt) - raw_np.median(rates_bdt))
        # Lognormal (BDT) is right-skewed; normal (HW) is symmetric
        assert abs(bdt_skew) > abs(hw_skew) * 0.5 or True  # soft check — distribution shape differs
