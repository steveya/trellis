"""Calibration and convergence tests for all numerical methods.

Each numerical method must pass:
1. Calibration: reprices its calibration instruments exactly
2. Convergence: refining discretization → analytical limit
3. Cross-method: different methods agree on the same instrument
4. Sensitivity: Greeks have correct sign and magnitude
"""

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.curves.yield_curve import YieldCurve
from trellis.models.trees.lattice import (
    RecombiningLattice,
    calibrate_lattice,
    lattice_backward_induction,
)
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.discretization import exact_simulation
from trellis.models.processes.gbm import GBM
from trellis.models.transforms.fft_pricer import fft_price
from trellis.models.transforms.cos_method import cos_price
from trellis.models.pde.grid import Grid
from trellis.models.pde.crank_nicolson import crank_nicolson_1d
from trellis.models.pde.operator import BlackScholesOperator
from trellis.models.pde.theta_method import theta_method_1d
from trellis.models.pde.psor import psor_1d
from tests.lattice_builders import build_equity_lattice, build_short_rate_lattice


# Common parameters
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


# ===================================================================
# 1. RATE TREE CALIBRATION
# ===================================================================

class TestRateTreeCalibration:
    """The calibrated rate tree must reprice ZCBs at all tenors."""

    @pytest.mark.parametrize("tenor", [1, 2, 5, 10])
    def test_zcb_repricing_flat_curve(self, tenor):
        """Tree reprices ZCBs on a flat curve to machine precision."""
        curve = YieldCurve.flat(0.05)
        n_steps = max(50, int(tenor * 20))
        sigma_hw = 0.01  # 1% HW vol
        lattice = build_short_rate_lattice(0.05, sigma_hw, 0.1, tenor, n_steps, discount_curve=curve)
        tree_zcb = lattice_backward_induction(lattice, lambda s, n, l: 1.0)
        market_zcb = float(curve.discount(tenor))
        assert tree_zcb == pytest.approx(market_zcb, abs=1e-10), (
            f"ZCB({tenor}Y): tree={tree_zcb:.10f}, market={market_zcb:.10f}"
        )

    @pytest.mark.parametrize("tenor", [1, 2, 5, 10])
    def test_zcb_repricing_upward_sloping(self, tenor):
        """Tree reprices ZCBs on a non-flat curve."""
        tenors = raw_np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        rates = raw_np.array([0.03, 0.032, 0.035, 0.04, 0.042, 0.045,
                               0.047, 0.05, 0.052, 0.053])
        curve = YieldCurve(tenors, rates)
        r0 = float(curve.zero_rate(0.01))
        n_steps = max(50, int(tenor * 20))
        lattice = build_short_rate_lattice(r0, 0.01, 0.1, tenor, n_steps, discount_curve=curve)
        tree_zcb = lattice_backward_induction(lattice, lambda s, n, l: 1.0)
        market_zcb = float(curve.discount(tenor))
        assert tree_zcb == pytest.approx(market_zcb, abs=1e-8), (
            f"ZCB({tenor}Y): tree={tree_zcb:.10f}, market={market_zcb:.10f}"
        )

    def test_straight_bond_on_tree_matches_analytical(self):
        """A coupon bond priced on the tree matches discounted-cashflow PV."""
        curve = YieldCurve.flat(0.05)
        T_bond = 5.0
        n_steps = 100
        coupon = 0.06  # above par
        notional = 100.0
        dt = T_bond / n_steps

        lattice = build_short_rate_lattice(0.05, 0.01, 0.1, T_bond, n_steps, discount_curve=curve)
        coupon_per_step = notional * coupon * dt

        def payoff(step, node, lat):
            return notional + coupon_per_step

        def cashflow(step, node, lat):
            return coupon_per_step

        tree_price = lattice_backward_induction(
            lattice, payoff, cashflow_at_node=cashflow,
        )

        # Analytical: PV of coupons + PV of principal
        analytical_pv = 0.0
        for i in range(1, n_steps + 1):
            t = i * dt
            analytical_pv += coupon_per_step * float(curve.discount(t))
        analytical_pv += notional * float(curve.discount(T_bond))

        # Tree coupon spacing is approximate, so allow 1% tolerance
        assert tree_price == pytest.approx(analytical_pv, rel=0.01), (
            f"tree={tree_price:.4f}, analytical={analytical_pv:.4f}"
        )

    def test_calibration_with_high_vol(self):
        """Calibration remains stable with high rate vol."""
        curve = YieldCurve.flat(0.05)
        sigma_hw = 0.03  # 3% — quite high
        lattice = build_short_rate_lattice(0.05, sigma_hw, 0.05, 10.0, 200, discount_curve=curve)
        tree_zcb = lattice_backward_induction(lattice, lambda s, n, l: 1.0)
        market_zcb = float(curve.discount(10.0))
        assert tree_zcb == pytest.approx(market_zcb, abs=1e-8)

    def test_calibration_with_zero_vol(self):
        """Zero vol → all rates equal to forward rates."""
        curve = YieldCurve.flat(0.05)
        lattice = build_short_rate_lattice(0.05, 1e-10, 0.1, 5.0, 100, discount_curve=curve)
        # At any step, all nodes should have nearly the same rate
        for step in [10, 50, 100]:
            n = lattice.n_nodes(step)
            rates = [lattice.get_state(step, j) for j in range(n)]
            assert raw_np.std(rates) < 1e-6


# ===================================================================
# 2. CALLABLE BOND ON RATE TREE
# ===================================================================

class TestCallableBondTree:
    """Callable bond pricing invariants on rate trees."""

    def _price_callable(self, rate, vol_hw, n_steps=200):
        """Helper: price a callable bond at given rate and HW vol."""
        curve = YieldCurve.flat(rate)
        T = 10.0
        notional = 100.0
        coupon = 0.05
        call_price = 100.0
        dt = T / n_steps

        lattice = build_short_rate_lattice(rate, vol_hw, 0.1, T, n_steps, discount_curve=curve)

        exercise_steps = [int(round(y / dt)) for y in [3, 5, 7]]
        exercise_steps = [s for s in exercise_steps if 0 < s < n_steps]

        coupon_per_step = notional * coupon * dt

        def payoff(step, node, lat):
            return notional + coupon_per_step

        def exercise(step, node, lat):
            return call_price + coupon_per_step

        def cashflow(step, node, lat):
            return coupon_per_step

        # Callable: exercise_fn=min (issuer minimizes liability)
        return lattice_backward_induction(
            lattice, payoff, exercise,
            exercise_type="bermudan", exercise_steps=exercise_steps,
            cashflow_at_node=cashflow, exercise_fn=min,
        )

    def _price_straight(self, rate, n_steps=200):
        """Helper: straight bond analytical PV."""
        curve = YieldCurve.flat(rate)
        T = 10.0
        notional = 100.0
        coupon = 0.05
        dt = T / n_steps
        pv = 0.0
        for i in range(1, n_steps + 1):
            pv += notional * coupon * dt * float(curve.discount(i * dt))
        pv += notional * float(curve.discount(T))
        return pv

    def test_callable_leq_straight(self):
        """Callable bond ≤ straight bond at all rate levels."""
        for rate in [0.03, 0.04, 0.05, 0.06, 0.07]:
            callable_pv = self._price_callable(rate, 0.01)
            straight_pv = self._price_straight(rate)
            assert callable_pv <= straight_pv + 0.5, (
                f"rate={rate}: callable={callable_pv:.2f} > straight={straight_pv:.2f}"
            )

    def test_higher_vol_lower_callable(self):
        """Higher vol → lower callable bond price (bigger call option value)."""
        p_low = self._price_callable(0.05, 0.005, n_steps=100)
        p_high = self._price_callable(0.05, 0.02, n_steps=100)
        assert p_high < p_low, (
            f"Expected p_high < p_low, got {p_high:.4f} >= {p_low:.4f}"
        )

    def test_callable_with_min_vs_max(self):
        """min (callable) < max (puttable) for same bond."""
        curve = YieldCurve.flat(0.05)
        T, n_steps = 10.0, 100
        dt = T / n_steps
        notional, coupon = 100.0, 0.05
        coupon_per_step = notional * coupon * dt

        lattice = build_short_rate_lattice(0.05, 0.01, 0.1, T, n_steps, discount_curve=curve)
        exercise_steps = [30, 50, 70]

        def payoff(s, n, l):
            return notional + coupon_per_step

        def exercise(s, n, l):
            return notional + coupon_per_step

        def cf(s, n, l):
            return coupon_per_step

        p_callable = lattice_backward_induction(
            lattice, payoff, exercise,
            exercise_type="bermudan", exercise_steps=exercise_steps,
            cashflow_at_node=cf, exercise_fn=min,
        )
        p_puttable = lattice_backward_induction(
            lattice, payoff, exercise,
            exercise_type="bermudan", exercise_steps=exercise_steps,
            cashflow_at_node=cf, exercise_fn=max,
        )
        assert p_callable < p_puttable


# ===================================================================
# 3. SPOT LATTICE CONVERGENCE
# ===================================================================

class TestSpotLatticeConvergence:
    """CRR spot lattice convergence across strikes and maturities."""

    @pytest.mark.parametrize("strike", [80, 100, 120])
    def test_call_at_various_strikes(self, strike):
        lattice = build_equity_lattice(S0, r, sigma, T, 300)

        def payoff(step, node, lat):
            return max(lat.get_state(step, node) - strike, 0)

        price = lattice_backward_induction(lattice, payoff)
        ref = bs_call(S0, strike, T, r, sigma)
        assert price == pytest.approx(ref, rel=0.01)

    @pytest.mark.parametrize("maturity", [0.25, 0.5, 1.0, 2.0])
    def test_call_at_various_maturities(self, maturity):
        n = max(100, int(maturity * 200))
        lattice = build_equity_lattice(S0, r, sigma, maturity, n)

        def payoff(step, node, lat):
            return max(lat.get_state(step, node) - K, 0)

        price = lattice_backward_induction(lattice, payoff)
        ref = bs_call(S0, K, maturity, r, sigma)
        assert price == pytest.approx(ref, rel=0.02)

    def test_put_call_parity(self):
        """C - P = S0 - K*exp(-rT) on the lattice."""
        lattice = build_equity_lattice(S0, r, sigma, T, 300)

        def call_payoff(s, n, l):
            return max(l.get_state(s, n) - K, 0)

        def put_payoff(s, n, l):
            return max(K - l.get_state(s, n), 0)

        C = lattice_backward_induction(lattice, call_payoff)
        P = lattice_backward_induction(lattice, put_payoff)
        parity = C - P
        expected = S0 - K * raw_np.exp(-r * T)
        assert parity == pytest.approx(expected, rel=0.01)

    def test_convergence_rate(self):
        """Error decreases as n_steps increases."""
        ref = bs_call(S0, K, T, r, sigma)
        errors = []
        for n in [50, 100, 200, 400]:
            lattice = build_equity_lattice(S0, r, sigma, T, n)

            def payoff(s, nd, l):
                return max(l.get_state(s, nd) - K, 0)

            price = lattice_backward_induction(lattice, payoff)
            errors.append(abs(price - ref))
        # Each doubling should roughly halve the error (O(1/n) convergence)
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1]


# ===================================================================
# 4. PDE CONVERGENCE
# ===================================================================

class TestPDEConvergence:
    """PDE solvers must converge to BS for European options."""

    def _pde_european_call(self, n_x, n_t):
        """Price European call via theta-method (Crank-Nicolson, theta=0.5)."""
        S_max = 4 * S0
        grid = Grid(x_min=0.0, x_max=S_max, n_x=n_x, T=T, n_t=n_t)
        S = grid.x
        terminal = raw_np.maximum(S - K, 0.0)
        op = BlackScholesOperator(lambda s, t: sigma, lambda t: r)
        V = theta_method_1d(
            grid,
            op,
            terminal,
            theta=0.5,
            lower_bc_fn=lambda t: 0.0,
            upper_bc_fn=lambda t: S_max - K * raw_np.exp(-r * (T - t)),
        )
        idx = raw_np.searchsorted(S, S0)
        idx = min(idx, n_x - 2)
        w = (S0 - S[idx]) / (S[idx + 1] - S[idx])
        return float(V[idx] * (1 - w) + V[idx + 1] * w)

    def test_cn_converges_to_bs(self):
        """Crank-Nicolson European call converges to BS with grid refinement."""
        price = self._pde_european_call(500, 500)
        assert price == pytest.approx(BS_CALL, rel=0.02), (
            f"CN={price:.4f}, BS={BS_CALL:.4f}"
        )

    def test_cn_convergence_improves(self):
        """Finer grid → smaller error."""
        ref = BS_CALL
        errors = []
        for n in [100, 200, 400]:
            price = self._pde_european_call(n, n)
            errors.append(abs(price - ref))
        # Monotone decrease (may not be strict due to oscillation)
        assert errors[-1] < errors[0]

    def test_psor_american_put_converges(self):
        """PSOR American put ≥ European put and converges to reasonable value."""
        S_max = 4 * S0
        n_x, n_t = 400, 400
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
        amer = float(V[idx] * (1 - w) + V[idx + 1] * w)

        assert amer >= BS_PUT - 0.5
        assert amer > BS_PUT  # early exercise premium


# ===================================================================
# 5. MONTE CARLO CALIBRATION TESTS
# ===================================================================

class TestMonteCarloCalibration:
    """MC must reprice vanilla European options (its calibration instruments)."""

    def test_call_within_confidence_interval(self):
        """MC call price within 2 SE of BS."""
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=200000, n_steps=1,
                                   seed=42, method="exact")

        def payoff_fn(paths):
            return raw_np.maximum(paths[:, -1] - K, 0)

        result = engine.price(S0, T, payoff_fn, discount_rate=r)
        assert abs(result["price"] - BS_CALL) < 2 * result["std_error"]

    def test_put_within_confidence_interval(self):
        """MC put price within 2 SE of BS."""
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=200000, n_steps=1,
                                   seed=42, method="exact")

        def payoff_fn(paths):
            return raw_np.maximum(K - paths[:, -1], 0)

        result = engine.price(S0, T, payoff_fn, discount_rate=r)
        assert abs(result["price"] - BS_PUT) < 2 * result["std_error"]

    def test_mc_put_call_parity(self):
        """C_MC - P_MC ≈ S0 - K*exp(-rT)."""
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=200000, n_steps=1,
                                   seed=42, method="exact")

        def call_pf(paths):
            return raw_np.maximum(paths[:, -1] - K, 0)

        def put_pf(paths):
            return raw_np.maximum(K - paths[:, -1], 0)

        c = engine.price(S0, T, call_pf, discount_rate=r)["price"]
        p = engine.price(S0, T, put_pf, discount_rate=r)["price"]
        expected = S0 - K * raw_np.exp(-r * T)
        assert (c - p) == pytest.approx(expected, abs=0.5)

    @pytest.mark.parametrize("strike", [80, 100, 120])
    def test_call_otm_itm(self, strike):
        """MC works for ITM, ATM, OTM calls."""
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=100000, n_steps=1,
                                   seed=42, method="exact")

        def payoff_fn(paths):
            return raw_np.maximum(paths[:, -1] - strike, 0)

        result = engine.price(S0, T, payoff_fn, discount_rate=r)
        ref = bs_call(S0, strike, T, r, sigma)
        assert abs(result["price"] - ref) < 3 * result["std_error"]


# ===================================================================
# 6. FFT/COS CALIBRATION
# ===================================================================

class TestTransformCalibration:
    """FFT and COS methods reprice BS exactly."""

    @staticmethod
    def _gbm_char_fn_fft(u):
        """Char fn of log(S_T) — used by FFT."""
        return raw_np.exp(
            1j * u * (raw_np.log(S0) + (r - 0.5 * sigma**2) * T)
            - 0.5 * sigma**2 * T * u**2
        )

    @staticmethod
    def _gbm_char_fn_cos(u):
        """Char fn of log(S_T / S0) — used by COS."""
        return raw_np.exp(
            1j * u * (r - 0.5 * sigma**2) * T
            - 0.5 * sigma**2 * T * u**2
        )

    @pytest.mark.parametrize("strike", [80, 100, 120])
    def test_fft_call(self, strike):
        price = fft_price(self._gbm_char_fn_fft, S0, strike, T, r)
        ref = bs_call(S0, strike, T, r, sigma)
        assert price == pytest.approx(ref, rel=0.02)

    @pytest.mark.parametrize("strike", [80, 100, 120])
    def test_cos_call(self, strike):
        price = cos_price(self._gbm_char_fn_cos, S0, strike, T, r, option_type="call")
        ref = bs_call(S0, strike, T, r, sigma)
        assert price == pytest.approx(ref, rel=0.02)


# ===================================================================
# 7. CROSS-METHOD CONSISTENCY
# ===================================================================

class TestCrossMethodConsistency:
    """Different numerical methods must agree on the same instrument."""

    def test_european_call_all_methods(self):
        """Tree, MC, FFT, COS all agree within 2% for ATM European call."""
        ref = BS_CALL

        # Tree
        lattice = build_equity_lattice(S0, r, sigma, T, 300)

        def payoff(s, n, l):
            return max(l.get_state(s, n) - K, 0)

        tree_price = lattice_backward_induction(lattice, payoff)

        # MC
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=100000, n_steps=1,
                                   seed=42, method="exact")
        mc_price = engine.price(S0, T,
                                 lambda p: raw_np.maximum(p[:, -1] - K, 0),
                                 discount_rate=r)["price"]

        # FFT (char fn of log(S_T))
        def char_fn_fft(u):
            return raw_np.exp(
                1j * u * (raw_np.log(S0) + (r - 0.5 * sigma**2) * T)
                - 0.5 * sigma**2 * T * u**2
            )

        fft_p = fft_price(char_fn_fft, S0, K, T, r)

        # COS (char fn of log(S_T / S0))
        def char_fn_cos(u):
            return raw_np.exp(
                1j * u * (r - 0.5 * sigma**2) * T
                - 0.5 * sigma**2 * T * u**2
            )

        cos_p = cos_price(char_fn_cos, S0, K, T, r, option_type="call")

        results = {
            "tree": tree_price, "mc": mc_price,
            "fft": fft_p, "cos": cos_p,
        }
        for name, price in results.items():
            assert price == pytest.approx(ref, rel=0.02), (
                f"{name}={price:.4f}, BS={ref:.4f}"
            )

    def test_european_put_tree_vs_mc(self):
        """Tree and MC agree on European put."""
        ref = BS_PUT

        lattice = build_equity_lattice(S0, r, sigma, T, 300)

        def payoff(s, n, l):
            return max(K - l.get_state(s, n), 0)

        tree_price = lattice_backward_induction(lattice, payoff)

        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=100000, n_steps=1,
                                   seed=42, method="exact")
        mc_price = engine.price(S0, T,
                                 lambda p: raw_np.maximum(K - p[:, -1], 0),
                                 discount_rate=r)["price"]

        assert tree_price == pytest.approx(ref, rel=0.02)
        assert mc_price == pytest.approx(ref, rel=0.03)


# ===================================================================
# 8. COPULA CALIBRATION
# ===================================================================

class TestCopulaCalibration:
    """Factor copula loss distribution must be self-consistent."""

    def test_expected_loss_bounds(self):
        """E[losses] for independent names = n * marginal_prob."""
        from trellis.models.copulas.factor import FactorCopula

        n_names = 100
        marginal = 0.05
        copula = FactorCopula(n_names=n_names, correlation=0.0)
        losses, probs = copula.loss_distribution(marginal)

        # Expected number of defaults
        el = sum(l * p for l, p in zip(losses, probs))
        expected = n_names * marginal
        assert el == pytest.approx(expected, rel=0.05)

    def test_higher_correlation_fatter_tails(self):
        """Higher correlation → more probability in tails (0 or many defaults)."""
        from trellis.models.copulas.factor import FactorCopula

        n_names = 100
        marginal = 0.05

        copula_low = FactorCopula(n_names=n_names, correlation=0.1)
        copula_high = FactorCopula(n_names=n_names, correlation=0.5)

        _, probs_low = copula_low.loss_distribution(marginal)
        _, probs_high = copula_high.loss_distribution(marginal)

        # Variance should be higher with more correlation
        var_low = sum((l / n_names)**2 * p for l, p
                      in zip(range(n_names + 1), probs_low))
        var_high = sum((l / n_names)**2 * p for l, p
                       in zip(range(n_names + 1), probs_high))
        assert var_high > var_low

    def test_probabilities_sum_to_one(self):
        from trellis.models.copulas.factor import FactorCopula

        copula = FactorCopula(n_names=50, correlation=0.3)
        _, probs = copula.loss_distribution(0.05)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)


# ===================================================================
# 9. SENSITIVITY SIGN TESTS
# ===================================================================

class TestGreeksSigns:
    """Greeks must have the correct sign for standard instruments."""

    def test_call_delta_positive(self):
        """Call delta > 0: price increases with spot."""
        lattice_lo = build_equity_lattice(S0 - 1, r, sigma, T, 200)
        lattice_hi = build_equity_lattice(S0 + 1, r, sigma, T, 200)

        def payoff(s, n, l):
            return max(l.get_state(s, n) - K, 0)

        p_lo = lattice_backward_induction(lattice_lo, payoff)
        p_hi = lattice_backward_induction(lattice_hi, payoff)
        assert p_hi > p_lo

    def test_put_delta_negative(self):
        """Put delta < 0: price decreases with spot."""
        lattice_lo = build_equity_lattice(S0 - 1, r, sigma, T, 200)
        lattice_hi = build_equity_lattice(S0 + 1, r, sigma, T, 200)

        def payoff(s, n, l):
            return max(K - l.get_state(s, n), 0)

        p_lo = lattice_backward_induction(lattice_lo, payoff)
        p_hi = lattice_backward_induction(lattice_hi, payoff)
        assert p_lo > p_hi

    def test_option_vega_positive(self):
        """Call vega > 0: price increases with vol."""
        def price_at_vol(v):
            lat = build_equity_lattice(S0, r, v, T, 200)

            def payoff(s, n, l):
                return max(l.get_state(s, n) - K, 0)

            return lattice_backward_induction(lat, payoff)

        p_low = price_at_vol(0.15)
        p_high = price_at_vol(0.25)
        assert p_high > p_low

    def test_bond_rate_sensitivity_negative(self):
        """Bond value decreases when rates increase."""
        T_bond, n_steps = 5.0, 100
        notional, coupon = 100.0, 0.05

        def price_bond(rate):
            curve = YieldCurve.flat(rate)
            dt = T_bond / n_steps
            lattice = build_short_rate_lattice(rate, 0.01, 0.1, T_bond, n_steps, discount_curve=curve)
            cpn = notional * coupon * dt

            def payoff(s, n, l):
                return notional + cpn

            def cf(s, n, l):
                return cpn

            return lattice_backward_induction(lattice, payoff, cashflow_at_node=cf)

        p_low = price_bond(0.04)
        p_high = price_bond(0.06)
        assert p_low > p_high  # higher rate → lower bond price
