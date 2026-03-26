"""T07: American put via PDE (PSOR) vs CRR tree vs LSM MC — three-way cross-check.

Parameters:
  S0=100, K=100, r=0.05, T=1.0
  Three vol levels: sigma = 0.20, 0.40, 0.60

Methods:
  A) CRR tree (2000 steps) — reference
  B) PSOR PDE via theta_method_1d with exercise_values
  C) LSM Monte Carlo with Laguerre basis, 100K paths

Cross-validation against QuantLib FD and binomial engines.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.backward_induction import backward_induction
from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import BlackScholesOperator
from trellis.models.pde.theta_method import theta_method_1d
from trellis.models.monte_carlo.lsm import longstaff_schwartz, laguerre_basis

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0 = 100.0
K = 100.0
R = 0.05
T = 1.0
VOLS = [0.20, 0.40, 0.60]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bs_put_price(S, K, r, sigma, T):
    """European Black-Scholes put price (lower bound for American put)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def crr_american_put(sigma, n_steps=2000):
    """Price American put via CRR binomial tree with backward induction."""
    tree = BinomialTree.crr(S0, T, n_steps, R, sigma)

    def payoff_fn(step, node):
        return max(K - tree.value_at(step, node), 0.0)

    def exercise_val_fn(step, node, t):
        return max(K - t.value_at(step, node), 0.0)

    return backward_induction(
        tree, payoff_fn,
        discount_rate=R,
        exercise_type="american",
        exercise_value_fn=exercise_val_fn,
    )


def pde_american_put(sigma, n_x=500, n_t=500, s_max_mult=4.0):
    """Price American put via PDE with PSOR (theta-method)."""
    s_max = s_max_mult * S0
    grid = Grid(x_min=0.0, x_max=s_max, n_x=n_x, T=T, n_t=n_t)
    op = BlackScholesOperator(
        sigma_fn=lambda S, t: sigma,
        r_fn=lambda t: R,
    )

    S_grid = grid.x
    terminal = np.maximum(K - S_grid, 0.0)
    exercise_vals = np.maximum(K - S_grid, 0.0)

    # Boundary conditions: at S=0, put = K*exp(-r*tau); at S=S_max, put = 0
    lower_bc = lambda t: K * np.exp(-R * (T - t))
    upper_bc = lambda t: 0.0

    V = theta_method_1d(
        grid, op, terminal, theta=0.5,
        lower_bc_fn=lower_bc, upper_bc_fn=upper_bc,
        exercise_values=exercise_vals,
        exercise_fn=max,
    )

    # Interpolate to S0
    idx = np.searchsorted(S_grid, S0) - 1
    idx = max(0, min(idx, len(S_grid) - 2))
    w = (S0 - S_grid[idx]) / (S_grid[idx + 1] - S_grid[idx])
    return V[idx] * (1 - w) + V[idx + 1] * w


def lsm_american_put(sigma, n_paths=100_000, n_steps=100, seed=42):
    """Price American put via LSM Monte Carlo with exact GBM simulation."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Vectorised exact GBM simulation (risk-neutral drift = r)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for i in range(n_steps):
        dw = rng.standard_normal(n_paths)
        paths[:, i + 1] = paths[:, i] * np.exp(
            (R - 0.5 * sigma ** 2) * dt + sigma * sqrt_dt * dw
        )

    exercise_dates = list(range(1, n_steps + 1))

    def payoff_fn(S):
        return np.maximum(K - S, 0.0)

    # Normalised Laguerre basis for better conditioning
    def scaled_laguerre(S):
        x = S / K  # normalise around strike
        L0 = np.ones_like(x)
        L1 = 1 - x
        L2 = 0.5 * (x ** 2 - 4 * x + 2)
        return np.column_stack([L0, L1, L2])

    return longstaff_schwartz(
        paths, exercise_dates, payoff_fn,
        discount_rate=R, dt=dt,
        basis_fn=scaled_laguerre,
    )


# Precompute prices (cached at module level for test reuse)
_cache: dict[tuple[str, float], float] = {}


def _get_price(method: str, sigma: float) -> float:
    key = (method, sigma)
    if key not in _cache:
        if method == "crr":
            _cache[key] = crr_american_put(sigma)
        elif method == "pde":
            _cache[key] = pde_american_put(sigma)
        elif method == "lsm":
            _cache[key] = lsm_american_put(sigma)
        elif method == "bs_euro":
            _cache[key] = bs_put_price(S0, K, R, sigma, T)
    return _cache[key]


# ---------------------------------------------------------------------------
# Tests 1–3: Three methods agree within tolerance
# ---------------------------------------------------------------------------

class TestThreeWayAgreement:
    """Tests 1–3: CRR tree, PDE (PSOR), and LSM MC agree at each vol."""

    @pytest.mark.parametrize("sigma,tol", [
        (0.20, 0.01),
        (0.40, 0.01),
        (0.60, 0.02),
    ])
    def test_crr_vs_pde(self, sigma, tol):
        ref = _get_price("crr", sigma)
        pde = _get_price("pde", sigma)
        rel_err = abs(pde - ref) / ref
        assert rel_err < tol, (
            f"sigma={sigma}: PDE={pde:.4f} vs CRR={ref:.4f}, rel_err={rel_err:.4f}"
        )

    @pytest.mark.parametrize("sigma,tol", [
        (0.20, 0.01),
        (0.40, 0.01),
        (0.60, 0.02),
    ])
    def test_crr_vs_lsm(self, sigma, tol):
        ref = _get_price("crr", sigma)
        lsm = _get_price("lsm", sigma)
        rel_err = abs(lsm - ref) / ref
        assert rel_err < tol, (
            f"sigma={sigma}: LSM={lsm:.4f} vs CRR={ref:.4f}, rel_err={rel_err:.4f}"
        )

    @pytest.mark.parametrize("sigma,tol", [
        (0.20, 0.01),
        (0.40, 0.01),
        (0.60, 0.02),
    ])
    def test_pde_vs_lsm(self, sigma, tol):
        pde = _get_price("pde", sigma)
        lsm = _get_price("lsm", sigma)
        avg = 0.5 * (pde + lsm)
        rel_err = abs(pde - lsm) / avg
        assert rel_err < tol, (
            f"sigma={sigma}: PDE={pde:.4f} vs LSM={lsm:.4f}, rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4: All prices >= European BS put
# ---------------------------------------------------------------------------

class TestAmericanGeEuropean:
    """Test 4: American put price >= European BS put price."""

    @pytest.mark.parametrize("sigma", VOLS)
    def test_crr_ge_european(self, sigma):
        am = _get_price("crr", sigma)
        eu = _get_price("bs_euro", sigma)
        assert am >= eu - 1e-6, f"CRR {am:.4f} < European {eu:.4f}"

    @pytest.mark.parametrize("sigma", VOLS)
    def test_pde_ge_european(self, sigma):
        am = _get_price("pde", sigma)
        eu = _get_price("bs_euro", sigma)
        assert am >= eu - 1e-6, f"PDE {am:.4f} < European {eu:.4f}"

    @pytest.mark.parametrize("sigma", VOLS)
    def test_lsm_ge_european(self, sigma):
        am = _get_price("lsm", sigma)
        eu = _get_price("bs_euro", sigma)
        # LSM is a lower bound, so allow small tolerance
        assert am >= eu - 0.15, f"LSM {am:.4f} < European {eu:.4f} (minus tolerance)"


# ---------------------------------------------------------------------------
# Test 5: All prices >= intrinsic value max(K - S0, 0)
# ---------------------------------------------------------------------------

class TestGeIntrinsic:
    """Test 5: All prices >= max(K - S0, 0) = 0 for ATM."""

    @pytest.mark.parametrize("sigma", VOLS)
    @pytest.mark.parametrize("method", ["crr", "pde", "lsm"])
    def test_ge_intrinsic(self, sigma, method):
        price = _get_price(method, sigma)
        intrinsic = max(K - S0, 0.0)
        assert price >= intrinsic - 1e-6, (
            f"{method} sigma={sigma}: price={price:.4f} < intrinsic={intrinsic:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests 6–7: QuantLib cross-validation
# ---------------------------------------------------------------------------

class TestQuantLibCrossValidation:
    """Tests 6–7: Cross-validate trellis against QuantLib."""

    @staticmethod
    def _ql_setup(sigma):
        """Build QuantLib BS process and American put option."""
        ql = pytest.importorskip("QuantLib")

        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today
        maturity = today + ql.Period(1, ql.Years)

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, R, ql.Actual365Fixed())
        )
        div_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        )
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
        )

        bs_process = ql.BlackScholesMertonProcess(
            spot_handle, div_handle, rate_handle, vol_handle
        )

        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
        exercise = ql.AmericanExercise(today, maturity)
        option = ql.VanillaOption(payoff, exercise)
        return ql, bs_process, option

    @pytest.mark.parametrize("sigma", VOLS)
    def test_trellis_tree_vs_quantlib_tree(self, sigma):
        """Test 6: Trellis CRR tree vs QuantLib binomial tree within 0.5%."""
        ql, bs_process, option = self._ql_setup(sigma)

        engine = ql.BinomialVanillaEngine(bs_process, "crr", 2000)
        option.setPricingEngine(engine)
        ql_price = option.NPV()

        trellis_price = _get_price("crr", sigma)
        rel_err = abs(trellis_price - ql_price) / ql_price
        assert rel_err < 0.005, (
            f"sigma={sigma}: Trellis CRR={trellis_price:.4f} vs "
            f"QuantLib CRR={ql_price:.4f}, rel_err={rel_err:.4f}"
        )

    @pytest.mark.parametrize("sigma", VOLS)
    def test_trellis_pde_vs_quantlib_fd(self, sigma):
        """Test 7: Trellis PDE vs QuantLib FD within 1%."""
        ql, bs_process, option = self._ql_setup(sigma)

        engine = ql.FdBlackScholesVanillaEngine(bs_process, 500, 500)
        option.setPricingEngine(engine)
        ql_price = option.NPV()

        trellis_price = _get_price("pde", sigma)
        rel_err = abs(trellis_price - ql_price) / ql_price
        assert rel_err < 0.01, (
            f"sigma={sigma}: Trellis PDE={trellis_price:.4f} vs "
            f"QuantLib FD={ql_price:.4f}, rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 8: Early exercise premium increases with vol
# ---------------------------------------------------------------------------

class TestEarlyExercisePremium:
    """Test 8: Early exercise premium (American - European) increases with vol."""

    def test_eep_increases_with_vol(self):
        premiums = []
        for sigma in VOLS:
            am = _get_price("crr", sigma)
            eu = _get_price("bs_euro", sigma)
            eep = am - eu
            premiums.append(eep)
            # Each premium must be positive
            assert eep > 0, f"sigma={sigma}: EEP={eep:.6f} not positive"

        # Premium should increase with vol
        for i in range(1, len(premiums)):
            assert premiums[i] > premiums[i - 1], (
                f"EEP not increasing: sigma={VOLS[i]}: {premiums[i]:.4f} "
                f"<= sigma={VOLS[i-1]}: {premiums[i-1]:.4f}"
            )
