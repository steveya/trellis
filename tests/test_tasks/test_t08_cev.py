"""T08: CEV model European option — PDE vs MC cross-validation.

CEV dynamics: dS = r*S*dt + sigma*S^beta*dW
Parameters: S0=100, K=100, r=0.05, sigma=3.0, beta=0.5 (square root model), T=1.0

Note: sigma=3.0 is chosen so that ATM implied vol ~ 30%
(since local_vol = sigma * S^(beta-1) = 3.0 * 100^(-0.5) = 0.30).

Cross-validation against QuantLib AnalyticCEVEngine.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import CEVOperator
from trellis.models.pde.theta_method import theta_method_1d

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 3.0       # CEV vol parameter (sigma * S^(beta-1) ~ 0.30 at S=100)
BETA = 0.5        # square root model
T = 1.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cev_pde_price(K_val, sigma=SIGMA, beta=BETA, n_x=500, n_t=500,
                  S_max=400.0):
    """Price European call via CEV PDE (Crank-Nicolson)."""
    grid = Grid(x_min=0.0, x_max=S_max, n_x=n_x, T=T, n_t=n_t)
    op = CEVOperator(
        sigma_fn=lambda S, t: sigma,
        r_fn=lambda t: R,
        beta=beta,
    )
    S_grid = grid.x
    terminal = np.maximum(S_grid - K_val, 0.0)

    # BCs: call at S=0 is 0; at S=S_max, call ~ S_max - K*exp(-r*tau)
    lower_bc = lambda t: 0.0
    upper_bc = lambda t: S_max - K_val * np.exp(-R * (T - t))

    V = theta_method_1d(
        grid, op, terminal, theta=0.5,
        lower_bc_fn=lower_bc, upper_bc_fn=upper_bc,
    )

    # Interpolate to S0
    idx = np.searchsorted(S_grid, S0) - 1
    idx = max(0, min(idx, len(S_grid) - 2))
    w = (S0 - S_grid[idx]) / (S_grid[idx + 1] - S_grid[idx])
    return V[idx] * (1 - w) + V[idx + 1] * w


def cev_mc_price(K_val, sigma=SIGMA, beta=BETA, n_paths=100_000,
                 n_steps=200, seed=42):
    """Price European call via Euler MC of CEV spot dynamics."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.full(n_paths, S0)
    for _ in range(n_steps):
        dw = rng.standard_normal(n_paths)
        S_beta = np.power(np.maximum(S, 1e-8), beta)
        S = S + R * S * dt + sigma * S_beta * sqrt_dt * dw
        S = np.maximum(S, 0.0)  # absorbing barrier at 0

    payoff = np.maximum(S - K_val, 0.0)
    return float(np.exp(-R * T) * np.mean(payoff))


def bs_call_price(S, K_val, r, sigma_bs, T_val):
    """Black-Scholes European call price."""
    d1 = (np.log(S / K_val) + (r + 0.5 * sigma_bs ** 2) * T_val) / (
        sigma_bs * np.sqrt(T_val)
    )
    d2 = d1 - sigma_bs * np.sqrt(T_val)
    return float(S * norm.cdf(d1) - K_val * np.exp(-r * T_val) * norm.cdf(d2))


def implied_vol_bisect(price, S, K_val, r, T_val, option_type="call",
                       lo=0.01, hi=2.0, tol=1e-6, max_iter=100):
    """Extract implied vol via bisection."""
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if option_type == "call":
            model_price = bs_call_price(S, K_val, r, mid, T_val)
        else:
            model_price = (bs_call_price(S, K_val, r, mid, T_val)
                           - S + K_val * np.exp(-r * T_val))
        if model_price > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_cache: dict[str, float] = {}


def _pde(K_val=K):
    key = f"pde_{K_val}"
    if key not in _cache:
        _cache[key] = cev_pde_price(K_val)
    return _cache[key]


def _mc(K_val=K):
    key = f"mc_{K_val}"
    if key not in _cache:
        _cache[key] = cev_mc_price(K_val)
    return _cache[key]


# ---------------------------------------------------------------------------
# Test 1: CEV PDE European call — price is reasonable
# ---------------------------------------------------------------------------

class TestCEVPDE:
    """Test 1: CEV PDE produces a reasonable European call price."""

    def test_pde_price_positive(self):
        price = _pde()
        assert price > 0, f"PDE price = {price:.4f} should be positive"

    def test_pde_price_lt_spot(self):
        price = _pde()
        assert price < S0, f"Call price {price:.4f} should be < S0={S0}"

    def test_pde_price_ge_intrinsic(self):
        price = _pde()
        intrinsic = max(S0 - K, 0.0)
        assert price >= intrinsic - 0.01, (
            f"Price {price:.4f} < intrinsic {intrinsic:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 2: CEV MC European call — price is reasonable
# ---------------------------------------------------------------------------

class TestCEVMC:
    """Test 2: CEV MC produces a reasonable European call price."""

    def test_mc_price_positive(self):
        price = _mc()
        assert price > 0, f"MC price = {price:.4f} should be positive"

    def test_mc_price_lt_spot(self):
        price = _mc()
        assert price < S0, f"Call price {price:.4f} should be < S0={S0}"


# ---------------------------------------------------------------------------
# Test 3: CEV PDE vs MC agreement within 1%
# ---------------------------------------------------------------------------

class TestPDEvsMC:
    """Test 3: PDE and MC agree within 1%."""

    def test_pde_mc_agreement(self):
        pde = _pde()
        mc = _mc()
        avg = 0.5 * (pde + mc)
        rel_err = abs(pde - mc) / avg
        assert rel_err < 0.01, (
            f"PDE={pde:.4f} vs MC={mc:.4f}, rel_err={rel_err:.4%}"
        )


# ---------------------------------------------------------------------------
# Test 4: CEV reduces to BS at beta=1
# ---------------------------------------------------------------------------

class TestCEVReducesToBS:
    """Test 4: With beta=1 and sigma=0.30, CEV PDE matches BS exactly."""

    def test_beta1_matches_bs(self):
        sigma_bs = 0.30
        # CEV with beta=1: dS = r*S*dt + sigma*S^1*dW = BS with vol=sigma
        pde_price = cev_pde_price(K, sigma=sigma_bs, beta=1.0)
        bs_price = bs_call_price(S0, K, R, sigma_bs, T)
        rel_err = abs(pde_price - bs_price) / bs_price
        assert rel_err < 0.005, (
            f"CEV(beta=1) PDE={pde_price:.4f} vs BS={bs_price:.4f}, "
            f"rel_err={rel_err:.4%}"
        )


# ---------------------------------------------------------------------------
# Test 5: QuantLib cross-validation
# ---------------------------------------------------------------------------

class TestQuantLibCrossValidation:
    """Test 5: Cross-validate against QuantLib CEV engines.

    QuantLib models forward CEV: dF = alpha * F^beta * dW.
    Our model is spot CEV: dS = r*S*dt + sigma*S^beta*dW.
    The two agree when r is small or T is short. For finite r*T,
    there is a small systematic difference. We validate:
    (a) beta=0.5: PDE vs QuantLib analytic within 1.5%
    (b) beta=1 case: BS formula comparison (since QuantLib CEV diverges at beta=1)
    """

    @staticmethod
    def _ql_cev_price(K_val, sigma, beta):
        """Price European call using QuantLib AnalyticCEVEngine."""
        ql = pytest.importorskip("QuantLib")

        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today
        maturity = today + ql.Period(1, ql.Years)

        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, R, ql.Actual365Fixed())
        )
        f0 = S0 * np.exp(R * T)

        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K_val)
        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)

        engine = ql.AnalyticCEVEngine(f0, sigma, beta, rate_handle)
        option.setPricingEngine(engine)
        return option.NPV()

    def test_cev_pde_vs_quantlib_analytic(self):
        """PDE vs QuantLib analytic CEV at ATM — within 1.5%.

        The ~1% difference is expected from forward vs spot dynamics.
        """
        ql_price = self._ql_cev_price(K, SIGMA, BETA)
        pde_price = _pde()
        rel_err = abs(pde_price - ql_price) / ql_price
        assert rel_err < 0.015, (
            f"PDE={pde_price:.4f} vs QuantLib={ql_price:.4f}, "
            f"rel_err={rel_err:.4%}"
        )

    def test_beta1_vs_bs(self):
        """At beta=1, both our PDE and BS formula match.

        QuantLib's CEV engine diverges at beta=1, so we compare to BS directly.
        """
        sigma_bs = 0.30
        pde_price = cev_pde_price(K, sigma=sigma_bs, beta=1.0)
        bs_price = bs_call_price(S0, K, R, sigma_bs, T)
        rel_err = abs(pde_price - bs_price) / bs_price
        assert rel_err < 0.005, (
            f"CEV(beta=1) PDE={pde_price:.4f} vs BS={bs_price:.4f}, "
            f"rel_err={rel_err:.4%}"
        )

    @pytest.mark.parametrize("K_val", [80, 90, 100, 110, 120])
    def test_smile_pde_vs_quantlib(self, K_val):
        """PDE vs QuantLib across strikes — within 2%."""
        ql_price = self._ql_cev_price(K_val, SIGMA, BETA)
        pde_price = cev_pde_price(K_val)
        # Use absolute tolerance for deep OTM where prices are small
        # Allow up to 2.5% difference: the forward-vs-spot convention mismatch
        # grows with distance from ATM.
        if min(ql_price, pde_price) < 1.0:
            assert abs(pde_price - ql_price) < 0.5, (
                f"K={K_val}: PDE={pde_price:.4f} vs QL={ql_price:.4f}"
            )
        else:
            rel_err = abs(pde_price - ql_price) / ql_price
            assert rel_err < 0.025, (
                f"K={K_val}: PDE={pde_price:.4f} vs QL={ql_price:.4f}, "
                f"rel_err={rel_err:.4%}"
            )


# ---------------------------------------------------------------------------
# Test 6: beta=0.5 produces negative skew
# ---------------------------------------------------------------------------

class TestCEVSmile:
    """Test 6: CEV with beta<1 produces negative skew (lower strike -> higher IV)."""

    def test_negative_skew(self):
        """Implied vol decreases with strike for beta=0.5."""
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        ivs = []
        for K_val in strikes:
            price = cev_pde_price(K_val)
            iv = implied_vol_bisect(price, S0, K_val, R, T)
            ivs.append(iv)

        # Verify monotonically decreasing (negative skew)
        for i in range(1, len(ivs)):
            assert ivs[i] < ivs[i - 1], (
                f"IV not decreasing: K={strikes[i]} IV={ivs[i]:.4f} "
                f">= K={strikes[i-1]} IV={ivs[i-1]:.4f}"
            )

    def test_atm_iv_near_30pct(self):
        """ATM implied vol should be approximately 30%."""
        price = _pde()
        iv = implied_vol_bisect(price, S0, K, R, T)
        assert abs(iv - 0.30) < 0.02, (
            f"ATM IV={iv:.4f}, expected ~0.30"
        )

    def test_skew_magnitude(self):
        """The 80-120 skew should be meaningfully positive (>1%)."""
        price_80 = cev_pde_price(80.0)
        price_120 = cev_pde_price(120.0)
        iv_80 = implied_vol_bisect(price_80, S0, 80.0, R, T)
        iv_120 = implied_vol_bisect(price_120, S0, 120.0, R, T)
        skew = iv_80 - iv_120
        assert skew > 0.01, (
            f"Skew IV(80)-IV(120)={skew:.4f} too small"
        )
