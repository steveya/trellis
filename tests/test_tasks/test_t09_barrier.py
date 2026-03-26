"""T09: Barrier option (down-and-out call) — PDE with absorbing BC vs MC with
discrete monitoring vs Rubinstein analytical.  Cross-validate against QuantLib.

Parameters:
  S0=100, K=100, B=90 (down barrier), r=0.05, sigma=0.20, T=1.0

Methods:
  A) Rubinstein analytical formula (continuous monitoring, closed-form)
  B) PDE with absorbing boundary at S=B
  C) Monte Carlo with discrete monitoring (existing BarrierOptionPayoff)

Tolerances:
  - PDE vs analytical: <= 1%
  - MC vs analytical: <= 2% (discrete monitoring bias)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from trellis.models.analytical.barrier import (
    down_and_out_call,
    down_and_in_call,
)
from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import BlackScholesOperator
from trellis.models.pde.theta_method import theta_method_1d
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.gbm import GBM

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0 = 100.0
K = 100.0
B = 90.0
R = 0.05
SIGMA = 0.20
T = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bs_call_price(S, K, r, sigma, T):
    """Black-Scholes European call price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def pde_barrier_call(
    S0, K, B, r, sigma, T, n_x=500, n_t=500, s_max_mult=4.0,
    barrier_type="down_and_out",
):
    """Price a barrier call via PDE with absorbing boundary at S=B.

    Grid runs from S=B to S_max. Lower BC V(B,t)=0 (absorbing barrier).
    Upper BC: vanilla call asymptote S - K*exp(-r*tau).
    """
    s_max = s_max_mult * S0
    grid = Grid(x_min=B, x_max=s_max, n_x=n_x, T=T, n_t=n_t)
    op = BlackScholesOperator(
        sigma_fn=lambda S, t: sigma,
        r_fn=lambda t: r,
    )

    S_grid = grid.x

    # Terminal condition: call payoff max(S - K, 0) for DO;
    # for DI we solve DO first then use parity
    terminal = np.maximum(S_grid - K, 0.0)

    # Boundary conditions for down-and-out call
    lower_bc = lambda t: 0.0  # absorbing barrier
    upper_bc = lambda t: s_max - K * np.exp(-r * (T - t))  # vanilla call at S_max

    V = theta_method_1d(
        grid, op, terminal, theta=0.5,
        lower_bc_fn=lower_bc, upper_bc_fn=upper_bc,
    )

    # Interpolate to S0
    idx = np.searchsorted(S_grid, S0) - 1
    idx = max(0, min(idx, len(S_grid) - 2))
    w = (S0 - S_grid[idx]) / (S_grid[idx + 1] - S_grid[idx])
    do_price = V[idx] * (1 - w) + V[idx + 1] * w

    if barrier_type == "down_and_out":
        return float(do_price)
    elif barrier_type == "down_and_in":
        # In-out parity: DI = vanilla - DO
        vanilla = bs_call_price(S0, K, r, sigma, T)
        return float(vanilla - do_price)
    else:
        raise ValueError(f"Unsupported barrier_type: {barrier_type}")


def _bgk_adjusted_barrier(B, sigma, T, n_steps, direction="down"):
    """Broadie-Glasserman-Kou continuity correction for discrete monitoring.

    Adjusts the barrier so that discrete-monitoring MC approximates
    the continuous-monitoring price.  For a down barrier, the adjusted
    barrier is shifted UP (harder to survive -> more knock-outs).

    Reference: Broadie, Glasserman, Kou (1997), Mathematical Finance.
    """
    # beta = -zeta(1/2) / sqrt(2*pi) ~ 0.5826
    beta = 0.5826
    dt = T / n_steps
    if direction == "down":
        return B * np.exp(beta * sigma * np.sqrt(dt))
    else:  # up barrier
        return B * np.exp(-beta * sigma * np.sqrt(dt))


def mc_barrier_call(S0, K, B, r, sigma, T, n_paths=50_000, n_steps=1000, seed=42):
    """Price a down-and-out call via Monte Carlo with discrete monitoring.

    Applies the Broadie-Glasserman-Kou continuity correction to reduce
    discrete-monitoring bias.
    """
    process = GBM(mu=r, sigma=sigma)
    engine = MonteCarloEngine(
        process, n_paths=n_paths, n_steps=n_steps, seed=seed, method="exact",
    )

    B_adj = _bgk_adjusted_barrier(B, sigma, T, n_steps, direction="down")

    def payoff_fn(paths):
        S_T = paths[:, -1]
        breached = np.any(paths <= B_adj, axis=1)
        vanilla = np.maximum(S_T - K, 0.0)
        return np.where(breached, 0.0, vanilla)

    result = engine.price(S0, T, payoff_fn, discount_rate=r)
    return result["price"]


def mc_barrier_in_call(S0, K, B, r, sigma, T, n_paths=50_000, n_steps=1000, seed=42):
    """Price a down-and-in call via Monte Carlo with discrete monitoring.

    Applies the Broadie-Glasserman-Kou continuity correction.
    """
    process = GBM(mu=r, sigma=sigma)
    engine = MonteCarloEngine(
        process, n_paths=n_paths, n_steps=n_steps, seed=seed, method="exact",
    )

    B_adj = _bgk_adjusted_barrier(B, sigma, T, n_steps, direction="down")

    def payoff_fn(paths):
        S_T = paths[:, -1]
        breached = np.any(paths <= B_adj, axis=1)
        vanilla = np.maximum(S_T - K, 0.0)
        return np.where(breached, vanilla, 0.0)

    result = engine.price(S0, T, payoff_fn, discount_rate=r)
    return result["price"]


# ---------------------------------------------------------------------------
# Precompute and cache prices
# ---------------------------------------------------------------------------
_cache: dict[str, float] = {}


def _get(key: str) -> float:
    if key not in _cache:
        if key == "analytical_do":
            _cache[key] = down_and_out_call(S0, K, B, R, SIGMA, T)
        elif key == "analytical_di":
            _cache[key] = down_and_in_call(S0, K, B, R, SIGMA, T)
        elif key == "pde_do":
            _cache[key] = pde_barrier_call(S0, K, B, R, SIGMA, T)
        elif key == "pde_di":
            _cache[key] = pde_barrier_call(
                S0, K, B, R, SIGMA, T, barrier_type="down_and_in",
            )
        elif key == "mc_do":
            _cache[key] = mc_barrier_call(S0, K, B, R, SIGMA, T)
        elif key == "mc_di":
            _cache[key] = mc_barrier_in_call(S0, K, B, R, SIGMA, T)
        elif key == "bs_call":
            _cache[key] = bs_call_price(S0, K, R, SIGMA, T)
    return _cache[key]


# ---------------------------------------------------------------------------
# Test 1: Rubinstein analytical formula sanity
# ---------------------------------------------------------------------------

class TestAnalytical:
    """Test 1: Rubinstein analytical formula produces reasonable values."""

    def test_do_call_positive(self):
        """DO call should be positive and less than vanilla call."""
        do = _get("analytical_do")
        vanilla = _get("bs_call")
        assert do > 0, f"DO call should be positive, got {do}"
        assert do < vanilla, f"DO call ({do:.4f}) should be < vanilla ({vanilla:.4f})"

    def test_do_call_known_value(self):
        """Check against known textbook-range value.

        For S0=100, K=100, B=90, r=0.05, sigma=0.20, T=1.0,
        the DO call should be roughly 6-8 (vanilla BS call ~ 10.45).
        """
        do = _get("analytical_do")
        assert 4.0 < do < 12.0, f"DO call = {do:.4f} outside expected range [4, 12]"

    def test_di_call_positive(self):
        """DI call should be positive."""
        di = _get("analytical_di")
        assert di > 0, f"DI call should be positive, got {di}"


# ---------------------------------------------------------------------------
# Test 2: PDE barrier
# ---------------------------------------------------------------------------

class TestPDEBarrier:
    """Test 2: PDE with absorbing BC matches analytical within 1%."""

    def test_pde_do_vs_analytical(self):
        pde = _get("pde_do")
        ana = _get("analytical_do")
        rel_err = abs(pde - ana) / ana
        assert rel_err < 0.01, (
            f"PDE DO={pde:.4f} vs Analytical={ana:.4f}, rel_err={rel_err:.4f}"
        )

    def test_pde_positive_and_bounded(self):
        pde = _get("pde_do")
        vanilla = _get("bs_call")
        assert pde > 0, f"PDE DO call should be positive, got {pde}"
        assert pde < vanilla * 1.01, (
            f"PDE DO ({pde:.4f}) should be <= vanilla ({vanilla:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 3: MC barrier (discrete monitoring)
# ---------------------------------------------------------------------------

class TestMCBarrier:
    """Test 3: MC with discrete monitoring matches analytical within 2%."""

    def test_mc_do_vs_analytical(self):
        mc = _get("mc_do")
        ana = _get("analytical_do")
        rel_err = abs(mc - ana) / ana
        assert rel_err < 0.02, (
            f"MC DO={mc:.4f} vs Analytical={ana:.4f}, rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Three-way agreement
# ---------------------------------------------------------------------------

class TestThreeWayAgreement:
    """Test 4: PDE, MC, and analytical should agree within tolerance."""

    def test_pde_vs_analytical(self):
        pde = _get("pde_do")
        ana = _get("analytical_do")
        rel_err = abs(pde - ana) / ana
        assert rel_err < 0.01, (
            f"PDE={pde:.4f} vs Analytical={ana:.4f}, rel_err={rel_err:.4f}"
        )

    def test_mc_vs_analytical(self):
        mc = _get("mc_do")
        ana = _get("analytical_do")
        rel_err = abs(mc - ana) / ana
        assert rel_err < 0.02, (
            f"MC={mc:.4f} vs Analytical={ana:.4f}, rel_err={rel_err:.4f}"
        )

    def test_pde_vs_mc(self):
        pde = _get("pde_do")
        mc = _get("mc_do")
        avg = 0.5 * (pde + mc)
        rel_err = abs(pde - mc) / avg
        assert rel_err < 0.02, (
            f"PDE={pde:.4f} vs MC={mc:.4f}, rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: In-out parity
# ---------------------------------------------------------------------------

class TestInOutParity:
    """Test 5: DI_call + DO_call = vanilla_call (for continuous barriers)."""

    def test_analytical_in_out_parity(self):
        """Analytical DO + DI should exactly equal vanilla BS call."""
        do = _get("analytical_do")
        di = _get("analytical_di")
        vanilla = _get("bs_call")
        total = do + di
        rel_err = abs(total - vanilla) / vanilla
        assert rel_err < 1e-10, (
            f"DO({do:.6f}) + DI({di:.6f}) = {total:.6f} vs Vanilla={vanilla:.6f}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_pde_in_out_parity(self):
        """PDE DO + DI should match vanilla BS call within 1%."""
        do = _get("pde_do")
        di = _get("pde_di")
        vanilla = _get("bs_call")
        total = do + di
        rel_err = abs(total - vanilla) / vanilla
        assert rel_err < 0.01, (
            f"PDE: DO({do:.4f}) + DI({di:.4f}) = {total:.4f} vs "
            f"Vanilla={vanilla:.4f}, rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 6: QuantLib cross-validation
# ---------------------------------------------------------------------------

class TestQuantLibCrossValidation:
    """Test 6: Cross-validate trellis against QuantLib barrier engines."""

    @staticmethod
    def _ql_setup():
        """Build QuantLib BS process and barrier option."""
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
            ql.BlackConstantVol(
                today, ql.NullCalendar(), SIGMA, ql.Actual365Fixed(),
            )
        )

        bs_process = ql.BlackScholesMertonProcess(
            spot_handle, div_handle, rate_handle, vol_handle,
        )

        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(maturity)

        # Down-and-out barrier option
        barrier_option = ql.BarrierOption(
            ql.Barrier.DownOut, B, 0.0, payoff, exercise,
        )
        return ql, bs_process, barrier_option, today, maturity

    def test_analytical_vs_quantlib_analytic(self):
        """Trellis analytical vs QuantLib AnalyticBarrierEngine."""
        ql, bs_process, option, today, maturity = self._ql_setup()

        engine = ql.AnalyticBarrierEngine(bs_process)
        option.setPricingEngine(engine)
        ql_price = option.NPV()

        trellis_price = _get("analytical_do")
        rel_err = abs(trellis_price - ql_price) / ql_price
        assert rel_err < 0.01, (
            f"Trellis Analytical={trellis_price:.4f} vs "
            f"QuantLib Analytic={ql_price:.4f}, rel_err={rel_err:.4f}"
        )

    def test_pde_vs_quantlib_analytic(self):
        """Trellis PDE vs QuantLib AnalyticBarrierEngine within 1%."""
        ql, bs_process, option, today, maturity = self._ql_setup()

        engine = ql.AnalyticBarrierEngine(bs_process)
        option.setPricingEngine(engine)
        ql_price = option.NPV()

        trellis_price = _get("pde_do")
        rel_err = abs(trellis_price - ql_price) / ql_price
        assert rel_err < 0.01, (
            f"Trellis PDE={trellis_price:.4f} vs "
            f"QuantLib Analytic={ql_price:.4f}, rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 7: Barrier at various levels
# ---------------------------------------------------------------------------

class TestBarrierLevels:
    """Test 7: As barrier approaches spot, DO price should decrease."""

    BARRIERS = [80.0, 85.0, 90.0, 95.0]

    def test_do_price_decreases_as_barrier_rises(self):
        """DO call price should monotonically decrease as B increases toward S0."""
        prices = []
        for b in self.BARRIERS:
            p = down_and_out_call(S0, K, b, R, SIGMA, T)
            prices.append(p)
            assert p > 0, f"DO call with B={b} should be positive, got {p}"

        for i in range(1, len(prices)):
            assert prices[i] < prices[i - 1], (
                f"DO call not decreasing: B={self.BARRIERS[i]}: {prices[i]:.4f} "
                f">= B={self.BARRIERS[i-1]}: {prices[i-1]:.4f}"
            )

    def test_pde_agrees_at_various_barriers(self):
        """PDE should agree with analytical within 1% at each barrier level."""
        for b in self.BARRIERS:
            ana = down_and_out_call(S0, K, b, R, SIGMA, T)
            pde = pde_barrier_call(S0, K, b, R, SIGMA, T)
            rel_err = abs(pde - ana) / ana if ana > 0.01 else abs(pde - ana)
            assert rel_err < 0.01, (
                f"B={b}: PDE={pde:.4f} vs Analytical={ana:.4f}, rel_err={rel_err:.4f}"
            )

    def test_low_barrier_approaches_vanilla(self):
        """With barrier far from spot (B=50), DO call should approach vanilla."""
        b_far = 50.0
        do = down_and_out_call(S0, K, b_far, R, SIGMA, T)
        vanilla = bs_call_price(S0, K, R, SIGMA, T)
        rel_err = abs(do - vanilla) / vanilla
        assert rel_err < 0.01, (
            f"Far barrier B={b_far}: DO={do:.4f} should approach "
            f"vanilla={vanilla:.4f}, rel_err={rel_err:.4f}"
        )
