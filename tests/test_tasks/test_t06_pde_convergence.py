"""T06: European call PDE theta-method convergence study.

Measures convergence order for theta=0.5 (Crank-Nicolson) and theta=1.0
(fully implicit Euler). Cross-validates against Black-Scholes analytical
and QuantLib FD engine.

Parameters:
  S0=100, K=100, r=0.05, sigma=0.20, T=1.0
  Grid: S_max=400, varying n_x = n_t from 50 to 800
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import BlackScholesOperator
from trellis.models.pde.theta_method import theta_method_1d

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0 = 100.0
K_ATM = 100.0
R = 0.05
SIGMA = 0.20
T = 1.0
S_MAX = 400.0
GRID_SIZES = [50, 100, 200, 400, 800]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bs_price(S, K, r, sigma, T, option_type="call"):
    """Black-Scholes analytical price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def pde_price(n, theta, K=K_ATM, option_type="call"):
    """Price a European option via the theta-method PDE solver."""
    grid = Grid(x_min=0.0, x_max=S_MAX, n_x=n, T=T, n_t=n)
    op = BlackScholesOperator(
        sigma_fn=lambda S, t: SIGMA,
        r_fn=lambda t: R,
    )

    S = grid.x
    if option_type == "call":
        terminal = np.maximum(S - K, 0.0)
        # Boundary conditions: call worth 0 at S=0, S-K*exp(-r*tau) at S_max
        lower_bc = lambda t: 0.0
        upper_bc = lambda t: S_MAX - K * np.exp(-R * (T - t))
    else:
        terminal = np.maximum(K - S, 0.0)
        lower_bc = lambda t: K * np.exp(-R * (T - t))
        upper_bc = lambda t: 0.0

    V = theta_method_1d(
        grid, op, terminal, theta=theta,
        lower_bc_fn=lower_bc, upper_bc_fn=upper_bc,
    )

    # Interpolate to S0
    idx = np.searchsorted(grid.x, S0) - 1
    idx = max(0, min(idx, len(grid.x) - 2))
    w = (S0 - grid.x[idx]) / (grid.x[idx + 1] - grid.x[idx])
    return V[idx] * (1 - w) + V[idx + 1] * w


def convergence_order_diagnostic(pairs):
    """Estimate convergence order from (grid_size, price) pairs.

    Fits log(error) = -p * log(n) + c via least-squares.
    Returns p (the convergence order).
    """
    ref = bs_price(S0, K_ATM, R, SIGMA, T)
    ns = np.array([p[0] for p in pairs], dtype=float)
    errors = np.array([abs(p[1] - ref) for p in pairs])

    # Filter out zero/tiny errors that would break log
    mask = errors > 1e-14
    if mask.sum() < 2:
        return float("inf")

    log_n = np.log(ns[mask])
    log_e = np.log(errors[mask])

    # Least-squares: log_e = -p * log_n + c
    A = np.column_stack([log_n, np.ones_like(log_n)])
    coeffs, *_ = np.linalg.lstsq(A, log_e, rcond=None)
    p = -coeffs[0]
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCNConvergence:
    """Test 1: Crank-Nicolson convergence to BS."""

    def test_cn_converges_to_bs(self):
        ref = bs_price(S0, K_ATM, R, SIGMA, T)
        prices = []
        for n in GRID_SIZES:
            p = pde_price(n, theta=0.5)
            prices.append((n, p))

        # All should converge
        errors = [abs(p - ref) for _, p in prices]
        # Errors should decrease monotonically (roughly)
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1] * 1.5, (
                f"Error not decreasing: n={GRID_SIZES[i]}, "
                f"err={errors[i]:.6f} vs prev {errors[i-1]:.6f}"
            )
        # Error at n=800 < 0.01
        assert errors[-1] < 0.01, f"Error at n=800: {errors[-1]:.6f}"


class TestImplicitConvergence:
    """Test 2: Implicit Euler convergence to BS."""

    def test_implicit_converges_to_bs(self):
        ref = bs_price(S0, K_ATM, R, SIGMA, T)
        for n in GRID_SIZES:
            p = pde_price(n, theta=1.0)
            err = abs(p - ref)
            # Should converge — at larger n, error smaller
            if n == 800:
                # Implicit is first-order, so larger error than CN
                assert err < 0.5, f"Implicit error at n=800: {err:.6f}"


class TestCNOrder:
    """Test 3: CN is O(dt^2) — second-order convergence."""

    def test_cn_second_order(self):
        pairs = [(n, pde_price(n, theta=0.5)) for n in GRID_SIZES]
        order = convergence_order_diagnostic(pairs)
        # CN: spatial O(dx^2) and temporal O(dt^2). Since n_x = n_t
        # and dx ~ 1/n, dt ~ 1/n, the overall order should be ~2.
        assert order > 1.5, f"CN convergence order {order:.2f}, expected ~2"


class TestImplicitOrder:
    """Test 4: Implicit is O(dt) — first-order convergence."""

    def test_implicit_first_order(self):
        pairs = [(n, pde_price(n, theta=1.0)) for n in GRID_SIZES]
        order = convergence_order_diagnostic(pairs)
        # Implicit: temporal O(dt), spatial O(dx^2). Since n_x = n_t,
        # the bottleneck is temporal first-order, so overall ~1.
        assert 0.5 < order < 1.8, f"Implicit convergence order {order:.2f}, expected ~1"


class TestCNMoreAccurate:
    """Test 5: CN more accurate than implicit at fine grid.

    At coarse grids, error cancellation between spatial O(dx^2) and temporal
    O(dt) can make implicit appear more accurate. At fine grids (n=800),
    the asymptotic regime dominates and CN's O(dt^2) advantage is clear.
    """

    def test_cn_beats_implicit_at_n800(self):
        ref = bs_price(S0, K_ATM, R, SIGMA, T)
        cn_price = pde_price(800, theta=0.5)
        impl_price = pde_price(800, theta=1.0)
        cn_err = abs(cn_price - ref)
        impl_err = abs(impl_price - ref)
        assert cn_err < impl_err, (
            f"CN error {cn_err:.6f} should be smaller than implicit {impl_err:.6f}"
        )


class TestPutCallParity:
    """Test 6: Put-call parity on PDE prices."""

    def test_put_call_parity_cn(self):
        n = 400
        call_price = pde_price(n, theta=0.5, option_type="call")
        put_price = pde_price(n, theta=0.5, option_type="put")
        parity_rhs = S0 - K_ATM * np.exp(-R * T)
        diff = call_price - put_price
        assert abs(diff - parity_rhs) < 0.05, (
            f"Put-call parity violated: C-P={diff:.4f}, S-Ke^{{-rT}}={parity_rhs:.4f}"
        )


class TestQuantLibCrossValidation:
    """Test 7: Cross-validate against QuantLib FD engine."""

    def test_quantlib_fd_bs(self):
        ql = pytest.importorskip("QuantLib")

        # QuantLib setup
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
            ql.BlackConstantVol(today, ql.NullCalendar(), SIGMA, ql.Actual365Fixed())
        )

        bs_process = ql.BlackScholesMertonProcess(
            spot_handle, div_handle, rate_handle, vol_handle
        )

        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K_ATM)
        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)

        engine = ql.FdBlackScholesVanillaEngine(bs_process, 800, 800)
        option.setPricingEngine(engine)
        ql_price = option.NPV()

        our_price = pde_price(800, theta=0.5)
        assert abs(our_price - ql_price) < 0.05, (
            f"Our price {our_price:.4f} vs QuantLib {ql_price:.4f}"
        )


class TestVariousStrikes:
    """Test 8: Price at various strikes, compare to BS."""

    @pytest.mark.parametrize("K", [80, 100, 120])
    def test_strike(self, K):
        ref = bs_price(S0, K, R, SIGMA, T)
        price = pde_price(400, theta=0.5, K=K)
        err = abs(price - ref)
        assert err < 0.05, f"K={K}: PDE={price:.4f}, BS={ref:.4f}, err={err:.4f}"
