"""T10: Callable bond — HW rate PDE (PSOR free boundary) vs HW tree.

Cross-validates:
  1. Straight bond on HW rate PDE vs analytical (sum of discounted coupons)
  2. Callable bond on HW rate PDE with PSOR (Bermudan exercise)
  3. HW PDE vs HW tree within 1%
  4. QuantLib TreeCallableFixedRateBondEngine cross-validation

Bond specification:
  5% coupon, 10Y maturity, semi-annual, face=100
  Callable at par at 3Y, 5Y, 7Y (Bermudan)
  Flat curve at 5%

HW parameters: a=0.1, sigma=0.01
PDE grid: r from -0.05 to 0.15, n_r=300, n_t=400
Tree: n_steps=400
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from trellis.conventions.schedule import generate_schedule
from trellis.core.types import Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.models.pde.grid import Grid
from trellis.models.pde.rate_operator import HullWhitePDEOperator
from trellis.models.pde.thomas import thomas_solve
from trellis.models.trees.lattice import (
    build_generic_lattice,
    lattice_backward_induction,
)
from trellis.models.trees.control import resolve_lattice_exercise_policy
from trellis.models.trees.models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

FLAT_RATE = 0.05
COUPON_RATE = 0.05
FACE = 100.0
T = 10.0
CALL_YEARS = [3, 5, 7]
CALL_PRICE = 100.0

HW_SIGMA = 0.01
HW_A = 0.1

# PDE grid
R_MIN = -0.10
R_MAX = 0.20
N_R = 400
N_T = 500

# Tree
N_TREE_STEPS = 400


@pytest.fixture(scope="module")
def flat_curve():
    return YieldCurve.flat(FLAT_RATE, max_tenor=max(T + 1, 31.0))


# ---------------------------------------------------------------------------
# Hull-White theta(t) calibration for flat curve
# ---------------------------------------------------------------------------

def hw_alpha_flat(a: float, sigma: float, r0: float) -> callable:
    """Long-run mean alpha(t) for HW under a flat curve.

    The HW process is: dr = (theta_HW(t) - a*r) dt + sigma dW
                      = a*(alpha(t) - r) dt + sigma dW

    where alpha(t) = theta_HW(t) / a is the time-dependent long-run mean.

    For a flat curve with rate r0:
        theta_HW(t) = a*r0 + sigma^2/(2*a) * (1 - exp(-2*a*t))
        alpha(t)    = r0 + sigma^2/(2*a^2) * (1 - exp(-2*a*t))

    The PDE operator uses a*(alpha - r), so theta_fn must return alpha(t).
    """
    def alpha_fn(t):
        return r0 + sigma ** 2 / (2 * a ** 2) * (1 - np.exp(-2 * a * t))
    return alpha_fn


# ---------------------------------------------------------------------------
# PDE backward solver with Bermudan exercise and discrete coupons
# ---------------------------------------------------------------------------

def hw_pde_backward(
    r_grid: np.ndarray,
    dt: float,
    n_t: int,
    operator: HullWhitePDEOperator,
    terminal_values: np.ndarray,
    coupon_steps: dict[int, float] | None = None,
    call_steps: set[int] | None = None,
    call_price: float = 100.0,
    theta: float = 0.5,
) -> np.ndarray:
    """Solve the HW PDE backward in time with discrete coupons and Bermudan call.

    Parameters
    ----------
    r_grid : ndarray of shape (n_r,)
        Spatial grid in short rate.
    dt : float
        Time step.
    n_t : int
        Number of time steps.
    operator : HullWhitePDEOperator
        PDE operator.
    terminal_values : ndarray of shape (n_r,)
        Value at maturity.
    coupon_steps : dict mapping step -> coupon amount
        Discrete coupon additions at specific time steps.
    call_steps : set of int
        Steps at which the bond is callable.
    call_price : float
        Call price (issuer redeems at this price).
    theta : float
        Theta-method parameter (0.5 = Crank-Nicolson).

    Returns
    -------
    V : ndarray of shape (n_r,)
        Solution at t=0.
    """
    n_x = len(r_grid)
    n_int = n_x - 2
    V = terminal_values.copy().astype(float)

    if coupon_steps is None:
        coupon_steps = {}
    if call_steps is None:
        call_steps = set()

    dr = r_grid[1] - r_grid[0]

    for step in range(n_t - 1, -1, -1):
        t = step * dt

        # Get operator coefficients
        a_coeff, b_coeff, c_coeff = operator.coefficients(r_grid, t, dt)

        # Boundary conditions via explicit one-step discounting at boundary
        # nodes.  At extreme r the PDE degenerates, so we discount directly:
        #   V_boundary(t) = V_boundary(t+dt) * exp(-r_boundary * dt)
        V_lower = V[0] * np.exp(-r_grid[0] * dt)
        V_upper = V[-1] * np.exp(-r_grid[-1] * dt)

        # Standard theta-method step (no exercise at this point)
        V = _theta_step(V, a_coeff, b_coeff, c_coeff, theta, V_lower, V_upper, n_int)

        # After solving interior, update boundaries by linear extrapolation
        # from the two nearest interior points (linearity / zero-gamma BC).
        V[0] = 2.0 * V[1] - V[2]
        V[-1] = 2.0 * V[-2] - V[-3]

        # Add discrete coupon at this step (if any)
        if step in coupon_steps:
            V += coupon_steps[step]

        # Bermudan call exercise: issuer calls to minimise liability
        # V = min(V_continuation, call_price + accrued_coupon)
        if step in call_steps:
            cpn = coupon_steps.get(step, 0.0)
            exercise_val = call_price + cpn
            V = np.minimum(V, exercise_val)

    return V


def _theta_step(
    V: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    theta: float,
    V_lower: float,
    V_upper: float,
    n_int: int,
) -> np.ndarray:
    """One time step of the theta-method (tridiagonal solve)."""
    # Implicit side: (I - theta * L*dt)
    a_impl = np.zeros(n_int - 1)
    b_impl = np.zeros(n_int)
    c_impl = np.zeros(n_int - 1)

    for idx in range(n_int):
        b_impl[idx] = 1.0 - theta * b[idx]
        if idx > 0:
            a_impl[idx - 1] = -theta * a[idx]
        if idx < n_int - 1:
            c_impl[idx] = -theta * c[idx]

    # Explicit side: (I + (1-theta) * L*dt) * V^{n+1}
    rhs = np.zeros(n_int)
    one_m_theta = 1.0 - theta
    for idx in range(n_int):
        i = idx + 1  # grid index
        rhs[idx] = (
            one_m_theta * a[idx] * V[i - 1]
            + (1.0 + one_m_theta * b[idx]) * V[i]
            + one_m_theta * c[idx] * V[i + 1]
        )

    # Boundary adjustments
    rhs[0] += theta * a[0] * V_lower
    rhs[-1] += theta * c[-1] * V_upper

    # Solve
    V_int = thomas_solve(a_impl, b_impl, c_impl, rhs)

    V_new = V.copy()
    V_new[0] = V_lower
    V_new[1:1 + n_int] = V_int
    V_new[-1] = V_upper
    return V_new


# ---------------------------------------------------------------------------
# Helpers: coupon and call step mappings
# ---------------------------------------------------------------------------

def _build_coupon_steps_pde(dt: float, n_t: int) -> dict[int, float]:
    """Map PDE time steps to semi-annual coupon amounts."""
    coupon_amount = FACE * COUPON_RATE / 2.0  # semi-annual coupon

    coupon_steps: dict[int, float] = {}
    # Semi-annual coupon dates at 0.5, 1.0, 1.5, ..., 10.0
    for k in range(1, int(T * 2) + 1):
        t_coupon = k * 0.5
        step = int(round(t_coupon / dt))
        if 0 < step < n_t:
            coupon_steps[step] = coupon_steps.get(step, 0.0) + coupon_amount
    return coupon_steps


def _build_call_steps_pde(dt: float) -> set[int]:
    """Convert call years to PDE time step indices."""
    return {int(round(y / dt)) for y in CALL_YEARS}


def _build_coupon_steps_tree(dt: float, n_steps: int) -> dict[int, float]:
    """Map tree step indices to semi-annual coupon amounts."""
    coupon_amount = FACE * COUPON_RATE / 2.0

    coupon_steps: dict[int, float] = {}
    for k in range(1, int(T * 2) + 1):
        t_coupon = k * 0.5
        step = int(round(t_coupon / dt))
        if 0 < step <= n_steps:
            coupon_steps[step] = coupon_steps.get(step, 0.0) + coupon_amount
    return coupon_steps


def _build_call_steps_tree(dt: float) -> set[int]:
    """Convert call years to tree step indices."""
    return {int(round(y / dt)) for y in CALL_YEARS}


# ---------------------------------------------------------------------------
# PDE pricing functions
# ---------------------------------------------------------------------------

def _price_straight_bond_pde(flat_curve) -> float:
    """Price a straight bond using the HW PDE."""
    dt = T / N_T
    r_grid = np.linspace(R_MIN, R_MAX, N_R)
    theta_fn = hw_alpha_flat(HW_A, HW_SIGMA, FLAT_RATE)

    operator = HullWhitePDEOperator(
        sigma=HW_SIGMA, a=HW_A, theta_fn=theta_fn, r0=FLAT_RATE,
    )

    coupon_steps = _build_coupon_steps_pde(dt, N_T)

    # Terminal condition: face + final coupon at maturity
    final_coupon = FACE * COUPON_RATE / 2.0  # last semi-annual coupon
    terminal = np.full(N_R, FACE + final_coupon)

    V = hw_pde_backward(
        r_grid, dt, N_T, operator, terminal,
        coupon_steps=coupon_steps,
    )

    # Interpolate at r = r0
    idx = np.searchsorted(r_grid, FLAT_RATE)
    idx = min(max(idx, 1), N_R - 2)
    # Linear interpolation
    r_lo, r_hi = r_grid[idx - 1], r_grid[idx]
    w = (FLAT_RATE - r_lo) / (r_hi - r_lo)
    price = V[idx - 1] * (1 - w) + V[idx] * w
    return float(price)


def _price_callable_bond_pde(flat_curve) -> float:
    """Price a callable bond using the HW PDE with Bermudan exercise."""
    dt = T / N_T
    r_grid = np.linspace(R_MIN, R_MAX, N_R)
    theta_fn = hw_alpha_flat(HW_A, HW_SIGMA, FLAT_RATE)

    operator = HullWhitePDEOperator(
        sigma=HW_SIGMA, a=HW_A, theta_fn=theta_fn, r0=FLAT_RATE,
    )

    coupon_steps = _build_coupon_steps_pde(dt, N_T)
    call_steps = _build_call_steps_pde(dt)

    # Terminal condition: face + final coupon at maturity
    final_coupon = FACE * COUPON_RATE / 2.0
    terminal = np.full(N_R, FACE + final_coupon)

    V = hw_pde_backward(
        r_grid, dt, N_T, operator, terminal,
        coupon_steps=coupon_steps,
        call_steps=call_steps,
        call_price=CALL_PRICE,
    )

    # Interpolate at r = r0
    idx = np.searchsorted(r_grid, FLAT_RATE)
    idx = min(max(idx, 1), N_R - 2)
    r_lo, r_hi = r_grid[idx - 1], r_grid[idx]
    w = (FLAT_RATE - r_lo) / (r_hi - r_lo)
    price = V[idx - 1] * (1 - w) + V[idx] * w
    return float(price)


# ---------------------------------------------------------------------------
# Tree pricing functions
# ---------------------------------------------------------------------------

def _price_callable_bond_tree(flat_curve) -> float:
    """Price a callable bond using the HW tree."""
    hw_model = MODEL_REGISTRY["hull_white"]
    lattice = build_generic_lattice(
        hw_model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
        T=T, n_steps=N_TREE_STEPS, discount_curve=flat_curve,
    )
    dt = lattice.dt
    n_steps = lattice.n_steps
    coupon_steps = _build_coupon_steps_tree(dt, n_steps)
    call_steps = _build_call_steps_tree(dt)

    final_coupon = coupon_steps.get(n_steps, 0.0)

    def terminal_payoff(step, node, lat):
        return FACE + final_coupon

    def cashflow_at_node(step, node, lat):
        return coupon_steps.get(step, 0.0)

    def exercise_value(step, node, lat):
        cpn = coupon_steps.get(step, 0.0)
        return CALL_PRICE + cpn

    exercise_policy = resolve_lattice_exercise_policy(
        "issuer_call",
        exercise_steps=sorted(call_steps),
    )

    price = lattice_backward_induction(
        lattice,
        terminal_payoff=terminal_payoff,
        exercise_value=exercise_value,
        cashflow_at_node=cashflow_at_node,
        exercise_policy=exercise_policy,
    )
    return price


def _price_straight_bond_tree(flat_curve) -> float:
    """Price a straight bond using the HW tree."""
    hw_model = MODEL_REGISTRY["hull_white"]
    lattice = build_generic_lattice(
        hw_model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
        T=T, n_steps=N_TREE_STEPS, discount_curve=flat_curve,
    )
    dt = lattice.dt
    n_steps = lattice.n_steps
    coupon_steps = _build_coupon_steps_tree(dt, n_steps)
    final_coupon = coupon_steps.get(n_steps, 0.0)

    def terminal_payoff(step, node, lat):
        return FACE + final_coupon

    def cashflow_at_node(step, node, lat):
        return coupon_steps.get(step, 0.0)

    return lattice_backward_induction(
        lattice,
        terminal_payoff=terminal_payoff,
        cashflow_at_node=cashflow_at_node,
    )


# ---------------------------------------------------------------------------
# Analytical straight bond price
# ---------------------------------------------------------------------------

def _analytical_straight_bond(rate: float) -> float:
    """Analytical price of a 10Y 5% semi-annual bond on a flat curve."""
    pv = 0.0
    coupon = FACE * COUPON_RATE / 2.0
    for k in range(1, int(T * 2) + 1):
        t_k = k * 0.5
        pv += coupon * math.exp(-rate * t_k)
    # Add face at maturity
    pv += FACE * math.exp(-rate * T)
    return pv


# ===================================================================
# Test 1: Straight bond on rate PDE vs analytical
# ===================================================================

class TestStraightBondPDE:
    """Price a 10Y 5% bond via HW PDE and compare to analytical."""

    def test_straight_bond_pde_vs_analytical(self, flat_curve):
        """HW PDE straight bond should match analytical within 1%."""
        pde_price = _price_straight_bond_pde(flat_curve)
        analytical_price = _analytical_straight_bond(FLAT_RATE)

        rel_err = abs(pde_price - analytical_price) / analytical_price
        assert rel_err < 0.01, (
            f"PDE={pde_price:.4f}, Analytical={analytical_price:.4f}, "
            f"rel_err={rel_err:.4%}"
        )

    def test_straight_bond_reasonable_range(self, flat_curve):
        """At par rate, straight bond should be near 100."""
        pde_price = _price_straight_bond_pde(flat_curve)
        # 5% coupon, 5% flat curve => price ~ 100
        assert 98 < pde_price < 102, f"PDE straight bond={pde_price:.4f}"


# ===================================================================
# Test 2: Callable bond on rate PDE with PSOR / Bermudan exercise
# ===================================================================

class TestCallableBondPDE:
    """Callable bond via HW PDE with Bermudan exercise (min constraint)."""

    def test_callable_leq_straight_pde(self, flat_curve):
        """Callable bond <= straight bond (issuer option has negative value)."""
        callable_price = _price_callable_bond_pde(flat_curve)
        straight_price = _price_straight_bond_pde(flat_curve)

        assert callable_price <= straight_price + 0.1, (
            f"Callable ({callable_price:.4f}) > Straight ({straight_price:.4f})"
        )

    def test_callable_reasonable_range_pde(self, flat_curve):
        """Callable bond should be in a reasonable range."""
        price = _price_callable_bond_pde(flat_curve)
        assert 85 < price < 102, f"PDE callable bond={price:.4f}"


# ===================================================================
# Test 3: HW PDE vs HW tree within 1%
# ===================================================================

class TestPDEvsTree:
    """Cross-validate HW PDE callable bond against HW tree."""

    def test_callable_pde_vs_tree(self, flat_curve):
        """HW PDE and HW tree callable bond prices within 1%."""
        pde_price = _price_callable_bond_pde(flat_curve)
        tree_price = _price_callable_bond_tree(flat_curve)

        rel_err = abs(pde_price - tree_price) / tree_price
        assert rel_err < 0.01, (
            f"PDE={pde_price:.4f}, Tree={tree_price:.4f}, "
            f"rel_err={rel_err:.4%}"
        )

    def test_straight_pde_vs_tree(self, flat_curve):
        """HW PDE and HW tree straight bond prices within 0.5%."""
        pde_price = _price_straight_bond_pde(flat_curve)
        tree_price = _price_straight_bond_tree(flat_curve)

        rel_err = abs(pde_price - tree_price) / tree_price
        assert rel_err < 0.005, (
            f"PDE={pde_price:.4f}, Tree={tree_price:.4f}, "
            f"rel_err={rel_err:.4%}"
        )


# ===================================================================
# Test 4: QuantLib cross-validation
# ===================================================================

class TestQuantLibCrossValidation:
    """Compare Trellis HW callable bond (PDE + tree) to QuantLib."""

    @pytest.fixture(autouse=True)
    def _require_quantlib(self):
        pytest.importorskip("QuantLib")

    def _quantlib_callable_price(self) -> float:
        """Price callable bond using QuantLib TreeCallableFixedRateBondEngine."""
        import QuantLib as ql

        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today

        # Build flat curve
        ql_curve = ql.FlatForward(today, FLAT_RATE, ql.Actual365Fixed())
        curve_handle = ql.YieldTermStructureHandle(ql_curve)

        # Bond schedule
        issue_date = today
        maturity_date = ql.Date(15, 1, 2035)

        schedule = ql.Schedule(
            issue_date, maturity_date,
            ql.Period(ql.Semiannual),
            ql.NullCalendar(),
            ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Backward, False,
        )

        # Call schedule
        call_schedule = ql.CallabilitySchedule()
        for y in CALL_YEARS:
            call_date = ql.Date(15, 1, 2025 + y)
            call_price_ql = ql.BondPrice(CALL_PRICE, ql.BondPrice.Clean)
            call_schedule.append(
                ql.Callability(call_price_ql, ql.Callability.Call, call_date)
            )

        # Build callable bond
        callable_bond = ql.CallableFixedRateBond(
            0,            # settlement days
            FACE,         # face
            schedule,
            [COUPON_RATE],
            ql.Actual365Fixed(),
            ql.Unadjusted,
            FACE,         # redemption
            issue_date,
            call_schedule,
        )

        # HW engine
        hw_model = ql.HullWhite(curve_handle, HW_A, HW_SIGMA)
        engine = ql.TreeCallableFixedRateBondEngine(hw_model, N_TREE_STEPS)
        callable_bond.setPricingEngine(engine)

        return callable_bond.cleanPrice()

    def test_pde_vs_quantlib(self, flat_curve):
        """Trellis HW PDE callable matches QuantLib within 1%."""
        ql_price = self._quantlib_callable_price()
        pde_price = _price_callable_bond_pde(flat_curve)

        rel_err = abs(pde_price - ql_price) / ql_price
        assert rel_err < 0.01, (
            f"PDE={pde_price:.4f}, QuantLib={ql_price:.4f}, "
            f"rel_err={rel_err:.4%}"
        )

    def test_tree_vs_quantlib(self, flat_curve):
        """Trellis HW tree callable matches QuantLib within 1%."""
        ql_price = self._quantlib_callable_price()
        tree_price = _price_callable_bond_tree(flat_curve)

        rel_err = abs(tree_price - ql_price) / ql_price
        assert rel_err < 0.01, (
            f"Tree={tree_price:.4f}, QuantLib={ql_price:.4f}, "
            f"rel_err={rel_err:.4%}"
        )
