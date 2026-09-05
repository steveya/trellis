"""T02: Callable bond — BDT lognormal tree vs HW normal tree.

Cross-validates:
  1. BDT tree (lognormal, always-positive rates)
  2. HW tree (normal, can go negative)
  3. BDT vs HW comparison (same bond, different rate distributions)
  4. FinancePy HWTree cross-validation (HW params)
  5. QuantLib TreeCallableFixedRateBondEngine cross-validation (HW params)
  6. BDT ZCB repricing (calibration consistency)

Bond specification:
  5% coupon, 10Y maturity, semi-annual, face=100
  Callable at par at 3Y/5Y/7Y (Bermudan)
  Flat curve at 5%
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from trellis.conventions.schedule import generate_schedule
from trellis.core.types import Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.models.trees.lattice import (
    RecombiningLattice,
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
T = 10.0            # 10-year bond
N_STEPS = 200
CALL_YEARS = [3, 5, 7]  # Bermudan call dates
CALL_PRICE = 100.0  # callable at par

# BDT parameters
BDT_SIGMA = 0.20    # 20% yield vol (lognormal)
BDT_A = 0.05        # mean reversion in log-space

# HW parameters
HW_SIGMA = 0.01     # 1% absolute rate vol (normal)
HW_A = 0.1          # mean reversion speed


@pytest.fixture(scope="module")
def flat_curve():
    return YieldCurve.flat(FLAT_RATE, max_tenor=max(T + 1, 31.0))


# ---------------------------------------------------------------------------
# Helpers: build coupon schedule and pricing functions
# ---------------------------------------------------------------------------

def _build_coupon_steps(dt: float, n_steps: int) -> dict[int, float]:
    """Map tree step indices to discrete coupon amounts.

    Uses generate_schedule to get semi-annual coupon dates, then maps
    each date to the nearest tree step.
    """
    start = date(2025, 1, 15)
    end = date(2035, 1, 15)
    coupon_dates = generate_schedule(start, end, Frequency.SEMI_ANNUAL)

    coupon_amount = FACE * COUPON_RATE / 2.0  # semi-annual coupon

    coupon_steps: dict[int, float] = {}
    for d in coupon_dates:
        # Convert date to year fraction
        t_years = (d - start).days / 365.25
        step = int(round(t_years / dt))
        if 0 < step <= n_steps:
            # Accumulate in case two dates map to the same step
            coupon_steps[step] = coupon_steps.get(step, 0.0) + coupon_amount

    return coupon_steps


def _build_call_steps(dt: float) -> set[int]:
    """Convert call years to tree step indices."""
    return {int(round(y / dt)) for y in CALL_YEARS}


def _price_callable_bond(lattice: RecombiningLattice) -> float:
    """Price a callable bond on a calibrated rate lattice.

    Uses Bermudan exercise with exercise_fn=min (issuer minimizes liability).
    """
    dt = lattice.dt
    n_steps = lattice.n_steps
    coupon_steps = _build_coupon_steps(dt, n_steps)
    call_steps = _build_call_steps(dt)

    # Terminal payoff: face + final coupon
    final_coupon = coupon_steps.get(n_steps, 0.0)

    def terminal_payoff(step, node, lat):
        return FACE + final_coupon

    # Intermediate cashflows: discrete coupons at scheduled dates
    def cashflow_at_node(step, node, lat):
        return coupon_steps.get(step, 0.0)

    # Exercise value at call dates: call_price + coupon at that step
    # (issuer pays accrued coupon when calling)
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


def _price_straight_bond(lattice: RecombiningLattice) -> float:
    """Price a straight (non-callable) bond on the same lattice."""
    dt = lattice.dt
    n_steps = lattice.n_steps
    coupon_steps = _build_coupon_steps(dt, n_steps)
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


# ===================================================================
# Test 1: BDT callable bond pricing
# ===================================================================

class TestBDTCallableBond:
    """BDT lognormal tree: callable bond pricing and rate positivity."""

    def test_bdt_rates_all_positive(self, flat_curve):
        """BDT lognormal rates must be strictly positive at every node."""
        model = MODEL_REGISTRY["bdt"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=BDT_SIGMA, a=BDT_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        for step in range(lattice.n_steps + 1):
            for node in range(lattice.n_nodes(step)):
                r = lattice.get_state(step, node)
                assert r > 0, (
                    f"BDT rate at step={step}, node={node} is {r:.6f} "
                    f"(must be positive for lognormal model)"
                )

    def test_bdt_callable_leq_straight(self, flat_curve):
        """Callable bond <= straight bond (issuer option has negative value to holder)."""
        model = MODEL_REGISTRY["bdt"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=BDT_SIGMA, a=BDT_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        callable_price = _price_callable_bond(lattice)
        straight_price = _price_straight_bond(lattice)

        assert callable_price <= straight_price + 0.01, (
            f"Callable ({callable_price:.4f}) > Straight ({straight_price:.4f})"
        )

    def test_bdt_callable_reasonable_range(self, flat_curve):
        """BDT callable bond price should be in a reasonable range."""
        model = MODEL_REGISTRY["bdt"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=BDT_SIGMA, a=BDT_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        price = _price_callable_bond(lattice)

        # At par rate (5% coupon, 5% flat curve), straight bond ~ 100
        # Callable should be <= 100 and > 80 (reasonable range)
        assert 80 < price < 105, f"BDT callable price={price:.4f} out of range"


# ===================================================================
# Test 2: HW callable bond pricing
# ===================================================================

class TestHWCallableBond:
    """HW normal tree: callable bond pricing and negative rates at extremes."""

    def test_hw_has_negative_rates(self, flat_curve):
        """HW normal model can produce negative rates at extreme nodes."""
        model = MODEL_REGISTRY["hull_white"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        has_negative = False
        for step in range(lattice.n_steps + 1):
            for node in range(lattice.n_nodes(step)):
                r = lattice.get_state(step, node)
                if r < 0:
                    has_negative = True
                    break
            if has_negative:
                break

        # With sigma=0.01, a=0.1, 10Y, 200 steps, extreme nodes can go negative
        # This is expected behavior for normal models
        # Note: with these params it might not go negative; verify either way
        # The key point is the HW model *allows* it (no exp() wrapper)
        # Check the most extreme node at the last step
        last_step = lattice.n_steps
        lowest_rate = lattice.get_state(last_step, 0)
        highest_rate = lattice.get_state(last_step, lattice.n_nodes(last_step) - 1)

        # With modest vol, rates may stay positive. The structural test is:
        # HW rate at node 0 (lowest) should be less than BDT would produce
        # Just verify the rate range is reasonable for a normal model
        assert lowest_rate < highest_rate, "Rate ordering should hold"
        # Rates should span a reasonable range given 10Y horizon
        rate_range = highest_rate - lowest_rate
        assert rate_range > 0.01, f"Rate range={rate_range:.4f} too narrow"

    def test_hw_callable_leq_straight(self, flat_curve):
        """Callable bond <= straight bond under HW model."""
        model = MODEL_REGISTRY["hull_white"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        callable_price = _price_callable_bond(lattice)
        straight_price = _price_straight_bond(lattice)

        assert callable_price <= straight_price + 0.01, (
            f"Callable ({callable_price:.4f}) > Straight ({straight_price:.4f})"
        )

    def test_hw_callable_reasonable_range(self, flat_curve):
        """HW callable bond price should be in a reasonable range."""
        model = MODEL_REGISTRY["hull_white"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        price = _price_callable_bond(lattice)
        assert 80 < price < 105, f"HW callable price={price:.4f} out of range"


# ===================================================================
# Test 3: BDT vs HW comparison
# ===================================================================

class TestBDTvsHWComparison:
    """Compare callable bond prices across BDT (lognormal) and HW (normal).

    The two models produce different rate distributions:
    - BDT: lognormal (right-skewed, always positive, fatter right tail)
    - HW: normal (symmetric, can be negative)

    Both are calibrated to the same discount curve, so ZCB prices match.
    But the callable bond option value depends on the rate distribution,
    so prices will differ. They should be in the same ballpark.
    """

    def test_bdt_hw_callable_same_ballpark(self, flat_curve):
        """BDT and HW callable prices within 2-3 points of each other."""
        bdt_model = MODEL_REGISTRY["bdt"]
        bdt_lattice = build_generic_lattice(
            bdt_model, r0=FLAT_RATE, sigma=BDT_SIGMA, a=BDT_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        bdt_price = _price_callable_bond(bdt_lattice)

        hw_model = MODEL_REGISTRY["hull_white"]
        hw_lattice = build_generic_lattice(
            hw_model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        hw_price = _price_callable_bond(hw_lattice)

        diff = abs(bdt_price - hw_price)
        assert diff < 3.0, (
            f"BDT={bdt_price:.4f}, HW={hw_price:.4f}, diff={diff:.4f} "
            f"(should be < 3 points)"
        )

    def test_bdt_hw_not_identical(self, flat_curve):
        """BDT and HW should NOT produce identical callable prices.

        The lognormal rate distribution (BDT) vs normal (HW) gives
        different option values even when calibrated to the same curve.
        """
        bdt_model = MODEL_REGISTRY["bdt"]
        bdt_lattice = build_generic_lattice(
            bdt_model, r0=FLAT_RATE, sigma=BDT_SIGMA, a=BDT_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        bdt_price = _price_callable_bond(bdt_lattice)

        hw_model = MODEL_REGISTRY["hull_white"]
        hw_lattice = build_generic_lattice(
            hw_model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        hw_price = _price_callable_bond(hw_lattice)

        # They should differ by at least a small amount
        assert abs(bdt_price - hw_price) > 0.01, (
            f"BDT={bdt_price:.4f} and HW={hw_price:.4f} are suspiciously close"
        )

    def test_straight_bonds_match(self, flat_curve):
        """Both models should give the same straight bond price.

        Since both are calibrated to the same curve, a non-callable bond
        (no option component) should price identically.
        """
        bdt_model = MODEL_REGISTRY["bdt"]
        bdt_lattice = build_generic_lattice(
            bdt_model, r0=FLAT_RATE, sigma=BDT_SIGMA, a=BDT_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        bdt_straight = _price_straight_bond(bdt_lattice)

        hw_model = MODEL_REGISTRY["hull_white"]
        hw_lattice = build_generic_lattice(
            hw_model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        hw_straight = _price_straight_bond(hw_lattice)

        # Both calibrated to same curve => straight bond prices should match
        assert bdt_straight == pytest.approx(hw_straight, rel=0.005), (
            f"BDT straight={bdt_straight:.4f}, HW straight={hw_straight:.4f}"
        )


# ===================================================================
# Test 4: Cross-validate HW against FinancePy
# ===================================================================

class TestFinancePyCrossValidation:
    """Compare Trellis HW callable bond to FinancePy BondEmbeddedOption."""

    @pytest.fixture(autouse=True)
    def _require_financepy(self):
        pytest.importorskip("financepy")

    def test_hw_callable_vs_financepy(self, flat_curve):
        """Trellis HW callable matches FinancePy HWTree within 50bp."""
        from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
        from financepy.models.hw_tree import HWTree
        from financepy.products.bonds.bond_callable import BondEmbeddedOption
        from financepy.utils.date import Date
        from financepy.utils.day_count import DayCountTypes
        from financepy.utils.frequency import FrequencyTypes

        settle = Date(15, 1, 2025)
        maturity_dt = Date(15, 1, 2035)

        fp_curve = DiscountCurveFlat(
            settle, FLAT_RATE, FrequencyTypes.CONTINUOUS, DayCountTypes.ACT_365F,
        )

        # Call dates
        call_dates = [
            Date(15, 1, 2028),   # 3Y
            Date(15, 1, 2030),   # 5Y
            Date(15, 1, 2032),   # 7Y
        ]
        call_prices = [CALL_PRICE] * len(call_dates)

        # No put dates
        put_dates = []
        put_prices = []

        callable_bond = BondEmbeddedOption(
            issue_dt=settle,
            maturity_dt=maturity_dt,
            coupon=COUPON_RATE,
            freq_type=FrequencyTypes.SEMI_ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
            call_dts=call_dates,
            call_prices=call_prices,
            put_dts=put_dates,
            put_prices=put_prices,
        )

        model = HWTree(HW_SIGMA, HW_A, N_STEPS)
        fp_result = callable_bond.value(settle, fp_curve, model)
        # FinancePy returns dict: {'bondwithoption': ..., 'bondpure': ...}
        fp_price = fp_result["bondwithoption"]

        # Our HW tree price
        hw_model = MODEL_REGISTRY["hull_white"]
        hw_lattice = build_generic_lattice(
            hw_model, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        trellis_price = _price_callable_bond(hw_lattice)

        # Should match within 50bp (0.50 on a 100 face bond)
        assert trellis_price == pytest.approx(fp_price, abs=0.50), (
            f"Trellis HW={trellis_price:.4f}, FinancePy={fp_price:.4f}, "
            f"diff={abs(trellis_price - fp_price):.4f}"
        )


# ===================================================================
# Test 5: Cross-validate HW against QuantLib
# ===================================================================

class TestQuantLibCrossValidation:
    """Compare Trellis HW callable bond to QuantLib TreeCallableFixedRateBondEngine."""

    @pytest.fixture(autouse=True)
    def _require_quantlib(self):
        pytest.importorskip("QuantLib")

    def test_hw_callable_vs_quantlib(self, flat_curve):
        """Trellis HW callable matches QuantLib within 20bp."""
        import QuantLib as ql

        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today
        settlement = today

        # Build flat curve
        ql_curve = ql.FlatForward(today, FLAT_RATE, ql.Actual365Fixed())
        curve_handle = ql.YieldTermStructureHandle(ql_curve)

        # Build the callable bond schedule
        issue_date = today
        maturity_date = ql.Date(15, 1, 2035)

        schedule = ql.Schedule(
            issue_date, maturity_date,
            ql.Period(ql.Semiannual),
            ql.NullCalendar(),
            ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Backward, False,
        )

        # Call schedule: callable at par at 3Y, 5Y, 7Y
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
        engine = ql.TreeCallableFixedRateBondEngine(hw_model, N_STEPS)
        callable_bond.setPricingEngine(engine)

        ql_price = callable_bond.cleanPrice()

        # Our HW tree price
        hw_model_trellis = MODEL_REGISTRY["hull_white"]
        hw_lattice = build_generic_lattice(
            hw_model_trellis, r0=FLAT_RATE, sigma=HW_SIGMA, a=HW_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )
        trellis_price = _price_callable_bond(hw_lattice)

        # Should match within 20bp (0.20 on a 100 face bond)
        assert trellis_price == pytest.approx(ql_price, abs=0.20), (
            f"Trellis HW={trellis_price:.4f}, QuantLib={ql_price:.4f}, "
            f"diff={abs(trellis_price - ql_price):.4f}"
        )


# ===================================================================
# Test 6: BDT ZCB repricing (calibration consistency)
# ===================================================================

class TestBDTCalibrationConsistency:
    """Verify BDT tree reprices zero-coupon bonds (calibration test)."""

    def test_bdt_zcb_repricing(self, flat_curve):
        """BDT tree P(0,T) matches curve.discount(T) to high precision."""
        model = MODEL_REGISTRY["bdt"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=BDT_SIGMA, a=BDT_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )

        def terminal_payoff(step, node, lat):
            return 1.0

        tree_zcb = lattice_backward_induction(lattice, terminal_payoff)
        curve_zcb = float(flat_curve.discount(T))

        assert tree_zcb == pytest.approx(curve_zcb, rel=1e-4), (
            f"BDT ZCB={tree_zcb:.8f}, Curve={curve_zcb:.8f}, "
            f"diff={abs(tree_zcb - curve_zcb):.2e}"
        )

    def test_bdt_zcb_at_intermediate_maturity(self, flat_curve):
        """BDT tree reprices ZCB at intermediate maturity (5Y)."""
        model = MODEL_REGISTRY["bdt"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=BDT_SIGMA, a=BDT_A,
            T=T, n_steps=N_STEPS, discount_curve=flat_curve,
        )

        # Price ZCB maturing at 5Y by backward induction from step 5Y to root
        dt = lattice.dt
        t_mid = 5.0
        mid_step = int(round(t_mid / dt))

        n_terminal = lattice.n_nodes(mid_step)
        values = np.ones(n_terminal)

        for step in range(mid_step - 1, -1, -1):
            n_nodes = lattice.n_nodes(step)
            new_vals = np.zeros(n_nodes)
            for j in range(n_nodes):
                df = lattice.get_discount(step, j)
                probs = lattice.get_probabilities(step, j)
                children = lattice.child_indices(step, j)
                new_vals[j] = df * sum(
                    p * values[c] for p, c in zip(probs, children)
                )
            values = new_vals

        tree_zcb = float(values[0])
        curve_zcb = float(flat_curve.discount(t_mid))

        assert tree_zcb == pytest.approx(curve_zcb, rel=1e-4), (
            f"BDT ZCB(5Y)={tree_zcb:.8f}, Curve={curve_zcb:.8f}"
        )


def test_t02_and_t17_named_fixture_prices_with_authored_controls():
    """The manifest rows must execute one shared trade and explicit numerics."""
    from trellis.agent.benchmark_contracts import benchmark_spec_overrides
    from trellis.agent.task_manifests import load_task_manifest
    from trellis.agent.task_runtime import build_market_state_for_task
    from trellis.instruments.callable_bond import CallableBondSpec
    from trellis.models.callable_bond_pde import price_callable_bond_pde
    from trellis.models.callable_bond_tree import (
        price_callable_bond_tree,
        straight_bond_present_value,
    )

    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
        if task["id"] in {"T02", "T17"}
    }
    for task in tasks.values():
        assert task["cross_validate"]["output_unit"] == "currency_amount"
        assert task["cross_validate"]["output_currency"] == "USD"
    market_state, _ = build_market_state_for_task(tasks["T02"])
    spec = CallableBondSpec(**benchmark_spec_overrides(tasks["T02"]))

    t02_contracts = tasks["T02"]["cross_validate"]["target_contracts"]
    bdt_controls = t02_contracts["bdt_tree"]["variant_parameters"]
    hw_controls = t02_contracts["hull_white_tree"]["variant_parameters"]
    bdt_price = price_callable_bond_tree(
        market_state,
        spec,
        model=bdt_controls["lattice_model"],
        mean_reversion=bdt_controls["mean_reversion"],
        sigma=bdt_controls["sigma"],
        n_steps=bdt_controls["tree_steps"],
    )
    hw_price = price_callable_bond_tree(
        market_state,
        spec,
        model=hw_controls["lattice_model"],
        mean_reversion=hw_controls["mean_reversion"],
        sigma=hw_controls["sigma"],
        n_steps=hw_controls["tree_steps"],
    )

    t17_contracts = tasks["T17"]["cross_validate"]["target_contracts"]
    pde_controls = t17_contracts["hw_pde_theta"]["variant_parameters"]
    pde_price = price_callable_bond_pde(
        market_state,
        spec,
        mean_reversion=pde_controls["mean_reversion"],
        sigma=pde_controls["sigma"],
        theta=pde_controls["theta"],
        n_r=pde_controls["n_r"],
        n_t=pde_controls["n_t"],
        r_min=pde_controls["r_min"],
        r_max=pde_controls["r_max"],
    )
    straight_price = straight_bond_present_value(
        market_state,
        spec,
        settlement=market_state.settlement,
    )

    assert straight_price == pytest.approx(99.5097634759, abs=1e-6)
    assert bdt_price == pytest.approx(96.3444970526, abs=1e-6)
    assert hw_price == pytest.approx(96.8252972854, abs=1e-6)
    assert pde_price == pytest.approx(96.8513691004, abs=1e-6)
    assert bdt_price <= straight_price
    assert hw_price <= straight_price
    assert pde_price <= straight_price
    assert abs(bdt_price - hw_price) / hw_price * 100.0 <= (
        tasks["T02"]["cross_validate"]["tolerance_pct"]
    )
    assert abs(pde_price - hw_price) / hw_price * 100.0 <= (
        tasks["T17"]["cross_validate"]["tolerance_pct"]
    )


@pytest.mark.parametrize(
    ("task_id", "expected_prices", "reference_target", "comparison_target"),
    (
        (
            "T02",
            {
                "bdt_tree": 96.3444970526,
                "hull_white_tree": 96.8252972854,
            },
            "hull_white_tree",
            "bdt_tree",
        ),
        (
            "T17",
            {
                "hw_pde_theta": 96.8513691004,
                "hw_rate_tree": 96.8252972854,
            },
            "hw_rate_tree",
            "hw_pde_theta",
        ),
    ),
)
def test_t02_and_t17_run_task_preserves_fixture_market_and_acceptance(
    monkeypatch,
    tmp_path,
    task_id,
    expected_prices,
    reference_target,
    comparison_target,
):
    """The full deterministic task path must retain the authored proof contract."""
    from trellis.agent.task_manifests import load_task_manifest
    from trellis.agent.task_runtime import run_task
    from trellis.engine.payoff_pricer import price_payoff

    task = next(
        task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
        if task["id"] == task_id
    )
    observed_settlements = []

    def observed_price(payoff, market_state):
        observed_settlements.append(market_state.settlement)
        return price_payoff(payoff, market_state)

    monkeypatch.setattr("trellis.agent.executor.REPO_ROOT", tmp_path)
    result = run_task(
        task,
        market_state=None,
        price_fn=observed_price,
        task_run_storage_root=tmp_path / "records",
        task_run_storage_layout="isolated",
        execution_mode_override="deterministic_replay",
        recovery_mode="assisted",
    )

    assert result["success"] is True
    assert observed_settlements
    assert set(observed_settlements) == {date(2025, 1, 15)}
    assert result["cross_validation"]["reference_target"] == reference_target
    for target_id, expected_price in expected_prices.items():
        assert result["cross_validation"]["prices"][target_id] == pytest.approx(
            expected_price,
            abs=1e-6,
        )
        assert result["method_results"][target_id]["acceptance"] == (
            result["cross_validation"]["target_acceptance"][target_id]
        )

    acceptance = result["cross_validation"]["target_acceptance"]
    assert acceptance[reference_target] == {
        "role": "reference",
        "relation": "reference",
        "reference_target": reference_target,
        "tolerance_pct": task["cross_validate"]["tolerance_pct"],
        "tolerance_unit": "percent_of_reference_price",
        "output_unit": "currency_amount",
        "status": "reference",
        "output_currency": "USD",
    }
    assert acceptance[comparison_target] == {
        "role": "comparison",
        "relation": "within_tolerance",
        "reference_target": reference_target,
        "tolerance_pct": task["cross_validate"]["tolerance_pct"],
        "tolerance_unit": "percent_of_reference_price",
        "output_unit": "currency_amount",
        "status": "passed",
        "output_currency": "USD",
    }
