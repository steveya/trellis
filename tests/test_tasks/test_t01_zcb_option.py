"""T01: ZCB option pricing — Jamshidian analytical + Ho-Lee tree + HW tree.

Cross-validates:
  1. Jamshidian closed-form (our implementation)
  2. Ho-Lee tree (a=0, equal probs) via build_generic_lattice
  3. Hull-White tree (a=0.1) via build_generic_lattice
  4. QuantLib HullWhite.discountBondOption (analytical benchmark)

Parameters (following Hull, Ch. 28):
  flat curve = 5%, sigma_HW = 0.01, a = 0.1 (HW) / 0 (Ho-Lee),
  T_exp = 3, T_bond = 9, K = 63 per $100 face (= 0.63 per unit face),
  face = 100, n_steps = 200.
"""

from __future__ import annotations

import numpy as np
import pytest

from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical.jamshidian import zcb_option_hw
from trellis.models.trees.lattice import (
    RecombiningLattice,
    build_generic_lattice,
    lattice_backward_induction,
)
from trellis.models.trees.models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

FLAT_RATE = 0.05
SIGMA = 0.01
A_HW = 0.1
A_HL = 0.0          # Ho-Lee = HW with a=0
T_EXP = 3.0
T_BOND = 9.0
K_UNIT = 0.63       # strike per unit face
K_100 = 63.0        # strike per $100 face
FACE = 100.0
N_STEPS = 200


@pytest.fixture(scope="module")
def flat_curve():
    return YieldCurve.flat(FLAT_RATE, max_tenor=max(T_BOND + 1, 31.0))


# ---------------------------------------------------------------------------
# Helper: price ZCB option on a rate lattice
# ---------------------------------------------------------------------------

def _price_zcb_option_on_lattice(
    lattice: RecombiningLattice,
    t_exp: float,
    t_bond: float,
    strike: float,
    face: float,
    option_type: str = "call",
) -> float:
    """Price a European option on a ZCB using a calibrated rate lattice.

    Strategy:
      1. At each node at the expiry step, compute the ZCB price
         P(T_exp, T_bond) by backward induction from the bond maturity step
         down to the expiry step.
      2. Compute option payoff = max(ZCB*face - strike, 0) for a call.
      3. Roll back the option payoff from expiry to today.

    The lattice must cover at least T_bond (i.e., n_steps * dt >= T_bond).
    """
    dt = lattice.dt
    n_steps = lattice.n_steps

    # Step indices for expiry and bond maturity
    exp_step = int(round(t_exp / dt))
    bond_step = int(round(t_bond / dt))

    if bond_step > n_steps:
        raise ValueError(
            f"Lattice too short: bond_step={bond_step} > n_steps={n_steps}"
        )

    # --- Phase 1: Compute ZCB price P(T_exp, T_bond) at each expiry node ---
    # Backward induction from bond_step to exp_step (terminal payoff = 1.0)
    n_terminal = lattice.n_nodes(bond_step)
    zcb_values = np.ones(n_terminal)  # ZCB pays 1 at maturity

    for step in range(bond_step - 1, exp_step - 1, -1):
        n_nodes = lattice.n_nodes(step)
        new_vals = np.zeros(n_nodes)
        for j in range(n_nodes):
            df = lattice.get_discount(step, j)
            probs = lattice.get_probabilities(step, j)
            children = lattice.child_indices(step, j)
            new_vals[j] = df * sum(
                p * zcb_values[c] for p, c in zip(probs, children)
            )
        zcb_values = new_vals

    # zcb_values now holds P(T_exp, T_bond) at each node of exp_step

    # --- Phase 2: Option payoff at expiry ---
    n_exp_nodes = lattice.n_nodes(exp_step)
    if option_type == "call":
        option_payoff = np.array([
            max(zcb_values[j] * face - strike, 0.0)
            for j in range(n_exp_nodes)
        ])
    else:  # put
        option_payoff = np.array([
            max(strike - zcb_values[j] * face, 0.0)
            for j in range(n_exp_nodes)
        ])

    # --- Phase 3: Roll back option from expiry to root ---
    values = option_payoff
    for step in range(exp_step - 1, -1, -1):
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

    return float(values[0])


# ===================================================================
# 1. Jamshidian analytical tests
# ===================================================================

class TestJamshidianAnalytical:
    """Jamshidian closed-form vs QuantLib analytical benchmark."""

    def test_call_hw(self, flat_curve):
        """Jamshidian call (a=0.1) matches QuantLib."""
        result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, A_HW)
        # QuantLib reference: 0.09549344
        assert result["call"] == pytest.approx(0.09549344, rel=1e-5)

    def test_put_hw(self, flat_curve):
        """Jamshidian put (a=0.1) matches QuantLib."""
        result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, A_HW)
        # QuantLib reference: 0.00011131
        assert result["put"] == pytest.approx(0.00011131, rel=1e-3)

    def test_call_ho_lee(self, flat_curve):
        """Jamshidian call (a~0, Ho-Lee limit) matches QuantLib."""
        result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, a=1e-10)
        # QuantLib reference (HW with a=1e-10): 0.09694275
        assert result["call"] == pytest.approx(0.09694275, rel=1e-4)

    def test_put_ho_lee(self, flat_curve):
        """Jamshidian put (a~0, Ho-Lee limit) matches QuantLib."""
        result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, a=1e-10)
        # QuantLib reference: 0.00156063
        assert result["put"] == pytest.approx(0.00156063, rel=1e-3)

    def test_put_call_parity(self, flat_curve):
        """Put-call parity: C - P = P(0,T_bond) - K * P(0,T_exp)."""
        result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, A_HW)
        P_exp = flat_curve.discount(T_EXP)
        P_bond = flat_curve.discount(T_BOND)
        parity = P_bond - K_UNIT * P_exp
        assert result["call"] - result["put"] == pytest.approx(parity, rel=1e-10)


# ===================================================================
# 2. Ho-Lee tree test
# ===================================================================

class TestHoLeeTree:
    """ZCB option on a Ho-Lee lattice vs Jamshidian analytical (a=0)."""

    def test_zcb_option_call(self, flat_curve):
        """Ho-Lee tree call matches Jamshidian (a~0) within 1%."""
        model = MODEL_REGISTRY["ho_lee"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=SIGMA, a=A_HL,
            T=T_BOND, n_steps=N_STEPS, discount_curve=flat_curve,
        )

        tree_call = _price_zcb_option_on_lattice(
            lattice, T_EXP, T_BOND, K_100, FACE, option_type="call",
        )

        analytical = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, a=1e-10)
        ref_call = analytical["call"] * FACE

        assert tree_call == pytest.approx(ref_call, rel=0.01), (
            f"Ho-Lee tree call={tree_call:.4f}, analytical={ref_call:.4f}"
        )

    def test_zcb_option_put(self, flat_curve):
        """Ho-Lee tree put matches Jamshidian (a~0) within 5%."""
        model = MODEL_REGISTRY["ho_lee"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=SIGMA, a=A_HL,
            T=T_BOND, n_steps=N_STEPS, discount_curve=flat_curve,
        )

        tree_put = _price_zcb_option_on_lattice(
            lattice, T_EXP, T_BOND, K_100, FACE, option_type="put",
        )

        analytical = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, a=1e-10)
        ref_put = analytical["put"] * FACE

        # Put is very small (~0.156) so use wider tolerance
        assert tree_put == pytest.approx(ref_put, rel=0.05), (
            f"Ho-Lee tree put={tree_put:.4f}, analytical={ref_put:.4f}"
        )


# ===================================================================
# 3. Hull-White tree test
# ===================================================================

class TestHullWhiteTree:
    """ZCB option on a HW lattice vs Jamshidian analytical (a=0.1)."""

    def test_zcb_option_call(self, flat_curve):
        """HW tree call matches Jamshidian (a=0.1) within 1%."""
        model = MODEL_REGISTRY["hull_white"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=SIGMA, a=A_HW,
            T=T_BOND, n_steps=N_STEPS, discount_curve=flat_curve,
        )

        tree_call = _price_zcb_option_on_lattice(
            lattice, T_EXP, T_BOND, K_100, FACE, option_type="call",
        )

        analytical = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, A_HW)
        ref_call = analytical["call"] * FACE

        assert tree_call == pytest.approx(ref_call, rel=0.01), (
            f"HW tree call={tree_call:.4f}, analytical={ref_call:.4f}"
        )

    def test_zcb_option_put(self, flat_curve):
        """HW tree put matches Jamshidian (a=0.1) within 10%."""
        model = MODEL_REGISTRY["hull_white"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=SIGMA, a=A_HW,
            T=T_BOND, n_steps=N_STEPS, discount_curve=flat_curve,
        )

        tree_put = _price_zcb_option_on_lattice(
            lattice, T_EXP, T_BOND, K_100, FACE, option_type="put",
        )

        analytical = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, A_HW)
        ref_put = analytical["put"] * FACE

        # Put is tiny (~0.011) so use wider tolerance
        assert tree_put == pytest.approx(ref_put, rel=0.10), (
            f"HW tree put={tree_put:.4f}, analytical={ref_put:.4f}"
        )


# ===================================================================
# 4. Cross-validation against QuantLib
# ===================================================================

class TestQuantLibCrossValidation:
    """Compare Trellis Jamshidian to QuantLib HullWhite.discountBondOption."""

    @pytest.fixture(autouse=True)
    def _require_quantlib(self):
        pytest.importorskip("QuantLib")

    def _ql_zcb_option(self, a: float, sigma: float):
        """Compute ZCB option price via QuantLib HW model."""
        import QuantLib as ql

        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today
        curve = ql.FlatForward(today, FLAT_RATE, ql.Actual365Fixed())
        handle = ql.YieldTermStructureHandle(curve)
        hw = ql.HullWhite(handle, a, sigma)

        call_price = hw.discountBondOption(
            ql.Option.Call, K_UNIT, T_EXP, T_BOND,
        )
        put_price = hw.discountBondOption(
            ql.Option.Put, K_UNIT, T_EXP, T_BOND,
        )
        return {"call": call_price, "put": put_price}

    def test_hw_call_vs_quantlib(self, flat_curve):
        """Trellis Jamshidian call (a=0.1) matches QuantLib to 6 digits."""
        ql_result = self._ql_zcb_option(A_HW, SIGMA)
        trellis_result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, A_HW)
        assert trellis_result["call"] == pytest.approx(ql_result["call"], rel=1e-6)

    def test_hw_put_vs_quantlib(self, flat_curve):
        """Trellis Jamshidian put (a=0.1) matches QuantLib to 6 digits."""
        ql_result = self._ql_zcb_option(A_HW, SIGMA)
        trellis_result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, A_HW)
        assert trellis_result["put"] == pytest.approx(ql_result["put"], rel=1e-6)

    def test_ho_lee_call_vs_quantlib(self, flat_curve):
        """Trellis Jamshidian call (a~0) matches QuantLib Ho-Lee."""
        a_small = 1e-10
        ql_result = self._ql_zcb_option(a_small, SIGMA)
        trellis_result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, a_small)
        assert trellis_result["call"] == pytest.approx(ql_result["call"], rel=1e-6)

    def test_ho_lee_put_vs_quantlib(self, flat_curve):
        """Trellis Jamshidian put (a~0) matches QuantLib Ho-Lee."""
        a_small = 1e-10
        ql_result = self._ql_zcb_option(a_small, SIGMA)
        trellis_result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, a_small)
        assert trellis_result["put"] == pytest.approx(ql_result["put"], rel=1e-6)


# ===================================================================
# 5. Cross-validation against FinancePy HWTree
# ===================================================================

class TestFinancePyCrossValidation:
    """Compare Trellis HW tree to FinancePy HWTree on a European ZCB option.

    FinancePy's BondOption requires a Bond (not BondZero), so we use a
    very small coupon (1e-8) as a proxy for a zero-coupon bond and compare
    to our analytical Jamshidian formula.
    """

    @pytest.fixture(autouse=True)
    def _require_financepy(self):
        pytest.importorskip("financepy")

    def test_hw_analytical_matches_financepy_tree(self, flat_curve):
        """FinancePy HWTree call on near-ZCB ~ Trellis Jamshidian analytical."""
        from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
        from financepy.models.hw_tree import HWTree
        from financepy.products.bonds.bond import Bond
        from financepy.products.bonds.bond_option import BondOption
        from financepy.utils.date import Date
        from financepy.utils.day_count import DayCountTypes
        from financepy.utils.frequency import FrequencyTypes
        from financepy.utils.global_types import OptionTypes

        settle = Date(15, 1, 2025)
        expiry_dt = Date(15, 1, 2028)
        maturity_dt = Date(15, 1, 2034)

        fp_curve = DiscountCurveFlat(
            settle, FLAT_RATE, FrequencyTypes.CONTINUOUS, DayCountTypes.ACT_365F,
        )

        # FinancePy requires Bond (not BondZero) for BondOption.
        # Use a tiny coupon as proxy for ZCB.
        bond = Bond(
            issue_dt=settle,
            maturity_dt=maturity_dt,
            coupon=1e-8,
            freq_type=FrequencyTypes.ANNUAL,
            dc_type=DayCountTypes.ACT_365F,
        )

        model = HWTree(SIGMA, A_HW, 200)

        call_opt = BondOption(bond, expiry_dt, K_100, OptionTypes.EUROPEAN_CALL)
        fp_call = call_opt.value(settle, fp_curve, model)

        put_opt = BondOption(bond, expiry_dt, K_100, OptionTypes.EUROPEAN_PUT)
        fp_put = put_opt.value(settle, fp_curve, model)

        # Our analytical reference
        trellis_result = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, A_HW)
        ref_call = trellis_result["call"] * FACE
        ref_put = trellis_result["put"] * FACE

        # FinancePy tree has discretization error; allow 5% for call
        assert fp_call == pytest.approx(ref_call, rel=0.05), (
            f"FinancePy call={fp_call:.4f}, Jamshidian={ref_call:.4f}"
        )
        # Put is tiny, so use absolute tolerance
        assert fp_put == pytest.approx(ref_put, abs=0.05), (
            f"FinancePy put={fp_put:.4f}, Jamshidian={ref_put:.4f}"
        )


# ===================================================================
# 6. Consistency: tree ZCB reprices the discount curve
# ===================================================================

class TestTreeCalibrationConsistency:
    """Verify that the calibrated lattice reprices ZCBs correctly."""

    def test_ho_lee_zcb_repricing(self, flat_curve):
        """Ho-Lee tree P(0,T_bond) matches curve.discount(T_bond)."""
        model = MODEL_REGISTRY["ho_lee"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=SIGMA, a=A_HL,
            T=T_BOND, n_steps=N_STEPS, discount_curve=flat_curve,
        )

        # Price a ZCB via full backward induction
        def terminal_payoff(step, node, lat):
            return 1.0

        tree_zcb = lattice_backward_induction(lattice, terminal_payoff)
        curve_zcb = float(flat_curve.discount(T_BOND))

        assert tree_zcb == pytest.approx(curve_zcb, rel=1e-4), (
            f"Tree ZCB={tree_zcb:.8f}, Curve={curve_zcb:.8f}"
        )

    def test_hw_zcb_repricing(self, flat_curve):
        """HW tree P(0,T_bond) matches curve.discount(T_bond)."""
        model = MODEL_REGISTRY["hull_white"]
        lattice = build_generic_lattice(
            model, r0=FLAT_RATE, sigma=SIGMA, a=A_HW,
            T=T_BOND, n_steps=N_STEPS, discount_curve=flat_curve,
        )

        def terminal_payoff(step, node, lat):
            return 1.0

        tree_zcb = lattice_backward_induction(lattice, terminal_payoff)
        curve_zcb = float(flat_curve.discount(T_BOND))

        assert tree_zcb == pytest.approx(curve_zcb, rel=1e-4), (
            f"Tree ZCB={tree_zcb:.8f}, Curve={curve_zcb:.8f}"
        )
