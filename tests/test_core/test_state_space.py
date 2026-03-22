"""Tests for StateSpace and ScenarioWeightedPayoff."""

from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.core.state_space import StateSpace
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.bond import Bond
from trellis.instruments.scenario_weighted import ScenarioWeightedPayoff


SETTLE = date(2024, 11, 15)


def _ms(rate):
    """MarketState with a flat curve."""
    return MarketState(as_of=SETTLE, settlement=SETTLE, discount=YieldCurve.flat(rate))


def _zero_coupon(maturity_years=10):
    return Bond(
        face=100, coupon=0.0,
        maturity_date=date(2024 + maturity_years, 11, 15),
        maturity=maturity_years, frequency=2,
        issue_date=SETTLE,
    )


class TestStateSpace:

    def test_construction(self):
        ss = StateSpace(states={
            "cut": (0.4, _ms(0.04)),
            "hold": (0.4, _ms(0.05)),
            "hike": (0.2, _ms(0.06)),
        })
        assert ss.state_names == ["cut", "hold", "hike"]
        assert ss.probability("cut") == 0.4
        assert ss.market_state("hold").discount is not None

    def test_probabilities_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to"):
            StateSpace(states={"a": (0.3, _ms(0.04)), "b": (0.3, _ms(0.05))})

    def test_probabilities_tolerance(self):
        """Probabilities within 1% of 1.0 are accepted."""
        ss = StateSpace(states={"a": (0.502, _ms(0.04)), "b": (0.502, _ms(0.05))})
        assert ss.state_names == ["a", "b"]


class TestScenarioWeightedPayoff:

    def test_requirements(self):
        bond = _zero_coupon()
        inner = DeterministicCashflowPayoff(bond)
        sw = ScenarioWeightedPayoff(inner)
        assert sw.requirements == {"state_space"}

    def test_two_state_zero_coupon(self):
        """Probability-weighted ZCB price matches manual calculation."""
        bond = _zero_coupon(10)
        inner = DeterministicCashflowPayoff(bond)

        ms_base = _ms(0.04)
        ms_stress = _ms(0.06)

        ss = StateSpace(states={
            "base": (0.5, ms_base),
            "stress": (0.5, ms_stress),
        })
        outer_ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.05),  # doesn't matter for outer
            state_space=ss,
        )

        sw = ScenarioWeightedPayoff(inner)
        pv = price_payoff(sw, outer_ms)

        # Manual: 0.5 * 100*exp(-0.04*10) + 0.5 * 100*exp(-0.06*10)
        expected = 0.5 * 100 * np.exp(-0.04 * 10) + 0.5 * 100 * np.exp(-0.06 * 10)
        assert pv == pytest.approx(expected, rel=0.01)

    def test_three_state_manual(self):
        """Three states: rally, unchanged, selloff."""
        bond = _zero_coupon(5)
        inner = DeterministicCashflowPayoff(bond)

        ss = StateSpace(states={
            "rally": (0.3, _ms(0.03)),
            "unchanged": (0.5, _ms(0.04)),
            "selloff": (0.2, _ms(0.06)),
        })
        outer_ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.04),
            state_space=ss,
        )

        pv = price_payoff(ScenarioWeightedPayoff(inner), outer_ms)
        expected = (
            0.3 * 100 * np.exp(-0.03 * 5)
            + 0.5 * 100 * np.exp(-0.04 * 5)
            + 0.2 * 100 * np.exp(-0.06 * 5)
        )
        assert pv == pytest.approx(expected, rel=0.01)

    def test_single_state_equals_direct(self):
        """Single state (p=1.0) should equal direct pricing."""
        bond = _zero_coupon(10)
        inner = DeterministicCashflowPayoff(bond)

        cond_ms = _ms(0.05)
        ss = StateSpace(states={"only": (1.0, cond_ms)})
        outer_ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            state_space=ss,
        )

        weighted_pv = price_payoff(ScenarioWeightedPayoff(inner), outer_ms)
        direct_pv = price_payoff(inner, cond_ms)
        assert weighted_pv == pytest.approx(direct_pv, rel=1e-10)

    def test_missing_capability_in_conditional_raises(self):
        """Inner payoff needs black_vol but conditional state doesn't have it."""
        from trellis.instruments.cap import CapFloorSpec, CapPayoff
        from trellis.core.types import Frequency

        cap = CapPayoff(CapFloorSpec(
            notional=1e6, strike=0.05,
            start_date=date(2025, 2, 15), end_date=date(2026, 2, 15),
            frequency=Frequency.QUARTERLY,
        ))

        # Conditional MarketState has no vol surface
        ss = StateSpace(states={"base": (1.0, _ms(0.05))})
        outer_ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            state_space=ss,
        )

        with pytest.raises(MissingCapabilityError) as exc_info:
            price_payoff(ScenarioWeightedPayoff(cap), outer_ms)
        assert "black_vol" in exc_info.value.missing


class TestSessionWithStateSpace:

    def test_session_with_state_space(self):
        from trellis.session import Session

        s = Session(curve=YieldCurve.flat(0.05), settlement=SETTLE)
        ss = StateSpace(states={
            "base": (0.5, _ms(0.04)),
            "stress": (0.5, _ms(0.06)),
        })
        s2 = s.with_state_space(ss)
        assert s2 is not s
        assert s2.state_space is not None

        bond = _zero_coupon(10)
        inner = DeterministicCashflowPayoff(bond)
        pv = s2.price_payoff(ScenarioWeightedPayoff(inner))
        assert pv > 0
