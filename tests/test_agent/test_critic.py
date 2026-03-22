"""Tests for critic agent and arbiter."""

from datetime import date
from unittest.mock import patch

import pytest

from trellis.agent.arbiter import run_critic_tests, ValidationResult
from trellis.agent.critic import CriticConcern
from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.bond import Bond
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class TestCriticConcern:

    def test_frozen(self):
        c = CriticConcern("test", "assert True", "error")
        with pytest.raises(AttributeError):
            c.description = "changed"


class TestRunCriticTests:

    def test_passing_assertion(self):
        concerns = [
            CriticConcern("price is positive", "assert price_payoff(payoff, ms) > 0", "error"),
        ]
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        payoff = DeterministicCashflowPayoff(bond)

        failures = run_critic_tests(concerns, payoff)
        assert failures == []

    def test_failing_assertion(self):
        concerns = [
            CriticConcern(
                "price should exceed 200",
                "assert price_payoff(payoff, ms) > 200",
                "error",
            ),
        ]
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        payoff = DeterministicCashflowPayoff(bond)

        failures = run_critic_tests(concerns, payoff)
        assert len(failures) == 1
        assert "price should exceed 200" in failures[0]

    def test_warning_severity_skipped(self):
        concerns = [
            CriticConcern("just a warning", "assert False", "warning"),
        ]
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        payoff = DeterministicCashflowPayoff(bond)

        failures = run_critic_tests(concerns, payoff)
        assert failures == []  # warnings are not run

    def test_broken_test_code_skipped(self):
        """If the critic's test code itself has an error, it's skipped."""
        concerns = [
            CriticConcern("bad code", "undefined_variable + 1", "error"),
        ]
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        payoff = DeterministicCashflowPayoff(bond)

        failures = run_critic_tests(concerns, payoff)
        assert failures == []  # broken test code is skipped, not a failure


class TestInvariantExpanded:

    def test_bounded_by_reference_pass(self):
        from trellis.agent.invariants import check_bounded_by_reference

        bond = Bond(face=100, coupon=0.03, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)

        def payoff_factory():
            return DeterministicCashflowPayoff(bond)

        # A higher-coupon bond as reference (always worth more)
        bond_ref = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                         maturity=10, frequency=2)

        def reference_factory():
            return DeterministicCashflowPayoff(bond_ref)

        def ms_factory(rate=0.05, vol=0.20):
            return MarketState(
                as_of=SETTLE, settlement=SETTLE,
                discount=YieldCurve.flat(rate),
                vol_surface=FlatVol(vol),
            )

        failures = check_bounded_by_reference(
            payoff_factory, reference_factory, ms_factory, relation="<=",
        )
        assert failures == []

    def test_bounded_by_reference_fail(self):
        from trellis.agent.invariants import check_bounded_by_reference

        # Swap: higher coupon as "payoff", lower as "reference" — should fail
        bond_high = Bond(face=100, coupon=0.08, maturity_date=date(2034, 11, 15),
                          maturity=10, frequency=2)
        bond_low = Bond(face=100, coupon=0.02, maturity_date=date(2034, 11, 15),
                         maturity=10, frequency=2)

        def payoff_factory():
            return DeterministicCashflowPayoff(bond_high)

        def reference_factory():
            return DeterministicCashflowPayoff(bond_low)

        def ms_factory(rate=0.05, vol=0.20):
            return MarketState(
                as_of=SETTLE, settlement=SETTLE,
                discount=YieldCurve.flat(rate),
                vol_surface=FlatVol(vol),
            )

        failures = check_bounded_by_reference(
            payoff_factory, reference_factory, ms_factory, relation="<=",
        )
        assert len(failures) > 0
