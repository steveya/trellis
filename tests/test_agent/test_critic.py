"""Tests for critic agent and arbiter."""

from datetime import date
from unittest.mock import patch

import pytest

from trellis.agent.arbiter import run_critic_tests, ValidationResult
from trellis.agent.critic import CriticConcern, available_critic_checks
from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.bond import Bond
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class TestCriticConcern:

    def test_frozen(self):
        c = CriticConcern("price_non_negative", "test concern", "error")
        with pytest.raises(AttributeError):
            c.description = "changed"


class TestRunCriticTests:

    def test_structured_non_negative_check_passes(self):
        concerns = [
            CriticConcern(
                "price_non_negative",
                "price should stay non-negative",
                "error",
            ),
        ]
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        payoff = DeterministicCashflowPayoff(bond)

        failures = run_critic_tests(concerns, payoff)
        assert failures == []

    def test_structured_volatility_input_check_fails(self):
        concerns = [
            CriticConcern(
                "volatility_input_usage",
                "volatility should affect the price",
                "error",
                evidence="evaluate() ignores market_state.vol_surface",
                remediation="Read vol_surface or use a vol-sensitive primitive.",
            ),
        ]

        class VolBlindOption:
            @property
            def requirements(self):
                return {"discount", "black_vol"}

            def evaluate(self, ms):
                return 10.0

        payoff = VolBlindOption()

        failures = run_critic_tests(concerns, payoff)
        assert len(failures) == 1
        assert "volatility_input_usage" in failures[0]
        assert "relative_change" in failures[0]

    def test_warning_severity_skipped(self):
        concerns = [
            CriticConcern("price_non_negative", "just a warning", "warning"),
        ]
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        payoff = DeterministicCashflowPayoff(bond)

        failures = run_critic_tests(concerns, payoff)
        assert failures == []  # warnings are not run

    def test_legacy_test_code_still_supported(self):
        concerns = [
            CriticConcern(
                "legacy_test_code",
                "price should exceed 200",
                "error",
                test_code="assert price_payoff(payoff, ms) > 200",
            ),
        ]
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        payoff = DeterministicCashflowPayoff(bond)

        failures = run_critic_tests(concerns, payoff)
        assert len(failures) == 1
        assert "price should exceed 200" in failures[0]

    def test_broken_legacy_test_code_skipped(self):
        concerns = [
            CriticConcern(
                "legacy_test_code",
                "bad code",
                "error",
                test_code="undefined_variable + 1",
            ),
        ]
        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                     maturity=10, frequency=2)
        payoff = DeterministicCashflowPayoff(bond)

        failures = run_critic_tests(concerns, payoff)
        assert failures == []


def test_critique_includes_shared_knowledge(monkeypatch):
    from trellis.agent.critic import critique

    captured = {}

    def fake_llm_generate_json(prompt, model=None, max_retries=None):
        captured["prompt"] = prompt
        captured["max_retries"] = max_retries
        return []

    monkeypatch.setattr("trellis.agent.critic.llm_generate_json", fake_llm_generate_json)
    critique(
        "def price():\n    return 0.0",
        "Demo instrument",
        knowledge_context="## Shared Failure Memory\n- Avoid double discounting.",
        available_checks=available_critic_checks(instrument_type="european_option"),
    )

    assert "Shared Failure Memory" in captured["prompt"]
    assert "Avoid double discounting" in captured["prompt"]
    assert "price_non_negative" in captured["prompt"]


def test_critique_can_disable_text_fallback(monkeypatch):
    from trellis.agent.critic import critique

    def fail_json(prompt, model=None, max_retries=None):
        raise TimeoutError("json timed out")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("text fallback should be disabled")

    monkeypatch.setattr("trellis.agent.critic.llm_generate_json", fail_json)
    monkeypatch.setattr("trellis.agent.critic.llm_generate", fail_if_called)

    with pytest.raises(TimeoutError, match="json timed out"):
        critique(
            "def price():\n    return 0.0",
            "Demo instrument",
            available_checks=available_critic_checks(instrument_type="callable_bond"),
            allow_text_fallback=False,
            json_max_retries=0,
        )


def test_critique_parses_legacy_test_code_payload(monkeypatch):
    from trellis.agent.critic import critique

    monkeypatch.setattr(
        "trellis.agent.critic.llm_generate_json",
        lambda prompt, model=None, max_retries=None: [
            {
                "description": "legacy finding",
                "test_code": "assert True",
                "severity": "error",
            }
        ],
    )

    concerns = critique(
        "def price():\n    return 0.0",
        "Demo instrument",
        available_checks=available_critic_checks(instrument_type="callable_bond"),
    )

    assert len(concerns) == 1
    assert concerns[0].check_id == "legacy_test_code"
    assert concerns[0].test_code == "assert True"


def test_available_critic_checks_for_callable_bond():
    checks = available_critic_checks(instrument_type="callable_bond")
    ids = [check.check_id for check in checks]
    assert ids == [
        "volatility_input_usage",
        "rate_sensitivity_present",
        "callable_bound_vs_straight_bond",
    ]


def test_available_critic_checks_for_european_option():
    checks = available_critic_checks(instrument_type="european_option", method="analytical")
    ids = [check.check_id for check in checks]
    assert ids == [
        "price_non_negative",
        "volatility_input_usage",
    ]


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
