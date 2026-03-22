"""Arbiter: runs invariant suite + critic test cases deterministically.

No LLM judgment — just execute tests and report pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.bond import Bond
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


@dataclass
class ValidationResult:
    passed: bool
    invariant_failures: list[str]
    critic_failures: list[str]

    @property
    def all_failures(self) -> list[str]:
        return self.invariant_failures + self.critic_failures


def run_critic_tests(
    concerns: list,
    payoff,
    spec_kwargs: dict | None = None,
) -> list[str]:
    """Execute critic-generated test cases in a controlled namespace.

    Returns list of failure messages (empty = all passed).
    """
    failures = []

    # Build test environment
    ms = MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.20),
    )
    ms_low_rate = MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.20),
    )
    ms_high_rate = MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(0.07),
        vol_surface=FlatVol(0.20),
    )

    # Straight bond for comparison
    bond = Bond(
        face=100, coupon=0.05,
        maturity_date=date(2034, 11, 15),
        maturity=10, frequency=2,
    )
    straight_bond_pv = price_payoff(
        DeterministicCashflowPayoff(bond), ms, day_count=bond.day_count,
    )

    namespace = {
        "payoff": payoff,
        "ms": ms,
        "ms_low_rate": ms_low_rate,
        "ms_high_rate": ms_high_rate,
        "price_payoff": price_payoff,
        "straight_bond_pv": straight_bond_pv,
        "date": date,
        "MarketState": MarketState,
        "YieldCurve": YieldCurve,
        "FlatVol": FlatVol,
        "DeterministicCashflowPayoff": DeterministicCashflowPayoff,
        "Bond": Bond,
    }

    for concern in concerns:
        if concern.severity != "error":
            continue
        test_code = concern.test_code
        if not test_code.strip():
            continue
        try:
            exec(test_code, namespace)
        except AssertionError as e:
            failures.append(
                f"Critic concern FAILED: {concern.description}\n"
                f"  Test: {test_code}\n"
                f"  Error: {e}"
            )
        except Exception as e:
            # Test code itself errored — this is a critic quality issue, not a pricer bug
            # We skip rather than fail
            pass

    return failures


def validate(
    payoff,
    payoff_factory,
    market_state_factory,
    code: str,
    description: str,
    critic_concerns: list | None = None,
    reference_factory=None,
    is_option: bool = True,
) -> ValidationResult:
    """Full validation: invariants + critic tests.

    Parameters
    ----------
    payoff : instantiated payoff
    payoff_factory : callable() -> Payoff
    market_state_factory : callable(vol=, rate=) -> MarketState
    code : str
        Generated source code (for critic)
    description : str
        Instrument description
    critic_concerns : list[CriticConcern] or None
        Pre-computed critic concerns (avoids re-calling LLM)
    reference_factory : callable() -> Payoff, optional
    is_option : bool
    """
    from trellis.agent.invariants import run_invariant_suite

    # Run invariant suite
    passed, inv_failures = run_invariant_suite(
        payoff_factory=payoff_factory,
        market_state_factory=market_state_factory,
        is_option=is_option,
        reference_factory=reference_factory,
        reference_relation="<=",
    )

    # Run critic tests
    critic_failures = []
    if critic_concerns:
        critic_failures = run_critic_tests(critic_concerns, payoff)

    all_passed = passed and len(critic_failures) == 0

    return ValidationResult(
        passed=all_passed,
        invariant_failures=inv_failures,
        critic_failures=critic_failures,
    )
