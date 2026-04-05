"""Arbiter: runs invariant suite + deterministic critic-selected checks.

No LLM judgment — just execute deterministic checks and report pass/fail.
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
    """Aggregate outcome of invariant checks plus critic-authored test cases."""
    passed: bool
    invariant_failures: list[str]
    critic_failures: list[str]

    @property
    def all_failures(self) -> list[str]:
        """Return invariant and critic failures concatenated into one list."""
        return self.invariant_failures + self.critic_failures


def run_critic_tests(
    concerns: list,
    payoff,
    spec_kwargs: dict | None = None,
    *,
    allowed_check_ids: set[str] | None = None,
) -> list[str]:
    """Execute critic-selected checks in a controlled namespace.

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
    ms_low_vol = MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.05),
    )
    ms_high_vol = MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.40),
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
        "ms_low_vol": ms_low_vol,
        "ms_high_vol": ms_high_vol,
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
        check_id = str(getattr(concern, "check_id", "") or "").strip()
        if allowed_check_ids is not None and check_id not in allowed_check_ids:
            continue
        dispatched = _run_structured_critic_check(concern, namespace)
        if dispatched is not None:
            if dispatched:
                failures.append(_format_structured_failure(concern, dispatched))
            continue

    return failures


def _format_structured_failure(concern, detail: str) -> str:
    """Format one deterministic structured-check failure for diagnostics."""

    lines = [f"Critic concern FAILED [{concern.check_id}]: {concern.description}"]
    if concern.evidence:
        lines.append(f"  Evidence: {concern.evidence}")
    lines.append(f"  Detail: {detail}")
    if concern.remediation:
        lines.append(f"  Remediation: {concern.remediation}")
    return "\n".join(lines)


def _run_structured_critic_check(concern, namespace: dict) -> str | None:
    """Run one structured critic-selected check.

    Returns:
    - ``None`` when the concern is unsupported.
    - ``""`` when the check passed.
    - non-empty string when the check failed.
    """

    check_id = str(getattr(concern, "check_id", "") or "").strip()
    if not check_id:
        return None
    checker = _CRITIC_CHECK_DISPATCH.get(check_id)
    if checker is None:
        return None
    try:
        return checker(namespace)
    except Exception:
        return ""


def _check_price_non_negative(namespace: dict) -> str:
    pv = float(namespace["price_payoff"](namespace["payoff"], namespace["ms"]))
    if pv < -1e-6:
        return f"price_payoff(payoff, ms)={pv:.6f} is negative"
    return ""


def _check_volatility_input_usage(namespace: dict) -> str:
    payoff = namespace["payoff"]
    price_payoff_fn = namespace["price_payoff"]
    low = float(price_payoff_fn(payoff, namespace["ms_low_vol"]))
    high = float(price_payoff_fn(payoff, namespace["ms_high_vol"]))
    scale = max(abs(low), abs(high), 1.0)
    change = abs(high - low) / scale
    if change < 1e-3:
        return (
            f"volatility sensitivity too small: low_vol={low:.6f}, "
            f"high_vol={high:.6f}, relative_change={change:.6%}"
        )
    return ""


def _check_rate_sensitivity_present(namespace: dict) -> str:
    payoff = namespace["payoff"]
    price_payoff_fn = namespace["price_payoff"]
    low = float(price_payoff_fn(payoff, namespace["ms_low_rate"]))
    high = float(price_payoff_fn(payoff, namespace["ms_high_rate"]))
    scale = max(abs(low), abs(high), 1.0)
    change = abs(high - low) / scale
    if change < 1e-4:
        return (
            f"discount-rate sensitivity too small: low_rate={low:.6f}, "
            f"high_rate={high:.6f}, relative_change={change:.6%}"
        )
    return ""


def _check_callable_bound_vs_straight_bond(namespace: dict) -> str:
    pv = float(namespace["price_payoff"](namespace["payoff"], namespace["ms"]))
    straight = float(namespace["straight_bond_pv"])
    if pv > straight + 1e-6:
        return (
            f"callable PV {pv:.6f} exceeds straight bond PV {straight:.6f}"
        )
    return ""


def _check_puttable_bound_vs_straight_bond(namespace: dict) -> str:
    pv = float(namespace["price_payoff"](namespace["payoff"], namespace["ms"]))
    straight = float(namespace["straight_bond_pv"])
    if pv + 1e-6 < straight:
        return (
            f"puttable PV {pv:.6f} is below straight bond PV {straight:.6f}"
        )
    return ""


_CRITIC_CHECK_DISPATCH = {
    "price_non_negative": _check_price_non_negative,
    "volatility_input_usage": _check_volatility_input_usage,
    "rate_sensitivity_present": _check_rate_sensitivity_present,
    "callable_bound_vs_straight_bond": _check_callable_bound_vs_straight_bond,
    "puttable_bound_vs_straight_bond": _check_puttable_bound_vs_straight_bond,
}


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
