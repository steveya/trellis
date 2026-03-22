"""Invariant checks for agent-built payoffs.

Three categories:
- Protocol conformance (structural)
- Price bounds and monotonicity (no-arbitrage)
- Scenario consistency (rate sensitivity, bounding)
"""

from __future__ import annotations

from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.payoff import Payoff
from trellis.engine.payoff_pricer import price_payoff


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def check_protocol_conformance(cls, market_state: MarketState) -> list[str]:
    """Verify cls satisfies the Payoff protocol."""
    failures = []
    if not hasattr(cls, "requirements"):
        failures.append(f"{cls.__name__} missing 'requirements' property")
        return failures
    if not hasattr(cls, "evaluate"):
        failures.append(f"{cls.__name__} missing 'evaluate' method")
        return failures
    try:
        instance = cls.__new__(cls)
        if not isinstance(instance, Payoff):
            failures.append(f"{cls.__name__} does not satisfy Payoff protocol")
    except Exception:
        pass
    return failures


# ---------------------------------------------------------------------------
# Price bounds
# ---------------------------------------------------------------------------

def check_non_negativity(payoff: Payoff, market_state: MarketState) -> list[str]:
    """Price must be non-negative for option-like payoffs."""
    failures = []
    try:
        pv = price_payoff(payoff, market_state)
        if pv < -1e-6:
            failures.append(f"Price is negative: {pv:.6f}")
    except Exception as e:
        failures.append(f"Pricing failed: {e}")
    return failures


def check_upper_bound(
    payoff_factory,
    bound_factory,
    market_state: MarketState,
    description: str = "upper bound",
) -> list[str]:
    """Payoff price must be ≤ bound price (e.g. callable ≤ straight bond)."""
    failures = []
    try:
        pv = price_payoff(payoff_factory(), market_state)
        bound = price_payoff(bound_factory(), market_state)
        if pv > bound + 1e-4:
            failures.append(
                f"Price ({pv:.4f}) exceeds {description} ({bound:.4f})"
            )
    except Exception as e:
        failures.append(f"Bound check failed: {e}")
    return failures


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

def check_vol_monotonicity(
    payoff_factory,
    market_state_factory,
    vol_range: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40),
) -> list[str]:
    """Price must be non-decreasing in vol for option payoffs."""
    failures = []
    prices = []
    for vol in vol_range:
        try:
            pv = price_payoff(payoff_factory(), market_state_factory(vol=vol))
            prices.append((vol, pv))
        except Exception as e:
            failures.append(f"Pricing failed at vol={vol}: {e}")
            return failures

    for i in range(len(prices) - 1):
        v1, p1 = prices[i]
        v2, p2 = prices[i + 1]
        if p2 < p1 - 1e-6:
            failures.append(
                f"Vol monotonicity violated: price({v1:.2f})={p1:.4f} > "
                f"price({v2:.2f})={p2:.4f}"
            )
    return failures


def check_rate_monotonicity(
    payoff_factory,
    market_state_factory,
    rate_range: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08),
    expected_direction: str = "decreasing",
) -> list[str]:
    """Price should move monotonically with rates.

    ``expected_direction``: "decreasing" (bonds) or "increasing" (receiver options).
    """
    failures = []
    prices = []
    for rate in rate_range:
        try:
            pv = price_payoff(payoff_factory(), market_state_factory(rate=rate))
            prices.append((rate, pv))
        except Exception as e:
            failures.append(f"Pricing failed at rate={rate}: {e}")
            return failures

    for i in range(len(prices) - 1):
        r1, p1 = prices[i]
        r2, p2 = prices[i + 1]
        if expected_direction == "decreasing" and p2 > p1 + 1e-4:
            failures.append(
                f"Rate monotonicity violated: price({r1:.2%})={p1:.4f} < "
                f"price({r2:.2%})={p2:.4f} (expected decreasing)"
            )
        elif expected_direction == "increasing" and p2 < p1 - 1e-4:
            failures.append(
                f"Rate monotonicity violated: price({r1:.2%})={p1:.4f} > "
                f"price({r2:.2%})={p2:.4f} (expected increasing)"
            )
    return failures


# ---------------------------------------------------------------------------
# Scenario / bounding checks
# ---------------------------------------------------------------------------

def check_bounded_by_reference(
    payoff_factory,
    reference_factory,
    market_state_factory,
    rate_range: tuple[float, ...] = (0.02, 0.05, 0.08),
    relation: str = "<=",
) -> list[str]:
    """Check that payoff price is bounded by a reference across rate scenarios.

    ``relation``: "<=" means payoff ≤ reference (e.g. callable ≤ straight).
    """
    failures = []
    for rate in rate_range:
        try:
            ms = market_state_factory(rate=rate)
            pv = price_payoff(payoff_factory(), ms)
            ref = price_payoff(reference_factory(), ms)
            if relation == "<=" and pv > ref + 1e-4:
                failures.append(
                    f"At rate={rate:.2%}: payoff ({pv:.4f}) > reference ({ref:.4f})"
                )
            elif relation == ">=" and pv < ref - 1e-4:
                failures.append(
                    f"At rate={rate:.2%}: payoff ({pv:.4f}) < reference ({ref:.4f})"
                )
        except Exception as e:
            failures.append(f"Bound check at rate={rate} failed: {e}")
    return failures


def check_zero_vol_intrinsic(
    payoff_factory,
    market_state_factory,
    intrinsic_fn,
    tol: float = 0.01,
) -> list[str]:
    """At zero vol, option price should equal intrinsic value."""
    failures = []
    try:
        pv = price_payoff(payoff_factory(), market_state_factory(vol=1e-10))
        intrinsic = intrinsic_fn(market_state_factory(vol=1e-10))
        if abs(pv - intrinsic) > tol * max(abs(intrinsic), 1.0):
            failures.append(
                f"Zero-vol price ({pv:.6f}) != intrinsic ({intrinsic:.6f}), "
                f"diff={abs(pv - intrinsic):.6f}"
            )
    except Exception as e:
        failures.append(f"Zero-vol check failed: {e}")
    return failures


# ---------------------------------------------------------------------------
# Suite runners
# ---------------------------------------------------------------------------

def run_invariant_suite(
    payoff_factory,
    market_state_factory,
    intrinsic_fn=None,
    is_option: bool = True,
    reference_factory=None,
    reference_relation: str = "<=",
) -> tuple[bool, list[str]]:
    """Run all applicable invariant checks.

    Parameters
    ----------
    payoff_factory : callable() -> Payoff
    market_state_factory : callable(vol=float, rate=float) -> MarketState
    intrinsic_fn : optional
    is_option : bool
    reference_factory : callable() -> Payoff, optional
        A reference payoff that bounds the agent-built payoff.
    reference_relation : str
        "<=" or ">=" — the expected relationship.

    Returns
    -------
    (all_passed, failure_messages)
    """
    all_failures = []

    # Non-negativity
    try:
        ms = market_state_factory(vol=0.20)
    except TypeError:
        ms = market_state_factory(rate=0.05)
    all_failures.extend(check_non_negativity(payoff_factory(), ms))

    # Vol monotonicity (option payoffs)
    if is_option:
        all_failures.extend(
            check_vol_monotonicity(payoff_factory, market_state_factory)
        )
        if intrinsic_fn is not None:
            all_failures.extend(
                check_zero_vol_intrinsic(
                    payoff_factory, market_state_factory, intrinsic_fn
                )
            )

    # Bounding by reference across rate scenarios
    if reference_factory is not None:
        all_failures.extend(
            check_bounded_by_reference(
                payoff_factory, reference_factory, market_state_factory,
                relation=reference_relation,
            )
        )

    return len(all_failures) == 0, all_failures
