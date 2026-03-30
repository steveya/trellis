"""Invariant checks for agent-built payoffs.

Three categories:
- Protocol conformance (structural)
- Price bounds and monotonicity (no-arbitrage)
- Scenario consistency (rate sensitivity, bounding)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

from trellis.core.market_state import MarketState
from trellis.core.payoff import Payoff
from trellis.engine.payoff_pricer import price_payoff


@dataclass(frozen=True)
class InvariantFailure:
    """Structured failure payload for deterministic validation checks."""

    check: str
    message: str
    actual: Any | None = None
    expected: Any | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)


def _market_state_context(market_state: MarketState) -> dict[str, Any]:
    """Capture a compact, machine-readable market-state snapshot for diagnostics."""
    return {
        "settlement": (
            market_state.settlement.isoformat()
            if getattr(market_state, "settlement", None) is not None
            else None
        ),
        "available_capabilities": sorted(market_state.available_capabilities),
        "spot": getattr(market_state, "spot", None),
        "underlier_spot_keys": sorted((market_state.underlier_spots or {}).keys()),
        "fx_pairs": sorted((market_state.fx_rates or {}).keys()),
        "forecast_curve_keys": sorted((market_state.forecast_curves or {}).keys()),
        "model_parameter_keys": sorted((market_state.model_parameters or {}).keys()),
        "discount_curve_type": (
            type(market_state.discount).__name__
            if getattr(market_state, "discount", None) is not None
            else None
        ),
        "vol_surface_type": (
            type(market_state.vol_surface).__name__
            if getattr(market_state, "vol_surface", None) is not None
            else None
        ),
    }


def _emit_failures(
    failures: list[InvariantFailure],
    *,
    return_diagnostics: bool,
) -> list[InvariantFailure] | list[str]:
    """Convert failure objects to the requested output format.

    If return_diagnostics is True, returns the full InvariantFailure objects
    (used by the validation bundle runner). Otherwise returns plain string
    messages (used by callers that just need human-readable output).
    """
    if return_diagnostics:
        return failures
    return [failure.message for failure in failures]


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

def check_price_sanity(
    payoff: Payoff,
    market_state: MarketState,
    max_multiple: float = 10.0,
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Price must be within a sane range (not 100× notional)."""
    failures: list[InvariantFailure] = []
    try:
        pv = price_payoff(payoff, market_state)
        # Heuristic: price shouldn't exceed max_multiple × 100 (typical notional)
        if abs(pv) > max_multiple * 100:
            failures.append(
                InvariantFailure(
                    check="check_price_sanity",
                    message=(
                        f"Price sanity check failed: |PV| = {abs(pv):.2f}, which exceeds "
                        f"{max_multiple}× typical notional (100). The model likely has a "
                        f"unit conversion error (e.g., passing Black vol to a rate tree)."
                    ),
                    actual=float(pv),
                    expected=f"|PV| <= {max_multiple * 100:.4f}",
                    context={
                        **_market_state_context(market_state),
                        "max_multiple": max_multiple,
                    },
                )
            )
    except Exception as e:
        failures.append(
            InvariantFailure(
                check="check_price_sanity",
                message=f"Price sanity check failed: {e}",
                exception_type=type(e).__name__,
                exception_message=str(e),
                context={
                    **_market_state_context(market_state),
                    "max_multiple": max_multiple,
                },
            )
        )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


def check_non_negativity(
    payoff: Payoff,
    market_state: MarketState,
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Price must be non-negative for option-like payoffs."""
    failures: list[InvariantFailure] = []
    try:
        pv = price_payoff(payoff, market_state)
        if pv < -1e-6:
            failures.append(
                InvariantFailure(
                    check="check_non_negativity",
                    message=f"Price is negative: {pv:.6f}",
                    actual=float(pv),
                    expected="PV >= 0.0",
                    context=_market_state_context(market_state),
                )
            )
    except Exception as e:
        failures.append(
            InvariantFailure(
                check="check_non_negativity",
                message=f"Pricing failed: {e}",
                exception_type=type(e).__name__,
                exception_message=str(e),
                context=_market_state_context(market_state),
            )
        )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


def check_upper_bound(
    payoff_factory,
    bound_factory,
    market_state: MarketState,
    description: str = "upper bound",
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Payoff price must be ≤ bound price (e.g. callable ≤ straight bond)."""
    failures: list[InvariantFailure] = []
    try:
        pv = price_payoff(payoff_factory(), market_state)
        bound = price_payoff(bound_factory(), market_state)
        if pv > bound + 1e-4:
            failures.append(
                InvariantFailure(
                    check="check_upper_bound",
                    message=f"Price ({pv:.4f}) exceeds {description} ({bound:.4f})",
                    actual=float(pv),
                    expected=f"<= {bound:.4f}",
                    context={
                        **_market_state_context(market_state),
                        "bound_description": description,
                        "bound_value": float(bound),
                    },
                )
            )
    except Exception as e:
        failures.append(
            InvariantFailure(
                check="check_upper_bound",
                message=f"Bound check failed: {e}",
                exception_type=type(e).__name__,
                exception_message=str(e),
                context={
                    **_market_state_context(market_state),
                    "bound_description": description,
                },
            )
        )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

def check_vol_monotonicity(
    payoff_factory,
    market_state_factory,
    vol_range: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40),
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Price must be non-decreasing in vol for option payoffs."""
    failures: list[InvariantFailure] = []
    prices = []
    for vol in vol_range:
        try:
            pv = price_payoff(payoff_factory(), market_state_factory(vol=vol))
            prices.append((vol, pv))
        except Exception as e:
            failures.append(
                InvariantFailure(
                    check="check_vol_monotonicity",
                    message=f"Pricing failed at vol={vol}: {e}",
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    context={
                        "vol": float(vol),
                        "sampled_prices": [[float(v), float(p)] for v, p in prices],
                    },
                )
            )
            return _emit_failures(failures, return_diagnostics=return_diagnostics)

    for i in range(len(prices) - 1):
        v1, p1 = prices[i]
        v2, p2 = prices[i + 1]
        if p2 < p1 - 1e-6:
            failures.append(
                InvariantFailure(
                    check="check_vol_monotonicity",
                    message=(
                        f"Vol monotonicity violated: price({v1:.2f})={p1:.4f} > "
                        f"price({v2:.2f})={p2:.4f}"
                    ),
                    actual=float(p2),
                    expected=f">= {p1:.4f}",
                    context={
                        "violating_pair": [[float(v1), float(p1)], [float(v2), float(p2)]],
                        "sampled_prices": [[float(v), float(p)] for v, p in prices],
                    },
                )
            )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


def check_rate_monotonicity(
    payoff_factory,
    market_state_factory,
    rate_range: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08),
    expected_direction: str = "decreasing",
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Price should move monotonically with rates.

    ``expected_direction``: "decreasing" (bonds) or "increasing" (receiver options).
    """
    failures: list[InvariantFailure] = []
    prices = []
    for rate in rate_range:
        try:
            pv = price_payoff(payoff_factory(), market_state_factory(rate=rate))
            prices.append((rate, pv))
        except Exception as e:
            failures.append(
                InvariantFailure(
                    check="check_rate_monotonicity",
                    message=f"Pricing failed at rate={rate}: {e}",
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    context={
                        "rate": float(rate),
                        "expected_direction": expected_direction,
                        "sampled_prices": [[float(r), float(p)] for r, p in prices],
                    },
                )
            )
            return _emit_failures(failures, return_diagnostics=return_diagnostics)

    for i in range(len(prices) - 1):
        r1, p1 = prices[i]
        r2, p2 = prices[i + 1]
        if expected_direction == "decreasing" and p2 > p1 + 1e-4:
            failures.append(
                InvariantFailure(
                    check="check_rate_monotonicity",
                    message=(
                        f"Rate monotonicity violated: price({r1:.2%})={p1:.4f} < "
                        f"price({r2:.2%})={p2:.4f} (expected decreasing)"
                    ),
                    actual=float(p2),
                    expected=f"<= {p1:.4f}",
                    context={
                        "expected_direction": expected_direction,
                        "violating_pair": [[float(r1), float(p1)], [float(r2), float(p2)]],
                        "sampled_prices": [[float(r), float(p)] for r, p in prices],
                    },
                )
            )
        elif expected_direction == "increasing" and p2 < p1 - 1e-4:
            failures.append(
                InvariantFailure(
                    check="check_rate_monotonicity",
                    message=(
                        f"Rate monotonicity violated: price({r1:.2%})={p1:.4f} > "
                        f"price({r2:.2%})={p2:.4f} (expected increasing)"
                    ),
                    actual=float(p2),
                    expected=f">= {p1:.4f}",
                    context={
                        "expected_direction": expected_direction,
                        "violating_pair": [[float(r1), float(p1)], [float(r2), float(p2)]],
                        "sampled_prices": [[float(r), float(p)] for r, p in prices],
                    },
                )
            )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


def check_vol_sensitivity(
    payoff_factory,
    market_state_factory,
    vol_low: float = 0.05,
    vol_high: float = 0.40,
    min_change_pct: float = 0.01,
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Price of an instrument with embedded optionality MUST change with vol.

    Any instrument whose requirements include 'black_vol' has embedded
    optionality. If the price doesn't change when vol changes, the model
    is not capturing the option component (e.g., using a spot tree for a
    rate derivative).

    Parameters
    ----------
    min_change_pct : float
        Minimum relative price change between vol_low and vol_high.
        Default 1%.
    """
    failures: list[InvariantFailure] = []
    try:
        p_low = price_payoff(payoff_factory(), market_state_factory(vol=vol_low))
        p_high = price_payoff(payoff_factory(), market_state_factory(vol=vol_high))
        base = max(abs(p_low), abs(p_high), 1.0)
        change = abs(p_high - p_low) / base
        if change < min_change_pct:
            failures.append(
                InvariantFailure(
                    check="check_vol_sensitivity",
                    message=(
                        f"Vol sensitivity too low: price({vol_low:.0%})={p_low:.4f}, "
                        f"price({vol_high:.0%})={p_high:.4f}, change={change:.4%}. "
                        f"Instruments with embedded options must have non-zero vega."
                    ),
                    actual=float(change),
                    expected=f">= {min_change_pct:.4%}",
                    context={
                        "vol_low": float(vol_low),
                        "vol_high": float(vol_high),
                        "price_low": float(p_low),
                        "price_high": float(p_high),
                    },
                )
            )
    except Exception as e:
        failures.append(
            InvariantFailure(
                check="check_vol_sensitivity",
                message=f"Vol sensitivity check failed: {e}",
                exception_type=type(e).__name__,
                exception_message=str(e),
                context={
                    "vol_low": float(vol_low),
                    "vol_high": float(vol_high),
                },
            )
        )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


# ---------------------------------------------------------------------------
# Scenario / bounding checks
# ---------------------------------------------------------------------------

def check_bounded_by_reference(
    payoff_factory,
    reference_factory,
    market_state_factory,
    rate_range: tuple[float, ...] = (0.02, 0.05, 0.08),
    relation: str = "<=",
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Check that payoff price is bounded by a reference across rate scenarios.

    ``relation``: "<=" means payoff ≤ reference (e.g. callable ≤ straight).
    """
    failures: list[InvariantFailure] = []
    for rate in rate_range:
        try:
            ms = market_state_factory(rate=rate)
            pv = price_payoff(payoff_factory(), ms)
            ref = price_payoff(reference_factory(), ms)
            if relation == "<=" and pv > ref + 1e-4:
                failures.append(
                    InvariantFailure(
                        check="check_bounded_by_reference",
                        message=f"At rate={rate:.2%}: payoff ({pv:.4f}) > reference ({ref:.4f})",
                        actual=float(pv),
                        expected=f"<= {ref:.4f}",
                        context={"rate": float(rate), "relation": relation},
                    )
                )
            elif relation == ">=" and pv < ref - 1e-4:
                failures.append(
                    InvariantFailure(
                        check="check_bounded_by_reference",
                        message=f"At rate={rate:.2%}: payoff ({pv:.4f}) < reference ({ref:.4f})",
                        actual=float(pv),
                        expected=f">= {ref:.4f}",
                        context={"rate": float(rate), "relation": relation},
                    )
                )
        except Exception as e:
            failures.append(
                InvariantFailure(
                    check="check_bounded_by_reference",
                    message=f"Bound check at rate={rate} failed: {e}",
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    context={"rate": float(rate), "relation": relation},
                )
            )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


def check_zero_vol_intrinsic(
    payoff_factory,
    market_state_factory,
    intrinsic_fn,
    tol: float = 0.01,
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """At zero vol, option price should equal intrinsic value."""
    failures: list[InvariantFailure] = []
    try:
        pv = price_payoff(payoff_factory(), market_state_factory(vol=1e-10))
        intrinsic = intrinsic_fn(market_state_factory(vol=1e-10))
        if abs(pv - intrinsic) > tol * max(abs(intrinsic), 1.0):
            failures.append(
                InvariantFailure(
                    check="check_zero_vol_intrinsic",
                    message=(
                        f"Zero-vol price ({pv:.6f}) != intrinsic ({intrinsic:.6f}), "
                        f"diff={abs(pv - intrinsic):.6f}"
                    ),
                    actual=float(pv),
                    expected=f"{intrinsic:.6f} +/- {tol:.4%}",
                    context={"intrinsic": float(intrinsic), "tol": float(tol)},
                )
            )
    except Exception as e:
        failures.append(
            InvariantFailure(
                check="check_zero_vol_intrinsic",
                message=f"Zero-vol check failed: {e}",
                exception_type=type(e).__name__,
                exception_message=str(e),
                context={"tol": float(tol)},
            )
        )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


def check_quanto_required_inputs(
    payoff: Payoff,
    market_state: MarketState,
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Quanto routes must bind the cross-currency inputs declared by the family contract."""
    required = {
        "discount_curve",
        "forward_curve",
        "black_vol_surface",
        "fx_rates",
        "spot",
        "model_parameters",
    }
    available = market_state.available_capabilities
    missing = sorted(required - available)
    if not missing:
        return _emit_failures([], return_diagnostics=return_diagnostics)
    return _emit_failures(
        [
            InvariantFailure(
                check="check_quanto_required_inputs",
                message="Quanto validation missing required market inputs: " + ", ".join(missing),
                expected="all quanto market inputs present",
                context={
                    **_market_state_context(market_state),
                    "missing_capabilities": list(missing),
                },
            )
        ],
        return_diagnostics=return_diagnostics,
    )


def check_quanto_cross_currency_semantics(
    payoff_factory,
    market_state_factory,
    corr_range: tuple[float, ...] = (-0.5, 0.0, 0.5),
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Quanto prices should react to the underlier/FX correlation input."""
    failures: list[InvariantFailure] = []
    prices = []
    for corr in corr_range:
        try:
            pv = price_payoff(payoff_factory(), market_state_factory(corr=corr))
            prices.append((corr, pv))
        except Exception as e:
            failures.append(
                InvariantFailure(
                    check="check_quanto_cross_currency_semantics",
                    message=f"Quanto correlation sensitivity check failed at corr={corr}: {e}",
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    context={
                        "corr": float(corr),
                        "sampled_prices": [[float(c), float(p)] for c, p in prices],
                    },
                )
            )
            return _emit_failures(failures, return_diagnostics=return_diagnostics)

    if len({round(price, 8) for _, price in prices}) <= 1:
        failures.append(
            InvariantFailure(
                check="check_quanto_cross_currency_semantics",
                message=(
                    "Quanto cross-currency semantics check failed: price is insensitive "
                    "to underlier/FX correlation."
                ),
                context={"sampled_prices": [[float(c), float(p)] for c, p in prices]},
            )
            )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


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

    # Vol sensitivity — any instrument with embedded optionality must have non-zero vega
    if is_option:
        all_failures.extend(
            check_vol_sensitivity(payoff_factory, market_state_factory)
        )
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
