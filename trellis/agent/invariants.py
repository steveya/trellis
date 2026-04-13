"""Invariant checks for agent-built payoffs.

Three categories:
- Protocol conformance (structural)
- Price bounds and monotonicity (no-arbitrage)
- Scenario consistency (rate sensitivity, bounding)
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
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


def _default_market_state(market_state_factory):
    """Instantiate a default validation market state from a factory with optional kwargs."""
    try:
        return market_state_factory()
    except TypeError:
        return market_state_factory(rate=0.05, vol=0.20, corr=0.35)


def _extract_spec(payoff: Payoff):
    """Return the bound spec object for a payoff when present."""
    spec = getattr(payoff, "spec", None)
    if spec is None:
        spec = getattr(payoff, "_spec", None)
    return spec


def _clone_spec_with_updates(spec, **updates):
    """Clone one spec object with field updates."""
    if spec is None:
        raise TypeError("Payoff does not expose a bound spec")
    if is_dataclass(spec):
        return replace(spec, **updates)
    if hasattr(spec, "__dict__"):
        values = dict(vars(spec))
        values.update(updates)
        return type(spec)(**values)
    raise TypeError(f"Unsupported spec clone type: {type(spec).__name__}")


def _clone_payoff_with_spec_updates(payoff: Payoff, **updates) -> Payoff:
    """Clone a payoff by rebuilding it with an updated spec."""
    spec = _extract_spec(payoff)
    cloned_spec = _clone_spec_with_updates(spec, **updates)
    return type(payoff)(cloned_spec)


def _is_cds_like(payoff: Payoff, spec) -> bool:
    """Return whether a payoff/spec pair looks like a single-name CDS route."""
    cls_name = type(payoff).__name__.lower()
    spec_name = type(spec).__name__.lower() if spec is not None else ""
    return (
        "cds" in cls_name
        or "credit_default_swap" in cls_name
        or "cds" in spec_name
        or (
            spec is not None
            and all(hasattr(spec, name) for name in ("spread", "recovery", "start_date", "end_date"))
        )
    )


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


def _cds_spread_unit_hint(payoff: Payoff) -> tuple[str, dict[str, float | str]]:
    """Return a targeted CDS spread-unit hint when the spec looks bps-like."""
    spec = getattr(payoff, "_spec", None)
    spread = getattr(spec, "spread", None)
    if not isinstance(spread, (int, float)):
        return "", {}

    cls_name = type(payoff).__name__.lower()
    spec_name = type(spec).__name__.lower() if spec is not None else ""
    is_cds_like = (
        "cds" in cls_name
        or "credit_default_swap" in cls_name
        or "cds" in spec_name
        or (
            spec is not None
            and all(hasattr(spec, name) for name in ("spread", "recovery", "start_date", "end_date"))
        )
    )
    if not is_cds_like or spread <= 1.0:
        return "", {}

    hint = (
        f" Likely CDS spread-unit issue: running spread {float(spread):.6g} looks like a "
        "basis-point quote. Convert basis points to decimal before accrual "
        "(for example 150 bp -> 0.015)."
    )
    context = {
        "cds_spread_hint": "basis_points_to_decimal",
        "reported_spread": float(spread),
    }
    return hint, context


def _parse_float_vector(raw: object) -> tuple[float, ...]:
    """Parse a comma-delimited float vector used by simple generated specs."""
    if raw in {None, ""}:
        return ()
    values: list[float] = []
    for token in str(raw).split(","):
        piece = token.strip()
        if not piece:
            continue
        try:
            values.append(float(piece))
        except ValueError:
            continue
    return tuple(values)


def _parse_name_vector(raw: object) -> tuple[str, ...]:
    """Parse a comma-delimited name vector used by simple generated specs."""
    if raw in {None, ""}:
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _spot_reference_scale(payoff: Payoff, market_state: MarketState) -> float | None:
    """Return a representative underlier spot scale for spot-based payoff sanity checks."""
    spec = _extract_spec(payoff)
    for attribute in ("spot", "s0", "underlier_spot"):
        value = getattr(spec, attribute, None)
        if isinstance(value, (int, float)) and abs(float(value)) > 0.0:
            return abs(float(value))

    spot_vector = _parse_float_vector(getattr(spec, "spots", None))
    if spot_vector:
        return max(abs(value) for value in spot_vector)

    names = _parse_name_vector(
        getattr(spec, "underliers", None)
        or getattr(spec, "underlyings", None)
        or getattr(spec, "constituents", None)
    )
    if names and getattr(market_state, "underlier_spots", None):
        matched = [
            float(market_state.underlier_spots[name])
            for name in names
            if name in market_state.underlier_spots
        ]
        if matched:
            return max(abs(value) for value in matched)

    spot = getattr(market_state, "spot", None)
    if isinstance(spot, (int, float)) and abs(float(spot)) > 0.0:
        return abs(float(spot))
    return None

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
        spread_hint, spread_context = _cds_spread_unit_hint(payoff)
        spec = getattr(payoff, "spec", None)
        notional = getattr(spec, "notional", None)
        try:
            base_notional = abs(float(notional)) if notional is not None else 100.0
        except (TypeError, ValueError):
            base_notional = 100.0
        if base_notional <= 0.0:
            base_notional = 100.0
        threshold = max_multiple * base_notional
        reference_spot = _spot_reference_scale(payoff, market_state)
        if reference_spot is not None and reference_spot > max_multiple:
            threshold = max(threshold, base_notional * reference_spot)
        if abs(pv) > threshold:
            failures.append(
                InvariantFailure(
                    check="check_price_sanity",
                    message=(
                        f"Price sanity check failed: |PV| = {abs(pv):.2f}, which exceeds "
                        f"{max_multiple}× reference scale ({threshold:.4f}). The model likely has a "
                        f"unit conversion error (e.g., passing Black vol to a rate tree)."
                        f"{spread_hint}"
                    ),
                    actual=float(pv),
                    expected=f"|PV| <= {threshold:.4f}",
                    context={
                        **_market_state_context(market_state),
                        "max_multiple": max_multiple,
                        "reference_notional": base_notional,
                        "reference_spot": reference_spot,
                        **spread_context,
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
        spread_hint, spread_context = _cds_spread_unit_hint(payoff)
        if pv < -1e-6:
            failures.append(
                InvariantFailure(
                    check="check_non_negativity",
                    message=f"Price is negative: {pv:.6f}.{spread_hint}" if spread_hint else f"Price is negative: {pv:.6f}",
                    actual=float(pv),
                    expected="PV >= 0.0",
                    context={**_market_state_context(market_state), **spread_context},
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
    expected_direction: str = "increasing",
    *,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Price should move monotonically with vol in the expected direction."""
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
                        "expected_direction": expected_direction,
                        "sampled_prices": [[float(v), float(p)] for v, p in prices],
                    },
                )
            )
            return _emit_failures(failures, return_diagnostics=return_diagnostics)

    for i in range(len(prices) - 1):
        v1, p1 = prices[i]
        v2, p2 = prices[i + 1]
        if expected_direction == "increasing" and p2 < p1 - 1e-6:
            failures.append(
                InvariantFailure(
                    check="check_vol_monotonicity",
                    message=(
                        f"Vol monotonicity violated: price({v1:.2f})={p1:.4f} > "
                        f"price({v2:.2f})={p2:.4f} (expected increasing)"
                    ),
                    actual=float(p2),
                    expected=f">= {p1:.4f}",
                    context={
                        "expected_direction": expected_direction,
                        "violating_pair": [[float(v1), float(p1)], [float(v2), float(p2)]],
                        "sampled_prices": [[float(v), float(p)] for v, p in prices],
                    },
                )
            )
        elif expected_direction == "decreasing" and p2 > p1 + 1e-6:
            failures.append(
                InvariantFailure(
                    check="check_vol_monotonicity",
                    message=(
                        f"Vol monotonicity violated: price({v1:.2f})={p1:.4f} < "
                        f"price({v2:.2f})={p2:.4f} (expected decreasing)"
                    ),
                    actual=float(p2),
                    expected=f"<= {p1:.4f}",
                    context={
                        "expected_direction": expected_direction,
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

    Any instrument whose requirements include 'black_vol_surface' has embedded
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


def check_cds_spread_quote_normalization(
    payoff_factory,
    market_state_factory,
    *,
    tolerance: float = 1e-4,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Single-name CDS should price equivalent decimal and bps spread quotes the same way."""
    failures: list[InvariantFailure] = []
    try:
        payoff = payoff_factory()
        spec = _extract_spec(payoff)
        if spec is None or not _is_cds_like(payoff, spec):
            return _emit_failures(failures, return_diagnostics=return_diagnostics)
        spread = getattr(spec, "spread", None)
        if not isinstance(spread, (int, float)) or spread <= 0.0:
            return _emit_failures(failures, return_diagnostics=return_diagnostics)

        equivalent_spread = float(spread) * 1e-4 if float(spread) > 1.0 else float(spread) * 1e4
        base_ms = _default_market_state(market_state_factory)
        equivalent_payoff = _clone_payoff_with_spec_updates(payoff, spread=equivalent_spread)
        base_pv = price_payoff(payoff, base_ms)
        equivalent_pv = price_payoff(equivalent_payoff, base_ms)
        scale = max(abs(base_pv), abs(equivalent_pv), 1.0)
        diff = abs(base_pv - equivalent_pv)
        if diff > tolerance * scale:
            failures.append(
                InvariantFailure(
                    check="check_cds_spread_quote_normalization",
                    message=(
                        "CDS spread quote normalization failed: semantically equivalent "
                        f"spreads {float(spread):.6g} and {float(equivalent_spread):.6g} "
                        f"produced materially different PVs ({base_pv:.6f} vs {equivalent_pv:.6f}). "
                        "Single-name CDS routes should treat basis-point and decimal quotes consistently."
                    ),
                    actual=float(base_pv),
                    expected=f"{equivalent_pv:.6f} +/- {tolerance:.4%}",
                    context={
                        "spread_quote": float(spread),
                        "equivalent_spread_quote": float(equivalent_spread),
                        "base_pv": float(base_pv),
                        "equivalent_pv": float(equivalent_pv),
                        "tolerance": float(tolerance),
                    },
                )
            )
    except Exception as e:
        failures.append(
            InvariantFailure(
                check="check_cds_spread_quote_normalization",
                message=f"CDS spread normalization check failed: {e}",
                exception_type=type(e).__name__,
                exception_message=str(e),
                context={"tolerance": float(tolerance)},
            )
        )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


def check_cds_credit_curve_sensitivity(
    payoff_factory,
    market_state_factory,
    hazard_shifts_bps: tuple[float, ...] = (0.0, 50.0, 100.0),
    *,
    tolerance: float = 1e-6,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Long-protection CDS routes should respond positively to higher hazard rates."""
    failures: list[InvariantFailure] = []
    try:
        base_ms = _default_market_state(market_state_factory)
        if getattr(base_ms, "credit_curve", None) is None:
            failures.append(
                InvariantFailure(
                    check="check_cds_credit_curve_sensitivity",
                    message="CDS credit sensitivity check requires a credit curve in MarketState.",
                    expected="market_state.credit_curve is present",
                    context=_market_state_context(base_ms),
                )
            )
            return _emit_failures(failures, return_diagnostics=return_diagnostics)

        prices: list[tuple[float, float]] = []
        for shift in hazard_shifts_bps:
            shifted_curve = base_ms.credit_curve.shift(float(shift))
            shifted_ms = replace(base_ms, credit_curve=shifted_curve)
            pv = price_payoff(payoff_factory(), shifted_ms)
            prices.append((float(shift), float(pv)))

        rounded_prices = {round(price, 8) for _, price in prices}
        if len(rounded_prices) <= 1:
            failures.append(
                InvariantFailure(
                    check="check_cds_credit_curve_sensitivity",
                    message=(
                        "CDS credit sensitivity check failed: price is insensitive to hazard-rate shifts."
                    ),
                    context={"sampled_prices": [[shift, price] for shift, price in prices]},
                )
            )
            return _emit_failures(failures, return_diagnostics=return_diagnostics)

        for index in range(len(prices) - 1):
            shift_a, pv_a = prices[index]
            shift_b, pv_b = prices[index + 1]
            if pv_b < pv_a - tolerance * max(abs(pv_a), 1.0):
                failures.append(
                    InvariantFailure(
                        check="check_cds_credit_curve_sensitivity",
                        message=(
                            "CDS credit sensitivity violated: price should increase for a long-protection "
                            f"CDS when hazard shifts higher, but PV moved from {pv_a:.6f} at +{shift_a:.0f}bp "
                            f"to {pv_b:.6f} at +{shift_b:.0f}bp."
                        ),
                        actual=float(pv_b),
                        expected=f">= {pv_a:.6f}",
                        context={
                            "violating_pair": [[float(shift_a), float(pv_a)], [float(shift_b), float(pv_b)]],
                            "sampled_prices": [[shift, price] for shift, price in prices],
                            "tolerance": float(tolerance),
                        },
                    )
                )
                break
    except Exception as e:
        failures.append(
            InvariantFailure(
                check="check_cds_credit_curve_sensitivity",
                message=f"CDS credit sensitivity check failed: {e}",
                exception_type=type(e).__name__,
                exception_message=str(e),
                context={
                    "hazard_shifts_bps": [float(shift) for shift in hazard_shifts_bps],
                    "tolerance": float(tolerance),
                },
            )
        )
    return _emit_failures(failures, return_diagnostics=return_diagnostics)


def check_rate_style_swaption_helper_consistency(
    payoff_factory,
    market_state_factory,
    *,
    scenarios: tuple[tuple[float, float], ...] = ((0.03, 0.15), (0.05, 0.20), (0.07, 0.30)),
    tolerance: float = 1e-4,
    comparison_kwargs: Mapping[str, object] | None = None,
    return_diagnostics: bool = False,
) -> list[InvariantFailure] | list[str]:
    """Analytical rate-style swaptions should agree with the checked helper surface."""
    from trellis.models.rate_style_swaption import price_swaption_black76

    failures: list[InvariantFailure] = []
    try:
        sample_payoff = payoff_factory()
        spec = _extract_spec(sample_payoff)
    except Exception as e:
        return _emit_failures(
            [
                InvariantFailure(
                    check="check_rate_style_swaption_helper_consistency",
                    message=f"Swaption helper consistency setup failed: {e}",
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    context={"relation": "within_tolerance", "tolerance": float(tolerance)},
                )
            ],
            return_diagnostics=return_diagnostics,
        )

    required_fields = (
        "notional",
        "strike",
        "swap_end",
        "swap_frequency",
        "day_count",
        "is_payer",
    )
    if spec is None or any(not hasattr(spec, field) for field in required_fields):
        return _emit_failures(failures, return_diagnostics=return_diagnostics)

    comparison_kwargs = dict(comparison_kwargs or {})
    sampled_prices: list[dict[str, float]] = []
    for rate, vol in scenarios:
        try:
            market_state = market_state_factory(rate=rate, vol=vol)
            payoff = payoff_factory()
            generated = float(price_payoff(payoff, market_state))
            reference = float(
                price_swaption_black76(
                    market_state,
                    _extract_spec(payoff),
                    **comparison_kwargs,
                )
            )
            sampled_prices.append(
                {
                    "rate": float(rate),
                    "vol": float(vol),
                    "generated": generated,
                    "reference": reference,
                }
            )
            scale = max(abs(reference), 1.0)
            if abs(generated - reference) > tolerance * scale:
                failures.append(
                    InvariantFailure(
                        check="check_rate_style_swaption_helper_consistency",
                        message=(
                            "Rate-style swaption helper consistency failed: generated payoff "
                            f"{generated:.6f} deviates from helper reference {reference:.6f} "
                            f"at rate={rate:.2%}, vol={vol:.2%}."
                        ),
                        actual=generated,
                        expected=f"{reference:.6f} +/- {tolerance:.4%}",
                        context={
                            "relation": "within_tolerance",
                            "tolerance": float(tolerance),
                            "rate": float(rate),
                            "vol": float(vol),
                            "comparison_kwargs": comparison_kwargs,
                            "sampled_prices": sampled_prices,
                        },
                    )
                )
                break
        except Exception as e:
            failures.append(
                InvariantFailure(
                    check="check_rate_style_swaption_helper_consistency",
                    message=f"Swaption helper consistency check failed: {e}",
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    context={
                        "relation": "within_tolerance",
                        "tolerance": float(tolerance),
                        "rate": float(rate),
                        "vol": float(vol),
                        "comparison_kwargs": comparison_kwargs,
                        "sampled_prices": sampled_prices,
                    },
                )
            )
            break
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
