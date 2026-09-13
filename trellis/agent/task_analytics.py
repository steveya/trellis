"""Bounded, declared analytics over an already-bound comparison payoff.

This is task orchestration, not a pricing adapter or an independent oracle.
The callable duration identity delegates both computations to public measures.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


def callable_duration_contract() -> dict[str, Any]:
    """Return the admitted same-payoff, constant-zero-OAS duration contract."""
    return {
        "profile": "callable_same_payoff_duration_v1",
        "output_name": "effective_duration",
        "output_unit": "years",
        "tolerance_unit": "percent_of_reference_duration",
        "bump_bps": 25.0,
        "market_price": None,
        "constant_oas_bps": 0.0,
        "risk_coordinate": "parallel_continuously_compounded_discount_zero_rate",
        "shock_convention": "symmetric_up_down",
        "denominator": "2_times_bump_decimal_times_current_callable_holder_pv",
        "derivative_method": "finite_difference",
        "forward_policy": "rederive_default_preserve_named_forecast_curves",
        "model_policy": "preserve_parameters_recalibrate_tree",
        "reference_role": "same_payoff_same_model_identity_not_independent_oracle",
        "measures": {
            "oas_bump_duration": "OASDuration",
            "same_payoff_parallel_duration": "Duration",
        },
    }


def validated_callable_anchor_price(value: Any) -> float:
    """Validate the authoritative evaluator output before scalar conversion."""
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value <= 0.0
    ):
        raise ValueError("Duration anchor must be a finite positive non-boolean holder PV")
    metadata = dict(getattr(value, "metadata", {}) or {})
    if any(
        key in metadata and metadata[key] != expected
        for key, expected in (
            ("status", "passed"), ("unit", "currency_amount"),
            ("output_unit", "currency_amount"), ("output_currency", "USD"),
        )
    ):
        raise ValueError("Duration anchor has invalid price units or failed status")
    return float(value)


def evaluate_comparison_analytics(
    payoff, market_state, *, target_id: str, contract: Mapping[str, Any],
    anchor_price: float,
) -> dict[str, dict[str, Any]]:
    """Execute the declared measure without substituting a price comparison."""
    from trellis.analytics.measures import Duration, OASDuration

    if dict(contract) != callable_duration_contract():
        raise ValueError("Unsupported comparison analytics contract")
    anchor_price = validated_callable_anchor_price(anchor_price)
    measure_name = contract["measures"][target_id]
    bump_bps = float(contract["bump_bps"])
    measure = (
        OASDuration(market_price=None, bump_bps=bump_bps)
        if measure_name == "OASDuration"
        else Duration(bump_bps=bump_bps)
    )
    value = measure.compute(
        payoff, market_state, base_price=anchor_price,
        _cache={"base_price": anchor_price},
    )
    metadata = dict(getattr(value, "metadata", {}) or {})
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("effective_duration must be finite and numeric")
    if (
        metadata.get("output_unit", "years") != "years"
        or metadata.get("unit", "years") != "years"
        or metadata.get("status", "passed") != "passed"
    ):
        raise ValueError("effective_duration has invalid units or failed status")
    if any(
        key in metadata and metadata[key] != expected
        for key, expected in (
            ("resolved_derivative_method", "parallel_curve_bump"),
            ("derivative_method_category", "finite_difference_bump"),
            ("derivative_method", "finite_difference"),
            ("bump_bps", bump_bps),
        )
    ):
        raise ValueError("effective_duration reports conflicting derivative provenance")
    if measure_name == "Duration" and (
        metadata.get("resolved_derivative_method") != "parallel_curve_bump"
        or metadata.get("derivative_method_category") != "finite_difference_bump"
        or metadata.get("bump_bps") != bump_bps
    ):
        raise ValueError("Callable Duration did not resolve the authored finite difference")
    # OASDuration(None) has one implementation: parallel shifts, no OAS solve.
    metadata.update({
        "measure": measure_name,
        "derivative_method": "finite_difference",
        "resolved_derivative_method": "parallel_curve_bump",
        "bump_bps": bump_bps,
        "constant_oas_bps": 0.0,
        "market_price": None,
        "anchor_price": anchor_price,
        "denominator": contract["denominator"],
        "risk_coordinate": contract["risk_coordinate"],
        "reference_role": contract["reference_role"],
    })
    return {"effective_duration": {
        "value": float(value), "unit": "years", "status": "passed", "metadata": metadata,
    }}


def validated_comparison_analytics_values(
    payload: Mapping[str, Any], *, target_id: str, contract: Mapping[str, Any],
    anchor_price: float,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Check required value, unit and execution evidence before scalar extraction."""
    if not isinstance(payload, Mapping) or set(payload) != {"effective_duration"}:
        raise ValueError("Missing required effective_duration output")
    output = payload["effective_duration"]
    if not isinstance(output, Mapping) or output.get("unit") != "years":
        raise ValueError("effective_duration must be reported in years")
    if output.get("status") != "passed":
        raise ValueError("effective_duration output did not pass")
    value = output.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("effective_duration must be finite and numeric")
    metadata = output.get("metadata")
    expected = {
        "measure": contract["measures"][target_id],
        "derivative_method": "finite_difference",
        "resolved_derivative_method": "parallel_curve_bump",
        "bump_bps": contract["bump_bps"],
        "constant_oas_bps": 0.0,
        "market_price": None,
        "anchor_price": anchor_price,
        "denominator": contract["denominator"],
        "risk_coordinate": contract["risk_coordinate"],
        "reference_role": contract["reference_role"],
    }
    if not isinstance(metadata, Mapping) or any(
        key not in metadata or metadata[key] != expected_value
        for key, expected_value in expected.items()
    ):
        raise ValueError("effective_duration execution metadata does not match its contract")
    if any(
        key in metadata and metadata[key] != expected_value
        for key, expected_value in (
            ("unit", "years"), ("output_unit", "years"), ("status", "passed"),
            ("derivative_method_category", "finite_difference_bump"),
        )
    ):
        raise ValueError("effective_duration has conflicting nested output metadata")
    return {"effective_duration": float(value)}, {"effective_duration": {
        **dict(metadata), "unit": "years", "status": "passed",
    }}
