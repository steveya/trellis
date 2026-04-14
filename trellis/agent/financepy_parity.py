"""Shared parity helpers for FinancePy-backed benchmark tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from trellis.agent.market_scenarios import market_scenario_contract_from_task
from trellis.agent.task_manifests import load_financepy_bindings
from trellis.core.market_state import MarketState
from trellis.curves.forward_curve import ForwardCurve


ROOT = Path(__file__).resolve().parents[2]


def financepy_binding_for_task(
    task: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Return the configured FinancePy binding payload for one benchmark task."""
    binding_id = str(task.get("financepy_binding_id") or "").strip()
    if not binding_id:
        return {}
    return dict(load_financepy_bindings(root=root).get(binding_id) or {})


def normalize_benchmark_outputs(
    task: Mapping[str, Any],
    outputs: Mapping[str, Any] | None,
    *,
    source: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Normalize benchmark outputs into one comparison-compatible shape."""
    normalized = dict(outputs or {})
    binding = financepy_binding_for_task(task, root=root)
    normalization = dict(binding.get("normalization") or {})
    contract = dict(task.get("benchmark_contract") or {})
    for output_name, rule in normalization.items():
        if output_name not in normalized or not isinstance(rule, Mapping):
            continue
        only_source = str(rule.get("source") or "").strip().lower()
        if only_source and only_source != source:
            continue
        scale_field = str(rule.get("scale_by_contract_field") or "").strip()
        if scale_field:
            scale = contract.get(scale_field)
            if scale not in {None, ""}:
                normalized[output_name] = float(normalized[output_name]) * float(scale)
        offset_field = str(rule.get("offset_by_contract_field") or "").strip()
        if offset_field:
            offset = contract.get(offset_field)
            if offset not in {None, ""}:
                normalized[output_name] = float(normalized[output_name]) + float(offset)
        sign = str(rule.get("sign") or "").strip().lower()
        if sign == "negate":
            normalized[output_name] = -float(normalized[output_name])
    return normalized


def align_market_state_for_financepy_parity(
    task: Mapping[str, Any],
    market_state: MarketState,
    *,
    root: Path = ROOT,
) -> tuple[MarketState, dict[str, Any]]:
    """Apply binding-driven market alignment for FinancePy parity tasks."""
    binding = financepy_binding_for_task(task, root=root)
    alignment = dict(binding.get("market_alignment") or {})
    if not alignment:
        return market_state, {}

    curve_regime = str(alignment.get("curve_regime") or "").strip().lower()
    if curve_regime not in {"single_curve_forecast", "single_curve_discount"}:
        return market_state, {}

    scenario_contract = market_scenario_contract_from_task(task, root=root)
    selected_curve_names = dict(getattr(market_state, "selected_curve_names", None) or {})
    forecast_curves = dict(getattr(market_state, "forecast_curves", None) or {})
    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})

    if curve_regime == "single_curve_forecast":
        forecast_name = (
            str(alignment.get("forecast_curve_name") or "").strip()
            or str(getattr(scenario_contract, "forecast_curve_name", None) or "").strip()
            or selected_curve_names.get("forecast_curve")
            or ""
        )
        forecast_curve = forecast_curves.get(forecast_name)
        if forecast_curve is None and market_state.forward_curve is not None:
            forecast_curve = getattr(market_state.forward_curve, "discount_curve", None)
        if forecast_curve is None:
            return market_state, {}
        discount_curve = forecast_curve
        selected_name = forecast_name or selected_curve_names.get("discount_curve") or "financepy_single_curve"
        if forecast_name:
            forecast_curves[forecast_name] = forecast_curve
        selected_curve_names["discount_curve"] = selected_name
        if forecast_name:
            selected_curve_names["forecast_curve"] = forecast_name
        aligned = replace(
            market_state,
            discount=discount_curve,
            forward_curve=ForwardCurve(discount_curve),
            forecast_curves=forecast_curves or None,
            selected_curve_names=selected_curve_names or None,
            market_provenance={
                **market_provenance,
                "financepy_parity_alignment": {
                    "curve_regime": curve_regime,
                    "selected_curve_name": selected_name,
                },
            },
        )
        return aligned, {
            "financepy_parity_alignment": True,
            "curve_regime": curve_regime,
            "selected_curve_name": selected_name,
        }

    discount_curve = getattr(market_state, "discount", None)
    if discount_curve is None:
        return market_state, {}
    forecast_name = (
        str(alignment.get("forecast_curve_name") or "").strip()
        or str(getattr(scenario_contract, "forecast_curve_name", None) or "").strip()
        or selected_curve_names.get("forecast_curve")
        or ""
    )
    if forecast_name:
        forecast_curves[forecast_name] = discount_curve
        selected_curve_names["forecast_curve"] = forecast_name
    selected_name = selected_curve_names.get("discount_curve") or forecast_name or "financepy_single_curve"
    selected_curve_names["discount_curve"] = selected_name
    aligned = replace(
        market_state,
        forward_curve=ForwardCurve(discount_curve),
        forecast_curves=forecast_curves or None,
        selected_curve_names=selected_curve_names or None,
        market_provenance={
            **market_provenance,
            "financepy_parity_alignment": {
                "curve_regime": curve_regime,
                "selected_curve_name": selected_name,
            },
        },
    )
    return aligned, {
        "financepy_parity_alignment": True,
        "curve_regime": curve_regime,
        "selected_curve_name": selected_name,
    }


__all__ = [
    "align_market_state_for_financepy_parity",
    "financepy_binding_for_task",
    "normalize_benchmark_outputs",
]
