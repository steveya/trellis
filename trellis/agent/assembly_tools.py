"""Structured assembly helpers for toolized build orchestration.

These helpers package deterministic route, adapter, invariant, comparison, and
cookbook-candidate logic into reusable structured outputs. They are used both
internally by the builder prompt path and externally through repo-aware tools.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Any, Mapping

from trellis.agent.codegen_guardrails import GenerationPlan, PrimitiveRef
from trellis.agent.knowledge.decompose import decompose_to_ir
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.quant import (
    PricingPlan,
    select_pricing_method,
    select_pricing_method_for_product_ir,
)


_MARKET_READ_HINTS = {
    "discount_curve": "`market_state.discount`",
    "forward_curve": "`market_state.forecast_forward_curve(self._spec.rate_index)` or `market_state.forward_curve`",
    "black_vol_surface": "`market_state.vol_surface.black_vol(...)`",
    "fx_rates": "`market_state.fx_rates[...]`",
    "spot": "`market_state.spot` or `market_state.underlier_spots[...]`",
    "credit_curve": "`market_state.credit_curve`",
    "state_space": "`market_state.state_space`",
    "local_vol_surface": "`market_state.local_vol_surface` or `market_state.local_vol_surfaces[...]`",
    "jump_parameters": "`market_state.jump_parameters` or `market_state.jump_parameter_sets[...]`",
    "model_parameters": "`market_state.model_parameters` or `market_state.model_parameter_sets[...]`",
}

_OPTION_LIKE_INSTRUMENTS = {
    "american_option",
    "asian_option",
    "barrier_option",
    "bermudan_swaption",
    "cap",
    "european_option",
    "floor",
    "swaption",
}

# Credit instruments price from survival probabilities / hazard rates, not vol.
# Vol sensitivity checks are not meaningful and will always produce false failures.
_CREDIT_INSTRUMENTS = {
    "credit_default_swap",
    "nth_to_default",
    "cds",
}


@dataclass(frozen=True)
class PrimitiveLookupResult:
    """Structured summary of the selected primitive route."""

    method: str
    instrument_type: str | None
    route: str | None
    engine_family: str | None
    approved_modules: tuple[str, ...]
    primitives: tuple[str, ...]
    adapters: tuple[str, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class ThinAdapterPlan:
    """Structured thin-adapter build plan for one generated payoff."""

    class_name: str
    market_reads: tuple[str, ...]
    primitive_calls: tuple[str, ...]
    adapter_steps: tuple[str, ...]
    return_contract: str


@dataclass(frozen=True)
class InvariantPack:
    """Named invariant checks applicable to one route/product family."""

    checks: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonHarnessTarget:
    """One concrete comparison target in a deterministic harness plan."""

    target_id: str
    preferred_method: str
    is_reference: bool = False
    relation: str | None = None


@dataclass(frozen=True)
class ComparisonHarnessPlan:
    """Deterministic cross-validation harness plan for a task."""

    targets: tuple[ComparisonHarnessTarget, ...]
    reference_target: str | None
    tolerance_pct: float


def normalize_comparison_relation(
    value: object | None,
    *,
    default: str | None = None,
) -> str | None:
    """Normalize user-facing comparison relation labels onto stable runtime ids."""
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    mapping = {
        "<=": "<=",
        "le": "<=",
        "lte": "<=",
        "upper_bound": "<=",
        "at_most": "<=",
        "no_greater_than": "<=",
        ">=": ">=",
        "ge": ">=",
        "gte": ">=",
        "lower_bound": ">=",
        "at_least": ">=",
        "no_less_than": ">=",
        "==": "within_tolerance",
        "=": "within_tolerance",
        "~=": "within_tolerance",
        "approx": "within_tolerance",
        "approx_equal": "within_tolerance",
        "close": "within_tolerance",
        "within_tolerance": "within_tolerance",
        "tolerance": "within_tolerance",
    }
    return mapping.get(text, text)


def lookup_primitive_route_from_context(
    *,
    generation_plan: GenerationPlan | None = None,
    pricing_plan: PricingPlan | None = None,
) -> PrimitiveLookupResult:
    """Summarize the selected route in a prompt- and tool-friendly structure."""
    method = pricing_plan.method if pricing_plan is not None else (
        generation_plan.method if generation_plan is not None else "unknown"
    )
    primitive_plan = generation_plan.primitive_plan if generation_plan is not None else None
    return PrimitiveLookupResult(
        method=method,
        instrument_type=generation_plan.instrument_type if generation_plan is not None else None,
        route=primitive_plan.route if primitive_plan is not None else None,
        engine_family=primitive_plan.engine_family if primitive_plan is not None else None,
        approved_modules=tuple(generation_plan.approved_modules if generation_plan is not None else ()),
        primitives=tuple(
            f"{primitive.module}.{primitive.symbol}"
            for primitive in primitive_plan.primitives
        ) if primitive_plan is not None else (),
        adapters=tuple(primitive_plan.adapters) if primitive_plan is not None else (),
        blockers=tuple(primitive_plan.blockers) if primitive_plan is not None else (),
    )


def lookup_primitive_route(
    *,
    description: str,
    instrument_type: str | None = None,
    preferred_method: str | None = None,
) -> PrimitiveLookupResult:
    """Plan a route deterministically from a description and optional method."""
    product_ir = decompose_to_ir(description, instrument_type=instrument_type)
    if preferred_method:
        pricing_plan = select_pricing_method_for_product_ir(
            product_ir,
            preferred_method=preferred_method,
            context_description=description,
        )
    else:
        pricing_plan = select_pricing_method(
            description,
            instrument_type=instrument_type,
        )
    from trellis.agent.codegen_guardrails import build_generation_plan

    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=instrument_type or getattr(product_ir, "instrument", None),
        inspected_modules=tuple(pricing_plan.method_modules),
        product_ir=product_ir,
    )
    return lookup_primitive_route_from_context(
        generation_plan=generation_plan,
        pricing_plan=pricing_plan,
    )


def build_thin_adapter_plan(
    spec_schema,
    *,
    pricing_plan: PricingPlan | None = None,
    generation_plan: GenerationPlan | None = None,
) -> ThinAdapterPlan:
    """Build a deterministic thin-adapter plan for the builder prompt."""
    required_market_data = sorted(
        getattr(pricing_plan, "required_market_data", ()) or ()
    )
    primitive_plan = generation_plan.primitive_plan if generation_plan is not None else None
    primitive_calls = tuple(
        f"{primitive.module}.{primitive.symbol}"
        for primitive in primitive_plan.primitives
        if primitive.required
    ) if primitive_plan is not None else ()
    adapter_steps = tuple(primitive_plan.adapters) if primitive_plan is not None else ()
    market_reads = tuple(
        _MARKET_READ_HINTS.get(requirement, f"`market_state` access for `{requirement}`")
        for requirement in required_market_data
    )
    return ThinAdapterPlan(
        class_name=spec_schema.class_name,
        market_reads=market_reads,
        primitive_calls=primitive_calls,
        adapter_steps=adapter_steps,
        return_contract="Return a Python `float` present value from `evaluate()`.",
    )


def render_thin_adapter_plan(
    spec_schema,
    *,
    pricing_plan: PricingPlan | None = None,
    generation_plan: GenerationPlan | None = None,
) -> str:
    """Render the structured thin-adapter plan for prompt injection."""
    plan = build_thin_adapter_plan(
        spec_schema,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
    )
    lines = [
        "## Thin Adapter Plan",
        f"- Target payoff class: `{plan.class_name}`",
    ]
    if plan.market_reads:
        lines.append("- Required market reads:")
        lines.extend(f"  - {item}" for item in plan.market_reads)
    if plan.primitive_calls:
        lines.append("- Required primitive calls:")
        lines.extend(f"  - `{item}`" for item in plan.primitive_calls[:8])
    if plan.adapter_steps:
        lines.append("- Required adapter steps:")
        lines.extend(f"  - `{item}`" for item in plan.adapter_steps[:8])
    lines.append(f"- Return contract: {plan.return_contract}")
    return "\n".join(lines)


def select_invariant_pack(
    *,
    instrument_type: str | None,
    method: str,
    product_ir=None,
) -> InvariantPack:
    """Select a deterministic invariant pack for one route/product family."""
    normalized_method = normalize_method(method)
    normalized_instrument = (instrument_type or getattr(product_ir, "instrument", None) or "").strip().lower()
    payoff_traits = set(getattr(product_ir, "payoff_traits", ()) or ())
    checks: list[str] = []

    if normalized_instrument in {"credit_default_swap", "cds"}:
        checks.extend(
            [
                "check_cds_spread_quote_normalization",
                "check_cds_credit_curve_sensitivity",
            ]
        )

    checks.extend(["check_non_negativity", "check_price_sanity"])

    if (
        normalized_instrument not in _CREDIT_INSTRUMENTS
        and (
            normalized_instrument in _OPTION_LIKE_INSTRUMENTS
            or normalized_method in {"analytical", "monte_carlo", "qmc", "pde_solver", "fft_pricing"}
            or {"callable", "puttable", "bermudan", "american"} & payoff_traits
        )
    ):
        checks.extend(["check_vol_sensitivity", "check_vol_monotonicity"])

    if normalized_method == "analytical" and normalized_instrument in {"european_option", "cap", "floor", "swaption"}:
        checks.append("check_zero_vol_intrinsic")
    if normalized_method == "analytical" and normalized_instrument == "swaption":
        checks.append("check_rate_style_swaption_helper_consistency")

    if normalized_instrument in {"callable_bond", "puttable_bond"}:
        checks.append("check_bounded_by_reference")

    return InvariantPack(checks=tuple(dict.fromkeys(checks)))


def render_invariant_pack(
    *,
    instrument_type: str | None,
    method: str,
    product_ir=None,
) -> str:
    """Render the selected invariant pack as prompt guidance."""
    pack = select_invariant_pack(
        instrument_type=instrument_type,
        method=method,
        product_ir=product_ir,
    )
    lines = [
        "## Invariant Pack",
        "- The generated payoff should be compatible with these deterministic validation checks:",
    ]
    lines.extend(f"  - `{check}`" for check in pack.checks)
    return "\n".join(lines)


def build_comparison_harness_plan(task: Mapping[str, Any]) -> ComparisonHarnessPlan:
    """Build a deterministic comparison harness plan from task metadata."""
    construct = task.get("construct")
    if isinstance(construct, str):
        construct_methods = [normalize_method(construct)]
    else:
        construct_methods = [normalize_method(item) for item in (construct or [])]

    cross_validate = task.get("cross_validate") or {}
    relation_overrides = cross_validate.get("relations") or {}
    internal_targets = list(cross_validate.get("internal") or ())
    analytical_target = cross_validate.get("analytical")

    targets: list[ComparisonHarnessTarget] = []
    if internal_targets:
        for target_id in internal_targets:
            relation = None
            normalized_target_id = str(target_id).strip()
            if normalized_target_id:
                relation = normalize_comparison_relation(
                    relation_overrides.get(normalized_target_id)
                    if isinstance(relation_overrides, Mapping)
                    else None
                )
            targets.append(
                ComparisonHarnessTarget(
                    target_id=normalized_target_id,
                    preferred_method=_preferred_method_for_target(normalized_target_id, construct_methods),
                    relation=relation,
                )
            )
    else:
        targets.extend(
            ComparisonHarnessTarget(
                target_id=method,
                preferred_method=method,
                relation=normalize_comparison_relation(
                    relation_overrides.get(method)
                    if isinstance(relation_overrides, Mapping)
                    else None
                ),
            )
            for method in construct_methods
        )

    if analytical_target:
        targets.append(
            ComparisonHarnessTarget(
                target_id=str(analytical_target),
                preferred_method="analytical",
                is_reference=True,
            )
        )

    deduped: list[ComparisonHarnessTarget] = []
    seen: set[str] = set()
    for target in targets:
        if target.target_id in seen:
            continue
        deduped.append(target)
        seen.add(target.target_id)

    reference_target = next((target.target_id for target in deduped if target.is_reference), None)
    tolerance_pct = float(cross_validate.get("tolerance_pct", 5.0))
    return ComparisonHarnessPlan(
        targets=tuple(deduped),
        reference_target=reference_target,
        tolerance_pct=tolerance_pct,
    )


def build_cookbook_candidate_payload(
    *,
    method: str,
    description: str,
    code: str,
) -> dict[str, Any] | None:
    """Build a deterministic cookbook-candidate payload from successful code."""
    template = _extract_evaluate_template(code)
    if not template:
        return None
    return {
        "method": method,
        "description": description,
        "source": "deterministic_fallback",
        "template": template,
    }


def _preferred_method_for_target(target_id: str, construct_methods: list[str]) -> str:
    normalized_target = normalize_method(target_id)
    if normalized_target in {"analytical", "rate_tree", "monte_carlo", "pde_solver", "fft_pricing"}:
        return normalized_target

    explicit_patterns = (
        ("tree", "rate_tree"),
        ("lattice", "rate_tree"),
        ("pde", "pde_solver"),
        ("psor", "pde_solver"),
        ("mc", "monte_carlo"),
        ("monte", "monte_carlo"),
        ("lsm", "monte_carlo"),
        ("dual", "monte_carlo"),
        ("mesh", "monte_carlo"),
        ("stochastic", "monte_carlo"),
        ("fft", "fft_pricing"),
        ("cos", "fft_pricing"),
        ("black", "analytical"),
        ("jamshidian", "analytical"),
        ("rubinstein", "analytical"),
    )
    lower = target_id.lower()
    for pattern, method in explicit_patterns:
        if pattern in lower:
            return method

    if len(construct_methods) == 1:
        return construct_methods[0]
    return "analytical"


def _extract_evaluate_template(code: str) -> str:
    if not code.strip():
        return ""

    lines = code.splitlines()
    start = None
    base_indent = 0
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)def evaluate\(", line)
        if match:
            start = index
            base_indent = len(match.group(1))
            break

    if start is None:
        return code[:2000]

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= base_indent:
            end = index
            break

    return textwrap.dedent("\n".join(lines[start:end])).strip()
