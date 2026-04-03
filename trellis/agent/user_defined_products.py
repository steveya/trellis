"""Structured user-defined product compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from trellis.agent.codegen_guardrails import build_generation_plan, render_generation_plan
from trellis.agent.knowledge import build_shared_knowledge_payload, retrieve_for_product_ir
from trellis.agent.knowledge.decompose import build_product_ir
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.quant import PricingPlan, select_pricing_method_for_product_ir


@dataclass(frozen=True)
class UserProductSpec:
    """Structured user-defined product specification."""

    name: str
    payoff_family: str
    payoff_traits: tuple[str, ...]
    exercise_style: str
    schedule_dependence: bool
    state_dependence: str
    model_family: str
    candidate_engine_families: tuple[str, ...]
    required_market_data: frozenset[str]
    preferred_method: str
    reusable_primitives: tuple[str, ...] = ()
    unresolved_primitives: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class UserDefinedProductCompilation:
    """Compiled structured user-defined product ready for the build path."""

    spec: UserProductSpec
    product_ir: Any
    pricing_plan: PricingPlan
    generation_plan: Any
    knowledge: dict[str, Any]
    knowledge_text: str
    review_knowledge_text: str
    routing_knowledge_text: str
    knowledge_summary: dict[str, Any]
    rendered_plan: str


def parse_user_product_spec(spec: UserProductSpec | dict[str, Any] | str) -> UserProductSpec:
    """Parse a structured user-defined product spec from YAML, dict, or object."""
    if isinstance(spec, UserProductSpec):
        return spec
    if isinstance(spec, str):
        payload = yaml.safe_load(spec) or {}
    else:
        payload = dict(spec)

    return UserProductSpec(
        name=payload["name"],
        payoff_family=payload["payoff_family"],
        payoff_traits=tuple(sorted(set(payload.get("payoff_traits", [])))),
        exercise_style=payload["exercise_style"],
        schedule_dependence=bool(payload["schedule_dependence"]),
        state_dependence=payload["state_dependence"],
        model_family=payload["model_family"],
        candidate_engine_families=tuple(payload.get("candidate_engine_families", [])),
        required_market_data=frozenset(payload.get("required_market_data", [])),
        preferred_method=normalize_method(payload["preferred_method"]),
        reusable_primitives=tuple(payload.get("reusable_primitives", [])),
        unresolved_primitives=tuple(payload.get("unresolved_primitives", [])),
        description=payload.get("description", payload["name"]),
    )


def compile_user_defined_product(
    spec: UserProductSpec | dict[str, Any] | str,
    *,
    inspected_modules: tuple[str, ...] = (),
    requested_measures: tuple[str, ...] | list[str] | None = None,
) -> UserDefinedProductCompilation:
    """Compile a structured user-defined product into Trellis planning artifacts."""
    parsed = parse_user_product_spec(spec)
    product_ir = build_product_ir(
        description=parsed.description or parsed.name,
        instrument=parsed.name,
        payoff_family=parsed.payoff_family,
        payoff_traits=parsed.payoff_traits,
        exercise_style=parsed.exercise_style,
        state_dependence=parsed.state_dependence,
        schedule_dependence=parsed.schedule_dependence,
        model_family=parsed.model_family,
        candidate_engine_families=parsed.candidate_engine_families,
        required_market_data=parsed.required_market_data,
        reusable_primitives=parsed.reusable_primitives,
        unresolved_primitives=parsed.unresolved_primitives or None,
        preferred_method=parsed.preferred_method,
    )
    pricing_plan = select_pricing_method_for_product_ir(
        product_ir,
        preferred_method=parsed.preferred_method,
        requested_measures=requested_measures,
    )
    inspected = inspected_modules or tuple(pricing_plan.method_modules)
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=product_ir.instrument,
        inspected_modules=inspected,
        product_ir=product_ir,
    )
    knowledge = retrieve_for_product_ir(
        product_ir,
        preferred_method=pricing_plan.method,
    )
    route_ids: tuple[str, ...] = ()
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    if primitive_plan is not None and getattr(primitive_plan, "route", None):
        route_ids = (primitive_plan.route,)
    shared_knowledge = build_shared_knowledge_payload(
        knowledge,
        pricing_method=pricing_plan.method,
        route_ids=route_ids,
        route_families=tuple(getattr(product_ir, "route_families", ()) or ()),
    )
    rendered_plan = render_generation_plan(generation_plan)
    return UserDefinedProductCompilation(
        spec=parsed,
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
        knowledge=knowledge,
        knowledge_text=shared_knowledge["builder_text_distilled"],
        review_knowledge_text=shared_knowledge["review_text_distilled"],
        routing_knowledge_text=shared_knowledge["routing_text_distilled"],
        knowledge_summary=shared_knowledge["summary"],
        rendered_plan=rendered_plan,
    )


def _method_modules_for(method: str) -> list[str]:
    """Return representative method modules for a canonical method label."""
    modules = {
        "analytical": ["trellis.models.black"],
        "rate_tree": ["trellis.models.trees.lattice"],
        "monte_carlo": ["trellis.models.monte_carlo.engine"],
        "qmc": ["trellis.models.qmc"],
        "fft_pricing": ["trellis.models.transforms.fft_pricer"],
        "pde_solver": ["trellis.models.pde.theta_method"],
    }
    return modules.get(method, [])
