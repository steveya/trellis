"""Compiled validation-contract surface for one route-ready request."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from trellis.agent.assembly_tools import normalize_comparison_relation
from trellis.agent.instrument_identity import resolve_authoritative_instrument_type
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.quant import quant_challenger_packet_summary
from trellis.agent import validation_bundles

_VOL_CHECK_INPUTS = frozenset({"black_vol_surface", "local_vol_surface"})
_CHECK_MARKET_DATA_REQUIREMENTS: dict[str, frozenset[str]] = {
    "check_vol_sensitivity": _VOL_CHECK_INPUTS,
    "check_vol_monotonicity": _VOL_CHECK_INPUTS,
    "check_zero_vol_intrinsic": _VOL_CHECK_INPUTS,
    "check_rate_monotonicity": frozenset({"discount_curve"}),
}


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable view of a shallow mapping."""
    return MappingProxyType(dict(mapping or {}))


def _tuple_strings(values) -> tuple[str, ...]:
    """Normalize any iterable of values into a deduplicated tuple of strings."""
    if not values:
        return ()
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


@dataclass(frozen=True)
class ValidationCheckSpec:
    """One deterministic validation check selected for execution."""

    check_id: str
    category: str
    harness_requirements: tuple[str, ...] = ()
    relation: str | None = None


@dataclass(frozen=True)
class ValidationRelationSpec:
    """One compiled comparison or bound relation attached to validation."""

    target_id: str
    relation: str
    source: str


@dataclass(frozen=True)
class ExecutableClaimSpec:
    """One validation-admitted claim that critic may select and arbiter may run."""

    claim_id: str
    validation_check_id: str
    category: str
    relation: str | None = None
    executable: bool = True


@dataclass(frozen=True)
class CompiledValidationContract:
    """Compiled validation state derived from route, lowering, and semantics."""

    contract_id: str
    instrument_type: str
    method: str
    bundle_id: str
    backend_binding_id: str | None = None
    exact_bundle_id: str | None = None
    route_id: str | None = None
    route_family: str | None = None
    required_market_data: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = ()
    deterministic_checks: tuple[ValidationCheckSpec, ...] = ()
    comparison_relations: tuple[ValidationRelationSpec, ...] = ()
    lowering_errors: tuple[str, ...] = ()
    admissibility_failures: tuple[str, ...] = ()
    residual_risks: tuple[str, ...] = ()
    quant_challenger_packet: Mapping[str, object] = field(default_factory=dict)
    review_hints: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Freeze mutable mapping fields for stability in traces and tests."""
        object.__setattr__(
            self,
            "quant_challenger_packet",
            _freeze_mapping(self.quant_challenger_packet),
        )
        object.__setattr__(self, "review_hints", _freeze_mapping(self.review_hints))


def compile_validation_contract(
    *,
    request=None,
    product_ir=None,
    pricing_plan=None,
    generation_plan=None,
    semantic_blueprint=None,
    comparison_spec=None,
    instrument_type: str | None = None,
) -> CompiledValidationContract | None:
    """Compile the deterministic validation contract for one request."""
    method = normalize_method(
        getattr(pricing_plan, "method", None)
        or getattr(semantic_blueprint, "preferred_method", None)
        or ""
    )
    resolved_instrument = _resolve_instrument_type(
        instrument_type=instrument_type,
        product_ir=product_ir,
        semantic_blueprint=semantic_blueprint,
    )
    if not method or not resolved_instrument:
        return None

    bundle = validation_bundles.select_validation_bundle(
        instrument_type=resolved_instrument,
        method=method,
        product_ir=product_ir,
        semantic_blueprint=semantic_blueprint,
    )
    route_id, route_family = _resolve_route_identity(
        generation_plan=generation_plan,
        semantic_blueprint=semantic_blueprint,
    )
    backend_binding_id = _resolve_backend_binding_identity(
        generation_plan=generation_plan,
        route_id=route_id,
        product_ir=product_ir,
        semantic_blueprint=semantic_blueprint,
    )
    exact_bundle_id = _exact_bundle_id_for(
        bundle_id=bundle.bundle_id,
        backend_binding_id=backend_binding_id,
    )
    lowering_errors = _lowering_errors_for(semantic_blueprint)
    admissibility_failures = _admissibility_failures_for(
        route_id=route_id,
        semantic_blueprint=semantic_blueprint,
        product_ir=product_ir,
    )
    comparison_relations = _comparison_relations_for(
        bundle=bundle,
        instrument_type=resolved_instrument,
        semantic_blueprint=semantic_blueprint,
        comparison_spec=comparison_spec,
    )
    residual_risks = _residual_risks_for(
        comparison_relations=comparison_relations,
        lowering_errors=lowering_errors,
        admissibility_failures=admissibility_failures,
        semantic_blueprint=semantic_blueprint,
    )
    requested_outputs = _requested_outputs_for(request=request, semantic_blueprint=semantic_blueprint)
    required_market_data = _required_market_data_for(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
        semantic_blueprint=semantic_blueprint,
    )
    deterministic_checks = tuple(
        ValidationCheckSpec(
            check_id=check_id,
            category=_category_for_check(bundle.categories, check_id),
            harness_requirements=_harness_requirements_for(check_id),
            relation=_relation_for_check(
                check_id,
                instrument_type=resolved_instrument,
                comparison_spec=comparison_spec,
            ),
        )
        for check_id in bundle.checks
        if _check_supported_by_market_data(check_id, required_market_data)
    )
    contract_id = f"{method}:{resolved_instrument}"
    quant_challenger_packet = quant_challenger_packet_summary(
        pricing_plan,
        deterministic_check_ids=tuple(check.check_id for check in deterministic_checks),
        validation_contract_id=contract_id,
        route_family=route_family or method,
        residual_risks=residual_risks,
    )
    review_hints = {
        "has_residual_risks": bool(residual_risks),
        "has_lowering_errors": bool(lowering_errors),
        "has_admissibility_failures": bool(admissibility_failures),
        "requires_relation_semantics": any(
            relation.relation == "unspecified" for relation in comparison_relations
        ),
        "has_exact_validation_identity": bool(exact_bundle_id),
        "has_quant_challenger_packet": bool(quant_challenger_packet),
        "has_quant_alternatives": _has_quant_alternatives(quant_challenger_packet),
    }
    return CompiledValidationContract(
        contract_id=contract_id,
        instrument_type=resolved_instrument,
        method=method,
        bundle_id=bundle.bundle_id,
        backend_binding_id=backend_binding_id,
        exact_bundle_id=exact_bundle_id,
        route_id=route_id,
        route_family=route_family,
        required_market_data=required_market_data,
        requested_outputs=requested_outputs,
        deterministic_checks=deterministic_checks,
        comparison_relations=comparison_relations,
        lowering_errors=lowering_errors,
        admissibility_failures=admissibility_failures,
        residual_risks=residual_risks,
        quant_challenger_packet=quant_challenger_packet,
        review_hints=review_hints,
    )


def validation_contract_summary(
    contract: CompiledValidationContract | None,
) -> dict[str, object] | None:
    """Project the compiled validation contract onto YAML-safe primitives."""
    if contract is None:
        return None
    return {
        "contract_id": contract.contract_id,
        "instrument_type": contract.instrument_type,
        "method": contract.method,
        "bundle_id": contract.bundle_id,
        "backend_binding_id": contract.backend_binding_id,
        "exact_bundle_id": contract.exact_bundle_id,
        "route_id": contract.route_id,
        "route_family": contract.route_family,
        "required_market_data": list(contract.required_market_data),
        "requested_outputs": list(contract.requested_outputs),
        "deterministic_checks": [
            {
                "check_id": check.check_id,
                "category": check.category,
                "harness_requirements": list(check.harness_requirements),
                "relation": check.relation,
            }
            for check in contract.deterministic_checks
        ],
        "comparison_relations": [
            {
                "target_id": relation.target_id,
                "relation": relation.relation,
                "source": relation.source,
            }
            for relation in contract.comparison_relations
        ],
        "executable_claims": [
            {
                "claim_id": claim.claim_id,
                "validation_check_id": claim.validation_check_id,
                "category": claim.category,
                "relation": claim.relation,
                "executable": claim.executable,
            }
            for claim in executable_claim_specs_for_contract(contract)
        ],
        "lowering_errors": list(contract.lowering_errors),
        "admissibility_failures": list(contract.admissibility_failures),
        "residual_risks": list(contract.residual_risks),
        "quant_challenger_packet": dict(contract.quant_challenger_packet),
        "review_hints": dict(contract.review_hints),
    }


def _has_quant_alternatives(packet: Mapping[str, object]) -> bool:
    """Return whether a quant challenger packet contains rejected alternatives."""
    for candidate in packet.get("candidate_methods", ()) or ():
        if isinstance(candidate, Mapping) and candidate.get("status") == "rejected":
            return True
    return False


def executable_claim_specs_for_contract(
    contract: CompiledValidationContract | None,
) -> tuple[ExecutableClaimSpec, ...]:
    """Return critic/arbiter executable claims admitted by validation."""
    if contract is None:
        return ()

    instrument = str(getattr(contract, "instrument_type", "") or "").strip().lower()
    specs: list[ExecutableClaimSpec] = []
    seen: set[str] = set()
    for check in getattr(contract, "deterministic_checks", ()) or ():
        check_id = str(getattr(check, "check_id", "") or "").strip()
        if not check_id:
            continue
        category = str(getattr(check, "category", "") or "").strip()
        relation = getattr(check, "relation", None)
        claim_id = _claim_id_for_validation_check(
            check_id,
            instrument=instrument,
            relation=None if relation is None else str(relation),
        )
        if claim_id is None or claim_id in seen:
            continue
        seen.add(claim_id)
        specs.append(
            ExecutableClaimSpec(
                claim_id=claim_id,
                validation_check_id=check_id,
                category=category,
                relation=relation,
                executable=True,
            )
        )
    return tuple(specs)


def _claim_id_for_validation_check(
    check_id: str,
    *,
    instrument: str,
    relation: str | None,
) -> str | None:
    """Map validation check ids onto bounded critic/arbiter claim ids."""
    if check_id == "check_non_negativity":
        return "price_non_negative"
    if check_id in {"check_vol_sensitivity", "check_vol_monotonicity"}:
        return "volatility_input_usage"
    if check_id == "check_rate_monotonicity":
        return "rate_sensitivity_present"
    if check_id == "check_bounded_by_reference":
        if instrument == "puttable_bond" or relation == ">=":
            return "puttable_bound_vs_straight_bond"
        if instrument == "callable_bond" or relation == "<=":
            return "callable_bound_vs_straight_bond"
    return None


def _resolve_instrument_type(
    *,
    instrument_type: str | None,
    product_ir=None,
    semantic_blueprint=None,
) -> str:
    """Return the most specific normalized instrument identifier available."""
    resolved = resolve_authoritative_instrument_type(
        instrument_type,
        getattr(product_ir, "instrument", None),
        getattr(semantic_blueprint, "semantic_id", None),
    )
    return resolved or ""


def _resolve_route_identity(
    *,
    generation_plan=None,
    semantic_blueprint=None,
) -> tuple[str | None, str | None]:
    """Return the selected route id and route family when available."""
    has_exact_backend_fit = _generation_plan_has_exact_backend_fit(generation_plan) and not _lowering_errors_for(
        semantic_blueprint
    )
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    if primitive_plan is not None and getattr(primitive_plan, "route", None):
        return (
            str(getattr(primitive_plan, "route", None) or "").strip() or None,
            (
                str(getattr(primitive_plan, "route_family", None) or "").strip()
                or (
                    str(getattr(generation_plan, "backend_route_family", None) or "").strip()
                    if has_exact_backend_fit
                    else ""
                )
                or None
            ),
        )
    if (
        has_exact_backend_fit
        and generation_plan is not None
        and getattr(generation_plan, "backend_route_family", None)
    ):
        return (
            None,
            str(getattr(generation_plan, "backend_route_family", None) or "").strip() or None,
        )
    lowering = getattr(semantic_blueprint, "dsl_lowering", None)
    if lowering is not None:
        return (
            str(getattr(lowering, "route_id", None) or "").strip() or None,
            str(getattr(lowering, "route_family", None) or "").strip() or None,
        )
    return None, None


def _resolve_backend_binding_identity(
    *,
    generation_plan=None,
    route_id: str | None = None,
    product_ir=None,
    semantic_blueprint=None,
) -> str | None:
    """Return the exact backend-binding identity when available."""
    if _lowering_errors_for(semantic_blueprint):
        return None
    has_exact_backend_fit = _generation_plan_has_exact_backend_fit(generation_plan)
    if (
        has_exact_backend_fit
        and generation_plan is not None
        and getattr(generation_plan, "backend_binding_id", None)
    ):
        return str(getattr(generation_plan, "backend_binding_id", None) or "").strip() or None
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    if (
        has_exact_backend_fit
        and primitive_plan is not None
        and getattr(primitive_plan, "backend_binding_id", None)
    ):
        return str(getattr(primitive_plan, "backend_binding_id", None) or "").strip() or None
    if not route_id:
        return None
    from trellis.agent.backend_bindings import resolve_backend_binding_by_route_id

    resolved = resolve_backend_binding_by_route_id(
        route_id,
        product_ir=product_ir,
        primitive_plan=primitive_plan,
    )
    if resolved is None:
        return None
    return str(getattr(resolved, "binding_id", "") or "").strip() or None


def _exact_bundle_id_for(
    *,
    bundle_id: str,
    backend_binding_id: str | None,
) -> str | None:
    """Return the binding-scoped exact validation identity when available."""
    normalized_bundle = str(bundle_id or "").strip()
    normalized_binding = str(backend_binding_id or "").strip()
    if not normalized_bundle or not normalized_binding:
        return None
    return f"{normalized_bundle}@{normalized_binding}"


def _generation_plan_has_exact_backend_fit(generation_plan) -> bool:
    """Return whether the generation plan represents an exact backend fit."""
    if generation_plan is None:
        return False
    if str(getattr(generation_plan, "lane_plan_kind", "") or "").strip() == "exact_target_binding":
        return True
    for attr in ("backend_exact_target_refs", "backend_helper_refs"):
        if tuple(getattr(generation_plan, attr, ()) or ()):
            return True
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    if primitive_plan is None:
        return False
    for attr in ("backend_exact_target_refs", "backend_helper_refs"):
        if tuple(getattr(primitive_plan, attr, ()) or ()):
            return True
    return False


def _requested_outputs_for(*, request=None, semantic_blueprint=None) -> tuple[str, ...]:
    """Return normalized requested outputs attached to validation."""
    request_outputs = getattr(request, "requested_outputs", None)
    if request_outputs:
        return _tuple_strings(request_outputs)
    blueprint_outputs = getattr(semantic_blueprint, "requested_outputs", None)
    return _tuple_strings(blueprint_outputs)


def _required_market_data_for(*, pricing_plan=None, product_ir=None, semantic_blueprint=None) -> tuple[str, ...]:
    """Return the normalized required market-data capabilities."""
    if semantic_blueprint is not None and getattr(semantic_blueprint, "required_market_data", None):
        return _tuple_strings(getattr(semantic_blueprint, "required_market_data", ()))
    if pricing_plan is not None and getattr(pricing_plan, "required_market_data", None):
        return _tuple_strings(sorted(getattr(pricing_plan, "required_market_data", ()) or ()))
    return _tuple_strings(sorted(getattr(product_ir, "required_market_data", ()) or ()))


def _category_for_check(categories: Mapping[str, tuple[str, ...]], check_id: str) -> str:
    """Return the first matching category for a check."""
    for category, values in categories.items():
        if check_id in values:
            return category
    return "uncategorized"


def _relation_for_check(
    check_id: str,
    *,
    instrument_type: str | None = None,
    comparison_spec=None,
) -> str | None:
    """Return the current relation semantics implied by one deterministic check."""
    if check_id == "check_bounded_by_reference":
        return _bound_relation_for(
            instrument_type=instrument_type,
            comparison_spec=comparison_spec,
        )
    if check_id == "check_rate_style_swaption_helper_consistency":
        return "within_tolerance"
    return None


def _harness_requirements_for(check_id: str) -> tuple[str, ...]:
    """Return the deterministic harness requirements for one check."""
    if check_id in {"check_non_negativity", "check_price_sanity", "check_quanto_required_inputs"}:
        return ("test_payoff", "market_state")
    if check_id == "check_rate_style_swaption_helper_consistency":
        return ("payoff_factory", "market_state_factory")
    if check_id in {
        "check_vol_sensitivity",
        "check_vol_monotonicity",
        "check_rate_monotonicity",
        "check_quanto_cross_currency_semantics",
        "check_cds_spread_quote_normalization",
        "check_cds_credit_curve_sensitivity",
    }:
        return ("payoff_factory", "market_state_factory")
    if check_id == "check_zero_vol_intrinsic":
        return ("payoff_factory", "market_state_factory", "intrinsic_fn")
    if check_id == "check_bounded_by_reference":
        return ("payoff_factory", "reference_factory", "market_state_factory")
    return ()


def _check_supported_by_market_data(
    check_id: str,
    required_market_data: tuple[str, ...],
) -> bool:
    """Return whether the compiled market-data surface can support the check."""
    requirements = _CHECK_MARKET_DATA_REQUIREMENTS.get(check_id)
    if not requirements:
        return True
    available = {str(item).strip() for item in required_market_data if str(item).strip()}
    if requirements is _VOL_CHECK_INPUTS:
        return bool(available & requirements)
    return requirements.issubset(available)


def _comparison_relations_for(
    *,
    bundle,
    instrument_type: str | None = None,
    semantic_blueprint=None,
    comparison_spec=None,
) -> tuple[ValidationRelationSpec, ...]:
    """Compile current comparison relations from available validation metadata."""
    relations: list[ValidationRelationSpec] = []
    seen: set[tuple[str, str, str]] = set()
    comparison_spec_relations = _comparison_spec_relations_for(comparison_spec)

    if "check_bounded_by_reference" in bundle.checks:
        relation = ValidationRelationSpec(
            target_id="bounded_by_reference",
            relation=_bound_relation_for(
                instrument_type=instrument_type,
                comparison_spec=comparison_spec,
            ),
            source="validation_bundle_bound",
        )
        relations.append(relation)
        seen.add((relation.target_id, relation.relation, relation.source))

    comparison_spec_targets = _comparison_spec_targets_for(comparison_spec)
    for target_id in comparison_spec_targets:
        relation_value = comparison_spec_relations.get(target_id, "within_tolerance")
        relation = ValidationRelationSpec(
            target_id=target_id,
            relation=relation_value,
            source="comparison_spec_target",
        )
        key = (relation.target_id, relation.relation, relation.source)
        if key not in seen:
            relations.append(relation)
            seen.add(key)

    for target_id in _semantic_comparison_targets_for(semantic_blueprint=semantic_blueprint):
        relation_value = comparison_spec_relations.get(target_id, "unspecified")
        relation = ValidationRelationSpec(
            target_id=target_id,
            relation=relation_value,
            source="declared_comparison_target",
        )
        key = (relation.target_id, relation.relation, relation.source)
        if key not in seen:
            relations.append(relation)
            seen.add(key)
    return tuple(relations)


def _semantic_comparison_targets_for(*, semantic_blueprint=None) -> tuple[str, ...]:
    """Return declared comparison-target ids from semantic validation metadata."""
    targets: list[str] = []
    sem_contract = getattr(semantic_blueprint, "contract", None)
    sem_validation = getattr(sem_contract, "validation", None)
    for value in getattr(sem_validation, "comparison_targets", ()) or ():
        text = str(value).strip()
        if text and text not in targets:
            targets.append(text)
    return tuple(targets)


def _comparison_spec_targets_for(comparison_spec) -> tuple[str, ...]:
    """Return declared comparison-target ids from request-level comparison metadata."""
    validation_targets = getattr(comparison_spec, "validation_targets", None) or {}
    targets: list[str] = []

    internal_targets = validation_targets.get("internal") or ()
    for value in internal_targets:
        text = str(value).strip()
        if text and text not in targets:
            targets.append(text)

    analytical_target = validation_targets.get("analytical")
    if analytical_target:
        text = str(analytical_target).strip()
        if text and text not in targets:
            targets.append(text)

    relations = validation_targets.get("relations") or {}
    if isinstance(relations, Mapping):
        for value in relations.keys():
            text = str(value).strip()
            if text and text not in targets:
                targets.append(text)

    reserved = {"internal", "external", "analytical", "tolerance_pct", "relations", "relation"}
    for key, value in validation_targets.items():
        if key in reserved:
            continue
        text = str(key).strip()
        if text and text not in targets:
            targets.append(text)
        if isinstance(value, Mapping):
            target_id = str(value.get("target_id") or value.get("target") or "").strip()
            if target_id and target_id not in targets:
                targets.append(target_id)
    return tuple(targets)


def _comparison_spec_relations_for(comparison_spec) -> dict[str, str]:
    """Return normalized request-level comparison relations keyed by target id."""
    validation_targets = getattr(comparison_spec, "validation_targets", None) or {}
    relations: dict[str, str] = {}

    raw_relations = validation_targets.get("relations") or {}
    if isinstance(raw_relations, Mapping):
        for target_id, relation in raw_relations.items():
            normalized = normalize_comparison_relation(relation)
            text = str(target_id).strip()
            if text and normalized:
                relations[text] = normalized

    reserved = {"internal", "external", "analytical", "tolerance_pct", "relations", "relation"}
    for key, value in validation_targets.items():
        if key in reserved or not isinstance(value, Mapping):
            continue
        normalized = normalize_comparison_relation(value.get("relation"))
        if not normalized:
            continue
        text = str(key).strip()
        if text:
            relations[text] = normalized
        target_id = str(value.get("target_id") or value.get("target") or "").strip()
        if target_id:
            relations[target_id] = normalized
    return relations


def _bound_relation_for(*, instrument_type: str | None = None, comparison_spec=None) -> str:
    """Return the directional bound relation for callable or puttable reference checks."""
    comparison_spec_relations = _comparison_spec_relations_for(comparison_spec)
    override = comparison_spec_relations.get("bounded_by_reference")
    if override in {"<=", ">="}:
        return override
    normalized_instrument = str(instrument_type or "").strip().lower()
    if normalized_instrument == "puttable_bond":
        return ">="
    return "<="


def _lowering_errors_for(semantic_blueprint) -> tuple[str, ...]:
    """Return structured lowering error strings from the semantic blueprint."""
    lowering = getattr(semantic_blueprint, "dsl_lowering", None)
    if lowering is None:
        return ()
    errors: list[str] = []
    for item in getattr(lowering, "errors", ()) or ():
        stage = str(getattr(item, "stage", "") or "").strip()
        code = str(getattr(item, "code", "") or "").strip()
        message = str(getattr(item, "message", "") or "").strip()
        parts = [part for part in (stage, code, message) if part]
        text = ":".join(parts)
        if text and text not in errors:
            errors.append(text)
    return tuple(errors)


def _admissibility_failures_for(
    *,
    route_id: str | None,
    semantic_blueprint=None,
    product_ir=None,
) -> tuple[str, ...]:
    """Return typed route-admissibility failures for the selected route."""
    if not route_id or semantic_blueprint is None:
        return ()
    from trellis.agent.route_registry import evaluate_route_admissibility, find_route_by_id

    route = find_route_by_id(route_id)
    if route is None:
        return ()
    decision = evaluate_route_admissibility(
        route,
        semantic_blueprint=semantic_blueprint,
        product_ir=product_ir,
    )
    return _tuple_strings(decision.failures)


def _residual_risks_for(
    *,
    comparison_relations: tuple[ValidationRelationSpec, ...],
    lowering_errors: tuple[str, ...],
    admissibility_failures: tuple[str, ...],
    semantic_blueprint=None,
) -> tuple[str, ...]:
    """Return compact residual-risk ids derived from the compiled state."""
    risks: list[str] = []
    if lowering_errors:
        risks.append("dsl_lowering_errors_present")
    if admissibility_failures:
        risks.append("route_admissibility_failures_present")
    if any(item.relation == "unspecified" for item in comparison_relations):
        risks.append("comparison_relations_unspecified")
    if getattr(semantic_blueprint, "measure_support_warnings", ()) or ():
        risks.append("measure_support_warnings_present")
    if getattr(semantic_blueprint, "unsupported_paths", ()) or ():
        risks.append("unsupported_paths_declared")
    return tuple(risks)
