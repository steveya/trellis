"""Typed semantic-contract schema for family-name-free synthesis."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any
import re

import yaml

from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.semantic_concepts import (
    get_semantic_concept_definition,
    semantic_concept_summary,
)
from trellis.core.capabilities import normalize_capability_name


def _tuple(values) -> tuple[str, ...]:
    """Return a deduplicated tuple preserving input order."""
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _freeze_mapping(mapping: MappingProxyType | dict[str, object] | None) -> MappingProxyType:
    """Convert a mutable dict (or None) into a read-only MappingProxyType for use in frozen dataclasses."""
    return MappingProxyType(dict(mapping or {}))


def _string_list(values) -> list[str]:
    """Normalize a sequence into YAML-safe string lists."""
    return [str(value).strip() for value in _tuple(values)]


DEFAULT_PHASE_ORDER = (
    "event",
    "observation",
    "decision",
    "determination",
    "settlement",
    "state_update",
)


@dataclass(frozen=True)
class ConventionEnv:
    """Compact contract-convention environment for semantic contracts."""

    calendar: str = ""
    business_day_convention: str = ""
    day_count_convention: str = ""
    settlement_lag: str = ""
    payment_currency: str = ""
    reporting_currency: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticTimeline:
    """Phase-aware contract timeline keyed by role-specific date sets."""

    phase_order: tuple[str, ...] = DEFAULT_PHASE_ORDER
    anchor_dates: tuple[str, ...] = ()
    event_dates: tuple[str, ...] = ()
    observation_dates: tuple[str, ...] = ()
    decision_dates: tuple[str, ...] = ()
    determination_dates: tuple[str, ...] = ()
    settlement_dates: tuple[str, ...] = ()
    state_update_dates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservableSpec:
    """One typed observable consumed by contract semantics."""

    observable_id: str
    observable_type: str
    description: str = ""
    source: str = ""
    schedule_role: str = ""
    availability_phase: str = "observation"
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class StateField:
    """One semantic state component, tagged as event state or contract memory."""

    field_name: str
    kind: str = "contract_memory"
    description: str = ""
    source_observables: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObligationSpec:
    """One emitted settlement obligation from the semantic contract."""

    obligation_id: str
    settle_date_rule: str
    amount_expression: str
    currency: str = ""
    settlement_kind: str = "cash"
    trigger: str = ""
    provenance: str = ""


@dataclass(frozen=True)
class ControllerProtocol:
    """Tranche-1 controller protocol for strategic rights."""

    controller_style: str = "identity"
    controller_role: str = "none"
    decision_phase: str = "decision"
    schedule_role: str = ""
    admissible_actions: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class SemanticAuditInfo:
    """Minimal audit and provenance metadata for one semantic slice."""

    semantic_origin: str = ""
    provenance: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    legacy_mirrors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImplementationHints:
    """Implementation-facing hints that do not change contract meaning."""

    preserve_route_behavior: bool = True
    event_machine_source: str = ""
    primary_schedule_role: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticMarketInputSpec:
    """One named market input in a semantic contract."""

    input_id: str
    description: str = ""
    capability: str | None = None
    aliases: tuple[str, ...] = ()
    connector_hint: str = ""
    derivable_from: tuple[str, ...] = ()
    allowed_provenance: tuple[str, ...] = ("observed",)


def _to_compact_dict(obj) -> dict:
    """Return a dict with only non-default, non-empty fields from a frozen dataclass."""
    result: dict = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        if val is None:
            continue
        if isinstance(val, tuple) and len(val) == 0:
            continue
        if val == f.default:
            continue
        if f.default is not dataclasses.MISSING and val == f.default:
            continue
        if f.default_factory is not dataclasses.MISSING and val == f.default_factory():
            continue
        result[f.name] = val
    return result


@dataclass(frozen=True)
class SemanticProductSemantics:
    """Typed product semantics for one canonical semantic slice."""

    # --- Identity ---
    semantic_id: str
    semantic_version: str
    instrument_class: str
    instrument_aliases: tuple[str, ...]
    payoff_family: str

    # --- Underlier & Payoff ---
    conventions: ConventionEnv = field(default_factory=ConventionEnv)
    timeline: SemanticTimeline = field(default_factory=SemanticTimeline)
    underlier_structure: str = ""
    payoff_rule: str = ""
    settlement_rule: str = ""
    payoff_traits: tuple[str, ...] = ()
    observables: tuple[ObservableSpec, ...] = ()
    state_fields: tuple[StateField, ...] = ()
    obligations: tuple[ObligationSpec, ...] = ()
    controller_protocol: ControllerProtocol = field(default_factory=ControllerProtocol)
    audit_info: SemanticAuditInfo = field(default_factory=SemanticAuditInfo)
    implementation_hints: ImplementationHints = field(default_factory=ImplementationHints)

    # --- Exercise & Path ---
    exercise_style: str = "none"
    path_dependence: str = "terminal_markov"
    schedule_dependence: bool = False
    state_dependence: str = "terminal_markov"
    model_family: str = "generic"
    multi_asset: bool = False

    # --- Schedule & Observation ---
    observation_schedule: tuple[str, ...] = ()
    observation_basis: str = ""
    selection_operator: str = ""
    selection_scope: str = ""
    selection_count: int = 0
    lock_rule: str = ""
    aggregation_rule: str = ""
    maturity_settlement_rule: str = ""

    # --- Composition ---
    constituents: tuple[str, ...] = ()
    state_variables: tuple[str, ...] = ()
    event_transitions: tuple[str, ...] = ()
    event_machine: object | None = None  # EventMachine when typed

    def to_compact_dict(self) -> dict:
        """Return a dict with only non-default, non-empty fields."""
        return _to_compact_dict(self)


@dataclass(frozen=True)
class SemanticMarketDataContract:
    """Market-data requirements and provenance policy."""

    required_inputs: tuple[SemanticMarketInputSpec, ...]
    optional_inputs: tuple[SemanticMarketInputSpec, ...] = ()
    derivable_inputs: tuple[str, ...] = ()
    estimation_policy: tuple[str, ...] = ()
    provenance_requirements: tuple[str, ...] = ()
    missing_data_error_policy: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticMethodContract:
    """Candidate and supported method families for a semantic slice."""

    candidate_methods: tuple[str, ...]
    reference_methods: tuple[str, ...] = ()
    production_methods: tuple[str, ...] = ()
    unsupported_variants: tuple[str, ...] = ()
    method_limitations: tuple[str, ...] = ()
    preferred_method: str | None = None


@dataclass(frozen=True)
class SemanticValidationContract:
    """Validation and cross-check expectations for a semantic slice."""

    bundle_hints: tuple[str, ...] = ()
    universal_checks: tuple[str, ...] = ()
    semantic_checks: tuple[str, ...] = ()
    comparison_targets: tuple[str, ...] = ()
    reduction_cases: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticBlueprintHints:
    """Compiler hints for later route and module selection."""

    target_modules: tuple[str, ...] = ()
    primitive_families: tuple[str, ...] = ()
    adapter_obligations: tuple[str, ...] = ()
    proving_tasks: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    spec_schema_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticContract:
    """Top-level typed semantic contract."""

    product: SemanticProductSemantics
    market_data: SemanticMarketDataContract
    methods: SemanticMethodContract
    validation: SemanticValidationContract
    blueprint: SemanticBlueprintHints
    description: str = ""
    calibration: object | None = None  # CalibrationContract when typed

    @property
    def semantic_id(self) -> str:
        """Return the normalized semantic identifier."""
        return self.product.semantic_id


def parse_semantic_contract(spec: SemanticContract | dict[str, Any] | str) -> SemanticContract:
    """Parse a semantic contract from an object, dict, or YAML string."""
    if isinstance(spec, SemanticContract):
        return spec
    if isinstance(spec, str):
        payload = yaml.safe_load(spec) or {}
    else:
        payload = dict(spec)

    return SemanticContract(
        product=_parse_product_semantics(payload["product"]),
        market_data=_parse_market_data_contract(payload["market_data"]),
        methods=_parse_method_contract(payload["methods"]),
        validation=_parse_validation_contract(payload.get("validation", {})),
        blueprint=_parse_blueprint_hints(payload.get("blueprint", {})),
        description=str(payload.get("description", "")).strip(),
        calibration=payload.get("calibration"),
    )


def semantic_contract_summary(contract: SemanticContract | dict[str, Any] | str) -> dict[str, Any]:
    """Return a YAML-safe summary of a semantic contract for request metadata."""
    parsed = parse_semantic_contract(contract)
    concept = get_semantic_concept_definition(parsed.product.semantic_id)
    return {
        "semantic_id": parsed.product.semantic_id,
        "semantic_version": parsed.product.semantic_version,
        "semantic_concept": semantic_concept_summary(concept),
        "product": {
            "instrument_class": parsed.product.instrument_class,
            "underlier_structure": parsed.product.underlier_structure,
            "payoff_family": parsed.product.payoff_family,
            "payoff_rule": parsed.product.payoff_rule,
            "settlement_rule": parsed.product.settlement_rule,
            "observation_schedule": list(parsed.product.observation_schedule),
            "constituents": list(parsed.product.constituents),
            "exercise_style": parsed.product.exercise_style,
            "path_dependence": parsed.product.path_dependence,
            "schedule_dependence": parsed.product.schedule_dependence,
            "state_dependence": parsed.product.state_dependence,
            "selection_scope": parsed.product.selection_scope,
            "selection_operator": parsed.product.selection_operator,
            "selection_count": parsed.product.selection_count,
            "lock_rule": parsed.product.lock_rule,
            "aggregation_rule": parsed.product.aggregation_rule,
            "multi_asset": parsed.product.multi_asset,
        },
        "typed_semantics": {
            "phase_order": list(parsed.product.timeline.phase_order),
            "observables": [
                {
                    "observable_id": item.observable_id,
                    "observable_type": item.observable_type,
                    "availability_phase": item.availability_phase,
                    "schedule_role": item.schedule_role,
                }
                for item in parsed.product.observables
            ],
            "state_fields": [
                {
                    "field_name": item.field_name,
                    "kind": item.kind,
                    "tags": list(item.tags),
                }
                for item in parsed.product.state_fields
            ],
            "obligations": [
                {
                    "obligation_id": item.obligation_id,
                    "settle_date_rule": item.settle_date_rule,
                    "settlement_kind": item.settlement_kind,
                }
                for item in parsed.product.obligations
            ],
            "controller_protocol": {
                "controller_style": parsed.product.controller_protocol.controller_style,
                "controller_role": parsed.product.controller_protocol.controller_role,
                "decision_phase": parsed.product.controller_protocol.decision_phase,
                "schedule_role": parsed.product.controller_protocol.schedule_role,
            },
            "implementation_hints": {
                "preserve_route_behavior": parsed.product.implementation_hints.preserve_route_behavior,
                "event_machine_source": parsed.product.implementation_hints.event_machine_source,
                "primary_schedule_role": parsed.product.implementation_hints.primary_schedule_role,
            },
            "event_machine_present": parsed.product.event_machine is not None,
        },
        "market_data": {
            "required_inputs": [item.input_id for item in parsed.market_data.required_inputs],
            "optional_inputs": [item.input_id for item in parsed.market_data.optional_inputs],
        },
        "methods": {
            "candidate_methods": list(parsed.methods.candidate_methods),
            "preferred_method": parsed.methods.preferred_method,
        },
        "blueprint": {
            "target_modules": list(parsed.blueprint.target_modules),
            "primitive_families": list(parsed.blueprint.primitive_families),
        },
    }


def make_ranked_observation_basket_contract(
    *,
    description: str,
    constituents: tuple[str, ...] | list[str],
    observation_schedule: tuple[str, ...] | list[str],
    preferred_method: str = "monte_carlo",
    include_correlation: bool = True,
) -> SemanticContract:
    """Construct the canonical ranked-observation basket semantic contract."""
    constituent_names = _tuple(constituents)
    schedule = _normalize_schedule(observation_schedule)
    if len(constituent_names) < 2:
        raise ValueError("Ranked observation basket contract requires at least two constituents.")
    if not schedule:
        raise ValueError("Ranked observation basket contract requires an observation schedule.")

    product = SemanticProductSemantics(
        semantic_id="ranked_observation_basket",
        semantic_version="c2.0",
        instrument_class="basket_path_payoff",
        instrument_aliases=("ranked_observation_basket", "ranked_selection_basket", "basket_path_payoff"),
        payoff_family="basket_path_payoff",
        timeline=_default_semantic_timeline(
            schedule,
            settlement_dates=schedule[-1:],
            state_update_dates=schedule,
        ),
        underlier_structure="multi_asset_basket",
        payoff_rule="ranked_observation_path_payoff",
        settlement_rule="settle_once_at_maturity",
        payoff_traits=(
            "ranked_observation",
            "remaining_selection",
            "remove_selected",
            "locked_returns",
            "maturity_settlement",
        ),
        observables=(
            ObservableSpec(
                observable_id="constituent_spots",
                observable_type="spot_vector",
                description="Observed constituent spots at each ranked observation date.",
                source="underlier_spots",
                schedule_role="observation_dates",
                availability_phase="observation",
            ),
            ObservableSpec(
                observable_id="ranked_constituent_return",
                observable_type="simple_return",
                description="Derived simple return used for ranked constituent selection.",
                source="derived_from_constituent_spots",
                schedule_role="determination_dates",
                availability_phase="determination",
                dependencies=("constituent_spots",),
            ),
        ),
        state_fields=(
            StateField(
                field_name="remaining_constituents",
                kind="contract_memory",
                description="Constituents that remain eligible for future observation dates.",
                source_observables=("ranked_constituent_return",),
                tags=("pathwise_only", "remaining_pool"),
            ),
            StateField(
                field_name="locked_returns",
                kind="contract_memory",
                description="Returns locked at prior ranked observations.",
                source_observables=("ranked_constituent_return",),
                tags=("pathwise_only", "locked_cashflow_state"),
            ),
        ),
        obligations=(
            ObligationSpec(
                obligation_id="maturity_cash_settlement",
                settle_date_rule="settle_once_at_maturity",
                amount_expression="average_locked_returns",
                settlement_kind="cash",
                trigger="all_observations_complete",
                provenance="semantic_contract",
            ),
        ),
        controller_protocol=ControllerProtocol(
            controller_style="identity",
            controller_role="none",
            decision_phase="decision",
            schedule_role="",
            admissible_actions=(),
            description="Automatic ranked-selection transitions only; no strategic controller.",
        ),
        audit_info=_default_audit_info(),
        implementation_hints=_default_implementation_hints(
            event_machine_source="derived_from_event_transitions",
            primary_schedule_role="observation_dates",
        ),
        exercise_style="none",
        path_dependence="path_dependent",
        schedule_dependence=True,
        state_dependence="path_dependent",
        model_family="equity_multi_asset",
        multi_asset=True,
        observation_schedule=schedule,
        observation_basis="simple_return",
        selection_operator="best_of_remaining",
        selection_scope="remaining_constituents",
        selection_count=1,
        lock_rule="remove_selected",
        aggregation_rule="average_locked_returns",
        maturity_settlement_rule="settle_once_at_maturity",
        constituents=constituent_names,
        state_variables=("remaining_constituents", "locked_returns"),
        event_transitions=(
            "rank_remaining_constituents",
            "remove_selected_constituent",
            "lock_simple_return",
            "settle_at_maturity",
        ),
        event_machine=_derive_event_machine(
            (
                "rank_remaining_constituents",
                "remove_selected_constituent",
                "lock_simple_return",
                "settle_at_maturity",
            ),
            state_dependence="path_dependent",
        ),
    )

    required_inputs = [
        SemanticMarketInputSpec(
            input_id="discount_curve",
            description="Risk-free discount curve for maturity settlement.",
            capability="discount_curve",
            aliases=("discount", "discount_rate"),
            connector_hint="Use the settlement discount curve.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="underlier_spots",
            description="Current spot levels for each basket constituent.",
            capability="spot",
            aliases=("spots", "basket_spots", "underlier_spot"),
            connector_hint="Provide one spot per constituent.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="black_vol_surface",
            description="Implied volatility surface for each basket constituent.",
            capability="black_vol_surface",
            aliases=("vol_surface", "volatility_surface"),
            connector_hint="Provide implied vols or a surface.",
            allowed_provenance=("observed",),
        ),
    ]
    if include_correlation:
        required_inputs.append(
        SemanticMarketInputSpec(
            input_id="correlation_matrix",
            description="Pairwise correlation matrix across basket constituents.",
            capability="model_parameters",
            aliases=("correlation", "corr", "basket_correlation"),
            connector_hint="Provide a positive-definite correlation matrix in model_parameters.",
            derivable_from=(),
            allowed_provenance=(
                "observed",
                "estimated",
                "calibrated",
                "implied",
                "sampled",
                "synthetic",
            ),
        )
    )

    market_data = SemanticMarketDataContract(
        required_inputs=tuple(required_inputs),
        optional_inputs=(),
        derivable_inputs=(),
        estimation_policy=(
            "never_fabricate_correlation_matrix",
            "derive_forward_curve_from_discount_curve_when_available",
        ),
        provenance_requirements=(
            "observed_or_estimated_correlation_required_for_multi_asset_mc",
        ),
        missing_data_error_policy=(
            "fail_fast_on_missing_discount_or_spot_or_vol_or_correlation",
        ),
    )

    methods = SemanticMethodContract(
        candidate_methods=(normalize_method(preferred_method),),
        reference_methods=(normalize_method(preferred_method),),
        production_methods=(normalize_method(preferred_method),),
        preferred_method=normalize_method(preferred_method),
    )

    validation = SemanticValidationContract(
        bundle_hints=("ranked_observation_basket_contract",),
        universal_checks=(
            "observation_schedule_present",
            "constituents_present",
            "selection_scope_remaining_constituents",
            "selection_count_is_one",
            "correlation_required_for_multi_asset_monte_carlo",
        ),
        semantic_checks=(
            "best_performer_selected_from_remaining_pool",
            "selected_constituent_removed_after_lock",
            "simple_return_locked_per_observation",
            "settle_once_at_maturity",
        ),
        comparison_targets=(normalize_method(preferred_method),),
        reduction_cases=("three_constituent_three_date_basket",),
    )

    blueprint = SemanticBlueprintHints(
        target_modules=(
            "trellis.models.resolution.basket_semantics",
            "trellis.models.monte_carlo.semantic_basket",
        ),
        primitive_families=("correlated_basket_monte_carlo",),
        adapter_obligations=(
            "resolve_basket_spots_for_ranked_selection",
            "resolve_basket_correlation_matrix",
            "build_ranked_observation_snapshot_state",
            "lock_selected_simple_return",
            "aggregate_locked_returns_at_maturity",
        ),
        proving_tasks=(
            "compile_request_to_product_ir",
            "validate_ranked_observation_contract",
            "emit_bounded_semantic_blueprint",
        ),
        blocked_by=(),
        spec_schema_hints=("basket_option",),
    )

    return SemanticContract(
        product=product,
        market_data=market_data,
        methods=methods,
        validation=validation,
        blueprint=blueprint,
        description=description,
    )


def _semantic_method_contract(
    candidate_methods: tuple[str, ...] | list[str],
    *,
    preferred_method: str | None = None,
) -> SemanticMethodContract:
    """Build a deterministic method contract with a stable default method."""
    candidates = _normalize_methods(candidate_methods)
    preferred = normalize_method(preferred_method) if preferred_method else (
        candidates[0] if candidates else None
    )
    reference_methods = (preferred,) if preferred else ()
    production_methods = (preferred,) if preferred else ()
    return SemanticMethodContract(
        candidate_methods=candidates,
        reference_methods=reference_methods,
        production_methods=production_methods,
        preferred_method=preferred,
    )


def _semantic_market_data_contract(
    required_inputs: tuple[SemanticMarketInputSpec, ...] | list[SemanticMarketInputSpec],
    *,
    optional_inputs: tuple[SemanticMarketInputSpec, ...] | list[SemanticMarketInputSpec] = (),
    derivable_inputs: tuple[str, ...] | list[str] = (),
    estimation_policy: tuple[str, ...] | list[str] = (),
    provenance_requirements: tuple[str, ...] | list[str] = (),
    missing_data_error_policy: tuple[str, ...] | list[str] = (),
) -> SemanticMarketDataContract:
    """Build a deterministic market-data contract."""
    return SemanticMarketDataContract(
        required_inputs=tuple(required_inputs),
        optional_inputs=tuple(optional_inputs),
        derivable_inputs=_tuple(derivable_inputs),
        estimation_policy=_tuple(estimation_policy),
        provenance_requirements=_tuple(provenance_requirements),
        missing_data_error_policy=_tuple(missing_data_error_policy),
    )


def _semantic_validation_contract(
    *,
    bundle_hints: tuple[str, ...] | list[str] = (),
    universal_checks: tuple[str, ...] | list[str] = (),
    semantic_checks: tuple[str, ...] | list[str] = (),
    comparison_targets: tuple[str, ...] | list[str] = (),
    reduction_cases: tuple[str, ...] | list[str] = (),
) -> SemanticValidationContract:
    """Build a deterministic validation contract."""
    return SemanticValidationContract(
        bundle_hints=_tuple(bundle_hints),
        universal_checks=_tuple(universal_checks),
        semantic_checks=_tuple(semantic_checks),
        comparison_targets=_tuple(comparison_targets),
        reduction_cases=_tuple(reduction_cases),
    )


def _semantic_blueprint_hints(
    *,
    target_modules: tuple[str, ...] | list[str] = (),
    primitive_families: tuple[str, ...] | list[str] = (),
    adapter_obligations: tuple[str, ...] | list[str] = (),
    proving_tasks: tuple[str, ...] | list[str] = (),
    blocked_by: tuple[str, ...] | list[str] = (),
    spec_schema_hints: tuple[str, ...] | list[str] = (),
) -> SemanticBlueprintHints:
    """Build deterministic compiler hints for a semantic contract."""
    return SemanticBlueprintHints(
        target_modules=_tuple(target_modules),
        primitive_families=_tuple(primitive_families),
        adapter_obligations=_tuple(adapter_obligations),
        proving_tasks=_tuple(proving_tasks),
        blocked_by=_tuple(blocked_by),
        spec_schema_hints=_tuple(spec_schema_hints),
    )


def _semantic_contract_from_sections(
    *,
    product: SemanticProductSemantics,
    required_inputs: tuple[SemanticMarketInputSpec, ...] | list[SemanticMarketInputSpec],
    optional_inputs: tuple[SemanticMarketInputSpec, ...] | list[SemanticMarketInputSpec] = (),
    derivable_inputs: tuple[str, ...] | list[str] = (),
    estimation_policy: tuple[str, ...] | list[str] = (),
    provenance_requirements: tuple[str, ...] | list[str] = (),
    missing_data_error_policy: tuple[str, ...] | list[str] = (),
    candidate_methods: tuple[str, ...] | list[str] = (),
    preferred_method: str | None = None,
    bundle_hints: tuple[str, ...] | list[str] = (),
    universal_checks: tuple[str, ...] | list[str] = (),
    semantic_checks: tuple[str, ...] | list[str] = (),
    comparison_targets: tuple[str, ...] | list[str] = (),
    reduction_cases: tuple[str, ...] | list[str] = (),
    target_modules: tuple[str, ...] | list[str] = (),
    primitive_families: tuple[str, ...] | list[str] = (),
    adapter_obligations: tuple[str, ...] | list[str] = (),
    proving_tasks: tuple[str, ...] | list[str] = (),
    blocked_by: tuple[str, ...] | list[str] = (),
    spec_schema_hints: tuple[str, ...] | list[str] = (),
    description: str = "",
    calibration: object | None = None,
) -> SemanticContract:
    """Assemble one typed semantic contract from section-level inputs."""
    return SemanticContract(
        product=product,
        market_data=_semantic_market_data_contract(
            required_inputs,
            optional_inputs=optional_inputs,
            derivable_inputs=derivable_inputs,
            estimation_policy=estimation_policy,
            provenance_requirements=provenance_requirements,
            missing_data_error_policy=missing_data_error_policy,
        ),
        methods=_semantic_method_contract(
            candidate_methods,
            preferred_method=preferred_method,
        ),
        validation=_semantic_validation_contract(
            bundle_hints=bundle_hints,
            universal_checks=universal_checks,
            semantic_checks=semantic_checks,
            comparison_targets=comparison_targets,
            reduction_cases=reduction_cases,
        ),
        blueprint=_semantic_blueprint_hints(
            target_modules=target_modules,
            primitive_families=primitive_families,
            adapter_obligations=adapter_obligations,
            proving_tasks=proving_tasks,
            blocked_by=blocked_by,
            spec_schema_hints=spec_schema_hints,
        ),
        description=description,
        calibration=calibration,
    )


def _split_supported_dates(
    text: str,
    term_sheet,
    *,
    parameter_keys: tuple[str, ...],
) -> tuple[str, ...]:
    """Extract ordered schedule dates from structured fields or free text."""
    parameters = getattr(term_sheet, "parameters", {}) or {}
    for key in parameter_keys:
        value = parameters.get(key)
        if value:
            if isinstance(value, str):
                return _parse_name_list(value)
            return _normalize_schedule(value)
    return _extract_observation_schedule(text, term_sheet)


def _extract_primary_underlier(text: str, term_sheet) -> tuple[str, ...]:
    """Extract the primary underlier for single-name option-style requests."""
    parameters = getattr(term_sheet, "parameters", {}) or {}
    for key in (
        "underlier",
        "underlier_name",
        "underlier_symbol",
        "asset",
        "spot_name",
    ):
        value = parameters.get(key)
        if value:
            if isinstance(value, str):
                names = _parse_name_list(value)
            else:
                names = _tuple(value)
            if names:
                return (names[0],)

    stopwords = {
        "CALL",
        "CALLABLE",
        "CURRENCY",
        "DISCOUNT",
        "EUR",
        "EURO",
        "FX",
        "FOR",
        "FROM",
        "IN",
        "ISSUER",
        "MATURITY",
        "OPTION",
        "PAYOFF",
        "PUT",
        "RATE",
        "SWAP",
        "SWAPTION",
        "THE",
        "UNDERLIER",
        "USD",
        "WITH",
    }
    tokens = re.findall(r"\b[A-Z][A-Z0-9_.-]{1,}\b", text)
    for token in tokens:
        upper = token.upper()
        if upper in stopwords:
            continue
        return (upper,)
    return ()


def make_vanilla_option_contract(
    *,
    description: str,
    underliers: tuple[str, ...] | list[str],
    observation_schedule: tuple[str, ...] | list[str],
    preferred_method: str = "analytical",
) -> SemanticContract:
    """Construct a generic vanilla option semantic contract."""
    underlier_names = _tuple(underliers)
    schedule = _normalize_schedule(observation_schedule)
    if not underlier_names:
        raise ValueError("Vanilla option contract requires at least one underlier.")
    if not schedule:
        raise ValueError("Vanilla option contract requires an expiry or exercise schedule.")

    product = SemanticProductSemantics(
        semantic_id="vanilla_option",
        semantic_version="c2.1",
        instrument_class="european_option",
        instrument_aliases=("vanilla_option", "european_option", "option"),
        payoff_family="vanilla_option",
        timeline=_default_semantic_timeline(
            schedule,
            includes_decision=True,
            settlement_dates=schedule,
        ),
        underlier_structure="single_underlier",
        payoff_rule="vanilla_option_payoff",
        settlement_rule="cash_settle_at_expiry",
        payoff_traits=("discounting", "vol_surface_dependence"),
        observables=(
            ObservableSpec(
                observable_id="terminal_underlier_spot",
                observable_type="spot",
                description="Observed underlier spot at option expiry.",
                source="underlier_spot",
                schedule_role="observation_dates",
                availability_phase="observation",
            ),
        ),
        state_fields=(
            StateField(
                field_name="underlier_price",
                kind="event_state",
                description="Observed terminal underlier price used for payoff determination.",
                source_observables=("terminal_underlier_spot",),
                tags=("terminal_markov", "recombining_safe"),
            ),
        ),
        obligations=(
            ObligationSpec(
                obligation_id="expiry_cash_settlement",
                settle_date_rule="cash_settle_at_expiry",
                amount_expression="vanilla_option_payoff",
                settlement_kind="cash",
                trigger="exercise_if_optimal_at_expiry",
                provenance="semantic_contract",
            ),
        ),
        controller_protocol=ControllerProtocol(
            controller_style="holder_max",
            controller_role="holder",
            decision_phase="decision",
            schedule_role="decision_dates",
            admissible_actions=("exercise", "continue"),
            description="Holder exercise decision at expiry.",
        ),
        audit_info=_default_audit_info(),
        implementation_hints=_default_implementation_hints(
            event_machine_source="derived_from_event_transitions",
            primary_schedule_role="decision_dates",
        ),
        exercise_style="european",
        path_dependence="terminal_markov",
        schedule_dependence=False,
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
        multi_asset=False,
        observation_schedule=schedule,
        observation_basis="terminal_payoff",
        selection_operator="",
        selection_scope="",
        selection_count=0,
        lock_rule="",
        aggregation_rule="",
        maturity_settlement_rule="cash_settle_at_expiry",
        constituents=underlier_names,
        state_variables=("underlier_price",),
        event_transitions=("evaluate_terminal_payoff", "settle_at_expiry"),
        event_machine=_derive_event_machine(
            ("evaluate_terminal_payoff", "settle_at_expiry"),
            state_dependence="terminal_markov",
        ),
    )

    required_inputs = (
        SemanticMarketInputSpec(
            input_id="discount_curve",
            description="Risk-free discount curve for present-value discounting.",
            capability="discount_curve",
            aliases=("discount", "discount_rate"),
            connector_hint="Use the settlement discount curve.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="underlier_spot",
            description="Spot level for the single underlier.",
            capability="spot",
            aliases=("spot", "underlier_spots", "underlier_price"),
            connector_hint="Provide the current spot for the priced underlier.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="black_vol_surface",
            description="Implied volatility surface for the underlier.",
            capability="black_vol_surface",
            aliases=("vol_surface", "volatility_surface"),
            connector_hint="Provide implied vol or a surface.",
            allowed_provenance=("observed",),
        ),
    )
    return _semantic_contract_from_sections(
        product=product,
        required_inputs=required_inputs,
        candidate_methods=("analytical", "rate_tree", "pde_solver", "monte_carlo"),
        preferred_method=preferred_method,
        bundle_hints=("vanilla_option_contract",),
        universal_checks=(
            "single_underlier_present",
            "expiry_or_exercise_date_present",
            "settlement_rule_present",
        ),
        semantic_checks=(
            "terminal_payoff_evaluated_from_single_underlier",
            "settlement_occurs_at_expiry",
        ),
        comparison_targets=(normalize_method(preferred_method),),
        reduction_cases=("single_underlier_terminal_payoff",),
        target_modules=("trellis.models.black",),
        primitive_families=("analytical_black76",),
        adapter_obligations=(
            "resolve_single_underlier_spot",
            "resolve_discount_curve",
            "map_terminal_payoff_to_black_kernel",
        ),
        proving_tasks=(
            "compile_request_to_product_ir",
            "validate_vanilla_option_contract",
            "emit_bounded_semantic_blueprint",
        ),
        spec_schema_hints=("european_option",),
        description=description,
    )


def make_quanto_option_contract(
    *,
    description: str,
    underliers: tuple[str, ...] | list[str],
    observation_schedule: tuple[str, ...] | list[str],
    preferred_method: str = "analytical",
) -> SemanticContract:
    """Construct a generic quanto-style semantic contract."""
    underlier_names = _tuple(underliers)
    schedule = _normalize_schedule(observation_schedule)
    if not underlier_names:
        raise ValueError("Quanto option contract requires at least one underlier.")
    if not schedule:
        raise ValueError("Quanto option contract requires an expiry or exercise schedule.")

    product = SemanticProductSemantics(
        semantic_id="quanto_option",
        semantic_version="c2.1",
        instrument_class="quanto_option",
        instrument_aliases=("quanto_option", "quanto", "fx_option"),
        payoff_family="vanilla_option",
        timeline=_default_semantic_timeline(
            schedule,
            includes_decision=True,
            settlement_dates=schedule,
        ),
        underlier_structure="cross_currency_single_underlier",
        payoff_rule="quanto_adjusted_vanilla_payoff",
        settlement_rule="cash_settle_at_expiry_after_fx_conversion",
        payoff_traits=("discounting", "vol_surface_dependence", "fx_translation"),
        observables=(
            ObservableSpec(
                observable_id="terminal_underlier_spot",
                observable_type="spot",
                description="Observed underlier spot at expiry.",
                source="underlier_spot",
                schedule_role="observation_dates",
                availability_phase="observation",
            ),
            ObservableSpec(
                observable_id="expiry_fx_rate",
                observable_type="fx_rate",
                description="Observed FX rate used for payout conversion at expiry.",
                source="fx_rates",
                schedule_role="observation_dates",
                availability_phase="observation",
            ),
        ),
        state_fields=(
            StateField(
                field_name="underlier_price",
                kind="event_state",
                description="Observed terminal underlier price.",
                source_observables=("terminal_underlier_spot",),
                tags=("terminal_markov", "recombining_safe"),
            ),
            StateField(
                field_name="fx_rate",
                kind="event_state",
                description="Observed FX rate applied at settlement conversion.",
                source_observables=("expiry_fx_rate",),
                tags=("terminal_markov", "recombining_safe"),
            ),
        ),
        obligations=(
            ObligationSpec(
                obligation_id="expiry_quanto_cash_settlement",
                settle_date_rule="cash_settle_at_expiry_after_fx_conversion",
                amount_expression="quanto_adjusted_vanilla_payoff",
                settlement_kind="cash",
                trigger="exercise_if_optimal_at_expiry",
                provenance="semantic_contract",
            ),
        ),
        controller_protocol=ControllerProtocol(
            controller_style="holder_max",
            controller_role="holder",
            decision_phase="decision",
            schedule_role="decision_dates",
            admissible_actions=("exercise", "continue"),
            description="Holder exercise decision at expiry after FX translation is known.",
        ),
        audit_info=_default_audit_info(),
        implementation_hints=_default_implementation_hints(
            event_machine_source="derived_from_event_transitions",
            primary_schedule_role="decision_dates",
        ),
        exercise_style="european",
        path_dependence="terminal_markov",
        schedule_dependence=False,
        state_dependence="terminal_markov",
        model_family="fx_cross_currency",
        multi_asset=False,
        observation_schedule=schedule,
        observation_basis="terminal_payoff",
        selection_operator="",
        selection_scope="",
        selection_count=0,
        lock_rule="",
        aggregation_rule="",
        maturity_settlement_rule="cash_settle_at_expiry_after_fx_conversion",
        constituents=underlier_names,
        state_variables=("underlier_price", "fx_rate"),
        event_transitions=("translate_payoff_through_fx", "settle_at_expiry"),
        event_machine=_derive_event_machine(
            ("translate_payoff_through_fx", "settle_at_expiry"),
            state_dependence="terminal_markov",
        ),
    )

    required_inputs = (
        SemanticMarketInputSpec(
            input_id="discount_curve",
            description="Domestic discount curve for present-value discounting.",
            capability="discount_curve",
            aliases=("discount", "discount_rate"),
            connector_hint="Use the payout-currency discount curve.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="forward_curve",
            description="Forward curve for the underlier or funding leg.",
            capability="forward_curve",
            aliases=("forecast_curve", "forward_rate_curve"),
            connector_hint="Provide the relevant forward curve.",
            derivable_from=("discount_curve",),
            allowed_provenance=("observed", "derived"),
        ),
        SemanticMarketInputSpec(
            input_id="underlier_spot",
            description="Spot level for the underlier asset.",
            capability="spot",
            aliases=("spot", "underlier_spots", "underlier_price"),
            connector_hint="Provide the current spot for the underlier.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="black_vol_surface",
            description="Implied volatility surface for the underlier.",
            capability="black_vol_surface",
            aliases=("vol_surface", "volatility_surface"),
            connector_hint="Provide implied vol or a surface.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="fx_rates",
            description="FX spot rates needed for payout conversion.",
            capability="fx_rates",
            aliases=("fx", "fx_pair"),
            connector_hint="Provide the domestic/foreign FX rate.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="model_parameters",
            description="Correlation or model parameters for the cross-currency route.",
            capability="model_parameters",
            aliases=("quanto_correlation", "correlation_matrix"),
            connector_hint="Provide correlation / model parameters for the joint route.",
            allowed_provenance=(
                "observed",
                "estimated",
                "calibrated",
                "implied",
                "sampled",
                "synthetic",
            ),
        ),
    )
    return _semantic_contract_from_sections(
        product=product,
        required_inputs=required_inputs,
        candidate_methods=("analytical", "monte_carlo"),
        preferred_method=preferred_method,
        bundle_hints=("quanto_option_contract",),
        universal_checks=(
            "single_underlier_present",
            "expiry_or_exercise_date_present",
            "fx_translation_present",
            "settlement_rule_present",
        ),
        semantic_checks=(
            "quanto_adjustment_applied",
            "fx_conversion_applied_before_settlement",
        ),
        comparison_targets=(normalize_method(preferred_method),),
        reduction_cases=("single_underlier_cross_currency_payoff",),
        target_modules=(
            "trellis.models.resolution.quanto",
            "trellis.models.analytical.quanto",
            "trellis.models.monte_carlo.quanto",
        ),
        primitive_families=("quanto_adjustment_analytical",),
        adapter_obligations=(
            "resolve_underlier_spot",
            "resolve_fx_rate",
            "resolve_forward_and_discount_curves",
            "apply_quanto_adjustment_terms",
        ),
        proving_tasks=(
            "compile_request_to_product_ir",
            "validate_quanto_option_contract",
            "emit_bounded_semantic_blueprint",
        ),
        spec_schema_hints=("quanto_option",),
        description=description,
    )


def make_callable_bond_contract(
    *,
    description: str,
    observation_schedule: tuple[str, ...] | list[str],
    preferred_method: str = "rate_tree",
) -> SemanticContract:
    """Construct a generic callable-bond semantic contract."""
    schedule = _normalize_schedule(observation_schedule)
    if not schedule:
        raise ValueError("Callable bond contract requires a call schedule.")

    product = SemanticProductSemantics(
        semantic_id="callable_bond",
        semantic_version="c2.1",
        instrument_class="callable_bond",
        instrument_aliases=("callable_bond", "callable_debt", "issuer_call_bond"),
        payoff_family="callable_fixed_income",
        timeline=_default_semantic_timeline(
            schedule,
            includes_decision=True,
            settlement_dates=schedule,
            state_update_dates=schedule,
        ),
        underlier_structure="single_issuer_bond",
        payoff_rule="issuer_call_contingent_cashflow",
        settlement_rule="settle_on_call_or_maturity",
        payoff_traits=("callable", "fixed_coupons", "mean_reversion"),
        observables=(
            ObservableSpec(
                observable_id="rate_tree_state",
                observable_type="discount_curve",
                description="Rate-state observation driving callable bond continuation values.",
                source="discount_curve",
                schedule_role="observation_dates",
                availability_phase="observation",
            ),
            ObservableSpec(
                observable_id="coupon_cashflow_schedule",
                observable_type="cashflow_schedule",
                description="Coupon dates and fixed cashflow amounts available on the bond schedule.",
                source="contract_terms",
                schedule_role="determination_dates",
                availability_phase="determination",
            ),
        ),
        state_fields=(
            StateField(
                field_name="call_schedule",
                kind="contract_memory",
                description="Issuer call schedule used for backward induction.",
                source_observables=(),
                tags=("schedule_state", "recombining_safe"),
            ),
            StateField(
                field_name="coupon_schedule",
                kind="contract_memory",
                description="Coupon schedule carried into exercise and settlement evaluation.",
                source_observables=("coupon_cashflow_schedule",),
                tags=("schedule_state", "recombining_safe"),
            ),
        ),
        obligations=(
            ObligationSpec(
                obligation_id="call_or_maturity_settlement",
                settle_date_rule="settle_on_call_or_maturity",
                amount_expression="issuer_call_contingent_cashflow",
                settlement_kind="cash",
                trigger="issuer_call_or_bond_maturity",
                provenance="semantic_contract",
            ),
        ),
        controller_protocol=ControllerProtocol(
            controller_style="issuer_min",
            controller_role="issuer",
            decision_phase="decision",
            schedule_role="decision_dates",
            admissible_actions=("call", "continue"),
            description="Issuer call decision on scheduled call dates.",
        ),
        audit_info=_default_audit_info(),
        implementation_hints=_default_implementation_hints(
            event_machine_source="derived_from_event_transitions",
            primary_schedule_role="decision_dates",
        ),
        exercise_style="issuer_call",
        path_dependence="schedule_dependent",
        schedule_dependence=True,
        state_dependence="schedule_dependent",
        model_family="interest_rate",
        multi_asset=False,
        observation_schedule=schedule,
        observation_basis="call_schedule",
        selection_operator="",
        selection_scope="",
        selection_count=0,
        lock_rule="",
        aggregation_rule="",
        maturity_settlement_rule="settle_on_call_or_maturity",
        constituents=(),
        state_variables=("call_schedule", "coupon_schedule"),
        event_transitions=("evaluate_call_decision", "backward_induction", "settle_on_call_or_maturity"),
        event_machine=_derive_event_machine(
            ("evaluate_call_decision", "backward_induction", "settle_on_call_or_maturity"),
            state_dependence="schedule_dependent",
        ),
    )

    required_inputs = (
        SemanticMarketInputSpec(
            input_id="discount_curve",
            description="Risk-free discount curve for bond cashflows.",
            capability="discount_curve",
            aliases=("discount", "yield_curve"),
            connector_hint="Use the issuer discount curve.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="black_vol_surface",
            description="Volatility / calibration surface for the rate tree.",
            capability="black_vol_surface",
            aliases=("vol_surface", "volatility_surface"),
            connector_hint="Provide a calibration surface for the tree.",
            allowed_provenance=("observed",),
        ),
    )
    # Attach Hull-White calibration contract if available
    _calibration = None
    try:
        from trellis.agent.calibration_contract import hull_white_calibration_contract
        _calibration = hull_white_calibration_contract(fitting="swaption")
    except Exception:
        pass

    return _semantic_contract_from_sections(
        product=product,
        required_inputs=required_inputs,
        candidate_methods=("rate_tree", "pde_solver", "monte_carlo"),
        preferred_method=preferred_method,
        bundle_hints=("callable_bond_contract",),
        universal_checks=(
            "call_schedule_present",
            "settlement_rule_present",
            "issuer_call_logic_present",
        ),
        semantic_checks=(
            "issuer_call_decision_compares_continuation_value",
            "call_decision_applied_on_schedule",
        ),
        comparison_targets=(normalize_method(preferred_method),),
        reduction_cases=("single_issuer_call_schedule",),
        target_modules=("trellis.models.trees.lattice",),
        primitive_families=("exercise_lattice",),
        adapter_obligations=(
            "resolve_call_schedule",
            "calibrate_rate_tree",
            "backward_induction_over_call_dates",
        ),
        proving_tasks=(
            "compile_request_to_product_ir",
            "validate_callable_bond_contract",
            "emit_bounded_semantic_blueprint",
        ),
        spec_schema_hints=("callable_bond",),
        description=description,
        calibration=_calibration,
    )


def make_rate_style_swaption_contract(
    *,
    description: str,
    observation_schedule: tuple[str, ...] | list[str],
    preferred_method: str = "analytical",
    exercise_style: str = "european",
) -> SemanticContract:
    """Construct a generic rate-style swaption semantic contract."""
    schedule = _normalize_schedule(observation_schedule)
    if not schedule:
        raise ValueError("Rate-style swaption contract requires an exercise schedule.")
    normalized_method = normalize_method(preferred_method)
    normalized_exercise = str(exercise_style).strip().lower()
    if normalized_exercise not in {"european", "bermudan"}:
        raise ValueError("Rate-style swaption contract only supports european or bermudan exercise.")
    if normalized_exercise == "european":
        target_modules = ("trellis.models.black",)
        primitive_families = ("analytical_black76",)
    elif normalized_method == "analytical":
        target_modules = ("trellis.models.rate_style_swaption",)
        primitive_families = ("analytical_black76",)
    else:
        target_modules = ("trellis.models.trees.lattice",)
        primitive_families = ("exercise_lattice",)
    spec_schema_hints = ("swaption",) if normalized_exercise == "european" else ("bermudan_swaption",)

    product = SemanticProductSemantics(
        semantic_id="rate_style_swaption",
        semantic_version="c2.1",
        instrument_class="swaption",
        instrument_aliases=("swaption", "rate_style_swaption", "bermudan_swaption"),
        payoff_family="swaption",
        timeline=_default_semantic_timeline(
            schedule,
            includes_decision=True,
            settlement_dates=schedule,
            state_update_dates=schedule,
        ),
        underlier_structure="single_curve_rate_style",
        payoff_rule="swaption_exercise_payoff",
        settlement_rule="cash_settle_at_exercise",
        payoff_traits=("floating_coupons", "vol_surface_dependence"),
        observables=(
            ObservableSpec(
                observable_id="forward_swap_rate",
                observable_type="forward_rate",
                description="Observed or implied forward/par swap rate at exercise.",
                source="forward_curve",
                schedule_role="observation_dates",
                availability_phase="observation",
            ),
            ObservableSpec(
                observable_id="discount_curve_state",
                observable_type="discount_curve",
                description="Discount curve state used to settle the exercised swaption.",
                source="discount_curve",
                schedule_role="observation_dates",
                availability_phase="observation",
            ),
        ),
        state_fields=(
            StateField(
                field_name="exercise_date",
                kind="contract_memory",
                description="Exercise date carried through schedule-dependent swaption semantics.",
                source_observables=(),
                tags=("schedule_state", "recombining_safe"),
            ),
            StateField(
                field_name="swap_rate",
                kind="event_state",
                description="Observed swap rate used for exercise payoff evaluation.",
                source_observables=("forward_swap_rate",),
                tags=("schedule_state", "recombining_safe"),
            ),
        ),
        obligations=(
            ObligationSpec(
                obligation_id="exercise_cash_settlement",
                settle_date_rule="cash_settle_at_exercise",
                amount_expression="swaption_exercise_payoff",
                settlement_kind="cash",
                trigger="holder_exercises_swaption",
                provenance="semantic_contract",
            ),
        ),
        controller_protocol=ControllerProtocol(
            controller_style="holder_max",
            controller_role="holder",
            decision_phase="decision",
            schedule_role="decision_dates",
            admissible_actions=("exercise", "continue"),
            description="Holder exercise decision on the swaption exercise schedule.",
        ),
        audit_info=_default_audit_info(),
        implementation_hints=_default_implementation_hints(
            event_machine_source="derived_from_event_transitions",
            primary_schedule_role="decision_dates",
        ),
        exercise_style=normalized_exercise,
        path_dependence="schedule_dependent",
        schedule_dependence=True,
        state_dependence="schedule_dependent",
        model_family="interest_rate",
        multi_asset=False,
        observation_schedule=schedule,
        observation_basis="exercise_date",
        selection_operator="",
        selection_scope="",
        selection_count=0,
        lock_rule="",
        aggregation_rule="",
        maturity_settlement_rule="cash_settle_at_exercise",
        constituents=(),
        state_variables=("exercise_date", "swap_rate"),
        event_transitions=("price_swaption_at_exercise", "settle_at_exercise"),
        event_machine=_derive_event_machine(
            ("price_swaption_at_exercise", "settle_at_exercise"),
            state_dependence="schedule_dependent",
        ),
    )

    required_inputs = (
        SemanticMarketInputSpec(
            input_id="discount_curve",
            description="Risk-free discount curve for swaption cashflows.",
            capability="discount_curve",
            aliases=("discount", "yield_curve"),
            connector_hint="Use the settlement discount curve.",
            allowed_provenance=("observed",),
        ),
        SemanticMarketInputSpec(
            input_id="forward_curve",
            description="Forward curve for the underlying swap.",
            capability="forward_curve",
            aliases=("forecast_curve", "forward_rate_curve"),
            connector_hint="Provide the underlying swap forward curve.",
            derivable_from=("discount_curve",),
            allowed_provenance=("observed", "derived"),
        ),
        SemanticMarketInputSpec(
            input_id="black_vol_surface",
            description="Implied volatility surface for the swaption.",
            capability="black_vol_surface",
            aliases=("vol_surface", "volatility_surface"),
            connector_hint="Provide the swaption volatility surface.",
            allowed_provenance=("observed",),
        ),
    )
    return _semantic_contract_from_sections(
        product=product,
        required_inputs=required_inputs,
        candidate_methods=("analytical", "rate_tree"),
        preferred_method=preferred_method,
        bundle_hints=("rate_style_swaption_contract",),
        universal_checks=(
            "exercise_schedule_present",
            "swaption_settlement_rule_present",
            "forward_curve_present",
        ),
        semantic_checks=(
            "exercise_payoff_derived_from_schedule",
            "settlement_occurs_at_exercise",
        ),
        comparison_targets=(normalize_method(preferred_method),),
        reduction_cases=("single_curve_rate_style_swaption",),
        target_modules=target_modules,
        primitive_families=primitive_families,
        adapter_obligations=(
            "resolve_forward_and_discount_curves",
            "derive_swaption_exercise_schedule",
            "map_swaption_to_black_route",
        ),
        proving_tasks=(
            "compile_request_to_product_ir",
            "validate_rate_style_swaption_contract",
            "emit_bounded_semantic_blueprint",
        ),
        spec_schema_hints=spec_schema_hints,
        description=description,
    )


def draft_semantic_contract(
    description: str,
    instrument_type: str | None = None,
    *,
    term_sheet=None,
) -> SemanticContract | None:
    """Draft the canonical semantic contract from a natural-language request."""
    text = _combined_request_text(description, instrument_type, term_sheet)
    if not _looks_like_ranked_observation_basket_request(text):
        maybe_contract = _draft_shape_contract(text, description, instrument_type, term_sheet)
        if maybe_contract is None:
            return None
        return maybe_contract

    constituents = _extract_constituents(text, term_sheet)
    observation_schedule = _extract_observation_schedule(text, term_sheet)

    if not observation_schedule:
        raise ValueError(
            "Semantic ranked observation basket request requires an observation schedule."
        )
    if len(constituents) < 2:
        raise ValueError(
            "Semantic ranked observation basket request requires at least two constituents."
        )

    return make_ranked_observation_basket_contract(
        description=description,
        constituents=constituents,
        observation_schedule=observation_schedule,
    )


def _parse_market_input_spec(payload: SemanticMarketInputSpec | dict[str, Any]) -> SemanticMarketInputSpec:
    """Normalize one market-input record."""
    if isinstance(payload, SemanticMarketInputSpec):
        return payload
    capability = payload.get("capability")
    normalized_capability = None
    if capability is not None and str(capability).strip():
        normalized_capability = normalize_capability_name(str(capability))
    return SemanticMarketInputSpec(
        input_id=str(payload["input_id"]).strip(),
        description=str(payload.get("description", "")).strip(),
        capability=normalized_capability,
        aliases=_tuple(payload.get("aliases", ())),
        connector_hint=str(payload.get("connector_hint", "")).strip(),
        derivable_from=_tuple(payload.get("derivable_from", ())),
        allowed_provenance=_tuple(payload.get("allowed_provenance", ("observed",))),
    )


def _parse_product_semantics(
    payload: SemanticProductSemantics | dict[str, Any],
) -> SemanticProductSemantics:
    """Normalize product semantics."""
    if isinstance(payload, SemanticProductSemantics):
        return payload
    semantic_id = str(payload["semantic_id"]).strip()
    instrument_class = str(payload.get("instrument_class", semantic_id)).strip()
    event_transitions = _tuple(payload.get("event_transitions", ()))
    state_dependence = str(payload.get("state_dependence", "terminal_markov")).strip()
    observation_schedule = _tuple(payload.get("observation_schedule", ()))
    return SemanticProductSemantics(
        semantic_id=semantic_id,
        semantic_version=str(payload.get("semantic_version", "c2.0")).strip(),
        instrument_class=instrument_class,
        instrument_aliases=_tuple(payload.get("instrument_aliases", ())),
        payoff_family=str(payload["payoff_family"]).strip(),
        conventions=_parse_convention_env(payload.get("conventions", {})),
        timeline=_parse_semantic_timeline(
            payload.get("timeline", {}),
            fallback_schedule=observation_schedule,
            includes_decision=str(payload.get("exercise_style", "none")).strip() != "none",
        ),
        underlier_structure=str(payload.get("underlier_structure", "")).strip(),
        payoff_rule=str(payload.get("payoff_rule", "")).strip(),
        settlement_rule=str(payload.get("settlement_rule", "")).strip(),
        payoff_traits=_tuple(payload.get("payoff_traits", ())),
        observables=tuple(
            _parse_observable_spec(item)
            for item in payload.get("observables", ())
        ),
        state_fields=tuple(
            _parse_state_field(item)
            for item in payload.get("state_fields", ())
        ) or _legacy_state_fields(payload.get("state_variables", ())),
        obligations=tuple(
            _parse_obligation_spec(item)
            for item in payload.get("obligations", ())
        ),
        controller_protocol=_parse_controller_protocol(payload.get("controller_protocol", {})),
        audit_info=_parse_audit_info(payload.get("audit_info", {})),
        implementation_hints=_parse_implementation_hints(payload.get("implementation_hints", {})),
        exercise_style=str(payload.get("exercise_style", "none")).strip(),
        path_dependence=str(payload.get("path_dependence", "terminal_markov")).strip(),
        schedule_dependence=bool(payload.get("schedule_dependence", False)),
        state_dependence=state_dependence,
        model_family=str(payload.get("model_family", "generic")).strip(),
        multi_asset=bool(payload.get("multi_asset", False)),
        observation_schedule=observation_schedule,
        observation_basis=str(payload.get("observation_basis", "")).strip(),
        selection_operator=str(payload.get("selection_operator", "")).strip(),
        selection_scope=str(payload.get("selection_scope", "")).strip(),
        selection_count=int(payload.get("selection_count", 0)),
        lock_rule=str(payload.get("lock_rule", "")).strip(),
        aggregation_rule=str(payload.get("aggregation_rule", "")).strip(),
        maturity_settlement_rule=str(payload.get("maturity_settlement_rule", "")).strip(),
        constituents=_tuple(payload.get("constituents", ())),
        state_variables=_tuple(payload.get("state_variables", ())),
        event_transitions=event_transitions,
        event_machine=_derive_event_machine(
            event_transitions,
            state_dependence=state_dependence,
            explicit_machine=payload.get("event_machine"),
        ),
    )


def _parse_market_data_contract(
    payload: SemanticMarketDataContract | dict[str, Any],
) -> SemanticMarketDataContract:
    """Normalize market-data contract fields."""
    if isinstance(payload, SemanticMarketDataContract):
        return payload
    return SemanticMarketDataContract(
        required_inputs=tuple(
            _parse_market_input_spec(item)
            for item in payload.get("required_inputs", ())
        ),
        optional_inputs=tuple(
            _parse_market_input_spec(item)
            for item in payload.get("optional_inputs", ())
        ),
        derivable_inputs=_tuple(payload.get("derivable_inputs", ())),
        estimation_policy=_tuple(payload.get("estimation_policy", ())),
        provenance_requirements=_tuple(payload.get("provenance_requirements", ())),
        missing_data_error_policy=_tuple(payload.get("missing_data_error_policy", ())),
    )


def _normalize_methods(values) -> tuple[str, ...]:
    """Normalize method labels while preserving order."""
    normalized: list[str] = []
    for value in values or ():
        method = normalize_method(str(value))
        if method and method not in normalized:
            normalized.append(method)
    return tuple(normalized)


def _parse_method_contract(
    payload: SemanticMethodContract | dict[str, Any],
) -> SemanticMethodContract:
    """Normalize method contract fields."""
    if isinstance(payload, SemanticMethodContract):
        return payload
    preferred = payload.get("preferred_method")
    return SemanticMethodContract(
        candidate_methods=_normalize_methods(payload.get("candidate_methods", ())),
        reference_methods=_normalize_methods(payload.get("reference_methods", ())),
        production_methods=_normalize_methods(payload.get("production_methods", ())),
        unsupported_variants=_tuple(payload.get("unsupported_variants", ())),
        method_limitations=_tuple(payload.get("method_limitations", ())),
        preferred_method=normalize_method(str(preferred)) if preferred else None,
    )


def _parse_validation_contract(
    payload: SemanticValidationContract | dict[str, Any],
) -> SemanticValidationContract:
    """Normalize validation fields."""
    if isinstance(payload, SemanticValidationContract):
        return payload
    return SemanticValidationContract(
        bundle_hints=_tuple(payload.get("bundle_hints", ())),
        universal_checks=_tuple(payload.get("universal_checks", ())),
        semantic_checks=_tuple(payload.get("semantic_checks", ())),
        comparison_targets=_tuple(payload.get("comparison_targets", ())),
        reduction_cases=_tuple(payload.get("reduction_cases", ())),
    )


def _parse_blueprint_hints(
    payload: SemanticBlueprintHints | dict[str, Any],
) -> SemanticBlueprintHints:
    """Normalize blueprint hint fields."""
    if isinstance(payload, SemanticBlueprintHints):
        return payload
    return SemanticBlueprintHints(
        target_modules=_tuple(payload.get("target_modules", ())),
        primitive_families=_tuple(payload.get("primitive_families", ())),
        adapter_obligations=_tuple(payload.get("adapter_obligations", ())),
        proving_tasks=_tuple(payload.get("proving_tasks", ())),
        blocked_by=_tuple(payload.get("blocked_by", ())),
        spec_schema_hints=_tuple(payload.get("spec_schema_hints", ())),
    )


def _parse_convention_env(payload: ConventionEnv | dict[str, Any] | None) -> ConventionEnv:
    """Normalize the compact convention environment."""
    if isinstance(payload, ConventionEnv):
        return payload
    payload = dict(payload or {})
    return ConventionEnv(
        calendar=str(payload.get("calendar", "")).strip(),
        business_day_convention=str(payload.get("business_day_convention", "")).strip(),
        day_count_convention=str(payload.get("day_count_convention", "")).strip(),
        settlement_lag=str(payload.get("settlement_lag", "")).strip(),
        payment_currency=str(payload.get("payment_currency", "")).strip(),
        reporting_currency=str(payload.get("reporting_currency", "")).strip(),
        tags=_tuple(payload.get("tags", ())),
    )


def _parse_semantic_timeline(
    payload: SemanticTimeline | dict[str, Any] | None,
    *,
    fallback_schedule: tuple[str, ...] = (),
    includes_decision: bool = False,
) -> SemanticTimeline:
    """Normalize phase-aware timeline metadata."""
    if isinstance(payload, SemanticTimeline):
        return payload
    payload = dict(payload or {})
    if not payload:
        return _default_semantic_timeline(
            fallback_schedule,
            includes_decision=includes_decision,
            settlement_dates=fallback_schedule[-1:] if fallback_schedule else (),
        )
    return SemanticTimeline(
        phase_order=_normalize_phase_order(payload.get("phase_order", DEFAULT_PHASE_ORDER)),
        anchor_dates=_normalize_schedule(payload.get("anchor_dates", fallback_schedule)),
        event_dates=_normalize_schedule(payload.get("event_dates", fallback_schedule)),
        observation_dates=_normalize_schedule(payload.get("observation_dates", fallback_schedule)),
        decision_dates=_normalize_schedule(payload.get("decision_dates", fallback_schedule if includes_decision else ())),
        determination_dates=_normalize_schedule(payload.get("determination_dates", fallback_schedule)),
        settlement_dates=_normalize_schedule(payload.get("settlement_dates", fallback_schedule[-1:] if fallback_schedule else ())),
        state_update_dates=_normalize_schedule(payload.get("state_update_dates", fallback_schedule)),
    )


def _parse_observable_spec(payload: ObservableSpec | dict[str, Any]) -> ObservableSpec:
    """Normalize one observable spec."""
    if isinstance(payload, ObservableSpec):
        return payload
    return ObservableSpec(
        observable_id=str(payload["observable_id"]).strip(),
        observable_type=str(payload["observable_type"]).strip(),
        description=str(payload.get("description", "")).strip(),
        source=str(payload.get("source", "")).strip(),
        schedule_role=str(payload.get("schedule_role", "")).strip(),
        availability_phase=str(payload.get("availability_phase", "observation")).strip().lower(),
        dependencies=_tuple(payload.get("dependencies", ())),
    )


def _parse_state_field(payload: StateField | dict[str, Any]) -> StateField:
    """Normalize one state-field record."""
    if isinstance(payload, StateField):
        return payload
    return StateField(
        field_name=str(payload["field_name"]).strip(),
        kind=str(payload.get("kind", "contract_memory")).strip(),
        description=str(payload.get("description", "")).strip(),
        source_observables=_tuple(payload.get("source_observables", ())),
        tags=_tuple(payload.get("tags", ())),
    )


def _parse_obligation_spec(payload: ObligationSpec | dict[str, Any]) -> ObligationSpec:
    """Normalize one obligation spec."""
    if isinstance(payload, ObligationSpec):
        return payload
    return ObligationSpec(
        obligation_id=str(payload["obligation_id"]).strip(),
        settle_date_rule=str(payload.get("settle_date_rule", "")).strip(),
        amount_expression=str(payload.get("amount_expression", "")).strip(),
        currency=str(payload.get("currency", "")).strip(),
        settlement_kind=str(payload.get("settlement_kind", "cash")).strip(),
        trigger=str(payload.get("trigger", "")).strip(),
        provenance=str(payload.get("provenance", "")).strip(),
    )


def _parse_controller_protocol(
    payload: ControllerProtocol | dict[str, Any] | None,
) -> ControllerProtocol:
    """Normalize controller protocol metadata."""
    if isinstance(payload, ControllerProtocol):
        return payload
    payload = dict(payload or {})
    return ControllerProtocol(
        controller_style=str(payload.get("controller_style", "identity")).strip(),
        controller_role=str(payload.get("controller_role", "none")).strip(),
        decision_phase=str(payload.get("decision_phase", "decision")).strip().lower(),
        schedule_role=str(payload.get("schedule_role", "")).strip(),
        admissible_actions=_tuple(payload.get("admissible_actions", ())),
        description=str(payload.get("description", "")).strip(),
    )


def _parse_audit_info(payload: SemanticAuditInfo | dict[str, Any] | None) -> SemanticAuditInfo:
    """Normalize semantic audit metadata."""
    if isinstance(payload, SemanticAuditInfo):
        return payload
    payload = dict(payload or {})
    return SemanticAuditInfo(
        semantic_origin=str(payload.get("semantic_origin", "")).strip(),
        provenance=_tuple(payload.get("provenance", ())),
        notes=_tuple(payload.get("notes", ())),
        legacy_mirrors=_tuple(payload.get("legacy_mirrors", ())),
    )


def _parse_implementation_hints(
    payload: ImplementationHints | dict[str, Any] | None,
) -> ImplementationHints:
    """Normalize implementation-hint metadata."""
    if isinstance(payload, ImplementationHints):
        return payload
    payload = dict(payload or {})
    return ImplementationHints(
        preserve_route_behavior=bool(payload.get("preserve_route_behavior", True)),
        event_machine_source=str(payload.get("event_machine_source", "")).strip(),
        primary_schedule_role=str(payload.get("primary_schedule_role", "")).strip(),
        notes=_tuple(payload.get("notes", ())),
    )


def _normalize_schedule(values) -> tuple[str, ...]:
    """Normalize observation schedule values to ISO-date strings."""
    normalized: list[str] = []
    for value in values or ():
        if value is None:
            continue
        if isinstance(value, date):
            text = value.isoformat()
        else:
            text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _normalize_phase_order(values) -> tuple[str, ...]:
    """Normalize phase-order metadata."""
    normalized: list[str] = []
    for value in values or DEFAULT_PHASE_ORDER:
        text = str(value).strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized) or DEFAULT_PHASE_ORDER


def _default_semantic_timeline(
    schedule: tuple[str, ...] | list[str],
    *,
    includes_decision: bool = False,
    settlement_dates: tuple[str, ...] | list[str] = (),
    state_update_dates: tuple[str, ...] | list[str] = (),
) -> SemanticTimeline:
    """Build the default phase-aware timeline used in tranche 1."""
    normalized_schedule = _normalize_schedule(schedule)
    normalized_settlement = _normalize_schedule(
        settlement_dates or (normalized_schedule[-1:],)
    ) if normalized_schedule else ()
    normalized_state_updates = _normalize_schedule(state_update_dates)
    return SemanticTimeline(
        phase_order=DEFAULT_PHASE_ORDER,
        anchor_dates=normalized_schedule,
        event_dates=normalized_schedule,
        observation_dates=normalized_schedule,
        decision_dates=normalized_schedule if includes_decision else (),
        determination_dates=normalized_schedule,
        settlement_dates=normalized_settlement,
        state_update_dates=normalized_state_updates,
    )


def _legacy_state_fields(values) -> tuple[StateField, ...]:
    """Derive typed state fields from legacy state_variables."""
    fields: list[StateField] = []
    for name in _tuple(values):
        lower = name.lower()
        if any(token in lower for token in ("spot", "price", "rate")):
            kind = "event_state"
            tags = ("recombining_safe",)
        else:
            kind = "contract_memory"
            tags = ("schedule_state",) if "schedule" in lower else ()
        fields.append(
            StateField(
                field_name=name,
                kind=kind,
                description=f"Legacy-derived typed state field for `{name}`.",
                source_observables=(),
                tags=tags,
            )
        )
    return tuple(fields)


def _derive_event_machine(
    event_transitions: tuple[str, ...] | list[str],
    *,
    state_dependence: str,
    explicit_machine: object | None = None,
) -> object | None:
    """Derive a typed event machine from legacy transitions when needed."""
    if explicit_machine is not None:
        return explicit_machine
    transitions = _tuple(event_transitions)
    if not transitions:
        return None
    try:
        from trellis.agent.event_machine import event_transitions_to_machine
        return event_transitions_to_machine(
            transitions,
            state_dependence=state_dependence,
        )
    except Exception:
        return None


def _default_audit_info() -> SemanticAuditInfo:
    """Return the default semantic audit info for checked-in factories."""
    return SemanticAuditInfo(
        semantic_origin="checked_in_factory",
        provenance=("checked_in_factory", "deterministic_semantic_contract"),
        notes=("legacy semantic string fields are retained as mirrors in tranche 1",),
        legacy_mirrors=("settlement_rule", "event_transitions", "state_variables"),
    )


def _default_implementation_hints(
    *,
    event_machine_source: str,
    primary_schedule_role: str,
) -> ImplementationHints:
    """Return default implementation hints for checked-in semantic factories."""
    return ImplementationHints(
        preserve_route_behavior=True,
        event_machine_source=event_machine_source,
        primary_schedule_role=primary_schedule_role,
        notes=("typed semantic surface added without route-helper behavior changes",),
    )


def _combined_request_text(
    description: str,
    instrument_type: str | None,
    term_sheet,
) -> str:
    """Assemble the request text used for semantic cue detection."""
    parts = [description or ""]
    if instrument_type:
        parts.append(str(instrument_type))
    raw_description = getattr(term_sheet, "raw_description", "")
    if raw_description:
        parts.append(raw_description)
    return "\n".join(part for part in parts if part)


def _looks_like_ranked_observation_basket_request(text: str) -> bool:
    """Return whether the request appears to describe the canonical basket slice."""
    lower = text.lower()
    return any(
        cue in lower
        for cue in (
            "himalaya",
            "ranked observation",
            "ranked selection",
            "best remaining",
            "remaining constituents",
            "remove selected",
            "lock selected",
            "basket path payoff",
        )
    )


def _extract_constituents(text: str, term_sheet) -> tuple[str, ...]:
    """Extract basket constituents from structured fields or request text."""
    parameters = getattr(term_sheet, "parameters", {}) or {}
    for key in ("constituents", "underliers", "basket_names"):
        value = parameters.get(key)
        if value:
            if isinstance(value, str):
                return _parse_name_list(value)
            return _tuple(value)

    stopwords = {
        "A",
        "AND",
        "AT",
        "BEST",
        "BASKET",
        "BESTOF",
        "CONSTITUENTS",
        "DATE",
        "DATES",
        "HIMALAYA",
        "LOCK",
        "MATURITY",
        "OBSERVATION",
        "OBSERVATIONS",
        "PAYOFF",
        "PERFORMER",
        "RANKED",
        "REMOVE",
        "RETURN",
        "SELECT",
        "SELECTED",
        "SELECTION",
        "SIMPLE",
        "THE",
        "WITH",
    }
    tokens = re.findall(r"\b[A-Z][A-Z0-9_.-]{1,}\b", text)
    constituents = []
    for token in tokens:
        upper = token.upper()
        if upper in stopwords:
            continue
        if upper not in constituents:
            constituents.append(upper)
    return tuple(constituents)


def _parse_name_list(value: str) -> tuple[str, ...]:
    """Parse a comma-separated or slash-separated name list."""
    tokens = re.split(r"[,;/]|(?:\band\b)", value)
    cleaned: list[str] = []
    for token in tokens:
        text = token.strip().strip(".")
        if text and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned)


def _extract_observation_schedule(text: str, term_sheet) -> tuple[str, ...]:
    """Extract the ordered observation schedule from structured fields or text."""
    parameters = getattr(term_sheet, "parameters", {}) or {}
    for key in ("observation_schedule", "observation_dates"):
        value = parameters.get(key)
        if value:
            if isinstance(value, str):
                return _parse_name_list(value)
            return _normalize_schedule(value)

    schedule = []
    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        value = match.group(0)
        if value not in schedule:
            schedule.append(value)
    return tuple(schedule)


def _looks_like_quanto_option_request(text: str, instrument_type: str | None) -> bool:
    """Return whether the request appears to describe a quanto option."""
    lower = text.lower()
    normalized_instrument = (instrument_type or "").strip().lower().replace(" ", "_")
    if normalized_instrument == "quanto_option":
        return True
    return any(
        cue in lower
        for cue in (
            "quanto option",
            "quanto",
            "cross currency option",
            "cross-currency option",
            "fx option",
            "fx-linked option",
        )
    )


def _looks_like_callable_bond_request(text: str, instrument_type: str | None) -> bool:
    """Return whether the request appears to describe a callable bond."""
    lower = text.lower()
    normalized_instrument = (instrument_type or "").strip().lower().replace(" ", "_")
    if normalized_instrument == "callable_bond":
        return True
    return any(
        cue in lower
        for cue in (
            "callable bond",
            "issuer call",
            "call schedule",
            "call dates",
            "callable debt",
        )
    )


def _looks_like_vanilla_option_request(text: str, instrument_type: str | None) -> bool:
    """Return whether the request appears to describe a vanilla option."""
    lower = text.lower()
    normalized_instrument = (instrument_type or "").strip().lower().replace(" ", "_")
    if normalized_instrument in {"european_option", "american_option"}:
        return True
    if "callable bond" in lower or "callable debt" in lower:
        return False
    return any(
        cue in lower
        for cue in (
            "vanilla option",
            "european option",
            "european call",
            "european put",
            "call on",
            "put on",
            "option on",
        )
    )


def _looks_like_rate_style_swaption_request(text: str, instrument_type: str | None) -> bool:
    """Return whether the request appears to describe a simple rate-style swaption."""
    lower = text.lower()
    normalized_instrument = (instrument_type or "").strip().lower().replace(" ", "_")
    if normalized_instrument in {"swaption", "bermudan_swaption"}:
        return True
    return any(
        cue in lower
        for cue in (
            "swaption",
            "fixed-for-floating",
            "forward swap",
            "swap rate",
            "swap exercise",
        )
    )


def _draft_shape_contract(
    text: str,
    description: str,
    instrument_type: str | None,
    term_sheet,
) -> SemanticContract | None:
    """Draft one generic shape-driven semantic contract, if recognized."""
    if _looks_like_quanto_option_request(text, instrument_type):
        underliers = _extract_primary_underlier(text, term_sheet)
        observation_schedule = _split_supported_dates(
            text,
            term_sheet,
            parameter_keys=("expiry_date", "expiry", "exercise_date"),
        )
        if not underliers:
            raise ValueError(
                "Semantic quanto option request requires an identifiable underlier."
            )
        if not observation_schedule:
            raise ValueError(
                "Semantic quanto option request requires an expiry or exercise schedule."
            )
        return make_quanto_option_contract(
            description=description,
            underliers=underliers,
            observation_schedule=observation_schedule,
        )

    if _looks_like_callable_bond_request(text, instrument_type):
        observation_schedule = _split_supported_dates(
            text,
            term_sheet,
            parameter_keys=("call_schedule", "call_dates", "observation_schedule", "observation_dates"),
        )
        if not observation_schedule:
            raise ValueError(
                "Semantic callable bond request requires a call or exercise schedule."
            )
        return make_callable_bond_contract(
            description=description,
            observation_schedule=observation_schedule,
        )

    if _looks_like_vanilla_option_request(text, instrument_type):
        underliers = _extract_primary_underlier(text, term_sheet)
        observation_schedule = _split_supported_dates(
            text,
            term_sheet,
            parameter_keys=("expiry_date", "expiry", "exercise_date", "observation_schedule", "observation_dates"),
        )
        if not underliers:
            raise ValueError(
                "Semantic vanilla option request requires an identifiable underlier."
            )
        if not observation_schedule:
            raise ValueError(
                "Semantic vanilla option request requires an expiry or exercise schedule."
            )
        return make_vanilla_option_contract(
            description=description,
            underliers=underliers,
            observation_schedule=observation_schedule,
        )

    if _looks_like_rate_style_swaption_request(text, instrument_type):
        observation_schedule = _split_supported_dates(
            text,
            term_sheet,
            parameter_keys=("expiry_date", "expiry", "exercise_date", "observation_schedule", "observation_dates"),
        )
        if not observation_schedule:
            raise ValueError(
                "Semantic rate-style swaption request requires an exercise schedule."
            )
        normalized_instrument = str(instrument_type or "").strip().lower()
        return make_rate_style_swaption_contract(
            description=description,
            observation_schedule=observation_schedule,
            preferred_method="rate_tree" if normalized_instrument == "bermudan_swaption" else "analytical",
            exercise_style="bermudan" if normalized_instrument == "bermudan_swaption" else "european",
        )

    return None
