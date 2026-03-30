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
    underlier_structure: str = ""
    payoff_rule: str = ""
    settlement_rule: str = ""
    payoff_traits: tuple[str, ...] = ()

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
        underlier_structure="single_underlier",
        payoff_rule="vanilla_option_payoff",
        settlement_rule="cash_settle_at_expiry",
        payoff_traits=("discounting", "vol_surface_dependence"),
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
        underlier_structure="cross_currency_single_underlier",
        payoff_rule="quanto_adjusted_vanilla_payoff",
        settlement_rule="cash_settle_at_expiry_after_fx_conversion",
        payoff_traits=("discounting", "vol_surface_dependence", "fx_translation"),
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
        underlier_structure="single_issuer_bond",
        payoff_rule="issuer_call_contingent_cashflow",
        settlement_rule="settle_on_call_or_maturity",
        payoff_traits=("callable", "fixed_coupons", "mean_reversion"),
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
    )


def make_rate_style_swaption_contract(
    *,
    description: str,
    observation_schedule: tuple[str, ...] | list[str],
    preferred_method: str = "analytical",
) -> SemanticContract:
    """Construct a generic rate-style swaption semantic contract."""
    schedule = _normalize_schedule(observation_schedule)
    if not schedule:
        raise ValueError("Rate-style swaption contract requires an exercise schedule.")

    product = SemanticProductSemantics(
        semantic_id="rate_style_swaption",
        semantic_version="c2.1",
        instrument_class="swaption",
        instrument_aliases=("swaption", "rate_style_swaption", "bermudan_swaption"),
        payoff_family="swaption",
        underlier_structure="single_curve_rate_style",
        payoff_rule="swaption_exercise_payoff",
        settlement_rule="cash_settle_at_exercise",
        payoff_traits=("floating_coupons", "vol_surface_dependence"),
        exercise_style="european",
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
        target_modules=("trellis.models.black",),
        primitive_families=("analytical_black76",),
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
        spec_schema_hints=("swaption",),
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
    return SemanticProductSemantics(
        semantic_id=semantic_id,
        semantic_version=str(payload.get("semantic_version", "c2.0")).strip(),
        instrument_class=instrument_class,
        instrument_aliases=_tuple(payload.get("instrument_aliases", ())),
        payoff_family=str(payload["payoff_family"]).strip(),
        underlier_structure=str(payload.get("underlier_structure", "")).strip(),
        payoff_rule=str(payload.get("payoff_rule", "")).strip(),
        settlement_rule=str(payload.get("settlement_rule", "")).strip(),
        payoff_traits=_tuple(payload.get("payoff_traits", ())),
        exercise_style=str(payload.get("exercise_style", "none")).strip(),
        path_dependence=str(payload.get("path_dependence", "terminal_markov")).strip(),
        schedule_dependence=bool(payload.get("schedule_dependence", False)),
        state_dependence=str(payload.get("state_dependence", "terminal_markov")).strip(),
        model_family=str(payload.get("model_family", "generic")).strip(),
        multi_asset=bool(payload.get("multi_asset", False)),
        observation_schedule=_tuple(payload.get("observation_schedule", ())),
        observation_basis=str(payload.get("observation_basis", "")).strip(),
        selection_operator=str(payload.get("selection_operator", "")).strip(),
        selection_scope=str(payload.get("selection_scope", "")).strip(),
        selection_count=int(payload.get("selection_count", 0)),
        lock_rule=str(payload.get("lock_rule", "")).strip(),
        aggregation_rule=str(payload.get("aggregation_rule", "")).strip(),
        maturity_settlement_rule=str(payload.get("maturity_settlement_rule", "")).strip(),
        constituents=_tuple(payload.get("constituents", ())),
        state_variables=_tuple(payload.get("state_variables", ())),
        event_transitions=_tuple(payload.get("event_transitions", ())),
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
        return make_rate_style_swaption_contract(
            description=description,
            observation_schedule=observation_schedule,
        )

    return None
