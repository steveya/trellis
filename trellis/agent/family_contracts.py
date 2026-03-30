"""Typed family-contract schema for known product families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from trellis.agent.knowledge.methods import normalize_method
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


@dataclass(frozen=True)
class MarketInputSpec:
    """One named market input in a family contract."""

    input_id: str
    description: str = ""
    capability: str | None = None
    aliases: tuple[str, ...] = ()
    connector_hint: str = ""
    derivable_from: tuple[str, ...] = ()
    allowed_provenance: tuple[str, ...] = ("observed",)


@dataclass(frozen=True)
class ProductSemantics:
    """Typed product semantics for one family."""

    family_id: str
    family_version: str
    instrument: str
    instrument_aliases: tuple[str, ...]
    payoff_family: str
    payoff_traits: tuple[str, ...] = ()
    exercise_style: str = "none"
    path_dependence: str = "terminal_markov"
    schedule_dependence: bool = False
    state_dependence: str = "terminal_markov"
    model_family: str = "generic"
    schedule_semantics: tuple[str, ...] = ()
    state_variables: tuple[str, ...] = ()
    event_transitions: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketDataContract:
    """Market-data requirements and provenance policy."""

    required_inputs: tuple[MarketInputSpec, ...]
    optional_inputs: tuple[MarketInputSpec, ...] = ()
    derivable_inputs: tuple[str, ...] = ()
    estimation_policy: tuple[str, ...] = ()
    provenance_requirements: tuple[str, ...] = ()
    missing_data_error_policy: tuple[str, ...] = ()


@dataclass(frozen=True)
class MethodContract:
    """Candidate and supported method families for a product family."""

    candidate_methods: tuple[str, ...]
    reference_methods: tuple[str, ...] = ()
    production_methods: tuple[str, ...] = ()
    unsupported_variants: tuple[str, ...] = ()
    method_limitations: tuple[str, ...] = ()
    preferred_method: str | None = None


@dataclass(frozen=True)
class SensitivityContract:
    """Family-level sensitivity-support claims."""

    support_level: str
    supported_measures: tuple[str, ...] = ()
    stability_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationContract:
    """Validation and cross-check expectations for a family."""

    bundle_hints: tuple[str, ...] = ()
    universal_checks: tuple[str, ...] = ()
    family_checks: tuple[str, ...] = ()
    comparison_targets: tuple[str, ...] = ()
    reduction_cases: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlueprintHints:
    """Compiler hints for later route and module selection."""

    target_modules: tuple[str, ...] = ()
    primitive_families: tuple[str, ...] = ()
    adapter_obligations: tuple[str, ...] = ()
    proving_tasks: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    spec_schema_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class FamilyContract:
    """Top-level typed family contract."""

    product: ProductSemantics
    market_data: MarketDataContract
    methods: MethodContract
    sensitivities: SensitivityContract
    validation: ValidationContract
    blueprint: BlueprintHints
    description: str = ""

    @property
    def family_id(self) -> str:
        """Return the normalized family identifier."""
        return self.product.family_id


def parse_family_contract(spec: FamilyContract | dict[str, Any] | str) -> FamilyContract:
    """Parse a family contract from an object, dict, or YAML string."""
    if isinstance(spec, FamilyContract):
        return spec
    if isinstance(spec, str):
        payload = yaml.safe_load(spec) or {}
    else:
        payload = dict(spec)

    return FamilyContract(
        product=_parse_product_semantics(payload["product"]),
        market_data=_parse_market_data_contract(payload["market_data"]),
        methods=_parse_method_contract(payload["methods"]),
        sensitivities=_parse_sensitivity_contract(payload["sensitivities"]),
        validation=_parse_validation_contract(payload.get("validation", {})),
        blueprint=_parse_blueprint_hints(payload.get("blueprint", {})),
        description=str(payload.get("description", "")).strip(),
    )


def _parse_market_input_spec(payload: MarketInputSpec | dict[str, Any]) -> MarketInputSpec:
    """Normalize one market-input record."""
    if isinstance(payload, MarketInputSpec):
        return payload
    capability = payload.get("capability")
    normalized_capability = None
    if capability is not None and str(capability).strip():
        normalized_capability = normalize_capability_name(str(capability))
    return MarketInputSpec(
        input_id=str(payload["input_id"]).strip(),
        description=str(payload.get("description", "")).strip(),
        capability=normalized_capability,
        aliases=_tuple(payload.get("aliases", ())),
        connector_hint=str(payload.get("connector_hint", "")).strip(),
        derivable_from=_tuple(payload.get("derivable_from", ())),
        allowed_provenance=_tuple(payload.get("allowed_provenance", ("observed",))),
    )


def _parse_product_semantics(payload: ProductSemantics | dict[str, Any]) -> ProductSemantics:
    """Normalize product semantics."""
    if isinstance(payload, ProductSemantics):
        return payload
    family_id = str(payload["family_id"]).strip()
    instrument = str(payload.get("instrument", family_id)).strip()
    return ProductSemantics(
        family_id=family_id,
        family_version=str(payload.get("family_version", "c0")).strip(),
        instrument=instrument,
        instrument_aliases=_tuple(payload.get("instrument_aliases", ())),
        payoff_family=str(payload["payoff_family"]).strip(),
        payoff_traits=_tuple(payload.get("payoff_traits", ())),
        exercise_style=str(payload.get("exercise_style", "none")).strip(),
        path_dependence=str(payload.get("path_dependence", "terminal_markov")).strip(),
        schedule_dependence=bool(payload.get("schedule_dependence", False)),
        state_dependence=str(payload.get("state_dependence", "terminal_markov")).strip(),
        model_family=str(payload.get("model_family", "generic")).strip(),
        schedule_semantics=_tuple(payload.get("schedule_semantics", ())),
        state_variables=_tuple(payload.get("state_variables", ())),
        event_transitions=_tuple(payload.get("event_transitions", ())),
    )


def _parse_market_data_contract(
    payload: MarketDataContract | dict[str, Any],
) -> MarketDataContract:
    """Normalize market-data contract fields."""
    if isinstance(payload, MarketDataContract):
        return payload
    return MarketDataContract(
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


def _parse_method_contract(payload: MethodContract | dict[str, Any]) -> MethodContract:
    """Normalize method contract fields."""
    if isinstance(payload, MethodContract):
        return payload
    preferred = payload.get("preferred_method")
    return MethodContract(
        candidate_methods=_normalize_methods(payload.get("candidate_methods", ())),
        reference_methods=_normalize_methods(payload.get("reference_methods", ())),
        production_methods=_normalize_methods(payload.get("production_methods", ())),
        unsupported_variants=_tuple(payload.get("unsupported_variants", ())),
        method_limitations=_tuple(payload.get("method_limitations", ())),
        preferred_method=normalize_method(str(preferred)) if preferred else None,
    )


def _parse_sensitivity_contract(
    payload: SensitivityContract | dict[str, Any],
) -> SensitivityContract:
    """Normalize sensitivity support fields."""
    if isinstance(payload, SensitivityContract):
        return payload
    return SensitivityContract(
        support_level=str(payload.get("support_level", "unsupported")).strip(),
        supported_measures=_tuple(payload.get("supported_measures", ())),
        stability_notes=_tuple(payload.get("stability_notes", ())),
    )


def _parse_validation_contract(
    payload: ValidationContract | dict[str, Any],
) -> ValidationContract:
    """Normalize validation contract fields."""
    if isinstance(payload, ValidationContract):
        return payload
    return ValidationContract(
        bundle_hints=_tuple(payload.get("bundle_hints", ())),
        universal_checks=_tuple(payload.get("universal_checks", ())),
        family_checks=_tuple(payload.get("family_checks", ())),
        comparison_targets=_tuple(payload.get("comparison_targets", ())),
        reduction_cases=_tuple(payload.get("reduction_cases", ())),
    )


def _parse_blueprint_hints(
    payload: BlueprintHints | dict[str, Any],
) -> BlueprintHints:
    """Normalize blueprint hints."""
    if isinstance(payload, BlueprintHints):
        return payload
    return BlueprintHints(
        target_modules=_tuple(payload.get("target_modules", ())),
        primitive_families=_tuple(payload.get("primitive_families", ())),
        adapter_obligations=_tuple(payload.get("adapter_obligations", ())),
        proving_tasks=_tuple(payload.get("proving_tasks", ())),
        blocked_by=_tuple(payload.get("blocked_by", ())),
        spec_schema_hints=_tuple(payload.get("spec_schema_hints", ())),
    )
