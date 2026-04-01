"""Semantic concept taxonomy and resolution policy for the internal DSL."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any


@dataclass(frozen=True)
class SemanticConceptDefinition:
    """Canonical semantic concept metadata.

    The registry keeps semantic concepts separate from product-name wrappers so
    the agent can resolve requests against an internal vocabulary first and only
    then map to compatibility names.
    """

    semantic_id: str
    semantic_version: str
    scope: str
    description: str
    concept_role: str = "product_contract"
    aliases: tuple[str, ...] = ()
    compatibility_wrappers: tuple[str, ...] = ()
    deprecated_wrappers: tuple[str, ...] = ()
    required_contract_fields: tuple[str, ...] = ()
    allowed_contract_fields: tuple[str, ...] = ()
    required_primitives: tuple[str, ...] = ()
    route_helpers: tuple[str, ...] = ()
    required_market_inputs: tuple[str, ...] = ()
    extension_policy: tuple[str, ...] = (
        "reuse_existing_concept",
        "new_attribute",
        "thin_compatibility_wrapper",
        "introduce_new_concept",
    )
    supersedes: tuple[str, ...] = ()
    status: str = "active"
    route_family: str = ""
    example_requests: tuple[str, ...] = ()
    cue_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticConceptResolution:
    """Deterministic resolution of one request against the concept registry."""

    request_text: str
    instrument_type: str = ""
    concept_id: str = ""
    concept_version: str = ""
    concept_status: str = ""
    concept_role: str = ""
    resolution_kind: str = "clarification"
    matched_alias: str = ""
    matched_wrapper: str = ""
    candidate_concepts: tuple[str, ...] = ()
    conflicting_concepts: tuple[str, ...] = ()
    superseded_concepts: tuple[str, ...] = ()
    policy_notes: tuple[str, ...] = ()
    summary: str = ""
    confidence: float = 0.0
    gap_ratio: float = 0.0


def _concept(
    *,
    semantic_id: str,
    semantic_version: str,
    scope: str,
    description: str,
    aliases: tuple[str, ...],
    concept_role: str = "product_contract",
    compatibility_wrappers: tuple[str, ...] = (),
    deprecated_wrappers: tuple[str, ...] = (),
    required_contract_fields: tuple[str, ...] = (),
    allowed_contract_fields: tuple[str, ...] = (),
    required_primitives: tuple[str, ...] = (),
    route_helpers: tuple[str, ...] = (),
    required_market_inputs: tuple[str, ...] = (),
    extension_policy: tuple[str, ...] = (
        "reuse_existing_concept",
        "new_attribute",
        "thin_compatibility_wrapper",
        "introduce_new_concept",
    ),
    supersedes: tuple[str, ...] = (),
    status: str = "active",
    route_family: str = "",
    example_requests: tuple[str, ...] = (),
    cue_phrases: tuple[str, ...] = (),
) -> SemanticConceptDefinition:
    return SemanticConceptDefinition(
        semantic_id=semantic_id,
        semantic_version=semantic_version,
        scope=scope,
        description=description,
        concept_role=concept_role,
        aliases=aliases,
        compatibility_wrappers=compatibility_wrappers,
        deprecated_wrappers=deprecated_wrappers,
        required_contract_fields=required_contract_fields,
        allowed_contract_fields=allowed_contract_fields,
        required_primitives=required_primitives,
        route_helpers=route_helpers,
        required_market_inputs=required_market_inputs,
        extension_policy=extension_policy,
        supersedes=supersedes,
        status=status,
        route_family=route_family,
        example_requests=example_requests,
        cue_phrases=cue_phrases,
    )


SEMANTIC_CONCEPT_REGISTRY: tuple[SemanticConceptDefinition, ...] = (
    _concept(
        semantic_id="ranked_observation_basket",
        semantic_version="c2.0",
        scope="ranked-observation basket contracts and their thin compatibility wrappers",
        description=(
            "Multi-asset ranked-observation basket with remaining-constituent selection, "
            "locked returns, and maturity aggregation."
        ),
        concept_role="product_contract",
        aliases=(
            "ranked_observation_basket",
            "basket_path_payoff",
            "ranked_selection_basket",
        ),
        compatibility_wrappers=("basket_option", "basket_path_payoff"),
        deprecated_wrappers=("himalaya_option",),
        required_contract_fields=(
            "underlier_structure",
            "payoff_rule",
            "settlement_rule",
            "observation_schedule",
            "selection_scope",
            "selection_operator",
            "selection_count",
            "lock_rule",
            "aggregation_rule",
        ),
        allowed_contract_fields=(
            "instrument_class",
            "underlier_structure",
            "payoff_family",
            "payoff_rule",
            "settlement_rule",
            "payoff_traits",
            "exercise_style",
            "path_dependence",
            "schedule_dependence",
            "state_dependence",
            "model_family",
            "multi_asset",
            "observation_schedule",
            "observation_basis",
            "selection_operator",
            "selection_scope",
            "selection_count",
            "lock_rule",
            "aggregation_rule",
            "maturity_settlement_rule",
            "constituents",
            "state_variables",
            "event_transitions",
        ),
        required_primitives=("correlated_basket_monte_carlo",),
        route_helpers=(
            "trellis.models.resolution.basket_semantics",
            "trellis.models.monte_carlo.semantic_basket",
        ),
        required_market_inputs=(
            "discount_curve",
            "underlier_spots",
            "black_vol_surface",
            "correlation_matrix",
        ),
        route_family="monte_carlo",
        example_requests=(
            "Himalaya-style ranked observation basket on AAPL, MSFT, and NVDA",
        ),
        cue_phrases=(
            "ranked observation",
            "ranked selection",
            "remaining constituents",
            "best remaining",
            "remove selected",
            "lock selected",
            "basket path payoff",
        ),
    ),
    _concept(
        semantic_id="vanilla_option",
        semantic_version="c2.1",
        scope="single-underlier European-style option contracts",
        description="Single-underlier European-style option routed through the shared analytical basis.",
        concept_role="product_contract",
        aliases=(
            "vanilla_option",
            "option",
            "european_option",
        ),
        compatibility_wrappers=("european_option",),
        required_contract_fields=(
            "underlier_structure",
            "payoff_rule",
            "settlement_rule",
            "observation_schedule",
        ),
        allowed_contract_fields=(
            "instrument_class",
            "underlier_structure",
            "payoff_family",
            "payoff_rule",
            "settlement_rule",
            "payoff_traits",
            "exercise_style",
            "path_dependence",
            "schedule_dependence",
            "state_dependence",
            "model_family",
            "multi_asset",
            "observation_schedule",
            "observation_basis",
            "constituents",
            "state_variables",
            "event_transitions",
        ),
        required_primitives=("analytical_black76",),
        route_helpers=("trellis.models.black",),
        required_market_inputs=(
            "discount_curve",
            "underlier_spot",
            "black_vol_surface",
        ),
        route_family="analytical",
        example_requests=("European call on AAPL with strike 120 and expiry 2025-11-15",),
        cue_phrases=(
            "european call",
            "european put",
            "vanilla option",
            "option on",
            "strike",
            "expiry",
        ),
    ),
    _concept(
        semantic_id="quanto_option",
        semantic_version="c2.1",
        scope="cross-currency single-underlier option contracts",
        description="Single-underlier cross-currency option with FX translation and quanto adjustment.",
        concept_role="product_contract",
        aliases=(
            "quanto_option",
            "quanto",
            "fx_option",
            "cross_currency_option",
        ),
        compatibility_wrappers=("fx_option",),
        required_contract_fields=(
            "underlier_structure",
            "payoff_rule",
            "settlement_rule",
            "observation_schedule",
        ),
        allowed_contract_fields=(
            "instrument_class",
            "underlier_structure",
            "payoff_family",
            "payoff_rule",
            "settlement_rule",
            "payoff_traits",
            "exercise_style",
            "path_dependence",
            "schedule_dependence",
            "state_dependence",
            "model_family",
            "multi_asset",
            "observation_schedule",
            "observation_basis",
            "constituents",
            "state_variables",
            "event_transitions",
        ),
        required_primitives=("quanto_adjustment_analytical",),
        route_helpers=(
            "trellis.models.resolution.quanto",
            "trellis.models.analytical.quanto",
            "trellis.models.monte_carlo.quanto",
        ),
        required_market_inputs=(
            "discount_curve",
            "forward_curve",
            "underlier_spot",
            "black_vol_surface",
            "fx_rates",
            "model_parameters",
        ),
        route_family="analytical",
        example_requests=(
            "Quanto option on SAP in USD with EUR underlier currency and expiry 2025-11-15",
        ),
        cue_phrases=(
            "quanto option",
            "cross currency option",
            "cross-currency option",
            "fx option",
            "fx-linked option",
        ),
    ),
    _concept(
        semantic_id="callable_bond",
        semantic_version="c2.1",
        scope="issuer-call fixed-income contracts with schedule-driven exercise",
        description="Callable fixed-income concept with issuer call dates and backward induction.",
        concept_role="product_contract",
        aliases=(
            "callable_bond",
            "callable_debt",
            "issuer_call_bond",
        ),
        compatibility_wrappers=("callable_debt",),
        required_contract_fields=(
            "underlier_structure",
            "payoff_rule",
            "settlement_rule",
            "observation_schedule",
        ),
        allowed_contract_fields=(
            "instrument_class",
            "underlier_structure",
            "payoff_family",
            "payoff_rule",
            "settlement_rule",
            "payoff_traits",
            "exercise_style",
            "path_dependence",
            "schedule_dependence",
            "state_dependence",
            "model_family",
            "multi_asset",
            "observation_schedule",
            "observation_basis",
            "constituents",
            "state_variables",
            "event_transitions",
        ),
        required_primitives=("exercise_lattice",),
        route_helpers=("trellis.models.trees.lattice",),
        required_market_inputs=(
            "discount_curve",
            "black_vol_surface",
        ),
        route_family="rate_lattice",
        example_requests=(
            "Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        ),
        cue_phrases=(
            "callable bond",
            "issuer call",
            "call schedule",
            "call dates",
            "callable debt",
        ),
    ),
    _concept(
        semantic_id="rate_style_swaption",
        semantic_version="c2.1",
        scope="rate-style swaption contracts with schedule-dependent exercise",
        description="Simple rate-style swaption with schedule-dependent exercise and Black76 basis.",
        concept_role="product_contract",
        aliases=(
            "rate_style_swaption",
            "swaption",
            "bermudan_swaption",
        ),
        compatibility_wrappers=("swaption", "bermudan_swaption"),
        required_contract_fields=(
            "underlier_structure",
            "payoff_rule",
            "settlement_rule",
            "observation_schedule",
        ),
        allowed_contract_fields=(
            "instrument_class",
            "underlier_structure",
            "payoff_family",
            "payoff_rule",
            "settlement_rule",
            "payoff_traits",
            "exercise_style",
            "path_dependence",
            "schedule_dependence",
            "state_dependence",
            "model_family",
            "multi_asset",
            "observation_schedule",
            "observation_basis",
            "constituents",
            "state_variables",
            "event_transitions",
        ),
        required_primitives=("analytical_black76",),
        route_helpers=("trellis.models.black",),
        required_market_inputs=(
            "discount_curve",
            "forward_curve",
            "black_vol_surface",
        ),
        route_family="analytical",
        example_requests=(
            "European swaption on a fixed-for-floating swap with expiry 2026-01-15",
        ),
        cue_phrases=(
            "swaption",
            "fixed-for-floating",
            "forward swap",
            "swap rate",
            "swap exercise",
        ),
    ),
    _concept(
        semantic_id="schedule",
        semantic_version="c1.0",
        scope="ordered date schedules used for observation, exercise, fixing, or settlement",
        description="Ordered dates that define when a semantic contract observes, exercises, or settles.",
        concept_role="supporting_atom",
        aliases=(
            "schedule",
            "observation_schedule",
            "exercise_schedule",
            "coupon_schedule",
            "fixing_schedule",
        ),
        required_contract_fields=("observation_schedule",),
        allowed_contract_fields=(
            "observation_schedule",
            "observation_basis",
            "settlement_rule",
            "maturity_settlement_rule",
        ),
        required_primitives=("generate_schedule",),
        route_helpers=("trellis.core.date_utils",),
        example_requests=(
            "Generate an observation schedule for quarterly coupon dates.",
        ),
        cue_phrases=("schedule", "observation date", "exercise date", "coupon date"),
    ),
    _concept(
        semantic_id="curve",
        semantic_version="c1.0",
        scope="curve-shaped market inputs such as discount, forward, or yield curves",
        description="Curve-shaped term-structure inputs used by pricing and calibration routes.",
        concept_role="market_input",
        aliases=(
            "curve",
            "discount_curve",
            "forward_curve",
            "yield_curve",
        ),
        required_market_inputs=("discount_curve",),
        allowed_contract_fields=("market_data", "connector_hint", "capability"),
        route_helpers=("trellis.curves",),
        example_requests=("Discount curve required for present-value discounting.",),
        cue_phrases=("discount curve", "forward curve", "yield curve", "term structure"),
    ),
    _concept(
        semantic_id="surface",
        semantic_version="c1.0",
        scope="surface-shaped volatility and calibration inputs",
        description="Surface-shaped market inputs such as implied-volatility or local-volatility surfaces.",
        concept_role="market_input",
        aliases=(
            "surface",
            "vol_surface",
            "black_vol_surface",
            "local_vol_surface",
        ),
        required_market_inputs=("black_vol_surface",),
        allowed_contract_fields=("market_data", "connector_hint", "capability"),
        example_requests=("Implied volatility surface required for the route.",),
        cue_phrases=("vol surface", "volatility surface", "local vol surface", "surface"),
    ),
    _concept(
        semantic_id="correlation",
        semantic_version="c1.0",
        scope="dependence and correlation inputs for multi-asset or joint routes",
        description="Correlation and covariance inputs used to bind multi-asset and joint market state.",
        concept_role="market_input",
        aliases=(
            "correlation",
            "correlation_matrix",
            "covariance",
        ),
        required_market_inputs=("correlation_matrix",),
        allowed_contract_fields=("market_data", "connector_hint", "capability"),
        example_requests=("Correlation matrix for a multi-asset basket route.",),
        cue_phrases=("correlation matrix", "covariance matrix", "correlation", "dependence"),
    ),
    _concept(
        semantic_id="event_state",
        semantic_version="c1.0",
        scope="state-machine and event-transition concepts used in path-dependent routes",
        description="Event-state and transition concepts for path-dependent and stateful semantic routes.",
        concept_role="supporting_atom",
        aliases=(
            "event_state",
            "path_state",
            "state_machine",
        ),
        required_primitives=("path_state_accumulator",),
        route_helpers=("trellis.models.monte_carlo.event_state",),
        cue_phrases=("event state", "path state", "state machine", "state transition"),
    ),
    _concept(
        semantic_id="payoff",
        semantic_version="c1.0",
        scope="payoff rules and settlement semantics",
        description="Payoff-rule and settlement concepts that define how value is realized.",
        concept_role="supporting_atom",
        aliases=(
            "payoff",
            "payoff_rule",
            "settlement_rule",
        ),
        allowed_contract_fields=("payoff_rule", "settlement_rule", "maturity_settlement_rule"),
        cue_phrases=("payoff", "settlement", "cashflow", "settle"),
    ),
    _concept(
        semantic_id="exercise_policy",
        semantic_version="c1.0",
        scope="exercise styles and decision policies for early-exercise semantics",
        description="Exercise-policy concepts for callable, puttable, Bermudan, and American-style decisions.",
        concept_role="supporting_atom",
        aliases=(
            "exercise_policy",
            "exercise_style",
            "early_exercise",
        ),
        required_primitives=("exercise_lattice",),
        cue_phrases=("exercise policy", "exercise style", "callable", "puttable", "bermudan", "american"),
    ),
    _concept(
        semantic_id="calibration_target",
        semantic_version="c1.0",
        scope="targets used when fitting or calibrating a model or curve",
        description="Calibration-target concepts for fitting a model to market data or observed prices.",
        concept_role="market_input",
        aliases=(
            "calibration_target",
            "calibration",
            "fit_target",
        ),
        required_primitives=("calibration_solver",),
        cue_phrases=("calibration target", "calibration", "fit", "repricing", "solve for"),
    ),
    # ---- Credit derivatives ----
    _concept(
        semantic_id="credit_default_swap",
        semantic_version="c1.0",
        scope="single-name CDS: protection buyer/seller on one reference entity",
        description=(
            "Credit default swap on a single reference entity. Priced via "
            "survival-probability discounting (analytical) or hazard-rate "
            "simulation (Monte Carlo)."
        ),
        concept_role="product_contract",
        aliases=(
            "cds",
            "credit default swap",
            "single name cds",
            "single-name cds",
            "credit_default_swap",
        ),
        required_contract_fields=(
            "protection_leg",
            "premium_leg",
            "recovery_rate",
        ),
        required_primitives=(
            "credit_curve_survival_probability",
        ),
        required_market_inputs=(
            "discount_curve",
            "credit_curve",
        ),
        route_family="credit_default_swap",
        example_requests=(
            "Price a 5-year CDS on Company X",
            "CDS pricing: hazard rate MC vs survival prob analytical",
        ),
        cue_phrases=(
            "protection leg",
            "premium leg",
            "survival probability",
            "hazard rate",
            "credit spread",
            "single name",
            "single-name",
            "reference entity",
        ),
    ),
    _concept(
        semantic_id="nth_to_default",
        semantic_version="c1.0",
        scope="basket credit derivative: triggers on the nth default among N reference entities",
        description=(
            "Nth-to-default basket credit derivative. Requires default "
            "correlation modeling via copula (Gaussian or Student-t) and "
            "Monte Carlo simulation of correlated default times."
        ),
        concept_role="product_contract",
        aliases=(
            "nth to default",
            "nth-to-default",
            "ntd",
            "first to default",
            "basket cds",
            "nth_to_default",
        ),
        required_contract_fields=(
            "reference_entities",
            "default_trigger_n",
            "recovery_rates",
        ),
        required_primitives=(
            "gaussian_copula",
            "factor_copula",
        ),
        required_market_inputs=(
            "discount_curve",
            "credit_curve",
        ),
        route_family="nth_to_default",
        example_requests=(
            "Price a first-to-default basket on 5 names",
            "Nth-to-default on a portfolio of investment-grade credits",
        ),
        cue_phrases=(
            "default correlation",
            "basket credit",
            "reference entities",
            "names",
            "copula",
            "first to default",
            "nth default",
            "n-th default",
        ),
    ),
    _concept(
        semantic_id="market_parameter_source",
        semantic_version="c1.0",
        scope="provenance and source policy for market parameters",
        description="Source-policy concepts for observed, estimated, derived, calibrated, implied, sampled, synthetic, or user-supplied market parameters.",
        concept_role="market_input",
        aliases=(
            "market_parameter_source",
            "market_inputs",
            "provenance",
        ),
        required_market_inputs=(
            "discount_curve",
            "market_parameter_source",
        ),
        allowed_contract_fields=("provenance_requirements", "allowed_provenance", "missing_data_error_policy"),
        cue_phrases=(
            "market parameter source",
            "provenance",
            "observed",
            "estimated",
            "derived",
            "calibrated",
            "implied",
            "sampled",
            "synthetic",
        ),
    ),
)


@lru_cache(maxsize=1)
def _definition_index() -> dict[str, SemanticConceptDefinition]:
    """Return the canonical semantic-concept registry by id."""
    return {definition.semantic_id: definition for definition in SEMANTIC_CONCEPT_REGISTRY}


def get_semantic_concept_definition(semantic_id: str | None) -> SemanticConceptDefinition | None:
    """Look up one semantic concept by id."""
    if not semantic_id:
        return None
    return _definition_index().get(str(semantic_id).strip())


def semantic_concept_summary(
    concept: SemanticConceptDefinition | SemanticConceptResolution | None,
) -> dict[str, Any] | None:
    """Return a YAML-safe concept summary for traces and request metadata."""
    if concept is None:
        return None
    if isinstance(concept, SemanticConceptResolution):
        return {
            "request_text": concept.request_text,
            "instrument_type": concept.instrument_type,
            "concept_id": concept.concept_id,
            "concept_version": concept.concept_version,
            "concept_status": concept.concept_status,
            "concept_role": concept.concept_role,
            "resolution_kind": concept.resolution_kind,
            "matched_alias": concept.matched_alias,
            "matched_wrapper": concept.matched_wrapper,
            "candidate_concepts": list(concept.candidate_concepts),
            "conflicting_concepts": list(concept.conflicting_concepts),
            "superseded_concepts": list(concept.superseded_concepts),
            "policy_notes": list(concept.policy_notes),
            "summary": concept.summary,
            "confidence": concept.confidence,
            "gap_ratio": concept.gap_ratio,
        }

    return {
        "semantic_id": concept.semantic_id,
        "semantic_version": concept.semantic_version,
        "scope": concept.scope,
        "description": concept.description,
        "concept_role": concept.concept_role,
        "status": concept.status,
        "aliases": list(concept.aliases),
        "compatibility_wrappers": list(concept.compatibility_wrappers),
        "required_contract_fields": list(concept.required_contract_fields),
        "allowed_contract_fields": list(concept.allowed_contract_fields),
        "required_primitives": list(concept.required_primitives),
        "route_helpers": list(concept.route_helpers),
        "required_market_inputs": list(concept.required_market_inputs),
        "extension_policy": list(concept.extension_policy),
        "supersedes": list(concept.supersedes),
        "route_family": concept.route_family,
        "example_requests": list(concept.example_requests),
    }


_CONCEPT_ROLE_PRIORITY: dict[str, int] = {
    "product_contract": 0,
    "supporting_atom": 1,
    "market_input": 2,
}


def resolve_semantic_concept(
    description: str,
    *,
    instrument_type: str | None = None,
    term_sheet=None,
) -> SemanticConceptResolution:
    """Resolve a request against the semantic concept registry."""
    request_text = _combined_request_text(description, instrument_type, term_sheet)
    normalized_text = _normalize_text(request_text)
    normalized_instrument = _normalize_label(instrument_type)

    scored_matches: list[tuple[int, int, str, SemanticConceptDefinition, str, str, str]] = []
    for definition in SEMANTIC_CONCEPT_REGISTRY:
        score, match_kind, matched_alias, matched_wrapper, note = _score_definition(
            definition,
            normalized_text,
            normalized_instrument,
        )
        if score > 0:
            scored_matches.append(
                (score, len(definition.aliases) + len(definition.compatibility_wrappers), definition.semantic_id, definition, match_kind, matched_alias, matched_wrapper or note)
            )

    if not scored_matches:
        return SemanticConceptResolution(
            request_text=request_text,
            instrument_type=(instrument_type or "").strip(),
            resolution_kind=_fallback_resolution_kind(normalized_text),
            summary=_resolution_summary(
                concept_id="",
                concept_version="",
                resolution_kind=_fallback_resolution_kind(normalized_text),
                matched_alias="",
                matched_wrapper="",
                candidate_concepts=(),
                conflicting_concepts=(),
                concept_status="",
                concept_role="",
            ),
        )

    scored_matches.sort(key=lambda item: (-item[0], _CONCEPT_ROLE_PRIORITY.get(item[3].concept_role, 99), -item[1], item[2]))
    top_score, _, _, definition, match_kind, matched_alias, matched_wrapper = scored_matches[0]
    candidate_concepts = tuple(item[2] for item in scored_matches)
    conflicting_concepts = tuple(item[2] for item in scored_matches[1:])
    concept_status = definition.status
    resolution_kind = "reuse_existing_concept"
    policy_notes = list(definition.extension_policy)
    superseded_concepts = list(definition.supersedes)

    if match_kind == "deprecated_wrapper":
        concept_status = "stale"
        resolution_kind = "thin_compatibility_wrapper"
        policy_notes.append("deprecated_wrapper")
        if matched_alias:
            superseded_concepts.append(matched_alias)
    elif match_kind == "compatibility_wrapper":
        resolution_kind = "thin_compatibility_wrapper"
        policy_notes.append("compatibility_wrapper")
    elif match_kind == "alias":
        resolution_kind = "reuse_existing_concept"
    elif match_kind == "cue":
        resolution_kind = "reuse_existing_concept"

    if len(scored_matches) > 1:
        second_score = scored_matches[1][0]
        if top_score == second_score and match_kind not in {"compatibility_wrapper", "deprecated_wrapper"}:
            resolution_kind = "ambiguous"
            policy_notes.append("conflicting_registry_guidance")

    confidence = min(1.0, top_score / 120.0) if top_score > 0 else 0.0
    if top_score > 0 and len(scored_matches) > 1:
        gap_ratio = scored_matches[1][0] / top_score
    else:
        gap_ratio = 0.0

    summary = _resolution_summary(
        concept_id=definition.semantic_id,
        concept_version=definition.semantic_version,
        resolution_kind=resolution_kind,
        matched_alias=matched_alias,
        matched_wrapper=matched_wrapper if match_kind in {"compatibility_wrapper", "deprecated_wrapper"} else "",
        candidate_concepts=candidate_concepts,
        conflicting_concepts=conflicting_concepts,
        concept_status=concept_status,
        concept_role=definition.concept_role,
    )
    return SemanticConceptResolution(
        request_text=request_text,
        instrument_type=(instrument_type or "").strip(),
        concept_id=definition.semantic_id,
        concept_version=definition.semantic_version,
        concept_status=concept_status,
        concept_role=definition.concept_role,
        resolution_kind=resolution_kind,
        matched_alias=matched_alias,
        matched_wrapper=matched_wrapper if match_kind in {"compatibility_wrapper", "deprecated_wrapper"} else "",
        candidate_concepts=candidate_concepts,
        conflicting_concepts=conflicting_concepts,
        superseded_concepts=tuple(dict.fromkeys(superseded_concepts)),
        policy_notes=tuple(dict.fromkeys(policy_notes)),
        summary=summary,
        confidence=confidence,
        gap_ratio=gap_ratio,
    )


def _combined_request_text(description: str, instrument_type: str | None, term_sheet) -> str:
    """Combine request hints into one text blob for registry resolution."""
    parts = [
        description,
        instrument_type,
        getattr(term_sheet, "raw_description", None),
        getattr(term_sheet, "instrument_type", None),
    ]
    return " ".join(str(part).strip() for part in parts if part).strip()


def _normalize_text(text: str) -> str:
    """Normalize request text for phrase matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_label(value: str | None) -> str:
    """Normalize a label to a comparable concept token."""
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return whether a normalized phrase appears in normalized request text."""
    normalized = _normalize_text(phrase).replace("_", " ")
    if not normalized:
        return False
    return normalized in text


def _score_definition(
    definition: SemanticConceptDefinition,
    normalized_text: str,
    normalized_instrument: str,
) -> tuple[int, str, str, str, str]:
    """Score one semantic concept against the request text."""
    score = 0
    match_kind = ""
    matched_alias = ""
    matched_wrapper = ""
    note = ""

    for wrapper in definition.compatibility_wrappers:
        normalized_wrapper = _normalize_label(wrapper)
        if normalized_instrument and normalized_instrument == normalized_wrapper:
            score += 120
            match_kind = "compatibility_wrapper"
            matched_wrapper = wrapper
            note = "compatibility_wrapper"
            break
    if not matched_wrapper:
        for wrapper in definition.deprecated_wrappers:
            normalized_wrapper = _normalize_label(wrapper)
            if normalized_instrument and normalized_instrument == normalized_wrapper:
                score += 110
                match_kind = "deprecated_wrapper"
                matched_alias = wrapper
                matched_wrapper = wrapper
                note = "deprecated_wrapper"
                break
    if not matched_wrapper:
        for alias in definition.aliases:
            normalized_alias = _normalize_label(alias)
            if normalized_instrument and normalized_instrument == normalized_alias:
                score += 100
                match_kind = "alias"
                matched_alias = alias
                note = "canonical_alias"
                break

    for phrase in (*definition.aliases, *definition.compatibility_wrappers, *definition.deprecated_wrappers):
        if _contains_phrase(normalized_text, phrase):
            score += 30 + len(_normalize_label(phrase))
            if not matched_alias:
                matched_alias = phrase
            if not match_kind:
                match_kind = "alias"

    for phrase in definition.cue_phrases:
        if _contains_phrase(normalized_text, phrase):
            score += 10 + len(_normalize_label(phrase))
            if not match_kind:
                match_kind = "cue"
            note = phrase

    if definition.semantic_id in normalized_text:
        score += 5
        if not match_kind:
            match_kind = "alias"

    return score, match_kind or "cue", matched_alias, matched_wrapper, note


def _fallback_resolution_kind(normalized_text: str) -> str:
    """Return a deterministic fallback resolution kind for unmatched requests."""
    if any(
        cue in normalized_text
        for cue in (
            "basket",
            "option",
            "callable",
            "swaption",
            "quanto",
            "barrier",
            "lookback",
            "schedule",
            "coupon",
            "memory",
            "resettable",
        )
    ):
        return "introduce_new_concept"
    return "clarification"


def _resolution_summary(
    *,
    concept_id: str,
    concept_version: str,
    resolution_kind: str,
    matched_alias: str,
    matched_wrapper: str,
    candidate_concepts: tuple[str, ...],
    conflicting_concepts: tuple[str, ...],
    concept_status: str,
    concept_role: str,
) -> str:
    """Render a stable one-line summary for traces and metadata."""
    parts = [
        f"concept={concept_id or 'unknown'}",
        f"role={concept_role or 'unknown'}",
        f"version={concept_version or 'unknown'}",
        f"status={concept_status or 'unknown'}",
        f"kind={resolution_kind}",
    ]
    if matched_alias:
        parts.append(f"alias={matched_alias}")
    if matched_wrapper:
        parts.append(f"wrapper={matched_wrapper}")
    if candidate_concepts:
        parts.append("candidates=" + ",".join(candidate_concepts))
    if conflicting_concepts:
        parts.append("conflicts=" + ",".join(conflicting_concepts))
    return "; ".join(parts)
