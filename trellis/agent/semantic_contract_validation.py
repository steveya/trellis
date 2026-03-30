"""Deterministic validation for typed semantic contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
import re

from trellis.agent.knowledge.methods import is_known_method
from trellis.agent.semantic_concepts import (
    get_semantic_concept_definition,
    SemanticConceptResolution,
    resolve_semantic_concept,
)
from trellis.agent.semantic_contracts import SemanticContract, parse_semantic_contract
from trellis.core.capabilities import MARKET_DATA, normalize_capability_name


_KNOWN_CAPABILITIES = frozenset(cap.name for cap in MARKET_DATA)
_ALLOWED_PROVENANCE = frozenset(
    {
        "observed",
        "derived",
        "estimated",
        "calibrated",
        "implied",
        "sampled",
        "synthetic",
        "user_supplied",
    }
)
_ALLOWED_SELECTION_SCOPES = frozenset({"remaining_constituents"})
_ALLOWED_SELECTION_OPERATORS = frozenset({"best_of_remaining", "ranked_best_of_remaining"})
_ALLOWED_LOCK_RULES = frozenset({"remove_selected"})
_ALLOWED_AGGREGATION_RULES = frozenset({"average_locked_returns"})
_ALLOWED_UNDERLIER_STRUCTURES = frozenset(
    {
        "multi_asset_basket",
        "single_underlier",
        "cross_currency_single_underlier",
        "single_issuer_bond",
        "single_curve_rate_style",
    }
)
_CANONICAL_REQUIRED_CAPABILITIES = frozenset({
    "discount_curve",
    "spot",
    "black_vol_surface",
    "model_parameters",
})


@dataclass(frozen=True)
class SemanticContractValidationReport:
    """Validation outcome for one semantic contract."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    normalized_contract: SemanticContract | None = None

    @property
    def ok(self) -> bool:
        """Whether the contract passed validation."""
        return self.normalized_contract is not None and not self.errors


@dataclass(frozen=True)
class SemanticGapReport:
    """Structured classification for a request the semantic DSL cannot yet express."""

    # --- Identity ---
    request_text: str
    instrument_type: str = ""

    # --- Semantic Concept Resolution ---
    semantic_concept_id: str = ""
    semantic_concept_version: str = ""
    semantic_concept_status: str = ""
    semantic_concept_resolution_kind: str = ""
    semantic_concept_extension_kind: str = ""
    semantic_concept_matched_alias: str = ""
    semantic_concept_matched_wrapper: str = ""
    semantic_concept_conflicts: tuple[str, ...] = ()
    semantic_concept_superseded_concepts: tuple[str, ...] = ()
    semantic_concept_policy_notes: tuple[str, ...] = ()

    # --- Gap Classification ---
    gap_types: tuple[str, ...] = ()
    missing_contract_fields: tuple[str, ...] = ()
    missing_market_inputs: tuple[str, ...] = ()
    missing_runtime_primitives: tuple[str, ...] = ()
    missing_route_helpers: tuple[str, ...] = ()
    missing_knowledge_artifacts: tuple[str, ...] = ()

    # --- Status ---
    requires_clarification: bool = False
    can_use_mock_inputs: bool = False
    summary: str = ""

    def to_compact_dict(self) -> dict:
        """Return a dict with only non-default, non-empty fields."""
        from trellis.agent.semantic_contracts import _to_compact_dict
        return _to_compact_dict(self)


@dataclass(frozen=True)
class SemanticExtensionProposal:
    """Deterministic proposal for the smallest internal-DSL extension."""

    # --- Identity ---
    request_text: str
    instrument_type: str = ""

    # --- Semantic Concept Resolution ---
    semantic_concept_id: str = ""
    semantic_concept_version: str = ""
    semantic_concept_status: str = ""
    semantic_concept_resolution_kind: str = ""
    semantic_concept_extension_kind: str = ""
    semantic_concept_matched_alias: str = ""
    semantic_concept_matched_wrapper: str = ""
    semantic_concept_conflicts: tuple[str, ...] = ()
    semantic_concept_superseded_concepts: tuple[str, ...] = ()
    semantic_concept_policy_notes: tuple[str, ...] = ()

    # --- Decision ---
    decision: str = "clarification"
    trace_key: str = ""

    # --- Missing Pieces ---
    missing_contract_fields: tuple[str, ...] = ()
    missing_market_inputs: tuple[str, ...] = ()
    missing_runtime_primitives: tuple[str, ...] = ()
    missing_route_helpers: tuple[str, ...] = ()
    missing_knowledge_artifacts: tuple[str, ...] = ()

    # --- Proposed Extensions ---
    proposed_contract_fields: tuple[str, ...] = ()
    proposed_market_inputs: tuple[str, ...] = ()
    proposed_runtime_primitives: tuple[str, ...] = ()
    proposed_route_helpers: tuple[str, ...] = ()
    proposed_knowledge_artifacts: tuple[str, ...] = ()

    # --- Outcome ---
    recommended_next_step: str = ""
    confidence: float = 0.0
    summary: str = ""

    def to_compact_dict(self) -> dict:
        """Return a dict with only non-default, non-empty fields."""
        from trellis.agent.semantic_contracts import _to_compact_dict
        return _to_compact_dict(self)


def validate_semantic_contract(spec) -> SemanticContractValidationReport:
    """Parse and validate a semantic contract."""
    try:
        contract = parse_semantic_contract(spec)
    except Exception as exc:
        return SemanticContractValidationReport(
            errors=(f"Could not parse semantic contract: {exc}",),
            warnings=(),
            normalized_contract=None,
        )

    errors: list[str] = []
    warnings: list[str] = []

    _validate_basic_structure(contract, errors, warnings)
    _validate_market_inputs(contract, errors, warnings)
    _validate_methods(contract, errors, warnings)
    _validate_semantic_shape(contract, errors, warnings)

    return SemanticContractValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        normalized_contract=contract,
    )


def classify_semantic_gap(
    description: str,
    *,
    instrument_type: str | None = None,
    term_sheet=None,
) -> SemanticGapReport:
    """Classify the missing semantic pieces for an unsupported request."""
    request_text = _combined_request_text(description, instrument_type, term_sheet)
    normalized_text = _normalize_text(request_text)
    cues = _semantic_cues(normalized_text)
    concept_resolution = resolve_semantic_concept(
        description,
        instrument_type=instrument_type,
        term_sheet=term_sheet,
    )
    if not _should_surface_semantic_concept_in_gap(concept_resolution, cues):
        concept_resolution = SemanticConceptResolution(
            request_text=request_text,
            instrument_type=(instrument_type or "").strip(),
        )

    missing_contract_fields: list[str] = []
    missing_market_inputs: list[str] = []
    missing_runtime_primitives: list[str] = []
    missing_route_helpers: list[str] = []
    missing_knowledge_artifacts: list[str] = []

    if cues["shape"]:
        missing_contract_fields.extend(("underlier_structure", "payoff_rule", "settlement_rule"))
        missing_knowledge_artifacts.append("semantic_contract_lesson")
    if cues["schedule"]:
        missing_contract_fields.append("observation_schedule")
        missing_runtime_primitives.append("generate_schedule")
    if cues["path"]:
        missing_contract_fields.extend(
            ("selection_scope", "selection_operator", "lock_rule", "aggregation_rule")
        )
        missing_runtime_primitives.append("path_state_accumulator")
    if cues["credit"]:
        missing_market_inputs.extend(("discount_curve", "credit_curve"))
    if cues["market"]:
        missing_market_inputs.extend(("discount_curve", "market_parameter_source"))
        if cues["basket"]:
            missing_market_inputs.append("correlation_matrix")
    if cues["basket"] and cues["path"]:
        missing_route_helpers.append("correlated_basket_route_helper")
    if instrument_type and not cues["shape"]:
        missing_knowledge_artifacts.append("instrument_specific_lesson")

    if not any(cues.values()):
        missing_contract_fields.append("semantic_product_shape")
        missing_knowledge_artifacts.append("cookbook_entry")

    semantic_concept_extension_kind = _semantic_concept_extension_kind(
        concept_resolution=concept_resolution,
        missing_contract_fields=missing_contract_fields,
        cues=cues,
    )

    gap_types: list[str] = []
    if missing_contract_fields:
        gap_types.append("missing_semantic_contract_field")
    if missing_market_inputs:
        gap_types.append("missing_market_input_source")
    if missing_runtime_primitives:
        gap_types.append("missing_runtime_primitive")
    if missing_route_helpers:
        gap_types.append("missing_route_helper")
    if missing_knowledge_artifacts:
        gap_types.append("missing_knowledge_lesson")

    requires_clarification = not any(cues.values())
    # Ambiguous concept resolution should also require clarification — even
    # when cues fire, two concepts scoring equally means the request is
    # underspecified (e.g., "credit derivative" matches both CDS and NTD).
    if (
        concept_resolution is not None
        and getattr(concept_resolution, "resolution_kind", None) == "ambiguous"
    ):
        requires_clarification = True
    can_use_mock_inputs = bool(
        not requires_clarification
        and (missing_market_inputs or missing_runtime_primitives or missing_route_helpers)
    )

    summary_parts = []
    if missing_contract_fields:
        summary_parts.append(
            "missing contract fields: "
            + ", ".join(dict.fromkeys(missing_contract_fields))
        )
    if missing_market_inputs:
        summary_parts.append(
            "missing market inputs: "
            + ", ".join(dict.fromkeys(missing_market_inputs))
        )
    if missing_runtime_primitives:
        summary_parts.append(
            "missing runtime primitives: "
            + ", ".join(dict.fromkeys(missing_runtime_primitives))
        )
    if missing_route_helpers:
        summary_parts.append(
            "missing route helpers: "
            + ", ".join(dict.fromkeys(missing_route_helpers))
        )
    if missing_knowledge_artifacts:
        summary_parts.append(
            "missing knowledge artifacts: "
            + ", ".join(dict.fromkeys(missing_knowledge_artifacts))
        )
    if requires_clarification:
        summary_parts.append("request is too vague and needs clarification")
    elif can_use_mock_inputs:
        summary_parts.append("mock inputs can likely bridge the gap")

    return SemanticGapReport(
        request_text=request_text,
        instrument_type=(instrument_type or "").strip(),
        semantic_concept_id=concept_resolution.concept_id,
        semantic_concept_version=concept_resolution.concept_version,
        semantic_concept_status=concept_resolution.concept_status,
        semantic_concept_resolution_kind=concept_resolution.resolution_kind,
        semantic_concept_extension_kind=semantic_concept_extension_kind,
        semantic_concept_matched_alias=concept_resolution.matched_alias,
        semantic_concept_matched_wrapper=concept_resolution.matched_wrapper,
        semantic_concept_conflicts=concept_resolution.conflicting_concepts,
        semantic_concept_superseded_concepts=concept_resolution.superseded_concepts,
        semantic_concept_policy_notes=concept_resolution.policy_notes,
        gap_types=tuple(dict.fromkeys(gap_types)) or ("unsupported_semantic_request",),
        missing_contract_fields=tuple(dict.fromkeys(missing_contract_fields)),
        missing_market_inputs=tuple(dict.fromkeys(missing_market_inputs)),
        missing_runtime_primitives=tuple(dict.fromkeys(missing_runtime_primitives)),
        missing_route_helpers=tuple(dict.fromkeys(missing_route_helpers)),
        missing_knowledge_artifacts=tuple(dict.fromkeys(missing_knowledge_artifacts)),
        requires_clarification=requires_clarification,
        can_use_mock_inputs=can_use_mock_inputs,
        summary="; ".join(summary_parts) if summary_parts else "unsupported semantic request",
    )


def semantic_gap_summary(report: SemanticGapReport) -> dict[str, object]:
    """Return a YAML-safe summary for request and trace metadata."""
    return {
        "request_text": report.request_text,
        "instrument_type": report.instrument_type,
        "semantic_concept": _semantic_concept_summary_from_report(report),
        "gap_types": list(report.gap_types),
        "missing_contract_fields": list(report.missing_contract_fields),
        "missing_market_inputs": list(report.missing_market_inputs),
        "missing_runtime_primitives": list(report.missing_runtime_primitives),
        "missing_route_helpers": list(report.missing_route_helpers),
        "missing_knowledge_artifacts": list(report.missing_knowledge_artifacts),
        "requires_clarification": report.requires_clarification,
        "can_use_mock_inputs": report.can_use_mock_inputs,
        "summary": report.summary,
    }


def propose_semantic_extension(report: SemanticGapReport) -> SemanticExtensionProposal:
    """Convert a semantic gap into a concrete DSL extension proposal."""
    proposed_contract_fields: list[str] = []
    proposed_market_inputs: list[str] = []
    proposed_runtime_primitives: list[str] = []
    proposed_route_helpers: list[str] = []
    proposed_knowledge_artifacts: list[str] = []
    semantic_concept_extension_kind = report.semantic_concept_extension_kind

    if report.requires_clarification:
        decision = "clarification"
        confidence = 0.35
        recommended_next_step = (
            "Ask for the missing product shape, schedule, and payoff assumptions before extending the DSL."
        )
        proposed_contract_fields.extend(report.missing_contract_fields)
        proposed_knowledge_artifacts.append("clarification_prompt")
    else:
        if report.missing_runtime_primitives or report.missing_route_helpers:
            decision = "new_primitive"
            confidence = 0.75 if report.missing_runtime_primitives else 0.7
        elif report.missing_market_inputs:
            decision = "mock_inputs"
            confidence = 0.68
        else:
            decision = "knowledge_artifact"
            confidence = 0.6

        proposed_contract_fields.extend(report.missing_contract_fields)
        proposed_market_inputs.extend(_propose_market_inputs(report))
        proposed_runtime_primitives.extend(_propose_runtime_primitives(report))
        proposed_route_helpers.extend(_propose_route_helpers(report))
        proposed_knowledge_artifacts.extend(_propose_knowledge_artifacts(report))
        recommended_next_step = _recommend_next_step(
            report,
            decision=decision,
            proposed_market_inputs=proposed_market_inputs,
            proposed_runtime_primitives=proposed_runtime_primitives,
            proposed_route_helpers=proposed_route_helpers,
            semantic_concept_extension_kind=semantic_concept_extension_kind,
        )

    proposal = SemanticExtensionProposal(
        request_text=report.request_text,
        instrument_type=report.instrument_type,
        semantic_concept_id=report.semantic_concept_id,
        semantic_concept_version=report.semantic_concept_version,
        semantic_concept_status=report.semantic_concept_status,
        semantic_concept_resolution_kind=report.semantic_concept_resolution_kind,
        semantic_concept_extension_kind=report.semantic_concept_extension_kind,
        semantic_concept_matched_alias=report.semantic_concept_matched_alias,
        semantic_concept_matched_wrapper=report.semantic_concept_matched_wrapper,
        semantic_concept_conflicts=report.semantic_concept_conflicts,
        semantic_concept_superseded_concepts=report.semantic_concept_superseded_concepts,
        semantic_concept_policy_notes=report.semantic_concept_policy_notes,
        decision=decision,
        trace_key=_semantic_extension_trace_key(report, decision=decision),
        missing_contract_fields=report.missing_contract_fields,
        missing_market_inputs=report.missing_market_inputs,
        missing_runtime_primitives=report.missing_runtime_primitives,
        missing_route_helpers=report.missing_route_helpers,
        missing_knowledge_artifacts=report.missing_knowledge_artifacts,
        proposed_contract_fields=tuple(dict.fromkeys(proposed_contract_fields)),
        proposed_market_inputs=tuple(dict.fromkeys(proposed_market_inputs)),
        proposed_runtime_primitives=tuple(dict.fromkeys(proposed_runtime_primitives)),
        proposed_route_helpers=tuple(dict.fromkeys(proposed_route_helpers)),
        proposed_knowledge_artifacts=tuple(dict.fromkeys(proposed_knowledge_artifacts)),
        recommended_next_step=recommended_next_step,
        confidence=confidence,
        summary=_semantic_extension_summary(
            decision=decision,
            report=report,
            recommended_next_step=recommended_next_step,
            proposed_contract_fields=tuple(dict.fromkeys(proposed_contract_fields)),
            proposed_market_inputs=tuple(dict.fromkeys(proposed_market_inputs)),
            proposed_runtime_primitives=tuple(dict.fromkeys(proposed_runtime_primitives)),
            proposed_route_helpers=tuple(dict.fromkeys(proposed_route_helpers)),
            proposed_knowledge_artifacts=tuple(dict.fromkeys(proposed_knowledge_artifacts)),
        ),
    )
    return proposal


def semantic_extension_summary(report: SemanticExtensionProposal) -> dict[str, object]:
    """Return a YAML-safe summary for semantic extension metadata."""
    return {
        "request_text": report.request_text,
        "instrument_type": report.instrument_type,
        "semantic_concept": _semantic_concept_summary_from_proposal(report),
        "decision": report.decision,
        "trace_key": report.trace_key,
        "missing_contract_fields": list(report.missing_contract_fields),
        "missing_market_inputs": list(report.missing_market_inputs),
        "missing_runtime_primitives": list(report.missing_runtime_primitives),
        "missing_route_helpers": list(report.missing_route_helpers),
        "missing_knowledge_artifacts": list(report.missing_knowledge_artifacts),
        "proposed_contract_fields": list(report.proposed_contract_fields),
        "proposed_market_inputs": list(report.proposed_market_inputs),
        "proposed_runtime_primitives": list(report.proposed_runtime_primitives),
        "proposed_route_helpers": list(report.proposed_route_helpers),
        "proposed_knowledge_artifacts": list(report.proposed_knowledge_artifacts),
        "recommended_next_step": report.recommended_next_step,
        "confidence": report.confidence,
        "summary": report.summary,
    }


def _combined_request_text(description: str, instrument_type: str | None, term_sheet) -> str:
    """Combine all available request hints into one text blob for gap classification."""
    parts = [
        description,
        instrument_type,
        getattr(term_sheet, "raw_description", None),
        getattr(term_sheet, "instrument_type", None),
    ]
    return " ".join(str(part).strip() for part in parts if part).strip()


def _normalize_text(text: str) -> str:
    """Normalize request text for keyword-based gap classification."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _semantic_cues(normalized_text: str) -> dict[str, bool]:
    """Return coarse semantic cues that indicate which DSL fragment the request touches."""
    return {
        "shape": _contains_any(
            normalized_text,
            (
                "basket",
                "himalaya",
                "quanto",
                "callable",
                "swaption",
                "barrier",
                "lookback",
                "autocall",
                "memory",
                "resettable",
                "note",
                "bond",
                "option",
                "american",
                "european",
                "vanilla",
                "digital",
            ),
        ),
        "credit": _contains_any(
            normalized_text,
            (
                "credit",
                "credit curve",
                "hazard rate",
                "survival probability",
                "default probability",
                "default protection",
                "protection leg",
                "premium leg",
                "recovery rate",
                "spread",
            ),
        ),
        "schedule": _contains_any(
            normalized_text,
            (
                "schedule",
                "observation",
                "exercise",
                "fixing",
                "coupon date",
                "call date",
                "reset date",
                "expiry",
                "maturity",
            ),
        ),
        "path": _contains_any(
            normalized_text,
            (
                "path dependent",
                "path-dependent",
                "remaining",
                "rank",
                "best of",
                "best-of",
                "lock",
                "remove",
                "memory",
                "reset",
                "barrier",
                "coupon",
            ),
        ),
        "market": _contains_any(
            normalized_text,
            (
                "correlation",
                "covariance",
                "vol surface",
                "volatility",
                "discount curve",
                "forward curve",
                "fx",
                "spot",
                "model parameters",
                "calibration",
            ),
        ),
        "basket": _contains_any(
            normalized_text,
            (
                "basket",
                "himalaya",
                "best of",
                "best-of",
                "remaining constituents",
                "remaining constituent",
                "ranked observation",
                "ranked-selection",
            ),
        ),
    }


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether any keyword or phrase appears in the normalized text."""
    return any(needle in text for needle in needles)


def _semantic_extension_trace_key(report: SemanticGapReport, *, decision: str) -> str:
    """Build a stable key for extension traces and lesson retrieval."""
    concept_key = "|".join(
        (
            report.semantic_concept_id,
            report.semantic_concept_resolution_kind,
            report.semantic_concept_extension_kind,
        )
    )
    seed = "|".join(
        (
            _normalize_text(report.request_text),
            report.instrument_type,
            decision,
            concept_key,
            ",".join(report.gap_types),
            ",".join(report.missing_contract_fields),
            ",".join(report.missing_market_inputs),
            ",".join(report.missing_runtime_primitives),
            ",".join(report.missing_route_helpers),
        )
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _propose_market_inputs(report: SemanticGapReport) -> list[str]:
    """Translate missing market inputs into reusable sourcing suggestions."""
    suggestions: list[str] = []
    if "discount_curve" in report.missing_market_inputs:
        suggestions.append("mock_discount_curve_provider")
    if "market_parameter_source" in report.missing_market_inputs:
        suggestions.append("mock_market_parameter_provider")
    if "correlation_matrix" in report.missing_market_inputs:
        suggestions.append("correlation_source_policy")
    return suggestions


def _propose_runtime_primitives(report: SemanticGapReport) -> list[str]:
    """Translate missing runtime primitives into reusable shared helpers."""
    suggestions: list[str] = []
    if "generate_schedule" in report.missing_runtime_primitives:
        suggestions.append("trellis.core.date_utils.generate_schedule")
    if "path_state_accumulator" in report.missing_runtime_primitives:
        suggestions.extend(
            (
                "trellis.models.monte_carlo.event_state",
                "trellis.models.monte_carlo.basket_state",
                "trellis.models.monte_carlo.ranked_observation_payoffs",
            )
        )
    return suggestions


def _propose_route_helpers(report: SemanticGapReport) -> list[str]:
    """Translate missing route helpers into reusable semantic route surfaces."""
    suggestions: list[str] = []
    if "correlated_basket_route_helper" in report.missing_route_helpers:
        suggestions.extend(
            (
                "trellis.models.resolution.basket_semantics",
                "trellis.models.monte_carlo.semantic_basket",
            )
        )
    return suggestions


def _propose_knowledge_artifacts(report: SemanticGapReport) -> list[str]:
    """Translate missing knowledge artifacts into retrievable lesson/cookbook hints."""
    suggestions: list[str] = []
    if "semantic_contract_lesson" in report.missing_knowledge_artifacts:
        suggestions.append("semantic_contract_lesson")
    if "instrument_specific_lesson" in report.missing_knowledge_artifacts:
        suggestions.append("instrument_specific_lesson")
    if "cookbook_entry" in report.missing_knowledge_artifacts:
        suggestions.append("cookbook_entry")
    return suggestions


def _recommend_next_step(
    report: SemanticGapReport,
    *,
    decision: str,
    proposed_market_inputs: list[str],
    proposed_runtime_primitives: list[str],
    proposed_route_helpers: list[str],
    semantic_concept_extension_kind: str,
) -> str:
    """Summarize the smallest next step for the extension loop."""
    if decision == "clarification":
        return "Request the missing assumptions explicitly and re-draft the semantic contract."
    if semantic_concept_extension_kind == "thin_compatibility_wrapper":
        return "Keep the canonical concept and surface the compatibility name as a thin wrapper."
    if semantic_concept_extension_kind == "new_attribute":
        return "Extend the existing semantic concept with the missing attribute and keep the wrapper thin."
    if semantic_concept_extension_kind == "introduce_new_concept":
        return "Define the smallest new semantic concept before adding any wrapper or route helper."
    if proposed_runtime_primitives:
        return (
            "Reuse the shared runtime primitive surface and add only the missing thin adapter around "
            f"{proposed_runtime_primitives[0]}."
        )
    if proposed_route_helpers:
        return (
            "Bind the request through the shared semantic route helper and keep product-specific glue thin."
        )
    if proposed_market_inputs:
        return (
            "Synthesize a mock or empirical market-input source for "
            f"{proposed_market_inputs[0]} before pricing."
        )
    if report.missing_knowledge_artifacts:
        return "Capture the shape as a lesson or cookbook entry and retrain retrieval."
    return "Extend the internal DSL with the smallest reusable semantic artifact."


def _semantic_extension_summary(
    *,
    decision: str,
    report: SemanticGapReport,
    recommended_next_step: str,
    proposed_contract_fields: tuple[str, ...],
    proposed_market_inputs: tuple[str, ...],
    proposed_runtime_primitives: tuple[str, ...],
    proposed_route_helpers: tuple[str, ...],
    proposed_knowledge_artifacts: tuple[str, ...],
) -> str:
    """Render one stable human-readable summary for an extension proposal."""
    parts = [f"decision={decision}"]
    if report.semantic_concept_id:
        parts.append(f"concept={report.semantic_concept_id}")
    if report.semantic_concept_extension_kind:
        parts.append(f"concept_kind={report.semantic_concept_extension_kind}")
    if proposed_contract_fields:
        parts.append("fields=" + ", ".join(proposed_contract_fields))
    if proposed_market_inputs:
        parts.append("inputs=" + ", ".join(proposed_market_inputs))
    if proposed_runtime_primitives:
        parts.append("primitives=" + ", ".join(proposed_runtime_primitives))
    if proposed_route_helpers:
        parts.append("helpers=" + ", ".join(proposed_route_helpers))
    if proposed_knowledge_artifacts:
        parts.append("knowledge=" + ", ".join(proposed_knowledge_artifacts))
    if report.summary:
        parts.append(f"gap={report.summary}")
    parts.append(f"next={recommended_next_step}")
    return "; ".join(parts)


def _semantic_concept_extension_kind(
    *,
    concept_resolution,
    missing_contract_fields: list[str],
    cues: dict[str, bool],
) -> str:
    """Derive the concept-level extension kind for one unsupported request."""
    if concept_resolution.concept_id:
        if concept_resolution.resolution_kind == "thin_compatibility_wrapper":
            return "thin_compatibility_wrapper"
        if missing_contract_fields:
            return "new_attribute"
        return "reuse_existing_concept"
    if any(cues.values()):
        return "introduce_new_concept"
    return "clarification"


def _should_surface_semantic_concept_in_gap(
    concept_resolution,
    cues: dict[str, bool],
) -> bool:
    """Return whether a resolved concept should appear in a novel-request gap."""
    if not concept_resolution.concept_id:
        return False
    definition = get_semantic_concept_definition(concept_resolution.concept_id)
    if definition is None:
        return True
    if definition.concept_role == "supporting_atom" and cues["shape"]:
        return False
    return True


def _semantic_concept_summary_from_report(report: SemanticGapReport) -> dict[str, object] | None:
    """Return a YAML-safe semantic-concept summary from a gap report."""
    if not report.semantic_concept_id:
        return None
    return {
        "semantic_id": report.semantic_concept_id,
        "semantic_version": report.semantic_concept_version,
        "status": report.semantic_concept_status,
        "concept_role": _semantic_concept_role(report.semantic_concept_id),
        "resolution_kind": report.semantic_concept_resolution_kind,
        "extension_kind": report.semantic_concept_extension_kind,
        "matched_alias": report.semantic_concept_matched_alias,
        "matched_wrapper": report.semantic_concept_matched_wrapper,
        "conflicts": list(report.semantic_concept_conflicts),
        "superseded_concepts": list(report.semantic_concept_superseded_concepts),
        "policy_notes": list(report.semantic_concept_policy_notes),
    }


def _semantic_concept_role(semantic_id: str) -> str:
    """Return the registry role for one semantic concept id."""
    definition = get_semantic_concept_definition(semantic_id)
    if definition is None:
        return ""
    return definition.concept_role


def _semantic_concept_summary_from_proposal(
    report: SemanticExtensionProposal,
) -> dict[str, object] | None:
    """Return a YAML-safe semantic-concept summary from an extension proposal."""
    if not report.semantic_concept_id:
        return None
    return {
        "semantic_id": report.semantic_concept_id,
        "semantic_version": report.semantic_concept_version,
        "status": report.semantic_concept_status,
        "concept_role": _semantic_concept_role(report.semantic_concept_id),
        "resolution_kind": report.semantic_concept_resolution_kind,
        "extension_kind": report.semantic_concept_extension_kind,
        "matched_alias": report.semantic_concept_matched_alias,
        "matched_wrapper": report.semantic_concept_matched_wrapper,
        "conflicts": list(report.semantic_concept_conflicts),
        "superseded_concepts": list(report.semantic_concept_superseded_concepts),
        "policy_notes": list(report.semantic_concept_policy_notes),
    }


def _validate_basic_structure(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate top-level required sections."""
    if not contract.semantic_id:
        errors.append("Semantic contract semantic_id must be provided.")
    if not contract.product.instrument_class:
        errors.append("Semantic product instrument_class must be provided.")
    if not contract.product.payoff_family:
        errors.append("Semantic product payoff_family must be provided.")
    if not contract.methods.candidate_methods:
        errors.append("Semantic method contract must declare at least one candidate method.")
    if not contract.market_data.required_inputs:
        errors.append("Semantic market-data contract must declare required_inputs.")
    if not contract.product.underlier_structure:
        errors.append("Semantic product underlier_structure must be provided.")
    if not contract.product.payoff_rule:
        errors.append("Semantic product payoff_rule must be provided.")
    if not contract.product.settlement_rule:
        errors.append("Semantic product settlement_rule must be provided.")
    if not contract.validation.bundle_hints:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit validation bundle hint."
        )
    if not contract.blueprint.target_modules:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit target module hint."
        )


def _validate_market_inputs(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate input ids, capabilities, and provenance."""
    seen_ids: set[str] = set()
    for input_spec in (*contract.market_data.required_inputs, *contract.market_data.optional_inputs):
        if not input_spec.input_id:
            errors.append(
                f"Semantic contract `{contract.semantic_id}` has a market input with no input_id."
            )
            continue
        if input_spec.input_id in seen_ids:
            errors.append(
                f"Semantic contract `{contract.semantic_id}` defines market input `{input_spec.input_id}` more than once."
            )
        seen_ids.add(input_spec.input_id)

        if input_spec.capability:
            if input_spec.capability not in _KNOWN_CAPABILITIES:
                errors.append(
                    f"Market input `{input_spec.input_id}` references unknown capability `{input_spec.capability}`."
                )
            normalized = normalize_capability_name(input_spec.capability)
            if normalized != input_spec.capability:
                warnings.append(
                    f"Market input `{input_spec.input_id}` normalizes capability `{input_spec.capability}` to `{normalized}`."
                )
        if any(p not in _ALLOWED_PROVENANCE for p in input_spec.allowed_provenance):
            errors.append(
                f"Market input `{input_spec.input_id}` uses unsupported provenance labels {input_spec.allowed_provenance}."
            )
        if input_spec.derivable_from and "derived" not in input_spec.allowed_provenance:
            errors.append(
                f"Market input `{input_spec.input_id}` lists derivable_from but does not allow `derived` provenance."
            )
        if not input_spec.connector_hint and input_spec.capability:
            warnings.append(
                f"Market input `{input_spec.input_id}` has capability `{input_spec.capability}` but no connector hint."
            )


def _validate_methods(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate candidate and preferred methods."""
    candidate = set(contract.methods.candidate_methods)
    for method in contract.methods.candidate_methods:
        if not is_known_method(method):
            errors.append(
                f"Unknown candidate method `{method}` in semantic contract `{contract.semantic_id}`."
            )
    for method in (*contract.methods.reference_methods, *contract.methods.production_methods):
        if method not in candidate:
            errors.append(
                f"Method `{method}` is listed as reference/production but not in candidate_methods."
            )
    if contract.methods.preferred_method and contract.methods.preferred_method not in candidate:
        errors.append(
            f"Preferred method `{contract.methods.preferred_method}` is not in candidate_methods."
        )
    if not contract.methods.reference_methods and contract.methods.preferred_method is None:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit preferred or reference method."
        )


def _validate_semantic_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate the canonical shape-driven semantic contract."""
    definition = get_semantic_concept_definition(contract.semantic_id)
    if definition is None:
        errors.append(
            f"Unsupported semantic_id `{contract.semantic_id}` for semantic contract validation."
        )
        return
    if (
        definition.semantic_version
        and contract.product.semantic_version
        and contract.product.semantic_version != definition.semantic_version
    ):
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` uses version `{contract.product.semantic_version}` "
            f"but the registry currently expects `{definition.semantic_version}`."
        )
    if contract.product.instrument_class in definition.deprecated_wrappers:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` uses deprecated wrapper `{contract.product.instrument_class}`."
        )

    dispatch = {
        "ranked_observation_basket": _validate_ranked_observation_basket_shape,
        "vanilla_option": _validate_vanilla_option_shape,
        "quanto_option": _validate_quanto_option_shape,
        "callable_bond": _validate_callable_bond_shape,
        "rate_style_swaption": _validate_rate_style_swaption_shape,
    }
    validator = dispatch.get(contract.semantic_id)
    if validator is None:
        errors.append(
            f"Unsupported semantic_id `{contract.semantic_id}` for semantic contract validation."
        )
        return
    validator(contract, errors, warnings)


def _required_capabilities(contract: SemanticContract) -> set[str]:
    """Return the normalized market-data capabilities required by a contract."""
    return {
        normalize_capability_name(item.capability or item.input_id)
        for item in contract.market_data.required_inputs
        if item.capability or item.input_id
    }


def _validate_market_capabilities(
    contract: SemanticContract,
    errors: list[str],
    expected_capabilities: frozenset[str],
    *,
    multi_asset_requires_model_parameters: bool = False,
) -> set[str]:
    """Validate that the required market data matches a profile."""
    required_capabilities = _required_capabilities(contract)
    missing_capabilities = sorted(expected_capabilities - required_capabilities)
    if missing_capabilities:
        errors.append(
            f"Semantic contract `{contract.semantic_id}` requires the following market-data capabilities: "
            f"{missing_capabilities}."
        )
    if multi_asset_requires_model_parameters and "model_parameters" not in required_capabilities:
        errors.append(
            "Multi-asset Monte Carlo semantics require correlation data in `model_parameters`."
        )
    return required_capabilities


def _validate_profile_fields(
    contract: SemanticContract,
    errors: list[str],
    *,
    expected_instrument_class: str,
    expected_payoff_family: str,
    expected_underlier_structure: str,
    expected_payoff_rule: str,
    expected_settlement_rule: str,
    expected_exercise_style: str | None = None,
    expected_multi_asset: bool | None = None,
    require_schedule: bool = True,
    require_constituents: int = 0,
    selection_scope: str | None = None,
    selection_operator: str | None = None,
    selection_count: int | None = None,
    path_dependence: str | None = None,
    state_dependence: str | None = None,
    schedule_dependence: bool | None = None,
) -> None:
    """Validate a product profile's deterministic structural fields."""
    product = contract.product
    if product.instrument_class != expected_instrument_class:
        errors.append(
            f"Semantic slice expects instrument_class `{expected_instrument_class}`, got `{product.instrument_class}`."
        )
    if product.payoff_family != expected_payoff_family:
        errors.append(
            f"Semantic slice expects payoff_family `{expected_payoff_family}`, got `{product.payoff_family}`."
        )
    if product.underlier_structure != expected_underlier_structure:
        errors.append(
            f"Semantic slice expects underlier structure `{expected_underlier_structure}`, got `{product.underlier_structure}`."
        )
    if product.payoff_rule != expected_payoff_rule:
        errors.append(
            f"Semantic slice expects payoff_rule `{expected_payoff_rule}`, got `{product.payoff_rule}`."
        )
    if product.settlement_rule != expected_settlement_rule:
        errors.append(
            f"Semantic slice expects settlement_rule `{expected_settlement_rule}`, got `{product.settlement_rule}`."
        )
    if expected_exercise_style is not None and product.exercise_style != expected_exercise_style:
        errors.append(
            f"Semantic slice expects exercise_style `{expected_exercise_style}`, got `{product.exercise_style}`."
        )
    if expected_multi_asset is not None and product.multi_asset != expected_multi_asset:
        errors.append(
            f"Semantic slice expects multi_asset={expected_multi_asset}, got {product.multi_asset}."
        )
    if path_dependence is not None and product.path_dependence != path_dependence:
        errors.append(
            f"Semantic slice expects path_dependence `{path_dependence}`, got `{product.path_dependence}`."
        )
    if state_dependence is not None and product.state_dependence != state_dependence:
        errors.append(
            f"Semantic slice expects state_dependence `{state_dependence}`, got `{product.state_dependence}`."
        )
    if schedule_dependence is not None and product.schedule_dependence != schedule_dependence:
        errors.append(
            f"Semantic slice expects schedule_dependence={schedule_dependence}, got {product.schedule_dependence}."
        )
    if require_schedule and not product.observation_schedule:
        errors.append("Semantic contract requires a schedule for this shape.")
    elif product.observation_schedule and tuple(product.observation_schedule) != tuple(sorted(product.observation_schedule)):
        errors.append("Semantic observation schedule must be ordered.")
    if len(product.constituents) < require_constituents:
        errors.append(
            f"Semantic contract requires at least {require_constituents} constituent(s) for this shape."
        )
    if selection_scope is not None and product.selection_scope != selection_scope:
        errors.append(
            f"Semantic slice expects selection_scope `{selection_scope}`, got `{product.selection_scope}`."
        )
    if selection_operator is not None and product.selection_operator != selection_operator:
        errors.append(
            f"Semantic slice expects selection_operator `{selection_operator}`, got `{product.selection_operator}`."
        )
    if selection_count is not None and product.selection_count != selection_count:
        errors.append(
            f"Semantic slice expects selection_count={selection_count}, got {product.selection_count}."
        )
    if product.underlier_structure not in _ALLOWED_UNDERLIER_STRUCTURES:
        errors.append(
            f"Semantic underlier_structure `{product.underlier_structure}` is not supported."
        )


def _validate_ranked_observation_basket_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate the canonical ranked-observation basket semantic shape."""
    required_capabilities = _validate_market_capabilities(
        contract,
        errors,
        _CANONICAL_REQUIRED_CAPABILITIES,
        multi_asset_requires_model_parameters=True,
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="basket_path_payoff",
        expected_payoff_family="basket_path_payoff",
        expected_underlier_structure="multi_asset_basket",
        expected_payoff_rule="ranked_observation_path_payoff",
        expected_settlement_rule="settle_once_at_maturity",
        expected_exercise_style="none",
        expected_multi_asset=True,
        require_schedule=True,
        require_constituents=2,
        selection_scope="remaining_constituents",
        selection_operator="best_of_remaining",
        selection_count=1,
        path_dependence="path_dependent",
        state_dependence="path_dependent",
        schedule_dependence=True,
    )
    if len(contract.product.observation_schedule) < 1:
        errors.append("Semantic ranked observation basket requires an observation schedule.")
    if len(contract.product.constituents) < 2:
        errors.append("Semantic ranked observation basket requires at least two constituents.")
    if product := contract.product:
        if product.lock_rule != "remove_selected":
            errors.append(
                f"Semantic lock_rule `{product.lock_rule}` must be `remove_selected`."
            )
        if product.aggregation_rule != "average_locked_returns":
            errors.append(
                f"Semantic aggregation_rule `{product.aggregation_rule}` must be `average_locked_returns`."
            )
        if product.selection_count != 1:
            errors.append("Semantic ranked observation basket currently supports selection_count=1.")
    missing_capabilities = sorted(_CANONICAL_REQUIRED_CAPABILITIES - required_capabilities)
    if missing_capabilities:
        errors.append(
            "Semantic ranked observation basket requires the following market-data capabilities: "
            f"{missing_capabilities}."
        )
    if contract.product.multi_asset and "monte_carlo" in contract.methods.candidate_methods:
        if "model_parameters" not in required_capabilities:
            errors.append(
                "Multi-asset Monte Carlo semantics require correlation data in `model_parameters`."
            )
    if not contract.blueprint.primitive_families:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit primitive-family hint."
        )


def _validate_vanilla_option_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a single-underlier vanilla option semantic shape."""
    required_capabilities = _validate_market_capabilities(
        contract,
        errors,
        frozenset({"discount_curve", "spot", "black_vol_surface"}),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="european_option",
        expected_payoff_family="vanilla_option",
        expected_underlier_structure="single_underlier",
        expected_payoff_rule="vanilla_option_payoff",
        expected_settlement_rule="cash_settle_at_expiry",
        expected_exercise_style="european",
        expected_multi_asset=False,
        require_schedule=True,
        require_constituents=1,
        path_dependence="terminal_markov",
        state_dependence="terminal_markov",
        schedule_dependence=False,
    )
    if "spot" not in required_capabilities:
        errors.append("Vanilla option semantics require a spot underlier input.")
    if not contract.blueprint.primitive_families:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit primitive-family hint."
        )


def _validate_quanto_option_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a cross-currency quanto option semantic shape."""
    required_capabilities = _validate_market_capabilities(
        contract,
        errors,
        frozenset(
            {"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot", "model_parameters"}
        ),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="quanto_option",
        expected_payoff_family="vanilla_option",
        expected_underlier_structure="cross_currency_single_underlier",
        expected_payoff_rule="quanto_adjusted_vanilla_payoff",
        expected_settlement_rule="cash_settle_at_expiry_after_fx_conversion",
        expected_exercise_style="european",
        expected_multi_asset=False,
        require_schedule=True,
        require_constituents=1,
        path_dependence="terminal_markov",
        state_dependence="terminal_markov",
        schedule_dependence=False,
    )
    if "fx_rates" not in required_capabilities:
        errors.append("Quanto option semantics require FX rates.")


def _validate_callable_bond_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a callable fixed-income semantic shape."""
    _validate_market_capabilities(
        contract,
        errors,
        frozenset({"discount_curve", "black_vol_surface"}),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="callable_bond",
        expected_payoff_family="callable_fixed_income",
        expected_underlier_structure="single_issuer_bond",
        expected_payoff_rule="issuer_call_contingent_cashflow",
        expected_settlement_rule="settle_on_call_or_maturity",
        expected_exercise_style="issuer_call",
        expected_multi_asset=False,
        require_schedule=True,
        require_constituents=0,
        path_dependence="schedule_dependent",
        state_dependence="schedule_dependent",
        schedule_dependence=True,
    )
    if not contract.product.observation_schedule:
        errors.append("Semantic callable bond requires a call schedule.")
    if not contract.blueprint.primitive_families:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit primitive-family hint."
        )


def _validate_rate_style_swaption_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a simple rate-style swaption semantic shape."""
    _validate_market_capabilities(
        contract,
        errors,
        frozenset({"discount_curve", "forward_curve", "black_vol_surface"}),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="swaption",
        expected_payoff_family="swaption",
        expected_underlier_structure="single_curve_rate_style",
        expected_payoff_rule="swaption_exercise_payoff",
        expected_settlement_rule="cash_settle_at_exercise",
        expected_exercise_style="european",
        expected_multi_asset=False,
        require_schedule=True,
        require_constituents=0,
        path_dependence="schedule_dependent",
        state_dependence="schedule_dependent",
        schedule_dependence=True,
    )
    if not contract.blueprint.primitive_families:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit primitive-family hint."
        )
