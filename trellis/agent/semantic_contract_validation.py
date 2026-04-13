"""Deterministic validation for typed semantic contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
import re
from types import MappingProxyType

import trellis.core.capabilities as capability_registry

from trellis.agent.knowledge.methods import is_known_method
from trellis.agent.semantic_concepts import (
    get_semantic_concept_definition,
    SemanticConceptResolution,
    resolve_semantic_concept,
)
from trellis.agent.semantic_contracts import (
    DEFAULT_PHASE_ORDER,
    SemanticContract,
    parse_semantic_contract,
)


_KNOWN_CAPABILITIES = frozenset(cap.name for cap in capability_registry.MARKET_DATA)
_ALLOWED_PHASES = frozenset(DEFAULT_PHASE_ORDER)
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
        "single_reference_entity",
    }
)
_CANONICAL_REQUIRED_CAPABILITIES = frozenset({
    "discount_curve",
    "spot",
    "black_vol_surface",
    "model_parameters",
})
_ALLOWED_CONTROLLER_STYLES = frozenset({"identity", "holder_max", "issuer_min"})
_ALLOWED_STATE_FIELD_KINDS = frozenset({"event_state", "contract_memory"})
_ALLOWED_STATE_TAGS = frozenset(
    {
        "terminal_markov",
        "schedule_state",
        "recombining_safe",
        "pathwise_only",
        "remaining_pool",
        "locked_cashflow_state",
    }
)
_AUTOMATIC_ACTION_HINTS = (
    "settle",
    "lock",
    "remove",
    "translate",
    "rank",
    "record",
    "observe",
    "coupon",
    "autocall",
    "trigger",
    "knock",
    "barrier",
)

@dataclass(frozen=True)
class SemanticContractValidationFinding:
    """One structured semantic-contract validation finding."""

    code: str
    severity: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class SemanticContractValidationReport:
    """Validation outcome for one semantic contract."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    findings: tuple[SemanticContractValidationFinding, ...] = ()
    normalized_contract: SemanticContract | None = None

    @property
    def ok(self) -> bool:
        """Whether the contract passed validation."""
        return self.normalized_contract is not None and not self.errors

    @property
    def error_findings(self) -> tuple[SemanticContractValidationFinding, ...]:
        """Return the structured error findings."""
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warning_findings(self) -> tuple[SemanticContractValidationFinding, ...]:
        """Return the structured warning findings."""
        return tuple(f for f in self.findings if f.severity == "warning")

    @property
    def confidence(self) -> float:
        """Expose a simple confidence score for build-gate compatibility."""
        if self.errors:
            return 0.0
        if self.warnings:
            return 0.9
        return 1.0

    @property
    def has_promoted_route(self) -> bool:
        """Duck-type to the build gate's readiness report."""
        return self.normalized_contract is not None and not self.errors

    @property
    def route_gap(self):
        """Semantic validation does not emit a route-gap object."""
        return None

    @property
    def missing(self) -> tuple[str, ...]:
        """Duck-type the gap-report missing tuple for build-gate reuse."""
        return self.errors


@dataclass(frozen=True)
class SettlementAuthorityProfile:
    """Declarative settlement-authority match profile for one semantic family."""

    semantic_id: str
    instrument_classes: tuple[str, ...] = ()
    payoff_families: tuple[str, ...] = ()
    exercise_styles: tuple[str, ...] = ()
    required_payoff_traits: tuple[str, ...] = ()
    required_primitive_families: tuple[str, ...] = ()


_SETTLEMENT_AUTHORITY_PROFILES = MappingProxyType(
    {
        "vanilla_option": SettlementAuthorityProfile(
            semantic_id="vanilla_option",
            instrument_classes=("european_option",),
            payoff_families=("vanilla_option",),
        ),
        "callable_bond": SettlementAuthorityProfile(
            semantic_id="callable_bond",
            instrument_classes=("callable_bond",),
        ),
        "ranked_observation_basket": SettlementAuthorityProfile(
            semantic_id="ranked_observation_basket",
            instrument_classes=("basket_path_payoff",),
            required_payoff_traits=("ranked_observation",),
        ),
        "rate_style_swaption": SettlementAuthorityProfile(
            semantic_id="rate_style_swaption",
            instrument_classes=("swaption",),
            exercise_styles=("bermudan",),
            required_primitive_families=("exercise_lattice",),
        ),
    }
)

_RATE_CAP_FLOOR_WRAPPER_PROFILES = MappingProxyType(
    {
        "cap": MappingProxyType(
            {
                "obligation_id": "cap_period_cashflow",
                "option_type": "call",
            }
        ),
        "floor": MappingProxyType(
            {
                "obligation_id": "floor_period_cashflow",
                "option_type": "put",
            }
        ),
    }
)


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
    missing_binding_helpers: tuple[str, ...] = ()
    missing_knowledge_artifacts: tuple[str, ...] = ()

    # --- Status ---
    requires_clarification: bool = False
    can_use_mock_inputs: bool = False
    summary: str = ""

    def to_compact_dict(self) -> dict:
        """Return a dict with only non-default, non-empty fields."""
        from trellis.agent.semantic_contracts import _to_compact_dict
        return _to_compact_dict(self)

    @property
    def missing_route_helpers(self) -> tuple[str, ...]:
        """Compatibility alias for older trace payloads."""
        return self.missing_binding_helpers


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
    missing_binding_helpers: tuple[str, ...] = ()
    missing_knowledge_artifacts: tuple[str, ...] = ()

    # --- Proposed Extensions ---
    proposed_contract_fields: tuple[str, ...] = ()
    proposed_market_inputs: tuple[str, ...] = ()
    proposed_runtime_primitives: tuple[str, ...] = ()
    proposed_binding_helpers: tuple[str, ...] = ()
    proposed_knowledge_artifacts: tuple[str, ...] = ()

    # --- Outcome ---
    recommended_next_step: str = ""
    confidence: float = 0.0
    summary: str = ""

    def to_compact_dict(self) -> dict:
        """Return a dict with only non-default, non-empty fields."""
        from trellis.agent.semantic_contracts import _to_compact_dict
        return _to_compact_dict(self)

    @property
    def missing_route_helpers(self) -> tuple[str, ...]:
        """Compatibility alias for older trace payloads."""
        return self.missing_binding_helpers

    @property
    def proposed_route_helpers(self) -> tuple[str, ...]:
        """Compatibility alias for older trace payloads."""
        return self.proposed_binding_helpers


def validate_semantic_contract(spec) -> SemanticContractValidationReport:
    """Parse and validate a semantic contract."""
    try:
        contract = parse_semantic_contract(spec)
    except Exception as exc:
        return SemanticContractValidationReport(
            errors=(f"Could not parse semantic contract: {exc}",),
            warnings=(),
            findings=(
                SemanticContractValidationFinding(
                    code="semantic.parse_failed",
                    severity="error",
                    message=f"Could not parse semantic contract: {exc}",
                    path="contract",
                ),
            ),
            normalized_contract=None,
        )

    errors: list[str] = []
    warnings: list[str] = []

    _validate_basic_structure(contract, errors, warnings)
    _validate_typed_semantic_surface(contract, errors, warnings)
    _validate_market_inputs(contract, errors, warnings)
    _validate_methods(contract, errors, warnings)
    _validate_semantic_shape(contract, errors, warnings)
    _validate_event_machine(contract, errors, warnings)
    _validate_calibration(contract, errors, warnings)

    return SemanticContractValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        findings=_messages_to_findings(errors=errors, warnings=warnings),
        normalized_contract=contract,
    )


def _messages_to_findings(
    *,
    errors: list[str] | tuple[str, ...],
    warnings: list[str] | tuple[str, ...],
) -> tuple[SemanticContractValidationFinding, ...]:
    """Build structured findings from the stable string error/warning surface."""
    findings: list[SemanticContractValidationFinding] = []
    for severity, messages in (("error", errors), ("warning", warnings)):
        for message in messages:
            findings.append(
                SemanticContractValidationFinding(
                    code=_message_code(message, severity),
                    severity=severity,
                    message=message,
                    path=_message_path(message),
                )
            )
    return tuple(findings)


def _message_code(message: str, severity: str) -> str:
    """Map a validation message to a stable finding code."""
    lower = str(message).strip().lower()
    mapping = (
        ("parse semantic contract", "semantic.parse_failed"),
        ("phase order", "semantic.phase_order_invalid"),
        ("future-peek", "semantic.future_peek"),
        ("automatic trigger represented as control", "semantic.automatic_trigger_as_control"),
        ("settlement-bearing semantic contracts", "semantic.missing_obligation"),
        ("typed obligation", "semantic.typed_obligation_invalid"),
        ("state-tag consistency", "semantic.state_tag_inconsistent"),
        ("unsupported controller_style", "semantic.controller_style_invalid"),
        ("unsupported decision phase", "semantic.controller_phase_invalid"),
        ("normalized legacy schedule semantics", "semantic.legacy_phase_order_normalized"),
        ("normalized legacy state_variables", "semantic.legacy_state_fields_normalized"),
        ("normalized legacy event_transitions", "semantic.legacy_event_machine_normalized"),
        ("without typed observables", "semantic.legacy_observables_missing"),
        ("strategic rights implicit", "semantic.legacy_controller_protocol_missing"),
    )
    for needle, code in mapping:
        if needle in lower:
            return code
    slug = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    return f"semantic.{severity}.{slug[:80]}"


def _message_path(message: str) -> str:
    """Best-effort path classification for semantic validation messages."""
    lower = str(message).strip().lower()
    if "observable" in lower:
        return "product.observables"
    if "state field" in lower or "state-tag" in lower:
        return "product.state_fields"
    if "controller" in lower or "strategic rights" in lower:
        return "product.controller_protocol"
    if "obligation" in lower or "settlement-bearing" in lower:
        return "product.obligations"
    if "phase order" in lower or "decision phase" in lower:
        return "product.timeline"
    if "event_machine" in lower or "event_transitions" in lower:
        return "product.event_machine"
    return "contract"


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
    has_shape_authority = cues["shape"] or _semantic_concept_supplies_shape_authority(
        concept_resolution
    )
    has_any_semantic_cue = any(cues.values()) or has_shape_authority
    if not _should_surface_semantic_concept_in_gap(concept_resolution, cues):
        concept_resolution = SemanticConceptResolution(
            request_text=request_text,
            instrument_type=(instrument_type or "").strip(),
        )

    missing_contract_fields: list[str] = []
    missing_market_inputs: list[str] = []
    missing_runtime_primitives: list[str] = []
    missing_binding_helpers: list[str] = []
    missing_knowledge_artifacts: list[str] = []

    if has_shape_authority:
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
        missing_binding_helpers.append("correlated_basket_binding_helper")
    if instrument_type and not has_shape_authority:
        missing_knowledge_artifacts.append("instrument_specific_lesson")

    if not has_any_semantic_cue:
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
    if missing_binding_helpers:
        gap_types.append("missing_binding_helper")
    if missing_knowledge_artifacts:
        gap_types.append("missing_knowledge_lesson")

    requires_clarification = not has_any_semantic_cue
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
        and (missing_market_inputs or missing_runtime_primitives or missing_binding_helpers)
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
    if missing_binding_helpers:
        summary_parts.append(
            "missing binding helpers: "
            + ", ".join(dict.fromkeys(missing_binding_helpers))
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
        missing_binding_helpers=tuple(dict.fromkeys(missing_binding_helpers)),
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
        "missing_binding_helpers": list(report.missing_binding_helpers),
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
    proposed_binding_helpers: list[str] = []
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
        if report.missing_runtime_primitives or report.missing_binding_helpers:
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
        proposed_binding_helpers.extend(_propose_binding_helpers(report))
        proposed_knowledge_artifacts.extend(_propose_knowledge_artifacts(report))
        recommended_next_step = _recommend_next_step(
            report,
            decision=decision,
            proposed_market_inputs=proposed_market_inputs,
            proposed_runtime_primitives=proposed_runtime_primitives,
            proposed_binding_helpers=proposed_binding_helpers,
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
        missing_binding_helpers=report.missing_binding_helpers,
        missing_knowledge_artifacts=report.missing_knowledge_artifacts,
        proposed_contract_fields=tuple(dict.fromkeys(proposed_contract_fields)),
        proposed_market_inputs=tuple(dict.fromkeys(proposed_market_inputs)),
        proposed_runtime_primitives=tuple(dict.fromkeys(proposed_runtime_primitives)),
        proposed_binding_helpers=tuple(dict.fromkeys(proposed_binding_helpers)),
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
            proposed_binding_helpers=tuple(dict.fromkeys(proposed_binding_helpers)),
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
        "missing_binding_helpers": list(report.missing_binding_helpers),
        "missing_knowledge_artifacts": list(report.missing_knowledge_artifacts),
        "proposed_contract_fields": list(report.proposed_contract_fields),
        "proposed_market_inputs": list(report.proposed_market_inputs),
        "proposed_runtime_primitives": list(report.proposed_runtime_primitives),
        "proposed_binding_helpers": list(report.proposed_binding_helpers),
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
            ",".join(report.missing_binding_helpers),
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


def _propose_binding_helpers(report: SemanticGapReport) -> list[str]:
    """Translate missing binding helpers into reusable semantic binding surfaces."""
    suggestions: list[str] = []
    if "correlated_basket_binding_helper" in report.missing_binding_helpers:
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
    proposed_binding_helpers: list[str],
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
        return "Define the smallest new semantic concept before adding any wrapper or binding helper."
    if proposed_runtime_primitives:
        return (
            "Reuse the shared runtime primitive surface and add only the missing thin adapter around "
            f"{proposed_runtime_primitives[0]}."
        )
    if proposed_binding_helpers:
        return (
            "Bind the request through the shared semantic binding helper and keep product-specific glue thin."
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
    proposed_binding_helpers: tuple[str, ...],
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
    if proposed_binding_helpers:
        parts.append("binding_helpers=" + ", ".join(proposed_binding_helpers))
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


def _semantic_concept_supplies_shape_authority(concept_resolution) -> bool:
    """Whether resolved semantic identity already nails down the product shape."""
    if not getattr(concept_resolution, "concept_id", ""):
        return False
    return getattr(concept_resolution, "resolution_kind", "") in {
        "reuse_existing_concept",
        "thin_compatibility_wrapper",
    }


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
        if _typed_surface_is_authoritative_for_settlement_rule(contract):
            warnings.append(
                "Semantic contract omitted the legacy settlement_rule mirror; typed obligations remain authoritative for this migrated route."
            )
        else:
            errors.append("Semantic product settlement_rule must be provided.")
    if not contract.validation.bundle_hints:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit validation bundle hint."
        )
    if not contract.blueprint.target_modules:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit target module hint."
        )


def _validate_typed_semantic_surface(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate the typed semantic surface added to SemanticContract."""
    product = contract.product
    phase_order = tuple(product.timeline.phase_order)
    phase_index = {phase: idx for idx, phase in enumerate(phase_order)}

    _validate_phase_order(product, errors, warnings)
    _validate_observables(product, phase_index, errors, warnings)
    _validate_state_fields(product, phase_index, errors, warnings)
    _validate_controller_protocol(product, phase_index, errors, warnings)
    _validate_obligations(product, errors, warnings)
    _validate_legacy_typed_normalization(product, warnings)


def _validate_phase_order(
    product,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate the tranche-1 default same-day phase order."""
    phase_order = tuple(product.timeline.phase_order)
    if not phase_order:
        errors.append(
            "Semantic timeline phase order must be provided as EVENT -> OBSERVATION -> DECISION -> DETERMINATION -> SETTLEMENT -> STATE_UPDATE."
        )
        return
    if tuple(phase_order) != DEFAULT_PHASE_ORDER:
        errors.append(
            "Semantic timeline phase order must preserve EVENT -> OBSERVATION -> DECISION -> DETERMINATION -> SETTLEMENT -> STATE_UPDATE in tranche 1."
        )
    unknown_phases = sorted(set(phase_order) - _ALLOWED_PHASES)
    if unknown_phases:
        errors.append(
            f"Semantic timeline phase order uses unsupported phase labels {unknown_phases}."
        )
    if product.observation_schedule and not product.implementation_hints.primary_schedule_role:
        warnings.append(
            "Semantic contract normalized legacy schedule semantics to the default phase order."
        )


def _validate_observables(
    product,
    phase_index: dict[str, int],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate typed observable metadata and future-peek dependencies."""
    observables = tuple(product.observables)
    if not observables:
        warnings.append(
            "Semantic contract normalized legacy payoff semantics without typed observables."
        )
        return

    observable_map: dict[str, object] = {}
    for observable in observables:
        if not observable.observable_id:
            errors.append("Semantic observable entries must define observable_id.")
            continue
        if observable.observable_id in observable_map:
            errors.append(
                f"Semantic observable `{observable.observable_id}` is defined more than once."
            )
        observable_map[observable.observable_id] = observable
        if observable.availability_phase not in phase_index:
            errors.append(
                f"Semantic observable `{observable.observable_id}` uses unsupported availability phase `{observable.availability_phase}`."
            )

    for observable in observables:
        if observable.observable_id not in observable_map:
            continue
        for dependency in observable.dependencies:
            dep = observable_map.get(dependency)
            if dep is None:
                errors.append(
                    f"Semantic observable `{observable.observable_id}` references unknown dependency `{dependency}`."
                )
                continue
            dep_index = phase_index.get(dep.availability_phase)
            observable_index = phase_index.get(observable.availability_phase)
            if dep_index is None or observable_index is None:
                continue
            if dep_index > observable_index:
                errors.append(
                    f"Semantic observable `{observable.observable_id}` has a future-peek dependency on `{dependency}`."
                )


def _validate_state_fields(
    product,
    phase_index: dict[str, int],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate typed state-field metadata and solver tags."""
    observables = {
        observable.observable_id: observable
        for observable in product.observables
        if observable.observable_id
    }
    has_typed_observables = bool(observables)

    for state_field in product.state_fields:
        if state_field.kind not in _ALLOWED_STATE_FIELD_KINDS:
            errors.append(
                f"Semantic state field `{state_field.field_name}` uses unsupported kind `{state_field.kind}`."
            )
            continue
        unknown_tags = sorted(set(state_field.tags) - _ALLOWED_STATE_TAGS)
        if unknown_tags:
            errors.append(
                f"Semantic state-tag consistency failed for `{state_field.field_name}`: unsupported tags {unknown_tags}."
            )
        if "recombining_safe" in state_field.tags and "pathwise_only" in state_field.tags:
            errors.append(
                f"Semantic state-tag consistency failed for `{state_field.field_name}`: `recombining_safe` and `pathwise_only` cannot both be present."
            )
        if state_field.kind == "event_state" and "pathwise_only" in state_field.tags:
            errors.append(
                f"Semantic state-tag consistency failed for `{state_field.field_name}`: event_state cannot be tagged `pathwise_only`."
            )
        if state_field.kind == "contract_memory" and "terminal_markov" in state_field.tags:
            errors.append(
                f"Semantic state-tag consistency failed for `{state_field.field_name}`: contract_memory cannot be tagged `terminal_markov`."
            )

        cutoff_phase = "determination" if state_field.kind == "event_state" else "state_update"
        cutoff_index = phase_index.get(cutoff_phase)
        if cutoff_index is None:
            errors.append(
                f"Semantic timeline phase order is missing required phase `{cutoff_phase}` for state-field validation."
            )
            continue
        for source in state_field.source_observables:
            if not has_typed_observables:
                continue
            observable = observables.get(source)
            if observable is None:
                errors.append(
                    f"Semantic state field `{state_field.field_name}` references unknown observable `{source}`."
                )
                continue
            observable_index = phase_index.get(observable.availability_phase)
            if observable_index is None:
                continue
            if observable_index > cutoff_index:
                errors.append(
                    f"Semantic state field `{state_field.field_name}` future-peeks observable `{source}` beyond `{cutoff_phase}`."
                )

        if state_field.description.startswith("Legacy-derived typed state field"):
            warnings.append(
                "Semantic contract normalized legacy state_variables into typed state_fields."
            )


def _validate_controller_protocol(
    product,
    phase_index: dict[str, int],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate that strategic rights use controller protocol and triggers do not."""
    protocol = product.controller_protocol
    if protocol.controller_style not in _ALLOWED_CONTROLLER_STYLES:
        errors.append(
            f"Semantic controller protocol uses unsupported controller_style `{protocol.controller_style}`."
        )
    if protocol.decision_phase not in phase_index:
        errors.append(
            f"Semantic controller protocol uses unsupported decision phase `{protocol.decision_phase}`."
        )
    if (
        product.exercise_style not in {"", "none"}
        and protocol.controller_style == "identity"
    ):
        warnings.append(
            "Semantic contract still leaves strategic rights implicit; add a typed controller protocol instead of relying on legacy exercise semantics."
        )
    if protocol.controller_style == "identity":
        return

    automatic_actions = tuple(
        action for action in protocol.admissible_actions
        if _looks_like_automatic_action(action)
    )
    if product.exercise_style in {"", "none"} or protocol.controller_role in {"", "none"}:
        errors.append(
            "Semantic automatic trigger represented as control: controller protocol should only encode strategic rights."
        )
    elif protocol.admissible_actions and len(automatic_actions) == len(protocol.admissible_actions):
        errors.append(
            "Semantic automatic trigger represented as control: automatic transitions should stay in event/state machinery."
        )


def _validate_obligations(
    product,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate typed obligation emission for settlement-bearing products."""
    obligations = tuple(product.obligations)
    settlement_rules = tuple(
        rule
        for rule in (product.settlement_rule, product.maturity_settlement_rule)
        if rule
    )
    if settlement_rules and not obligations:
        errors.append(
            "Settlement-bearing semantic contracts must emit at least one typed obligation."
        )
        return
    if not obligations:
        return

    settle_rules = {obligation.settle_date_rule for obligation in obligations if obligation.settle_date_rule}
    for obligation in obligations:
        if not obligation.obligation_id:
            errors.append("Typed obligations must define obligation_id.")
        if not obligation.settle_date_rule:
            errors.append(
                f"Typed obligation `{obligation.obligation_id or '<unknown>'}` must define settle_date_rule."
            )
        if not obligation.amount_expression:
            errors.append(
                f"Typed obligation `{obligation.obligation_id or '<unknown>'}` must define amount_expression."
            )
    if settlement_rules and settle_rules and not any(rule in settle_rules for rule in settlement_rules):
        warnings.append(
            "Typed obligations are present but do not mirror the legacy settlement_rule exactly."
        )


def _typed_surface_is_authoritative_for_settlement_rule(contract: SemanticContract) -> bool:
    """Return whether typed obligations/timeline are authoritative for settlement on this slice."""
    product = contract.product
    instrument = str(getattr(product, "instrument_class", "")).strip().lower()
    payoff_family = str(getattr(product, "payoff_family", "")).strip().lower()
    exercise_style = str(getattr(product, "exercise_style", "")).strip().lower()
    payoff_traits = {
        str(item).strip().lower()
        for item in getattr(product, "payoff_traits", ()) or ()
    }
    primitive_families = {
        str(item).strip().lower()
        for item in getattr(contract.blueprint, "primitive_families", ()) or ()
    }
    profile = _SETTLEMENT_AUTHORITY_PROFILES.get(str(contract.semantic_id or "").strip())
    if profile is None:
        return False
    if profile.instrument_classes and instrument not in profile.instrument_classes:
        return False
    if profile.payoff_families and payoff_family not in profile.payoff_families:
        return False
    if profile.exercise_styles and exercise_style not in profile.exercise_styles:
        return False
    if profile.required_payoff_traits and not set(profile.required_payoff_traits).issubset(payoff_traits):
        return False
    if profile.required_primitive_families and not set(profile.required_primitive_families).issubset(primitive_families):
        return False
    return True


def _typed_settlement_rules(product) -> tuple[str, ...]:
    """Return typed settlement rules emitted by obligations, deduplicated in order."""
    rules: list[str] = []
    for obligation in getattr(product, "obligations", ()) or ():
        rule = str(getattr(obligation, "settle_date_rule", "")).strip()
        if rule and rule not in rules:
            rules.append(rule)
    return tuple(rules)


def _validate_legacy_typed_normalization(
    product,
    warnings: list[str],
) -> None:
    """Warn when validation had to rely on legacy semantic mirrors."""
    if (
        product.event_machine is not None
        and product.event_transitions
        and not product.implementation_hints.event_machine_source
    ):
        warnings.append(
            "Semantic contract normalized legacy event_transitions into a typed event_machine."
        )


def _looks_like_automatic_action(action: str) -> bool:
    """Return whether an admissible action looks automatic rather than strategic."""
    lower = str(action).strip().lower()
    return any(hint in lower for hint in _AUTOMATIC_ACTION_HINTS)


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
            normalized = capability_registry.normalize_capability_name(input_spec.capability)
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


def _validate_calibration(
    contract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate the calibration contract on the semantic contract, if present."""
    calibration = getattr(contract, "calibration", None)
    if calibration is None:
        return
    try:
        from trellis.agent.calibration_contract import validate_calibration_contract
        cal_errors = validate_calibration_contract(calibration)
        for err in cal_errors:
            errors.append(f"calibration: {err}")
    except Exception as exc:
        warnings.append(f"Could not validate calibration contract: {exc}")


def _validate_event_machine(
    contract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate the event machine on the contract product, if present."""
    product = getattr(contract, "product", None)
    if product is None:
        return
    machine = getattr(product, "event_machine", None)
    state_dep = getattr(product, "state_dependence", "terminal_markov")

    if machine is not None:
        try:
            from trellis.agent.event_machine import validate_event_machine
            machine_errors = validate_event_machine(machine)
            for err in machine_errors:
                errors.append(f"event_machine: {err}")
        except Exception as exc:
            warnings.append(f"Could not validate event machine: {exc}")
    elif state_dep not in ("none", "terminal_markov") and not getattr(product, "event_transitions", ()):
        warnings.append(
            f"Product has state_dependence='{state_dep}' but no event_machine "
            f"or event_transitions declared"
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
        "american_option": _validate_american_option_shape,
        "quanto_option": _validate_quanto_option_shape,
        "callable_bond": _validate_callable_bond_shape,
        "range_accrual": _validate_range_accrual_shape,
        "rate_style_swaption": _validate_rate_style_swaption_shape,
        "rate_cap_floor_strip": _validate_rate_cap_floor_strip_shape,
        "credit_default_swap": _validate_credit_default_swap_shape,
        "nth_to_default": _validate_nth_to_default_shape,
        "credit_basket_tranche": _validate_credit_basket_tranche_shape,
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
        capability_registry.normalize_capability_name(item.capability or item.input_id)
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
    expected_instrument_class: str | None,
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
    if expected_instrument_class is not None and product.instrument_class != expected_instrument_class:
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
        typed_rules = _typed_settlement_rules(product)
        if not (
            _typed_surface_is_authoritative_for_settlement_rule(contract)
            and expected_settlement_rule in typed_rules
        ):
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


def _validate_american_option_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a single-underlier holder-controlled vanilla option shape."""
    required_capabilities = _validate_market_capabilities(
        contract,
        errors,
        frozenset({"discount_curve", "spot", "black_vol_surface"}),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="american_option",
        expected_payoff_family="vanilla_option",
        expected_underlier_structure="single_underlier",
        expected_payoff_rule="vanilla_option_payoff",
        expected_settlement_rule="cash_settle_at_expiry",
        expected_multi_asset=False,
        require_schedule=True,
        require_constituents=1,
        path_dependence="terminal_markov",
        state_dependence="terminal_markov",
    )
    if contract.product.exercise_style not in {"american", "bermudan"}:
        errors.append(
            f"American-option semantics require exercise_style `american` or `bermudan`, got `{contract.product.exercise_style}`."
        )
    if contract.product.exercise_style == "bermudan" and not contract.product.schedule_dependence:
        errors.append("Bermudan option semantics require schedule_dependence=True.")
    if contract.product.controller_protocol.controller_style != "holder_max":
        errors.append("American-option semantics require holder_max controller protocol.")
    if "spot" not in required_capabilities:
        errors.append("American-option semantics require a spot underlier input.")
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


def _validate_range_accrual_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate the first range-accrual trade-entry semantic shape."""
    required_capabilities = _validate_market_capabilities(
        contract,
        errors,
        frozenset({"discount_curve", "forward_curve"}),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="range_accrual",
        expected_payoff_family="range_accrual_coupon",
        expected_underlier_structure="single_curve_rate_style",
        expected_payoff_rule="range_accrual_coupon_payment",
        expected_settlement_rule="coupon_period_cash_settlement",
        expected_exercise_style="none",
        expected_multi_asset=False,
        require_schedule=True,
        require_constituents=1,
        path_dependence="schedule_dependent",
        state_dependence="schedule_dependent",
        schedule_dependence=True,
    )
    if "forward_curve" not in required_capabilities:
        errors.append("Range-accrual semantics require a forward curve input.")
    required_input_ids = {item.input_id for item in contract.market_data.required_inputs}
    if "fixing_history" not in required_input_ids:
        errors.append("Range-accrual semantics require a fixing_history input.")

    product = contract.product
    term_fields = dict(getattr(product, "term_fields", {}) or {})
    reference_index = str(term_fields.get("reference_index", "")).strip()
    if not reference_index:
        errors.append("Range-accrual semantics require term_fields.reference_index.")

    coupon_definition = dict(term_fields.get("coupon_definition") or {})
    coupon_rate = coupon_definition.get("coupon_rate")
    if coupon_rate is None:
        errors.append("Range-accrual semantics require term_fields.coupon_definition.coupon_rate.")
    if not str(coupon_definition.get("coupon_style", "")).strip():
        errors.append("Range-accrual semantics require term_fields.coupon_definition.coupon_style.")

    range_condition = dict(term_fields.get("range_condition") or {})
    lower_bound = range_condition.get("lower_bound")
    upper_bound = range_condition.get("upper_bound")
    if lower_bound is None or upper_bound is None:
        errors.append("Range-accrual semantics require lower and upper accrual bounds.")
    elif float(lower_bound) > float(upper_bound):
        errors.append("Range-accrual semantics require lower_bound <= upper_bound.")

    settlement_profile = dict(term_fields.get("settlement_profile") or {})
    if not str(settlement_profile.get("coupon_settlement", "")).strip():
        errors.append("Range-accrual semantics require term_fields.settlement_profile.coupon_settlement.")
    if not str(settlement_profile.get("principal_settlement", "")).strip():
        errors.append("Range-accrual semantics require term_fields.settlement_profile.principal_settlement.")

    callability = dict(term_fields.get("callability") or {})
    if callability and not callability.get("call_schedule"):
        errors.append("Range-accrual callability hooks must include call_schedule when present.")

    observable_types = {
        str(getattr(item, "observable_type", "")).strip().lower()
        for item in product.observables
    }
    if "forward_rate" not in observable_types:
        errors.append("Range-accrual semantics require a typed forward_rate observable.")
    if "cashflow_schedule" not in observable_types:
        errors.append("Range-accrual semantics require a typed cashflow_schedule observable.")

    obligation_ids = {item.obligation_id for item in product.obligations}
    if "coupon_period_cashflow" not in obligation_ids:
        errors.append("Range-accrual semantics require a typed coupon-period obligation.")
    if "principal_repayment" not in obligation_ids:
        errors.append("Range-accrual semantics require a typed principal repayment obligation.")
    if product.controller_protocol.controller_style != "identity":
        errors.append("Range-accrual semantics cannot declare a strategic controller in the first slice.")
    if callability:
        warnings.append(
            "Range-accrual callability hooks are captured as trade-entry metadata; callable execution remains a later slice."
        )
    if not contract.blueprint.primitive_families:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit primitive-family hint."
        )


def _validate_rate_cap_floor_strip_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a schedule-driven cap/floor strip semantic shape."""
    required_capabilities = _validate_market_capabilities(
        contract,
        errors,
        frozenset({"discount_curve", "forward_curve", "black_vol_surface"}),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class=None,
        expected_payoff_family="rate_cap_floor_strip",
        expected_underlier_structure="single_curve_rate_style",
        expected_payoff_rule="period_rate_option_strip_payoff",
        expected_settlement_rule="coupon_period_cash_settlement",
        expected_exercise_style="none",
        expected_multi_asset=False,
        require_schedule=True,
        require_constituents=0,
        path_dependence="schedule_dependent",
        state_dependence="schedule_dependent",
        schedule_dependence=True,
    )
    product = contract.product
    wrapper_profile = _RATE_CAP_FLOOR_WRAPPER_PROFILES.get(
        str(product.instrument_class or "").strip().lower()
    )
    if wrapper_profile is None:
        errors.append(
            "Rate cap/floor strip semantics require instrument_class `cap` or `floor`."
        )
    if "forward_curve" not in required_capabilities:
        errors.append("Rate cap/floor strip semantics require a forward curve input.")
    observable_types = {
        str(getattr(item, "observable_type", "")).strip().lower()
        for item in product.observables
    }
    if "forward_rate" not in observable_types:
        errors.append("Rate cap/floor strip semantics require a typed forward_rate observable.")
    if "discount_curve" not in observable_types:
        errors.append("Rate cap/floor strip semantics require a typed discount_curve observable.")
    obligation_ids = {item.obligation_id for item in product.obligations}
    expected_obligation_id = str((wrapper_profile or {}).get("obligation_id", ""))
    if expected_obligation_id not in obligation_ids:
        errors.append(
            f"Rate cap/floor strip semantics require obligation `{expected_obligation_id}`."
        )
    term_fields = dict(getattr(product, "term_fields", {}) or {})
    option_type = str(term_fields.get("option_type", "")).strip().lower()
    expected_option_type = str((wrapper_profile or {}).get("option_type", ""))
    if option_type != expected_option_type:
        errors.append(
            f"Rate cap/floor strip semantics require option_type `{expected_option_type}`, got `{option_type}`."
        )
    if product.controller_protocol.controller_style != "identity":
        errors.append("Rate cap/floor strip semantics cannot declare a strategic controller.")
    if not contract.blueprint.primitive_families:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit primitive-family hint."
        )


def _validate_credit_default_swap_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a single-name CDS semantic shape."""
    required_capabilities = _validate_market_capabilities(
        contract,
        errors,
        frozenset({"discount_curve", "credit_curve"}),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="cds",
        expected_payoff_family="credit_default_swap",
        expected_underlier_structure="single_reference_entity",
        expected_payoff_rule="single_name_cds_legs",
        expected_settlement_rule="premium_schedule_and_default_settlement",
        expected_exercise_style="none",
        expected_multi_asset=False,
        require_schedule=True,
        require_constituents=0,
        path_dependence="schedule_dependent",
        state_dependence="schedule_dependent",
        schedule_dependence=True,
    )
    if "credit_curve" not in required_capabilities:
        errors.append("Credit-default-swap semantics require a credit curve input.")
    product = contract.product
    observable_types = {
        str(getattr(item, "observable_type", "")).strip().lower()
        for item in product.observables
    }
    if "credit_curve" not in observable_types:
        errors.append("Credit-default-swap semantics require a typed credit_curve observable.")
    if "cashflow_schedule" not in observable_types:
        errors.append("Credit-default-swap semantics require a typed cashflow_schedule observable.")
    obligation_ids = {item.obligation_id for item in product.obligations}
    if "premium_leg_cashflow" not in obligation_ids:
        errors.append("Credit-default-swap semantics require a typed premium-leg obligation.")
    if "protection_leg_cashflow" not in obligation_ids:
        errors.append("Credit-default-swap semantics require a typed protection-leg obligation.")
    if product.controller_protocol.controller_style != "identity":
        errors.append("Credit-default-swap semantics cannot declare a strategic controller.")
    if not contract.blueprint.primitive_families:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit primitive-family hint."
        )


def _validate_nth_to_default_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate an nth-to-default basket-credit semantic shape."""
    required_capabilities = _validate_market_capabilities(
        contract,
        errors,
        frozenset({"discount_curve", "credit_curve"}),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="nth_to_default",
        expected_payoff_family="nth_to_default",
        expected_underlier_structure="multi_asset_basket",
        expected_payoff_rule="nth_default_loss_payment",
        expected_settlement_rule="settle_at_nth_default_or_maturity",
        expected_exercise_style="none",
        expected_multi_asset=True,
        require_schedule=True,
        require_constituents=2,
        path_dependence="path_dependent",
        state_dependence="path_dependent",
        schedule_dependence=True,
    )
    if "credit_curve" not in required_capabilities:
        errors.append("Nth-to-default semantics require a credit curve input.")
    product = contract.product
    observable_types = {
        str(getattr(item, "observable_type", "")).strip().lower()
        for item in product.observables
    }
    if "credit_curve" not in observable_types:
        errors.append("Nth-to-default semantics require a typed credit_curve observable.")
    obligation_ids = {item.obligation_id for item in product.obligations}
    if "nth_default_cash_settlement" not in obligation_ids:
        errors.append("Nth-to-default semantics require a typed nth-default settlement obligation.")
    if product.controller_protocol.controller_style != "identity":
        errors.append("Nth-to-default semantics cannot declare a strategic controller.")
    if product.selection_count < 1:
        errors.append("Nth-to-default semantics require trigger_rank >= 1.")
    if product.selection_count > len(product.constituents):
        errors.append("Nth-to-default trigger_rank cannot exceed the reference-entity pool.")
    if not contract.blueprint.primitive_families:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit primitive-family hint."
        )


def _validate_credit_basket_tranche_shape(
    contract: SemanticContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a tranche-style basket-credit semantic shape."""
    required_capabilities = _validate_market_capabilities(
        contract,
        errors,
        frozenset({"discount_curve", "credit_curve"}),
    )
    _validate_profile_fields(
        contract,
        errors,
        expected_instrument_class="cdo",
        expected_payoff_family="credit_basket_tranche",
        expected_underlier_structure="multi_asset_basket",
        expected_payoff_rule="tranche_loss_payment",
        expected_settlement_rule="settle_expected_tranche_loss_at_maturity",
        expected_exercise_style="none",
        expected_multi_asset=True,
        require_schedule=True,
        require_constituents=2,
        path_dependence="path_dependent",
        state_dependence="path_dependent",
        schedule_dependence=True,
    )
    if "credit_curve" not in required_capabilities:
        errors.append("Credit-basket tranche semantics require a credit curve input.")
    product = contract.product
    observable_types = {
        str(getattr(item, "observable_type", "")).strip().lower()
        for item in product.observables
    }
    if "credit_curve" not in observable_types:
        errors.append("Credit-basket tranche semantics require a typed credit_curve observable.")
    obligation_ids = {item.obligation_id for item in product.obligations}
    if "tranche_loss_cash_settlement" not in obligation_ids:
        errors.append(
            "Credit-basket tranche semantics require a typed tranche-loss settlement obligation."
        )
    if product.controller_protocol.controller_style != "identity":
        errors.append("Credit-basket tranche semantics cannot declare a strategic controller.")

    term_fields = dict(getattr(product, "term_fields", {}) or {})
    reference_pool_size = int(term_fields.get("reference_pool_size") or 0)
    attachment = term_fields.get("attachment")
    detachment = term_fields.get("detachment")
    if reference_pool_size < 2:
        errors.append("Credit-basket tranche semantics require reference_pool_size >= 2.")
    if reference_pool_size and reference_pool_size != len(product.constituents):
        errors.append(
            "Credit-basket tranche reference_pool_size must match the constituent count."
        )
    if attachment is None or detachment is None:
        errors.append("Credit-basket tranche semantics require attachment and detachment.")
    else:
        attachment_value = float(attachment)
        detachment_value = float(detachment)
        if not 0.0 <= attachment_value < detachment_value <= 1.0:
            errors.append(
                "Credit-basket tranche semantics require 0 <= attachment < detachment <= 1."
            )
    if product.selection_count != len(product.constituents):
        errors.append(
            "Credit-basket tranche semantics require selection_count to equal the reference pool size."
        )
    if "cumulative_portfolio_loss_fraction" not in product.state_variables:
        errors.append(
            "Credit-basket tranche semantics require cumulative_portfolio_loss_fraction state."
        )
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
        expected_exercise_style=None,
        expected_multi_asset=False,
        require_schedule=True,
        require_constituents=0,
        path_dependence="schedule_dependent",
        state_dependence="schedule_dependent",
        schedule_dependence=True,
    )
    if contract.product.exercise_style not in {"european", "bermudan"}:
        errors.append(
            "Rate-style swaption semantics require exercise_style `european` or `bermudan`."
        )
    if not contract.blueprint.primitive_families:
        warnings.append(
            f"Semantic contract `{contract.semantic_id}` has no explicit primitive-family hint."
        )
