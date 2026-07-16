"""Typed policy and evidence for task artifact generation provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class GenerationPolicy(str, Enum):
    """Controls whether deterministic source construction may satisfy a build."""

    DETERMINISTIC_ALLOWED = "deterministic_allowed"
    BUILDER_SYNTHESIS_REQUIRED = "builder_synthesis_required"


class ArtifactOrigin(str, Enum):
    """Identifies how the executed pricing artifact was produced."""

    NONE = "none"
    REUSED_ADAPTER = "reused_adapter"
    SEMANTIC_SHIM = "semantic_shim"
    DETERMINISTIC_MATERIALIZATION = "deterministic_materialization"
    MODEL_GENERATED_SOURCE = "model_generated_source"
    DETERMINISTIC_PROOF_ARTIFACT = "deterministic_proof_artifact"


class GenerationPolicyError(RuntimeError):
    """Raised when the requested proving policy cannot be satisfied safely."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class GenerationEvidence:
    """Stable evidence separating artifact origin from path freshness."""

    policy: GenerationPolicy
    artifact_origin: ArtifactOrigin = ArtifactOrigin.NONE
    agent_synthesis_attempted: bool = False
    agent_synthesis_observed: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Return the machine-readable evidence shape persisted by task runs."""
        return {
            "policy": self.policy.value,
            "artifact_origin": self.artifact_origin.value,
            "agent_synthesis_attempted": self.agent_synthesis_attempted,
            "agent_synthesis_observed": self.agent_synthesis_observed,
        }


def normalize_generation_policy(
    value: str | GenerationPolicy,
) -> GenerationPolicy:
    """Normalize a public policy value and reject unknown policy names."""
    if isinstance(value, GenerationPolicy):
        return value
    try:
        return GenerationPolicy(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in GenerationPolicy)
        raise ValueError(
            f"Unknown generation policy {value!r}; expected one of: {allowed}"
        ) from exc


def generation_evidence_payload(
    *,
    policy: str | GenerationPolicy,
    artifact_origin: str | ArtifactOrigin = ArtifactOrigin.NONE,
    agent_synthesis_attempted: bool = False,
    agent_synthesis_observed: bool = False,
) -> dict[str, Any]:
    """Build one normalized evidence payload."""
    normalized_origin = (
        artifact_origin
        if isinstance(artifact_origin, ArtifactOrigin)
        else ArtifactOrigin(str(artifact_origin).strip().lower())
    )
    return GenerationEvidence(
        policy=normalize_generation_policy(policy),
        artifact_origin=normalized_origin,
        agent_synthesis_attempted=bool(agent_synthesis_attempted),
        agent_synthesis_observed=bool(agent_synthesis_observed),
    ).to_payload()


def record_generation_evidence(
    build_meta: dict[str, Any] | None,
    *,
    policy: str | GenerationPolicy,
    artifact_origin: str | ArtifactOrigin | None = None,
    agent_synthesis_attempted: bool | None = None,
    agent_synthesis_observed: bool | None = None,
) -> dict[str, Any]:
    """Update build metadata while preserving evidence not changed by this stage."""
    current = dict((build_meta or {}).get("generation_evidence") or {})
    payload = generation_evidence_payload(
        policy=policy,
        artifact_origin=(
            artifact_origin
            if artifact_origin is not None
            else current.get("artifact_origin", ArtifactOrigin.NONE.value)
        ),
        agent_synthesis_attempted=(
            bool(agent_synthesis_attempted)
            if agent_synthesis_attempted is not None
            else bool(current.get("agent_synthesis_attempted"))
        ),
        agent_synthesis_observed=(
            bool(agent_synthesis_observed)
            if agent_synthesis_observed is not None
            else bool(current.get("agent_synthesis_observed"))
        ),
    )
    if build_meta is not None:
        build_meta["generation_evidence"] = payload
    return payload


def generation_evidence_from_result(
    result: object,
    *,
    default_policy: str | GenerationPolicy = GenerationPolicy.DETERMINISTIC_ALLOWED,
) -> dict[str, Any]:
    """Read normalized generation evidence from a BuildResult-like object."""
    raw = getattr(result, "generation_evidence", None)
    if not isinstance(raw, Mapping):
        return generation_evidence_payload(policy=default_policy)
    return generation_evidence_payload(
        policy=raw.get("policy", default_policy),
        artifact_origin=raw.get("artifact_origin", ArtifactOrigin.NONE.value),
        agent_synthesis_attempted=bool(raw.get("agent_synthesis_attempted")),
        agent_synthesis_observed=bool(raw.get("agent_synthesis_observed")),
    )


def validate_generation_policy_request(
    *,
    policy: str | GenerationPolicy,
    fresh_build: bool,
    recovery_mode: str,
    execution_mode: str,
) -> None:
    """Fail closed when task controls cannot prove required builder synthesis."""
    normalized = normalize_generation_policy(policy)
    if normalized is GenerationPolicy.DETERMINISTIC_ALLOWED:
        return
    if not fresh_build:
        raise GenerationPolicyError(
            "Builder synthesis requires fresh_build=True so model source is isolated from admitted adapters.",
            reason="fresh_build_required",
        )
    normalized_recovery = str(recovery_mode).strip().lower()
    if normalized_recovery == "strict":
        raise GenerationPolicyError(
            "Builder synthesis is disabled in strict recovery mode.",
            reason="recovery_mode_strict",
        )
    normalized_execution = str(execution_mode).strip().lower()
    if normalized_execution not in {"live", "cassette_record"}:
        raise GenerationPolicyError(
            "Builder synthesis is unavailable in execution mode "
            f"{normalized_execution or '<empty>'!r}.",
            reason=f"execution_mode_{normalized_execution or 'unknown'}",
        )
