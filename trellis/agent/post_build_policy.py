"""Deterministic policy for optional post-build model stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Mapping

from trellis.agent.intra_run_learning import normalize_recovery_mode


_MODEL_ELIGIBLE_EXECUTION_MODES = frozenset({"live", "cassette_record"})
_KNOWN_ARTIFACT_POLICIES = frozenset(
    {"cached_existing", "forced_rebuild", "fresh_generated"}
)
_DETERMINISTIC_EXECUTION_MODES = frozenset(
    {"cassette_replay", "deterministic_replay", "offline_local_agents"}
)


@dataclass(frozen=True)
class PostBuildLearningPolicy:
    """Eligibility and evidence for reflection and consolidation stages."""

    execution_mode: str
    artifact_policy: str
    recovery_mode: str
    run_reflection: bool
    run_consolidation: bool
    reflection_reason: str
    consolidation_reason: str
    skip_reasons: tuple[str, ...] = ()
    policy_source: str = "task_execution_policy"

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe lifecycle payload."""
        payload = asdict(self)
        payload["skip_reasons"] = list(self.skip_reasons)
        return payload


def artifact_policy_for_build(*, force_rebuild: bool, fresh_build: bool) -> str:
    """Classify whether the builder may generate or only reuse an artifact."""
    if fresh_build:
        return "fresh_generated"
    if force_rebuild:
        return "forced_rebuild"
    return "cached_existing"


def determine_post_build_learning_policy(
    *,
    execution_mode: str,
    artifact_policy: str,
    recovery_mode: str,
    policy_source: str = "task_execution_policy",
) -> PostBuildLearningPolicy:
    """Fail closed unless a model-eligible assisted/remediation run admits stages."""
    normalized_execution = _normalize_policy_token(execution_mode, default="live")
    normalized_artifact = _normalize_policy_token(
        artifact_policy,
        default="cached_existing",
    )
    normalized_recovery = normalize_recovery_mode(recovery_mode).value

    skip_reasons: list[str] = []
    if normalized_execution not in _MODEL_ELIGIBLE_EXECUTION_MODES:
        if normalized_execution in _DETERMINISTIC_EXECUTION_MODES:
            skip_reasons.append(f"execution_mode_{normalized_execution}")
        else:
            skip_reasons.append(
                f"execution_mode_{normalized_execution}_not_model_eligible"
            )
    if normalized_artifact not in _KNOWN_ARTIFACT_POLICIES:
        skip_reasons.append(
            f"artifact_policy_{normalized_artifact}_not_model_eligible"
        )
    if normalized_recovery == "strict":
        skip_reasons.append("recovery_mode_strict")

    allowed = not skip_reasons
    reason = (
        f"{normalized_recovery}_model_backed_stage"
        if allowed
        else skip_reasons[0]
    )
    return PostBuildLearningPolicy(
        execution_mode=normalized_execution,
        artifact_policy=normalized_artifact,
        recovery_mode=normalized_recovery,
        run_reflection=allowed,
        run_consolidation=allowed,
        reflection_reason=reason,
        consolidation_reason=reason,
        skip_reasons=tuple(skip_reasons),
        policy_source=str(policy_source or "task_execution_policy"),
    )


def post_build_policy_from_request_metadata(
    request_metadata: Mapping[str, object] | None,
) -> PostBuildLearningPolicy:
    """Resolve and revalidate a policy carried by task request metadata."""
    raw = (
        request_metadata.get("post_build_learning_policy")
        if isinstance(request_metadata, Mapping)
        else None
    )
    if isinstance(raw, Mapping):
        return determine_post_build_learning_policy(
            execution_mode=str(raw.get("execution_mode") or "live"),
            artifact_policy=str(raw.get("artifact_policy") or "cached_existing"),
            recovery_mode=str(raw.get("recovery_mode") or "strict"),
            policy_source=str(raw.get("policy_source") or "task_execution_policy"),
        )

    if _env_flag("TRELLIS_OFFLINE_LOCAL_AGENTS"):
        return determine_post_build_learning_policy(
            execution_mode="deterministic_replay",
            artifact_policy="forced_rebuild",
            recovery_mode="strict",
            policy_source="offline_execution_policy",
        )

    # Direct library callers historically received post-build learning. Keep
    # that behavior explicit while task-runtime callers always carry a policy.
    return determine_post_build_learning_policy(
        execution_mode="live",
        artifact_policy="forced_rebuild",
        recovery_mode="assisted",
        policy_source="standalone_build_default",
    )


def _normalize_policy_token(value: object, *, default: str) -> str:
    text = str(value or default).strip().lower().replace("-", "_")
    return text or default


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


__all__ = [
    "PostBuildLearningPolicy",
    "artifact_policy_for_build",
    "determine_post_build_learning_policy",
    "post_build_policy_from_request_metadata",
]
