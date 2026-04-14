"""Runtime revision metadata helpers for persisted task and benchmark records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def runtime_revision_metadata(result: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return the best available repo and knowledge revisions for one run."""
    environment = _environment_payload(result)
    git_sha = str(environment.get("repo_revision") or "").strip() or _safe_repo_revision()
    knowledge_revision = (
        str(environment.get("knowledge_hash") or environment.get("knowledge_revision") or "").strip()
        or _safe_knowledge_revision()
    )
    return {
        "git_sha": git_sha or "unknown",
        "knowledge_revision": knowledge_revision or "unknown",
    }


def _environment_payload(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {}

    for container_key in ("framework_result", "framework"):
        container = result.get(container_key)
        if not isinstance(container, Mapping):
            continue
        environment = container.get("environment")
        if isinstance(environment, Mapping):
            return dict(environment)
    return {}


def _safe_repo_revision() -> str:
    try:
        from trellis.agent.knowledge.import_registry import get_repo_revision

        return str(get_repo_revision() or "").strip()
    except Exception:
        return ""


def _safe_knowledge_revision() -> str:
    try:
        from trellis.agent.knowledge import get_store

        return str(get_store().compute_knowledge_hash() or "").strip()
    except Exception:
        return ""
