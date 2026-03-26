"""Best-effort GitHub issue sync for request audit records."""

from __future__ import annotations

import os
from typing import Any

from trellis.agent.config import load_env
from trellis.agent.request_issue_format import (
    build_event_comment,
    build_issue_body,
    build_issue_title,
)


GITHUB_API_ROOT = "https://api.github.com"
_TRACKED_EVENTS = {
    "build_started",
    "request_blocked",
    "request_failed",
}
_COMMENT_EVENTS = {
    "build_started",
    "quant_selected_method",
    "planner_completed",
    "builder_attempt_started",
    "builder_attempt_failed",
    "builder_attempt_succeeded",
    "critic_completed",
    "arbiter_completed",
    "model_validator_completed",
    "model_validator_skipped",
    "build_completed",
    "existing_generated_module_reused",
    "request_blocked",
    "request_failed",
    "request_succeeded",
}


def sync_request_issue(
    trace: dict[str, Any],
    compiled_request,
    event_record: dict[str, Any],
) -> dict[str, Any] | None:
    """Create/update a GitHub issue for notable request events."""
    settings = _settings()
    if not settings["enabled"]:
        return None

    event = event_record.get("event", "")
    current_issue = trace.get("github_issue") or {}
    if not current_issue and event not in _TRACKED_EVENTS:
        return None

    issue_ref = dict(current_issue)
    if not issue_ref:
        issue_ref = _create_issue(compiled_request, trace, event_record, settings)
    if not issue_ref:
        return None

    if event in _COMMENT_EVENTS:
        _create_comment(issue_ref["number"], build_event_comment(trace, event_record), settings)
    return issue_ref


def _settings() -> dict[str, Any]:
    """Read GitHub issue-sync configuration from the environment."""
    load_env()
    token = (
        os.environ.get("GITHUB_REQUEST_AUDIT_TOKEN", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )
    repository = (
        os.environ.get("GITHUB_REQUEST_AUDIT_REPOSITORY", "").strip()
        or os.environ.get("GITHUB_REPOSITORY", "").strip()
    )
    labels = tuple(
        label.strip()
        for label in os.environ.get("GITHUB_REQUEST_AUDIT_LABELS", "").split(",")
        if label.strip()
    )
    return {
        "enabled": bool(token and repository),
        "token": token,
        "repository": repository,
        "labels": labels,
    }


def _create_issue(
    compiled_request,
    trace: dict[str, Any],
    event_record: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    """Create the initial GitHub issue representing a traced platform request."""
    payload = {
        "title": build_issue_title(trace, compiled_request),
        "body": build_issue_body(trace, compiled_request, event_record),
    }
    if settings["labels"]:
        payload["labels"] = list(settings["labels"])

    issue = _request(
        "POST",
        f"/repos/{settings['repository']}/issues",
        settings,
        json=payload,
    )
    if not issue.get("number"):
        return None
    return {
        "id": issue.get("id"),
        "number": issue["number"],
        "url": issue.get("html_url"),
        "repository": settings["repository"],
    }


def _create_comment(issue_number: int, body: str, settings: dict[str, Any]) -> None:
    """Append one event comment to the tracked GitHub issue."""
    _request(
        "POST",
        f"/repos/{settings['repository']}/issues/{issue_number}/comments",
        settings,
        json={"body": body},
    )


def _request(
    method: str,
    path: str,
    settings: dict[str, Any],
    *,
    json: dict[str, Any],
) -> dict[str, Any]:
    """Perform an authenticated GitHub REST request and return the parsed JSON body."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for GitHub sync") from exc

    response = requests.request(
        method,
        f"{GITHUB_API_ROOT}{path}",
        headers={
            "Authorization": f"Bearer {settings['token']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json=json,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()
