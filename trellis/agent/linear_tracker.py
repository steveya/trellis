"""Best-effort Linear issue sync for request audit records."""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

from trellis.agent.config import load_env
from trellis.agent.request_issue_format import (
    build_event_comment,
    build_issue_body,
    build_issue_title,
)


LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
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
    """Create/update a Linear issue for notable request events."""
    settings = _settings()
    if not settings["enabled"]:
        return None

    event = event_record.get("event", "")
    current_issue = trace.get("linear_issue") or {}
    if not current_issue and event not in _TRACKED_EVENTS:
        return None

    issue_ref = dict(current_issue)
    if not issue_ref:
        issue_ref = _create_issue(compiled_request, trace, event_record, settings)
    if not issue_ref:
        return None

    if event in _COMMENT_EVENTS:
        _create_comment(issue_ref["id"], build_event_comment(trace, event_record), settings)
    return issue_ref


def _settings() -> dict[str, Any]:
    """Read Linear issue-sync configuration from the environment."""
    load_env()
    api_key = os.environ.get("LINEAR_API_KEY", "").strip()
    team_ref = (
        os.environ.get("LINEAR_TEAM_ID", "").strip()
        or os.environ.get("LINEAR_REQUEST_AUDIT_TEAM_ID", "").strip()
    )
    return {
        "enabled": bool(api_key and team_ref),
        "api_key": api_key,
        "team_ref": team_ref,
        "project_id": os.environ.get("LINEAR_REQUEST_AUDIT_PROJECT_ID", "").strip() or None,
    }


def _create_issue(
    compiled_request,
    trace: dict[str, Any],
    event_record: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    """Create the initial Linear issue representing a traced platform request."""
    title = build_issue_title(trace, compiled_request)
    description = build_issue_body(trace, compiled_request, event_record)
    team_id = _resolve_team_id(settings)
    if not team_id:
        return None
    variables = {
        "input": {
            "teamId": team_id,
            "title": title,
            "description": description,
        }
    }
    if settings["project_id"]:
        variables["input"]["projectId"] = settings["project_id"]

    payload = _graphql(
        settings["api_key"],
        """
        mutation IssueCreate($input: IssueCreateInput!) {
          issueCreate(input: $input) {
            success
            issue {
              id
              identifier
              url
            }
          }
        }
        """,
        variables,
    )
    issue = ((payload.get("issueCreate") or {}).get("issue")) or {}
    if not issue.get("id"):
        return None
    return {
        "id": issue["id"],
        "identifier": issue.get("identifier"),
        "url": issue.get("url"),
    }


def _resolve_team_id(settings: dict[str, Any]) -> str | None:
    """Resolve a Linear team key or UUID-like reference to the concrete team id."""
    team_ref = settings["team_ref"]
    if _is_uuid(team_ref):
        return team_ref

    payload = _graphql(
        settings["api_key"],
        """
        query TeamByKey($key: String!) {
          teams(filter: { key: { eq: $key } }) {
            nodes {
              id
            }
          }
        }
        """,
        {"key": team_ref},
    )
    nodes = ((payload.get("teams") or {}).get("nodes")) or []
    if not nodes:
        raise RuntimeError(f"Linear team '{team_ref}' not found")
    return nodes[0].get("id")


def _create_comment(issue_id: str, body: str, settings: dict[str, Any]) -> None:
    """Append one lifecycle event comment to the tracked Linear issue."""
    _graphql(
        settings["api_key"],
        """
        mutation CommentCreate($input: CommentCreateInput!) {
          commentCreate(input: $input) {
            success
            comment {
              id
            }
          }
        }
        """,
        {
            "input": {
                "issueId": issue_id,
                "body": body,
            }
        },
    )


def _graphql(api_key: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Execute a Linear GraphQL request and return its ``data`` payload."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for Linear sync") from exc

    response = requests.post(
        LINEAR_GRAPHQL_URL,
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
        },
        json={"query": query, "variables": variables},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"]))
    return payload.get("data") or {}


def _is_uuid(value: str) -> bool:
    """Return whether ``value`` parses as a UUID."""
    try:
        UUID(value)
        return True
    except Exception:
        return False
