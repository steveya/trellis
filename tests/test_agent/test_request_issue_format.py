from __future__ import annotations

from types import SimpleNamespace


def test_issue_format_includes_task_and_target_context():
    from trellis.agent.platform_requests import PlatformRequest
    from trellis.agent.request_issue_format import (
        build_event_comment,
        build_issue_body,
        build_issue_title,
    )

    request = PlatformRequest(
        request_id="executor_build_20260325_deadbeef",
        request_type="build",
        entry_point="executor",
        description="Build a pricer for: European equity call under local vol: PDE vs MC",
        instrument_type="european_option",
        metadata={
            "task_id": "E23",
            "task_title": "European equity call under local vol: PDE vs MC",
            "comparison_target": "local_vol_pde",
            "preferred_method": "pde_solver",
        },
    )
    compiled = SimpleNamespace(request=request)
    trace = {
        "action": "compile_only",
        "route_method": "pde_solver",
        "requires_build": True,
        "outcome": "request_failed",
        "events": [
            {
                "timestamp": "2026-03-25T17:56:16+00:00",
                "event": "build_started",
                "status": "info",
            }
        ],
        "request_metadata": dict(request.metadata),
    }
    event = {
        "event": "request_failed",
        "status": "error",
        "details": {"reason": "semantic_validation", "failure_count": 2},
    }

    title = build_issue_title(trace, compiled)
    body = build_issue_body(trace, compiled, event)
    comment = build_event_comment(trace, event)

    assert title.startswith("Trellis audit: E23")
    assert "local_vol_pde" in title
    assert "pde_solver" in title
    assert "- task: `E23` — European equity call under local vol: PDE vs MC" in body
    assert "- trigger_event: `request_failed`" in body
    assert "- comparison_target: `local_vol_pde`" in body
    assert "- failure_count: `2`" in body
    assert "- task: `E23` — European equity call under local vol: PDE vs MC" in comment
    assert "- route_method: `pde_solver`" in comment
