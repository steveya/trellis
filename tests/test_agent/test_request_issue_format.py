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
            "semantic_role_ownership": {
                "selected_stage": "route_assembly",
                "selected_role": "quant",
                "trigger_condition": "pricing_plan_selection",
                "artifact_kind": "GenerationPlan",
                "executed": True,
                "fail_closed": False,
                "scope": "bounded_semantic_to_pricing_assembly",
            },
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
    assert "## Ownership" in body
    assert "- selected_role: `quant`" in body
    assert "- trigger_condition: `pricing_plan_selection`" in body
    assert "- trigger_event: `request_failed`" in body
    assert "- comparison_target: `local_vol_pde`" in body
    assert "- failure_count: `2`" in body
    assert "- task: `E23` — European equity call under local vol: PDE vs MC" in comment
    assert "- route_method: `pde_solver`" in comment
    assert "- ownership_role: `quant`" in comment


def test_stress_follow_on_format_includes_repeat_context_and_artifacts():
    from trellis.agent.request_issue_format import (
        build_stress_follow_on_body,
        build_stress_follow_on_title,
    )

    summary = {
        "task_id": "E23",
        "task_title": "European equity call under local vol: PDE vs MC",
        "outcome_class": "honest_block",
        "failure_bucket": "blocked",
        "comparison_status": "insufficient_results",
        "observed_blocker_categories": [
            "missing_foundational_primitive",
            "missing_binding_surface",
        ],
        "repeat_count": 3,
        "signature": "E23::blocked::missing_foundational_primitive,missing_binding_surface",
        "task_run_latest_path": "/tmp/task_runs/latest/E23.json",
        "task_run_history_path": "/tmp/task_runs/history/E23/run.json",
        "diagnosis_dossier_path": "/tmp/task_runs/diagnostics/latest/E23.md",
        "diagnosis_packet_path": "/tmp/task_runs/diagnostics/latest/E23.json",
        "linked_linear_issues": ["QUA-500"],
    }

    title = build_stress_follow_on_title(summary)
    body = build_stress_follow_on_body(summary)

    assert title.startswith("Connector stress: E23")
    assert "missing foundational primitive" in title
    assert "- source: `connector_stress_gate`" in body
    assert "- repeat_count: `3`" in body
    assert "- blocker_categories: `missing_foundational_primitive, missing_binding_surface`" in body
    assert "- linked_linear_issues: `QUA-500`" in body
    assert "/tmp/task_runs/diagnostics/latest/E23.md" in body
