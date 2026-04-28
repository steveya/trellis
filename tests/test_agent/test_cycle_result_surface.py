from __future__ import annotations


def _cycle_report(*, success: bool = True, failed_stage: str | None = None) -> dict[str, object]:
    stage_statuses = {
        "quant": "passed",
        "validation_bundle": "passed",
        "critic": "passed",
        "arbiter": "passed",
        "model_validator": "skipped",
    }
    if failed_stage:
        stage_statuses[failed_stage] = "failed"
    return {
        "request_id": "executor_cycle_surface",
        "status": "succeeded" if success else "failed",
        "outcome": "build_completed" if success else "request_failed",
        "success": success,
        "pricing_method": "analytical",
        "validation_contract_id": "validation:vanilla_option:analytical",
        "stage_statuses": stage_statuses,
        "stages": [
            {
                "stage": stage,
                "status": status,
                "event": f"{stage}_completed",
                "summary": f"{stage} {status}",
                "details": {},
            }
            for stage, status in stage_statuses.items()
        ],
        "failure_count": 0 if success else 1,
        "deterministic_blockers": (
            [] if success else [{"source": "arbiter", "check_id": "price_bound"}]
        ),
        "conceptual_blockers": [],
        "calibration_blockers": [],
        "residual_limitations": [{"risk_id": "quant:multiple_valid_methods_available"}],
        "residual_risks": ["quant:multiple_valid_methods_available"],
    }


def test_cycle_result_surface_projects_product_safe_claims():
    from trellis.agent.cycle_governance import evaluate_cycle_promotion_governance
    from trellis.agent.cycle_surface import build_cycle_result_surface

    report = _cycle_report()
    governance = evaluate_cycle_promotion_governance(report).to_dict()

    surface = build_cycle_result_surface(report, promotion_governance=governance)

    assert surface["schema_version"] == "agent_cycle_result.v1"
    assert surface["available"] is True
    assert surface["status"] == "passed"
    assert surface["headline"] == "Governed agent-review cycle passed."
    assert surface["promotion"]["eligible"] is True
    assert surface["evidence_counts"]["residual_limitations"] == 1
    assert surface["stage_statuses"]["model_validator"] == "skipped"
    assert "governed agent-review cycle completed" in surface["claim"]["certifies"]
    assert "external model approval" in surface["claim"]["does_not_certify"]
    assert "regulatory certification" in surface["claim"]["does_not_certify"]


def test_cycle_result_surface_fails_closed_on_blocking_buckets():
    from trellis.agent.cycle_surface import build_cycle_result_surface

    surface = build_cycle_result_surface(
        _cycle_report(success=False, failed_stage="arbiter")
    )

    assert surface["status"] == "failed"
    assert surface["headline"] == "Governed agent-review cycle found blocking issues."
    assert surface["evidence_counts"]["deterministic_blockers"] == 1
    assert "cycle_stage_failed:arbiter" in surface["blockers"]
    assert surface["operator_actions"][0] == "Resolve blocking cycle evidence before promotion or desk-safe reuse."


def test_cycle_result_surface_handles_missing_report_as_not_available():
    from trellis.agent.cycle_surface import build_cycle_result_surface

    surface = build_cycle_result_surface(None)

    assert surface["available"] is False
    assert surface["status"] == "not_available"
    assert surface["headline"] == "No governed agent-review cycle report is available."
    assert "agent-cycle evidence" in surface["claim"]["does_not_certify"]


def test_cycle_behavior_scorecard_reports_trigger_rates_and_blockers():
    from trellis.agent.cycle_surface import summarize_cycle_behavior

    scorecard = summarize_cycle_behavior(
        [
            {"cold_agent_cycle_report": _cycle_report()},
            {"cold_agent_cycle_report": _cycle_report(success=False, failed_stage="model_validator")},
            {"task_id": "missing_cycle"},
        ]
    )

    assert scorecard["schema_version"] == "agent_cycle_scorecard.v1"
    assert scorecard["run_count"] == 3
    assert scorecard["available_count"] == 2
    assert scorecard["not_available_count"] == 1
    assert scorecard["passed_count"] == 1
    assert scorecard["failed_count"] == 1
    assert scorecard["stage_trigger_rates"]["model_validator"]["triggered_count"] == 2
    assert scorecard["stage_trigger_rates"]["model_validator"]["failed_count"] == 1
    assert scorecard["blocker_counts"]["deterministic_blockers"] == 1
