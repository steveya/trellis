"""Tests for the binding-first exotic proof manifest and grading helpers."""

from __future__ import annotations

import json
from pathlib import Path


def _proof_tasks() -> dict[str, dict]:
    from trellis.agent.evals import load_binding_first_exotic_proof_manifest
    from trellis.agent.task_runtime import load_tasks

    manifest = load_binding_first_exotic_proof_manifest()
    task_ids = tuple(manifest)
    return {
        task["id"]: task
        for task in load_tasks(status=None)
        if task["id"] in task_ids
    }


def _expected_comparison_targets(task: dict) -> tuple[str, ...]:
    cross_validate = dict(task.get("cross_validate") or {})
    internal = tuple(cross_validate.get("internal") or ())
    analytical = cross_validate.get("analytical")
    if analytical:
        return internal + (str(analytical),)
    return internal


def test_binding_first_proof_manifest_matches_task_inventory():
    from trellis.agent.evals import load_binding_first_exotic_proof_manifest

    tasks = _proof_tasks()
    manifest = load_binding_first_exotic_proof_manifest()

    assert set(manifest) == set(tasks)


def test_binding_first_proof_manifest_matches_task_contracts():
    from trellis.agent.evals import load_binding_first_exotic_proof_manifest

    tasks = _proof_tasks()
    manifest = load_binding_first_exotic_proof_manifest()

    for task_id, task in tasks.items():
        expectation = manifest[task_id]
        assert tuple(expectation.get("required_mock_capabilities") or ()) == tuple(
            (task.get("market_assertions") or {}).get("requires") or ()
        ), task_id
        assert tuple(expectation.get("comparison_targets") or ()) == _expected_comparison_targets(task), task_id


def test_select_binding_first_exotic_proof_tasks_rejects_unknown_task_ids():
    from trellis.agent.evals import (
        load_binding_first_exotic_proof_manifest,
        select_binding_first_exotic_proof_tasks,
    )

    manifest = load_binding_first_exotic_proof_manifest()

    try:
        select_binding_first_exotic_proof_tasks(
            manifest,
            cohort="event_control_schedule",
            task_ids=["T17", "DOES_NOT_EXIST"],
        )
    except ValueError as exc:
        assert "DOES_NOT_EXIST" in str(exc)
    else:
        raise AssertionError("unknown proof task ids should raise ValueError")


def test_binding_first_proof_result_flags_missing_binding_ids_for_proved_task():
    from trellis.agent.evals import (
        grade_binding_first_exotic_proof_result,
        load_binding_first_exotic_proof_manifest,
    )

    task = _proof_tasks()["T105"]
    expectation = load_binding_first_exotic_proof_manifest()["T105"]
    report = grade_binding_first_exotic_proof_result(
        task,
        expectation,
        {
            "task_id": "T105",
            "success": True,
            "cross_validation": {"status": "passed"},
        },
    )

    assert report["outcome_class_alignment"].passed
    assert not report["binding_identity_alignment"].passed
    assert "did not persist any binding ids" in report["binding_identity_alignment"].details[0]


def test_summarize_binding_first_exotic_proof_captures_binding_ids_and_honest_block(tmp_path):
    from trellis.agent.evals import summarize_binding_first_exotic_proof

    proved_record = tmp_path / "T105_latest.json"
    proved_record.write_text(
        json.dumps(
            {
                "telemetry": {
                    "binding_observations": [
                        {
                            "binding_id": "trellis.models.fx_vanilla.price_quanto_option_analytical_from_market_state",
                            "binding_family": "analytical",
                            "route_id": "quanto_adjustment_analytical",
                            "route_family": "analytical",
                        }
                    ]
                }
            }
        )
    )
    blocked_record = tmp_path / "E27_latest.json"
    blocked_record.write_text(json.dumps({"telemetry": {"binding_observations": []}}))

    tasks = {
        "T105": _proof_tasks()["T105"],
        "E27": _proof_tasks()["E27"],
    }
    manifest = {
        "T105": {
            "cohort": "event_control_schedule",
            "outcome_class": "proved",
            "required_mock_capabilities": ["discount_curve", "forward_curve", "fx_rates", "spot"],
            "comparison_targets": ["quanto_bs", "mc_quanto"],
            "requires_binding_ids": True,
        },
        "E27": {
            "cohort": "event_control_schedule",
            "outcome_class": "honest_block",
            "required_mock_capabilities": ["discount_curve", "black_vol_surface", "model_parameters", "spot"],
            "comparison_targets": ["american_pathdep_pde", "american_pathdep_mc", "american_pathdep_fft"],
            "expected_blocker_categories": ["unsupported_composite"],
            "requires_binding_ids": False,
        },
    }
    results = [
        {
            "task_id": "T105",
            "success": True,
            "attempts": 1,
            "elapsed_seconds": 12.5,
            "cross_validation": {"status": "passed"},
            "task_run_latest_path": str(proved_record),
            "token_usage_summary": {"total_tokens": 100, "call_count": 2},
        },
        {
            "task_id": "E27",
            "success": False,
            "attempts": 1,
            "elapsed_seconds": 4.0,
            "cross_validation": {"status": "insufficient_results"},
            "blocker_details": {
                "blocker_report": {
                    "blockers": [{"category": "unsupported_composite"}]
                }
            },
            "task_run_latest_path": str(blocked_record),
            "token_usage_summary": {"total_tokens": 10, "call_count": 1},
        },
    ]

    summary = summarize_binding_first_exotic_proof(
        tasks,
        results,
        manifest=manifest,
    )

    assert summary["totals"]["passed_gate"] == 2
    assert summary["totals"]["proved"] == 1
    assert summary["totals"]["honest_block"] == 1
    assert summary["by_task"]["T105"]["binding_ids"] == [
        "trellis.models.fx_vanilla.price_quanto_option_analytical_from_market_state"
    ]
    assert summary["by_task"]["E27"]["failure_bucket"] == "blocked"


def test_summarize_binding_first_exotic_program_closeout_aggregates_cohorts():
    from trellis.agent.evals import summarize_binding_first_exotic_program_closeout

    closeout = summarize_binding_first_exotic_program_closeout(
        [
            {
                "cohort": "event_control_schedule",
                "status": "failed_gate",
                "task_ids": ["T105", "E27"],
                "report_json_path": "/tmp/qua808.json",
                "report_md_path": "/tmp/qua808.md",
                "raw_results_path": "/tmp/qua808_results.json",
                "proof_summary": {
                    "totals": {"tasks": 2, "passed_gate": 2, "failed_gate": 0, "proved": 1, "honest_block": 1},
                    "failure_buckets": {"success": 1, "blocked": 1},
                    "by_task": {
                        "T105": {
                            "title": "Quanto option",
                            "cohort": "event_control_schedule",
                            "outcome_class": "proved",
                            "passed_gate": True,
                            "failure_bucket": "success",
                            "comparison_status": "passed",
                            "binding_ids": ["binding.quanto"],
                            "route_ids": ["quanto_adjustment_analytical"],
                            "first_pass": True,
                            "attempts_to_success": 1,
                            "retry_taxonomy": [],
                            "elapsed_seconds": 10.0,
                            "token_usage": {"total_tokens": 100, "call_count": 2},
                        },
                        "E27": {
                            "title": "American Asian barrier",
                            "cohort": "event_control_schedule",
                            "outcome_class": "honest_block",
                            "passed_gate": True,
                            "failure_bucket": "blocked",
                            "comparison_status": "insufficient_results",
                            "binding_ids": [],
                            "route_ids": [],
                            "first_pass": True,
                            "attempts_to_success": 0,
                            "retry_taxonomy": [],
                            "elapsed_seconds": 5.0,
                            "token_usage": {"total_tokens": 10, "call_count": 1},
                        },
                    },
                },
                "task_summary": {},
            },
            {
                "cohort": "basket_credit_loss",
                "status": "failed_gate",
                "task_ids": ["T50"],
                "report_json_path": "/tmp/qua809.json",
                "report_md_path": "/tmp/qua809.md",
                "raw_results_path": "/tmp/qua809_results.json",
                "proof_summary": {
                    "totals": {"tasks": 1, "passed_gate": 0, "failed_gate": 1, "proved": 1, "honest_block": 0},
                    "failure_buckets": {"comparison_insufficient_results": 1},
                    "by_task": {
                        "T50": {
                            "title": "Nth-to-default",
                            "cohort": "basket_credit_loss",
                            "outcome_class": "proved",
                            "passed_gate": False,
                            "failure_bucket": "comparison_insufficient_results",
                            "comparison_status": "insufficient_results",
                            "binding_ids": ["binding.ntd"],
                            "route_ids": [],
                            "first_pass": False,
                            "attempts_to_success": 3,
                            "retry_taxonomy": ["code_generation"],
                            "elapsed_seconds": 20.0,
                            "token_usage": {"total_tokens": 200, "call_count": 3},
                        }
                    },
                },
                "task_summary": {},
            },
        ]
    )

    assert closeout["totals"]["tasks"] == 3
    assert closeout["totals"]["passed_gate"] == 2
    assert closeout["totals"]["failed_gate"] == 1
    assert closeout["honest_block_certified"] == 1
    assert closeout["first_pass"]["first_pass_successes"] == 2
    assert closeout["attempts_to_success"]["average"] == 1.0
    assert closeout["elapsed_seconds_total"] == 35.0
    assert closeout["token_usage"]["total_tokens"] == 310
    assert closeout["unknown_route_tasks"] == []


def test_summarize_binding_first_exotic_program_closeout_rejects_duplicate_cohorts():
    from trellis.agent.evals import summarize_binding_first_exotic_program_closeout

    report = {
        "cohort": "event_control_schedule",
        "proof_summary": {
            "totals": {"tasks": 0, "passed_gate": 0, "failed_gate": 0, "proved": 0, "honest_block": 0},
            "failure_buckets": {},
            "by_task": {},
        },
        "task_summary": {},
    }

    try:
        summarize_binding_first_exotic_program_closeout([report, report])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate cohort names should raise ValueError")
