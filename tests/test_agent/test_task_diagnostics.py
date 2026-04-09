from __future__ import annotations

from pathlib import Path


def _sample_record(root: Path) -> dict[str, object]:
    storage = {
        "history_path": str(root / "task_runs" / "history" / "T999" / "run.json"),
        "latest_path": str(root / "task_runs" / "latest" / "T999.json"),
        "latest_index_path": str(root / "task_results_latest.json"),
        "diagnosis_history_packet_path": str(
            root / "task_runs" / "diagnostics" / "history" / "T999" / "run.json"
        ),
        "diagnosis_history_dossier_path": str(
            root / "task_runs" / "diagnostics" / "history" / "T999" / "run.md"
        ),
        "diagnosis_latest_packet_path": str(
            root / "task_runs" / "diagnostics" / "latest" / "T999.json"
        ),
        "diagnosis_latest_dossier_path": str(
            root / "task_runs" / "diagnostics" / "latest" / "T999.md"
        ),
    }
    return {
        "task_id": "T999",
        "task_kind": "pricing",
        "run_id": "20260329T120000000000Z",
        "persisted_at": "2026-03-29T12:00:00+00:00",
        "task": {
            "id": "T999",
            "title": "Callable bond stress test",
            "construct": ["tree"],
        },
        "summary": {
            "success": False,
            "status": "failed",
            "comparison_status": "insufficient_results",
            "next_action": "Inspect which methods failed to produce valid results, then rerun the narrowest broken route.",
        },
        "workflow": {
            "status": "failed",
            "next_action": "Inspect which methods failed to produce valid results, then rerun the narrowest broken route.",
            "latest_trace": {"request_id": "executor_build_demo"},
        },
        "comparison": {
            "status": "insufficient_results",
            "reference_target": "black_scholes",
            "prices": {"psor": 1.01, "black_scholes": 1.0},
            "deviations_pct": {"psor": 1.0},
        },
        "learning": {
            "task_kind": "pricing",
            "retrieved_lesson_ids": [],
            "retrieved_lesson_titles": [],
            "retrieval_stages": ["semantic_validation_failed"],
            "retrieval_sources": ["callback"],
            "selected_artifact_ids": ["route_hint:callable_bond_tree"],
            "selected_artifact_titles": ["Callable bond tree route"],
            "selected_artifacts_by_audience": {
                "builder": [
                    {
                        "id": "route_hint:callable_bond_tree",
                        "title": "Callable bond tree route",
                        "kind": "route_hint",
                    }
                ]
            },
            "captured_lesson_ids": [],
            "lesson_contract_reports": [],
            "lesson_contract_count": 0,
            "lesson_contract_outcome": "not_attempted",
            "lesson_contract_errors": [],
            "lesson_contract_warnings": [],
            "lesson_promotion_outcomes": [],
            "lessons_attributed": 0,
            "cookbook_enriched": False,
            "cookbook_candidate_paths": [],
            "promotion_candidate_paths": [],
            "knowledge_trace_paths": [],
            "knowledge_gap_log_paths": [],
            "reusable_artifact_count": 0,
            "knowledge_outcome": "blocked_without_learning",
            "knowledge_outcome_reason": "task failed before any reusable learning artifact was captured",
        },
        "telemetry": {
            "task_kind": "pricing",
            "run_outcome": "comparison:insufficient_results",
            "retried": True,
            "retry_count": 1,
            "degraded": True,
            "comparison_status": "insufficient_results",
            "selected_artifacts": [
                {
                    "artifact_id": "route_hint:callable_bond_tree",
                    "title": "Callable bond tree route",
                    "kind": "route_hint",
                    "audiences": ["builder"],
                    "outcome": "comparison:insufficient_results",
                    "success": False,
                    "retried": True,
                    "retry_count": 1,
                    "degraded": True,
                    "route_ids": ["callable_bond_tree"],
                    "route_families": ["pde"],
                }
            ],
            "route_observations": [
                {
                    "route_id": "callable_bond_tree",
                    "route_family": "pde",
                    "primary_kind": "family_ir",
                    "primary_label": "EventAwarePDEIR",
                    "backend_binding_id": "",
                    "route_alias": "callable_bond_tree",
                    "trace_kind": "platform",
                    "trace_status": "failed",
                    "outcome": "comparison:insufficient_results",
                    "success": False,
                    "retried": True,
                    "retry_count": 1,
                    "degraded": True,
                    "selected_artifact_ids": ["route_hint:callable_bond_tree"],
                    "instruction_ids": [],
                    "effective_instruction_count": 0,
                    "hard_constraint_count": 0,
                    "conflict_count": 0,
                }
            ],
        },
        "post_build": {
            "latest_method": "psor",
            "latest_phase": "reflection_completed",
            "latest_status": "error",
            "active_flags": {
                "skip_post_build_reflection": False,
                "skip_post_build_consolidation": False,
            },
            "methods": {
                "psor": {
                    "latest_phase": "reflection_completed",
                    "latest_status": "error",
                    "event_count": 4,
                    "active_flags": {"skip_reflection": False},
                }
            },
        },
        "trace_summaries": [
            {
                "path": str(root / "task_runs" / "platform" / "executor_build_demo.yaml"),
                "exists": True,
                "trace_kind": "platform",
                "request_id": "executor_build_demo",
                "status": "failed",
                "outcome": "request_failed",
                "route_method": "pde",
                "latest_event": "builder_attempt_failed",
                "latest_event_status": "error",
                "updated_at": "2026-03-29T12:00:00+00:00",
                "route_health": {
                    "route_id": "callable_bond_tree",
                    "route_family": "pde",
                    "primary_kind": "family_ir",
                    "primary_label": "EventAwarePDEIR",
                    "backend_binding_id": "",
                    "route_alias": "callable_bond_tree",
                    "trace_status": "failed",
                    "effective_instruction_ids": [],
                    "effective_instruction_count": 0,
                    "hard_constraint_count": 0,
                    "conflict_count": 0,
                },
                "construction_identity": {
                    "primary_kind": "family_ir",
                    "primary_label": "EventAwarePDEIR",
                    "lane_family": "pde_solver",
                    "plan_kind": "family_lowering_only",
                    "family_ir_type": "EventAwarePDEIR",
                    "backend_binding_id": None,
                    "backend_engine_family": None,
                    "backend_exact_fit": False,
                    "route_alias": "callable_bond_tree",
                    "route_authority_kind": "route_registry_binding",
                    "state_obligations": ["schedule_state"],
                    "control_obligations": ["issuer_min"],
                },
                "route_binding_authority": {
                    "route_id": "callable_bond_tree",
                    "route_family": "pde",
                    "engine_family": "pde_solver",
                    "authority_kind": "route_registry_binding",
                    "exact_backend_fit": False,
                    "validation_bundle_id": "pde_solver:callable_bond",
                    "canary_task_ids": ["T02"],
                },
            }
        ],
        "method_runs": {
            "psor": {
                "success": False,
                "preferred_method": "pde",
                "attempts": 2,
                "route_method": "pde",
                "platform_trace_path": str(root / "task_runs" / "platform" / "executor_build_demo.yaml"),
                "trace_summary": {
                    "path": str(root / "task_runs" / "platform" / "executor_build_demo_method.yaml"),
                    "exists": True,
                    "trace_kind": "platform",
                    "request_id": "executor_build_demo_method",
                    "status": "failed",
                    "outcome": "request_failed",
                    "route_method": "pde",
                    "latest_event": "builder_attempt_failed",
                    "latest_event_status": "error",
                    "updated_at": "2026-03-29T12:00:00+00:00",
                },
            "cross_validation": {"status": "failed"},
            "token_usage_summary": {"total_tokens": 120},
            "failures": ["name 'absorbing' is not defined"],
            "error": "name 'absorbing' is not defined",
            "post_build_tracking": {
                "last_phase": "reflection_completed",
                "last_status": "error",
                "active_flags": {"skip_reflection": False},
            },
        },
        "black_scholes": {
                "success": True,
                "preferred_method": "analytical",
                "attempts": 1,
                "route_method": "analytical",
                "platform_trace_path": str(root / "task_runs" / "platform" / "executor_build_black_scholes.yaml"),
                "trace_summary": {
                    "path": str(root / "task_runs" / "platform" / "executor_build_black_scholes.yaml"),
                    "exists": True,
                    "trace_kind": "platform",
                    "request_id": "executor_build_black_scholes",
                    "status": "succeeded",
                    "outcome": "build_completed",
                    "route_method": "analytical",
                    "latest_event": "build_completed",
                    "latest_event_status": "ok",
                    "updated_at": "2026-03-29T12:00:00+00:00",
                },
            "cross_validation": {"status": "passed"},
            "token_usage_summary": {"total_tokens": 40},
            "failures": [],
            "error": None,
            "post_build_tracking": {
                "last_phase": "consolidation_dispatched",
                "last_status": "backgrounded",
                "active_flags": {"skip_reflection": False},
            },
        },
        },
        "result": {
            "success": False,
            "attempts": 2,
            "elapsed_seconds": 12.3,
            "error": "comparison failed",
            "failures": ["comparison failed"],
            "blocker_details": {},
            "cross_validation": {
                "status": "insufficient_results",
                "reference_target": "black_scholes",
                "prices": {"psor": 1.01, "black_scholes": 1.0},
                "deviations_pct": {"psor": 1.0},
            },
            "method_results": {},
            "knowledge_gaps": ["missing_cookbook"],
            "gap_confidence": 0.6,
            "runtime_controls": {
                "skip_post_build_reflection": True,
                "skip_post_build_consolidation": False,
                "skip_task_diagnosis_persist": False,
                "llm_wait_log_path": "/tmp/t999_waits.jsonl",
            },
            "task_contract_error": {},
            "reflection": {},
            "artifacts": {
                "platform_trace_paths": [
                    str(root / "task_runs" / "platform" / "executor_build_demo.yaml")
                ],
                "analytical_trace_paths": [],
            },
        },
        "storage": storage,
    }


def test_build_task_diagnosis_packet_summarizes_failure(tmp_path):
    from trellis.agent.task_diagnostics import build_task_diagnosis_packet, render_task_diagnosis_dossier

    packet = build_task_diagnosis_packet(_sample_record(tmp_path))

    assert packet["schema_version"] == 2
    assert packet["task"]["id"] == "T999"
    assert packet["outcome"]["failure_bucket"] == "comparison_insufficient_results"
    assert packet["outcome"]["decision_stage"] == "comparison"
    assert packet["primary_failure"]["likely_cause"].startswith(
        "The comparison task did not produce a valid enough method set"
    )
    assert packet["method_outcomes"][0]["method"] == "psor"
    assert packet["method_outcomes"][0]["post_build_latest_phase"] == "reflection_completed"
    assert packet["trace_index"][0]["scope"] == "task"
    assert packet["trace_index"][1]["scope"] == "method"
    assert packet["post_build"]["latest_phase"] == "reflection_completed"
    assert packet["runtime_controls"]["llm_wait_log_path"] == "/tmp/t999_waits.jsonl"
    assert packet["telemetry"]["selected_artifacts"][0]["artifact_id"] == "route_hint:callable_bond_tree"
    assert packet["telemetry"]["route_observations"][0]["route_id"] == "callable_bond_tree"
    assert packet["telemetry"]["route_observations"][0]["primary_label"] == "EventAwarePDEIR"
    assert packet["telemetry"]["route_observations"][0]["task_ids"] == ["T02"]
    assert packet["trace_index"][0]["construction_label"] == "EventAwarePDEIR"

    rendered = render_task_diagnosis_dossier(packet)
    assert "## Primary Diagnosis" in rendered
    assert "## Runtime Controls" in rendered
    assert "## Method Outcomes" in rendered
    assert "## Skill Telemetry" in rendered
    assert "## Post-build" in rendered
    assert "## Storage" in rendered
    assert "comparison_insufficient_results" in rendered
    assert "semantic_validation_failed" in rendered
    assert "route_hint:callable_bond_tree" in rendered
    assert "EventAwarePDEIR" in rendered
    assert "/tmp/t999_waits.jsonl" in rendered


def test_save_task_diagnosis_artifacts_writes_packet_and_dossier(tmp_path):
    from trellis.agent.task_diagnostics import save_task_diagnosis_artifacts

    artifacts = save_task_diagnosis_artifacts(_sample_record(tmp_path), root=tmp_path)

    assert artifacts.packet_path.exists()
    assert artifacts.dossier_path.exists()
    assert artifacts.latest_packet_path.exists()
    assert artifacts.latest_dossier_path.exists()
    assert artifacts.packet_path.parent.name == "T999"
    assert artifacts.packet["outcome"]["headline"].startswith("Callable bond stress test")
    assert "Task diagnosis" in artifacts.dossier_path.read_text()


def test_build_task_diagnosis_packet_detects_comparator_build_failure(tmp_path):
    from trellis.agent.task_diagnostics import build_task_diagnosis_packet

    record = _sample_record(tmp_path)
    record["task_id"] = "T013"
    record["task"] = {
        "id": "T013",
        "title": "European call comparator stress",
        "construct": "pde",
    }
    record["summary"] = {
        "success": False,
        "status": "failed",
        "comparison_status": "passed",
    }
    record["comparison"] = {
        "status": "passed",
        "reference_target": "black_scholes",
    }
    record["method_runs"] = {
        "theta_0.5": {
            "success": True,
            "attempts": 1,
            "preferred_method": "pde_solver",
            "route_method": "pde_solver",
            "trace_summary": {
                "path": str(tmp_path / "task_runs" / "platform" / "theta_05.yaml"),
                "exists": True,
                "trace_kind": "platform",
                "request_id": "theta_05",
                "status": "succeeded",
                "outcome": "build_completed",
                "route_method": "pde_solver",
                "latest_event": "build_completed",
                "latest_event_status": "ok",
                "updated_at": "2026-03-29T12:00:00+00:00",
            },
            "cross_validation": {"status": "passed"},
            "token_usage_summary": {"total_tokens": 120},
            "failures": [],
            "error": None,
        },
        "black_scholes": {
            "success": False,
            "attempts": 3,
            "preferred_method": "analytical",
            "route_method": "analytical",
            "platform_trace_path": str(tmp_path / "task_runs" / "platform" / "black_scholes.yaml"),
            "trace_summary": {
                "path": str(tmp_path / "task_runs" / "platform" / "black_scholes.yaml"),
                "exists": True,
                "trace_kind": "platform",
                "request_id": "black_scholes",
                "status": "failed",
                "outcome": "request_failed",
                "route_method": "analytical",
                "latest_event": "request_failed",
                "latest_event_status": "error",
                "updated_at": "2026-03-29T12:00:00+00:00",
            },
            "cross_validation": {"status": "passed"},
            "token_usage_summary": {"total_tokens": 40},
            "failures": [
                "Failed to build payoff after 3 attempts: SyntaxError at line 83, column 16: unexpected indent",
            ],
            "error": None,
        },
    }
    record["result"] = {
        "success": False,
        "attempts": 3,
        "elapsed_seconds": 12.3,
        "error": "comparison failed",
        "failures": ["comparison failed"],
        "blocker_details": {},
        "cross_validation": {"status": "passed"},
        "comparison_task": True,
        "method_results": {},
        "knowledge_gaps": [],
        "gap_confidence": 0.6,
        "task_contract_error": {},
        "reflection": {},
        "artifacts": {
            "platform_trace_paths": [],
            "analytical_trace_paths": [],
        },
    }

    packet = build_task_diagnosis_packet(record)

    assert packet["outcome"]["failure_bucket"] == "comparator_build_failure"
    assert packet["outcome"]["decision_stage"] == "comparison"
    assert packet["primary_failure"]["likely_cause"].startswith(
        "One comparison/comparator lane failed to build while other methods completed"
    )
    assert packet["primary_failure"]["confidence"] == "high"
    assert packet["outcome"]["next_action"].startswith(
        "Repair the failing comparator route or scaffold"
    )


def test_build_task_diagnosis_packet_keeps_platform_action_out_of_route_ids(tmp_path):
    from trellis.agent.task_diagnostics import build_task_diagnosis_packet

    record = _sample_record(tmp_path)
    record.pop("telemetry", None)
    trace = record["trace_summaries"][0]
    trace["action"] = "build_then_price"
    trace["route_method"] = "analytical"
    trace["construction_identity"] = {}
    trace["route_binding_authority"] = {}
    trace["route_health"] = {
        "route_id": "",
        "route_family": "analytical",
        "trace_status": "failed",
        "effective_instruction_ids": [],
        "effective_instruction_count": 0,
        "hard_constraint_count": 0,
        "conflict_count": 0,
        "canary_task_ids": [],
    }

    packet = build_task_diagnosis_packet(record)

    assert packet["telemetry"]["route_observations"][0]["route_id"] == ""
    assert packet["telemetry"]["route_observations"][0]["route_family"] == "analytical"
    assert packet["telemetry"]["route_observations"][0]["primary_label"] == "analytical"
