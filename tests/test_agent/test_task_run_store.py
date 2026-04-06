from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _write_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_json_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_persist_task_run_record_writes_latest_and_enriches_traces(tmp_path):
    from trellis.agent.task_run_store import load_task_run_record, persist_task_run_record

    trace_root = tmp_path / "trellis" / "agent" / "knowledge" / "traces" / "platform"
    fft_trace = trace_root / "executor_build_fft.yaml"
    analytical_trace = trace_root / "executor_build_black_scholes.yaml"

    _write_trace(
        fft_trace,
        {
            "request_id": "executor_build_fft",
            "status": "failed",
            "outcome": "request_failed",
            "action": "compile_only",
            "route_method": "fft_pricing",
            "updated_at": "2026-03-25T18:00:00+00:00",
            "request_metadata": {
                "task_id": "T104",
                "task_title": "FFT vs COS comparison",
                "comparison_target": "fft",
                "semantic_role_ownership": {
                    "selected_stage": "route_assembly",
                    "selected_role": "quant",
                    "trigger_condition": "pricing_plan_selection",
                    "artifact_kind": "GenerationPlan",
                },
            },
            "semantic_role_ownership": {
                "selected_stage": "route_assembly",
                "selected_role": "quant",
                "trigger_condition": "pricing_plan_selection",
                "artifact_kind": "GenerationPlan",
            },
            "generation_boundary": {
                "route_binding_authority": {
                    "route_id": "fft_black_scholes",
                    "route_family": "fft_pricing",
                    "engine_family": "fft_pricing",
                    "authority_kind": "route_registry_binding",
                    "exact_backend_fit": False,
                    "validation_bundle_id": "fft_pricing:european_option",
                    "canary_task_ids": ["T39"],
                }
            },
            "token_usage": {
                "call_count": 2,
                "calls_with_usage": 2,
                "calls_without_usage": 0,
                "prompt_tokens": 180,
                "completion_tokens": 70,
                "total_tokens": 250,
                "by_stage": {"code_generation": {"total_tokens": 250}},
                "by_provider": {"anthropic": {"total_tokens": 250}},
            },
            "events": [
                {
                    "event": "builder_attempt_failed",
                    "status": "error",
                    "timestamp": "2026-03-25T18:00:00+00:00",
                    "details": {"reason": "validation"},
                }
            ],
            "linear_issue": {
                "id": "lin-1",
                "identifier": "QUA-99",
                "url": "https://linear.app/quant-macro/issue/QUA-99/demo",
            },
            "github_issue": {
                "id": 123,
                "number": 77,
                "url": "https://github.com/steveya/trellis/issues/77",
                "repository": "steveya/trellis",
            },
        },
    )
    _write_trace(
        analytical_trace,
        {
            "request_id": "executor_build_black_scholes",
            "status": "succeeded",
            "outcome": "build_completed",
            "action": "compile_only",
            "route_method": "analytical",
            "updated_at": "2026-03-25T17:59:00+00:00",
            "request_metadata": {
                "task_id": "T104",
                "task_title": "FFT vs COS comparison",
                "comparison_target": "black_scholes",
                "semantic_role_ownership": {
                    "selected_stage": "route_assembly",
                    "selected_role": "quant",
                    "trigger_condition": "pricing_plan_selection",
                    "artifact_kind": "GenerationPlan",
                },
            },
            "semantic_role_ownership": {
                "selected_stage": "route_assembly",
                "selected_role": "quant",
                "trigger_condition": "pricing_plan_selection",
                "artifact_kind": "GenerationPlan",
            },
            "events": [
                {
                    "event": "build_completed",
                    "status": "ok",
                    "timestamp": "2026-03-25T17:59:00+00:00",
                    "details": {"attempts": 1},
                }
            ],
        },
    )

    task = {
        "id": "T104",
        "title": "FFT vs COS comparison",
        "construct": ["transforms"],
        "cross_validate": {"internal": ["fft", "cos"], "analytical": "black_scholes"},
    }
    result = {
        "task_id": "T104",
        "title": "FFT vs COS comparison",
        "success": False,
        "start_time": "2026-03-25T14:05:00",
        "comparison_task": True,
        "comparison_targets": ["fft", "cos", "black_scholes"],
        "token_usage_summary": {
            "call_count": 3,
            "calls_with_usage": 3,
            "calls_without_usage": 0,
            "prompt_tokens": 220,
            "completion_tokens": 90,
            "total_tokens": 310,
            "by_stage": {
                "decomposition": {"total_tokens": 30},
                "code_generation": {"total_tokens": 250},
                "reflection": {"total_tokens": 30},
            },
            "by_provider": {"anthropic": {"total_tokens": 310}},
        },
        "cross_validation": {
            "status": "failed",
            "prices": {"fft": 10.12, "cos": 10.02, "black_scholes": 10.0},
            "deviations_pct": {"fft": 1.2, "cos": 0.2},
            "reference_target": "black_scholes",
        },
        "market_context": {
            "source": "mock",
            "as_of": "2024-11-15",
            "selected_components": {"model_parameters": "heston_equity"},
            "selected_curve_names": {"discount_curve": "usd_ois"},
            "available_capabilities": ["discount_curve", "model_parameters"],
            "metadata": {},
            "provenance": {"source_kind": "synthetic_snapshot", "prior_seed": 1337},
            "market_parameter_trace": {
                "selected_parameter_set": "heston_equity",
                "selected_source_kind": "synthetic_prior",
                "sources": {
                    "heston_equity": {
                        "source_kind": "synthetic_prior",
                        "source_ref": "embedded_regime_snapshot",
                        "parameter_keys": ["kappa", "theta", "xi", "rho", "v0"],
                        "details": {
                            "synthetic_generation_contract_version": "v2",
                            "prior_seed": 1337,
                        },
                    }
                },
            },
        },
        "runtime_contract": {
            "task_id": "T104",
            "market_parameter_trace": {
                "selected_parameter_set": "heston_equity",
                "selected_source_kind": "synthetic_prior",
                "sources": {
                    "heston_equity": {
                        "source_kind": "synthetic_prior",
                        "parameter_keys": ["kappa", "theta", "xi", "rho", "v0"],
                    }
                },
            },
        },
        "artifacts": {
            "platform_trace_paths": [str(fft_trace), str(analytical_trace)],
        },
        "method_results": {
            "fft": {
                "success": False,
                "preferred_method": "fft_pricing",
                "attempts": 1,
                "failures": ["validation failed"],
                "platform_trace_path": str(fft_trace),
                "token_usage_summary": {
                    "call_count": 2,
                    "calls_with_usage": 2,
                    "calls_without_usage": 0,
                    "prompt_tokens": 180,
                    "completion_tokens": 70,
                    "total_tokens": 250,
                    "by_stage": {"code_generation": {"total_tokens": 250}},
                    "by_provider": {"anthropic": {"total_tokens": 250}},
                },
                "artifacts": {"platform_trace_paths": [str(fft_trace)]},
                "reflection": {},
            },
            "black_scholes": {
                "success": True,
                "preferred_method": "analytical",
                "attempts": 1,
                "failures": [],
                "platform_trace_path": str(analytical_trace),
                "artifacts": {"platform_trace_paths": [str(analytical_trace)]},
                "reflection": {},
            },
        },
        "failures": ["comparison failed"],
        "reflection": {},
    }

    persisted = persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 3, 25, 18, 1, 0, tzinfo=timezone.utc),
    )

    latest = load_task_run_record(persisted["latest_path"])

    assert Path(persisted["history_path"]).exists()
    assert Path(persisted["latest_path"]).exists()
    assert Path(persisted["latest_index_path"]).exists()
    assert Path(persisted["diagnosis_packet_path"]).exists()
    assert Path(persisted["diagnosis_dossier_path"]).exists()
    assert Path(persisted["latest_diagnosis_packet_path"]).exists()
    assert Path(persisted["latest_diagnosis_dossier_path"]).exists()
    assert latest["task_id"] == "T104"
    assert latest["summary"]["comparison_status"] == "failed"
    assert latest["summary"]["prices"]["fft"] == 10.12
    assert latest["method_runs"]["fft"]["trace_summary"]["linear_issue"]["identifier"] == "QUA-99"
    assert latest["method_runs"]["fft"]["trace_summary"]["semantic_role_ownership"]["selected_role"] == "quant"
    assert latest["method_runs"]["fft"]["trace_summary"]["route_binding_authority"]["route_id"] == "fft_black_scholes"
    assert latest["trace_summaries"][0]["route_binding_authority"]["canary_task_ids"] == ["T39"]
    assert latest["method_runs"]["fft"]["issue_refs"]["github"]["number"] == 77
    assert latest["method_runs"]["fft"]["token_usage"]["total_tokens"] == 250
    assert latest["trace_summaries"][0]["token_usage"]["total_tokens"] == 250
    assert latest["token_usage"]["task"]["total_tokens"] == 310
    assert latest["workflow"]["status"] == "failed"
    assert latest["workflow"]["linked_issues"]["linear"][0]["identifier"] == "QUA-99"
    assert latest["workflow"]["latest_trace"]["request_id"] == "executor_build_fft"
    assert latest["market"]["market_parameter_trace"]["selected_parameter_set"] == "heston_equity"
    assert latest["result"]["runtime_contract"]["market_parameter_trace"]["selected_source_kind"] == (
        "synthetic_prior"
    )
    assert Path(latest["storage"]["diagnosis_history_packet_path"]).parent.name == "T104"
    assert Path(latest["storage"]["diagnosis_history_packet_path"]).suffix == ".json"
    assert latest["storage"]["diagnosis_latest_packet_path"].endswith("/task_runs/diagnostics/latest/T104.json")
    assert latest["result"]["learning"]["knowledge_outcome"] == "no_new_knowledge"
    assert latest["learning"]["knowledge_outcome"] == "no_new_knowledge"


def test_persist_task_run_record_treats_terminal_result_as_not_running_even_with_running_trace(tmp_path):
    from trellis.agent.task_run_store import persist_task_run_record, load_task_run_record

    trace_root = tmp_path / "trellis" / "agent" / "knowledge" / "traces" / "platform"
    trace_path = trace_root / "executor_build_demo.yaml"
    _write_trace(
        trace_path,
        {
            "request_id": "executor_build_demo",
            "status": "running",
            "outcome": "",
            "action": "compile_only",
            "route_method": "analytical",
            "updated_at": "2026-03-25T18:31:02+00:00",
            "events": [
                {
                    "event": "planner_completed",
                    "status": "ok",
                    "timestamp": "2026-03-25T18:31:02+00:00",
                    "details": {},
                }
            ],
        },
    )

    task = {"id": "T200", "title": "Demo terminal failure", "construct": "analytical"}
    result = {
        "task_id": "T200",
        "title": "Demo terminal failure",
        "success": False,
        "failures": ["model not found"],
        "cross_validation": {"status": "insufficient_results"},
        "artifacts": {"platform_trace_paths": [str(trace_path)]},
        "reflection": {},
    }

    persisted = persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 3, 25, 18, 32, 0, tzinfo=timezone.utc),
    )
    latest = load_task_run_record(persisted["latest_path"])

    assert latest["workflow"]["status"] == "failed"
    assert latest["workflow"]["active_trace_count"] == 1


def test_skill_telemetry_rollups_capture_selected_artifacts_and_route_health(tmp_path):
    from trellis.agent.task_run_store import (
        load_latest_route_ranking_inputs,
        load_latest_route_health_rollup,
        load_latest_skill_ranking_inputs,
        load_latest_telemetry_rollups,
        load_latest_skill_telemetry_rollup,
        persist_task_run_record,
    )

    analytical_trace = tmp_path / "task_runs" / "traces" / "analytical_demo.json"
    _write_json_trace(
        analytical_trace,
        {
            "trace_id": "trace_analytical_demo",
            "trace_type": "analytical",
            "status": "succeeded",
            "created_at": "2026-03-29T12:00:00+00:00",
            "updated_at": "2026-03-29T12:00:05+00:00",
            "route": {
                "family": "analytical",
                "name": "analytical_black76",
                "model": "black76",
            },
            "steps": [],
            "context": {
                "generation_plan": {
                    "instruction_resolution": {
                        "effective_instructions": [
                            {
                                "id": "route_hint:analytical_black76",
                                "instruction_type": "hard_constraint",
                            }
                        ],
                        "conflicts": [{"winner": "route_hint:analytical_black76"}],
                    }
                }
            },
        },
    )

    task = {
        "id": "T301",
        "title": "Telemetry demo",
        "construct": "analytical",
    }
    result = {
        "task_id": "T301",
        "success": True,
        "attempts": 2,
        "cross_validation": {"status": "passed"},
        "artifacts": {"analytical_trace_paths": [str(analytical_trace)]},
        "knowledge_summary": {
            "selected_artifact_ids": ["route_hint:analytical_black76"],
            "selected_artifact_titles": ["Analytical Black76 route"],
            "selected_artifacts_by_audience": {
                "builder": [
                    {
                        "id": "route_hint:analytical_black76",
                        "title": "Analytical Black76 route",
                        "kind": "route_hint",
                    }
                ]
            },
        },
        "reflection": {},
    }

    persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 3, 29, 12, 1, 0, tzinfo=timezone.utc),
    )

    skill_rollup = load_latest_skill_telemetry_rollup(root=tmp_path)
    route_rollup = load_latest_route_health_rollup(root=tmp_path)
    bundle = load_latest_telemetry_rollups(root=tmp_path)
    skill_ranking = load_latest_skill_ranking_inputs(root=tmp_path)
    route_ranking = load_latest_route_ranking_inputs(root=tmp_path)

    assert skill_rollup["run_count"] == 1
    assert skill_rollup["artifacts"][0]["artifact_id"] == "route_hint:analytical_black76"
    assert skill_rollup["artifacts"][0]["selection_count"] == 1
    assert skill_rollup["artifacts"][0]["success_count"] == 1
    assert skill_rollup["artifacts"][0]["retried_count"] == 1
    assert skill_rollup["artifacts"][0]["retry_count_total"] == 1
    assert skill_rollup["artifacts"][0]["audiences"] == ["builder"]
    assert skill_rollup["artifacts"][0]["first_seen_at"] == "2026-03-29T12:01:00+00:00"
    assert skill_rollup["artifacts"][0]["last_seen_at"] == "2026-03-29T12:01:00+00:00"
    assert skill_rollup["ranking_inputs"][0]["success_rate"] == 1.0
    assert skill_rollup["ranking_inputs"][0]["retry_rate"] == 1.0
    assert skill_rollup["ranking_inputs"][0]["avg_retry_count"] == 1.0

    assert route_rollup["run_count"] == 1
    assert route_rollup["routes"][0]["route_id"] == "analytical_black76"
    assert route_rollup["routes"][0]["route_family"] == "analytical"
    assert route_rollup["routes"][0]["success_count"] == 1
    assert route_rollup["routes"][0]["retried_count"] == 1
    assert route_rollup["routes"][0]["retry_count_total"] == 1
    assert route_rollup["routes"][0]["hard_constraint_count_total"] == 1
    assert route_rollup["routes"][0]["conflict_count_total"] == 1
    assert route_rollup["ranking_inputs"][0]["avg_effective_instruction_count"] == 1.0
    assert route_rollup["ranking_inputs"][0]["avg_hard_constraint_count"] == 1.0
    assert route_rollup["ranking_inputs"][0]["avg_conflict_count"] == 1.0

    assert bundle["skill_telemetry"]["artifacts"][0]["artifact_id"] == "route_hint:analytical_black76"
    assert bundle["route_health"]["routes"][0]["route_id"] == "analytical_black76"
    assert skill_ranking["artifacts"][0]["artifact_id"] == "route_hint:analytical_black76"
    assert skill_ranking["artifacts"][0]["last_seen_at"] == "2026-03-29T12:01:00+00:00"
    assert route_ranking["routes"][0]["route_id"] == "analytical_black76"
    assert route_ranking["routes"][0]["last_seen_at"] == "2026-03-29T12:01:00+00:00"


def test_persist_task_run_record_supports_framework_task_contract(tmp_path):
    from trellis.agent.task_run_store import (
        load_latest_task_run,
        load_latest_task_run_records,
        load_task_run_record,
        persist_task_run_record,
    )

    task = {
        "id": "E17",
        "title": "Extract: FX pricing framework (GK base, barrier, digital, quanto adjustments)",
        "construct": "framework",
        "trigger_after": ["T108", "T109"],
    }
    result = {
        "task_id": "E17",
        "task_kind": "framework",
        "title": task["title"],
        "success": True,
        "framework_result": {
            "outcome_type": "extraction_candidate",
            "candidate_name": "FX pricing framework (GK base, barrier, digital, quanto adjustments)",
            "summary": "ready to review",
            "next_action": "Review the extraction candidate.",
            "related_task_ids": ["T108", "T109"],
            "related_components": ["garman_kohlhagen_formula", "fx_market_bridge"],
            "related_issue_refs": {
                "linear": [{"identifier": "QUA-101", "id": "lin-101"}],
                "github": [{"number": 88, "id": 88}],
            },
        },
        "artifacts": {
            "related_task_latest_paths": [
                str(tmp_path / "task_runs" / "latest" / "T108.json"),
                str(tmp_path / "task_runs" / "latest" / "T109.json"),
            ]
        },
        "reflection": {},
        "knowledge_summary": {},
    }

    persisted = persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 3, 26, 20, 0, tzinfo=timezone.utc),
    )

    latest = load_task_run_record(persisted["latest_path"])
    direct = load_latest_task_run("E17", root=tmp_path)
    records = load_latest_task_run_records(root=tmp_path, task_kind="framework")

    assert latest["task_kind"] == "framework"
    assert latest["framework"]["outcome_type"] == "extraction_candidate"
    assert latest["workflow"]["status"] == "proposed"
    assert latest["issue_refs"]["linear"][0]["identifier"] == "QUA-101"
    assert latest["summary"]["framework_outcome"] == "extraction_candidate"
    assert latest["learning"]["reusable_artifact_count"] == 0
    assert latest["learning"]["knowledge_outcome"] == "no_new_knowledge"
    assert "without new reusable knowledge" in latest["learning"]["knowledge_outcome_reason"]
    assert direct["storage"]["latest_path"].endswith("/task_runs/latest/E17.json")
    assert len(records) == 1
    assert records[0]["task_id"] == "E17"


def test_persist_task_run_record_retains_runtime_contract_metadata(tmp_path):
    from trellis.agent.task_run_store import load_task_run_record, persist_task_run_record

    task = {
        "id": "T998",
        "title": "Himalaya ranked observation basket",
        "construct": ["monte_carlo"],
    }
    result = {
        "task_id": "T998",
        "title": task["title"],
        "success": True,
        "start_time": "2026-03-27T20:45:00",
        "runtime_contract": {
            "task_id": "T998",
            "task_title": task["title"],
            "description": "Build a pricer for: Himalaya ranked observation basket",
            "instrument_type": "basket_option",
            "semantic_contract_id": "ranked_observation_basket",
            "snapshot_reference": {
                "source": "mock",
                "as_of": "2024-11-15",
                "selected_components": {
                    "discount_curve": "usd_ois",
                    "forecast_curve": "USD-SOFR-3M",
                },
                "selected_curve_names": {
                    "discount_curve": "usd_ois",
                    "forecast_curve": "USD-SOFR-3M",
                },
            },
            "evaluation_tags": [
                "task_runtime",
                "construct:monte_carlo",
                "market:mock",
            ],
            "selected_curve_names": {
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M",
            },
            "trace_identifier": "executor_build_20260327_deadbeef",
            "trace_path": "/tmp/executor_build_20260327_deadbeef.yaml",
        },
        "market_context": {
            "source": "mock",
            "as_of": "2024-11-15",
            "selected_components": {
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M",
            },
            "selected_curve_names": {
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M",
            },
        },
        "reflection": {},
    }

    persisted = persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 3, 27, 20, 46, tzinfo=timezone.utc),
    )

    latest = load_task_run_record(persisted["latest_path"])

    assert latest["result"]["runtime_contract"]["trace_identifier"] == "executor_build_20260327_deadbeef"
    assert latest["result"]["runtime_contract"]["snapshot_reference"]["source"] == "mock"
    assert latest["result"]["runtime_contract"]["selected_curve_names"] == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
    }
    assert latest["result"]["runtime_contract"]["evaluation_tags"] == [
        "task_runtime",
        "construct:monte_carlo",
        "market:mock",
    ]
    assert latest["market"]["selected_curve_names"] == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
    }


def test_persist_task_run_record_summarizes_post_build_markers(tmp_path):
    from trellis.agent.task_run_store import load_task_run_record, persist_task_run_record

    task = {
        "id": "T777",
        "title": "Post-build marker demo",
    }
    result = {
        "task_id": "T777",
        "title": task["title"],
        "success": False,
        "start_time": "2026-03-31T14:00:00",
        "comparison_task": True,
        "method_results": {
            "mc_demo": {
                "success": False,
                "attempts": 2,
                "failures": ["timeout"],
                "reflection": {},
                "post_build_tracking": {
                    "last_phase": "reflection_completed",
                    "last_status": "error",
                    "updated_at": "2026-03-31T14:00:30+00:00",
                    "active_flags": {"skip_reflection": False},
                    "events": [
                        {"phase": "build_completed", "status": "ok"},
                        {"phase": "reflection_completed", "status": "error"},
                    ],
                },
            }
        },
        "reflection": {},
        "post_build_tracking": {
            "last_phase": "decision_checkpoint_emitted",
            "last_status": "ok",
            "updated_at": "2026-03-31T14:00:10+00:00",
            "active_flags": {"skip_task_diagnosis_persist": False},
            "events": [
                {"phase": "build_completed", "status": "ok"},
                {"phase": "decision_checkpoint_emitted", "status": "ok"},
            ],
        },
    }

    persisted = persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 3, 31, 14, 1, tzinfo=timezone.utc),
    )

    latest = load_task_run_record(persisted["latest_path"])

    assert latest["post_build"]["latest_method"] == "mc_demo"
    assert latest["post_build"]["latest_phase"] == "reflection_completed"
    assert latest["post_build"]["latest_status"] == "error"
    assert latest["workflow"]["post_build_latest_phase"] == "reflection_completed"
    assert latest["workflow"]["post_build_latest_method"] == "mc_demo"


def test_persist_task_run_record_can_skip_diagnosis_artifacts(monkeypatch, tmp_path):
    from trellis.agent.task_run_store import persist_task_run_record

    monkeypatch.setenv("TRELLIS_SKIP_TASK_DIAGNOSIS_PERSIST", "1")

    persisted = persist_task_run_record(
        {"id": "T778", "title": "Diagnosis skip demo"},
        {"task_id": "T778", "title": "Diagnosis skip demo", "success": True, "reflection": {}},
        root=tmp_path,
        persisted_at=datetime(2026, 3, 31, 14, 5, tzinfo=timezone.utc),
    )

    assert persisted["diagnosis_persist_skipped"] == "env:TRELLIS_SKIP_TASK_DIAGNOSIS_PERSIST"
    assert persisted["diagnosis_persist_error"] == ""


def test_collect_trace_summaries_reads_analytical_trace_json(tmp_path):
    import json

    from trellis.agent.task_run_store import _collect_trace_summaries

    trace_path = tmp_path / "analytical_trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "trace_id": "trace_black76",
                "trace_type": "analytical",
                "status": "ok",
                "route": {
                    "family": "analytical",
                    "name": "analytical_black76",
                    "model": "black76",
                },
                "updated_at": "2026-03-28T00:00:00Z",
                "steps": [
                    {
                        "id": "trace_black76:root",
                        "kind": "trace",
                        "label": "Analytical build",
                        "status": "ok",
                        "parent_id": None,
                        "inputs": {},
                        "outputs": {},
                        "notes": [],
                    }
                ],
                "context": {
                    "route_card": "Black76 route card",
                    "selected_curve_names": {
                        "discount_curve": "usd_ois",
                    },
                    "generation_plan": {
                        "method": "analytical",
                        "instrument_type": "swaption",
                        "instruction_resolution": {
                            "route": "analytical_black76",
                            "effective_instruction_count": 1,
                            "dropped_instruction_count": 0,
                            "conflict_count": 0,
                            "effective_instructions": [
                                {
                                    "id": "analytical_black76:route-helper",
                                    "instruction_type": "hard_constraint",
                                }
                            ],
                            "dropped_instructions": [],
                            "conflicts": [],
                        },
                    },
                },
            },
            indent=2,
        )
    )

    summaries = _collect_trace_summaries(
        {
            "artifacts": {
                "analytical_trace_paths": [str(trace_path)],
            }
        }
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["trace_kind"] == "analytical"
    assert summary["request_id"] == "trace_black76"
    assert summary["route_method"] == "analytical"
    assert summary["action"] == "analytical_black76"
    assert summary["selected_curve_names"] == {"discount_curve": "usd_ois"}
    assert summary["instruction_resolution"]["route"] == "analytical_black76"
    assert summary["instruction_resolution_effective_count"] == 1
    assert summary["instruction_resolution_conflict_count"] == 0


def test_collect_trace_summaries_reads_analytical_curve_names_from_runtime_contract_snapshot(tmp_path):
    import json

    from trellis.agent.task_run_store import _collect_trace_summaries

    trace_path = tmp_path / "analytical_trace_runtime_contract.json"
    trace_path.write_text(
        json.dumps(
            {
                "trace_id": "trace_black76_runtime_contract",
                "trace_type": "analytical",
                "status": "ok",
                "route": {
                    "family": "analytical",
                    "name": "analytical_black76",
                    "model": "black76",
                },
                "updated_at": "2026-04-06T00:00:00Z",
                "steps": [],
                "context": {
                    "runtime_contract": {
                        "snapshot_reference": {
                            "selected_curve_names": {
                                "discount_curve": "usd_ois",
                                "forecast_curve": "USD-SOFR-3M",
                                "credit_curve": "usd_ig",
                            }
                        }
                    }
                },
            },
            indent=2,
        )
    )

    summaries = _collect_trace_summaries(
        {
            "artifacts": {
                "analytical_trace_paths": [str(trace_path)],
            }
        }
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["trace_kind"] == "analytical"
    assert summary["selected_curve_names"] == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
        "credit_curve": "usd_ig",
    }


def test_persist_task_run_record_does_not_promote_platform_action_to_fake_route_id(tmp_path):
    from trellis.agent.task_run_store import load_task_run_record, persist_task_run_record

    trace_root = tmp_path / "trellis" / "agent" / "knowledge" / "traces" / "platform"
    trace_path = trace_root / "executor_build_range_accrual.yaml"
    _write_trace(
        trace_path,
        {
            "request_id": "executor_build_range_accrual",
            "status": "failed",
            "outcome": "request_blocked",
            "action": "build_then_price",
            "route_method": "analytical",
            "updated_at": "2026-04-05T12:00:00+00:00",
            "request_metadata": {
                "task_id": "T301",
                "task_title": "Range accrual route gap",
                "semantic_blueprint": {
                    "dsl_route": None,
                    "dsl_route_family": None,
                },
            },
            "generation_boundary": {
                "method": "analytical",
                "lowering": {
                    "route_id": None,
                    "route_family": None,
                    "primitive_routes": [],
                    "route_modules": [
                        "trellis.models.range_accrual",
                        "trellis.models.contingent_cashflows",
                    ],
                    "expr_kind": "RangeAccrualCouponExpr",
                    "family_ir_type": "RangeAccrualIR",
                    "helper_refs": [],
                    "target_bindings": [],
                    "lowering_errors": [
                        {
                            "code": "missing_primitive_routes",
                        }
                    ],
                },
                "route_binding_authority": {},
                "primitive_plan": {},
            },
            "events": [
                {
                    "event": "request_blocked",
                    "status": "error",
                    "timestamp": "2026-04-05T12:00:00+00:00",
                    "details": {"blocker": "missing primitive route"},
                }
            ],
        },
    )

    task = {
        "id": "T301",
        "title": "Range accrual route gap",
        "construct": ["analytical"],
    }
    result = {
        "task_id": "T301",
        "title": "Range accrual route gap",
        "success": False,
        "failures": ["missing primitive route"],
        "cross_validation": {"status": "insufficient_results"},
        "artifacts": {"platform_trace_paths": [str(trace_path)]},
        "reflection": {},
    }

    persisted = persist_task_run_record(
        task,
        result,
        root=tmp_path,
        persisted_at=datetime(2026, 4, 5, 12, 1, 0, tzinfo=timezone.utc),
    )
    latest = load_task_run_record(persisted["latest_path"])

    assert latest["trace_summaries"][0]["route_health"]["route_id"] == ""
    assert latest["trace_summaries"][0]["route_health"]["route_family"] == "analytical"
    assert latest["telemetry"]["route_observations"][0]["route_id"] == ""
    assert latest["telemetry"]["route_observations"][0]["route_family"] == "analytical"
