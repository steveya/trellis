"""Tests for deterministic agent eval/grade helpers."""

from __future__ import annotations

from pathlib import Path

from trellis.agent.quant import PricingPlan


ROOT = Path(__file__).resolve().parents[2]
AGENT_ARTIFACTS = ROOT / "trellis" / "instruments" / "_agent"
BAD_AMERICAN_SOURCE = """\
from __future__ import annotations

import numpy as np

from trellis.core.market_state import MarketState
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.lsm import LaguerreBasis
from trellis.models.processes.gbm import GBM


class BadAmericanOptionPayoff:
    def __init__(self, spec):
        self._spec = spec

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        T = 1.0
        r = 0.05
        sigma = 0.2
        process = GBM(mu=r, sigma=sigma)
        engine = MonteCarloEngine(process, n_paths=4096, n_steps=64, method="lsm")

        def payoff_fn(paths):
            return np.maximum(spec.strike - paths, 0.0)

        basis = LaguerreBasis()
        _ = basis
        return float(engine.price(spec.spot, T, payoff_fn, discount_rate=r)["price"])
"""


def _plan(method: str = "analytical", instrument_type: str | None = "swaption"):
    from trellis.agent.codegen_guardrails import build_generation_plan

    pricing_plan = PricingPlan(
        method=method,
        method_modules=["trellis.models.black"] if method == "analytical" else [],
        required_market_data={"discount"},
        model_to_build=instrument_type,
        reasoning="test",
    )
    return build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type=instrument_type,
        inspected_modules=("trellis.models.black", "trellis.instruments.cap"),
    )


def test_grade_generation_plan_accepts_inspected_plan():
    from trellis.agent.evals import grade_generation_plan

    report = grade_generation_plan(_plan())
    assert report["inspection_evidence_present"].passed
    assert report["test_selection"].passed
    assert report["test_scope"].passed


def test_grade_generation_plan_flags_missing_inspection():
    from trellis.agent.codegen_guardrails import GenerationPlan
    from trellis.agent.evals import grade_generation_plan

    plan = GenerationPlan(
        method="analytical",
        instrument_type="swaption",
        inspected_modules=(),
        approved_modules=("trellis.models.black",),
        symbols_to_reuse=("black76_call",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
    )
    report = grade_generation_plan(plan)
    assert not report["inspection_evidence_present"].passed


def test_grade_generated_module_reports_import_correctness():
    from trellis.agent.evals import grade_generated_module

    source = """\
from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.models.black import black76_call, black76_put


def evaluate(settlement, expiry, start, end, freq, day_count):
    schedule = generate_schedule(start, end, freq)
    t = year_fraction(settlement, expiry, day_count)
    if schedule:
        return black76_call(0.05, 0.04, 0.2, t)
    return black76_put(0.05, 0.04, 0.2, t)
    """
    report = grade_generated_module(source, _plan())
    assert report["import_correctness"].passed
    assert report["import_correctness"].blocking is False
    assert report["semantic_validity"].passed


def test_grade_generated_module_flags_semantic_invalidity():
    from trellis.agent.evals import grade_generated_module

    report = grade_generated_module(
        BAD_AMERICAN_SOURCE,
        _plan(method="monte_carlo", instrument_type="american_option"),
    )
    assert not report["semantic_validity"].passed
    assert report["semantic_validity"].blocking
    assert any(
        "mc.invalid_method_mode" in detail for detail in report["semantic_validity"].details
    )


def test_grade_generated_module_flags_unsupported_claims():
    from trellis.agent.evals import grade_eval_claims

    claims = ["supports_bsde", "supports_quantization", "analytical"]
    report = grade_eval_claims(claims)
    assert not report["unsupported_claims"].passed
    assert report["unsupported_claims"].blocking
    assert "supports_bsde" in report["unsupported_claims"].details[0]


def test_grade_generation_plan_flags_unknown_test_scope():
    from trellis.agent.codegen_guardrails import GenerationPlan
    from trellis.agent.evals import grade_generation_plan

    plan = GenerationPlan(
        method="analytical",
        instrument_type="swaption",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black",),
        symbols_to_reuse=("black76_call",),
        proposed_tests=("tests/not_a_real_test.py",),
    )

    report = grade_generation_plan(plan)
    assert not report["test_scope"].passed
    assert report["test_scope"].blocking


def test_load_stress_task_manifest():
    from trellis.agent.evals import load_stress_task_manifest

    manifest = load_stress_task_manifest()

    assert set(manifest) >= {"E21", "E22", "E23", "E24", "E25", "E26", "E27", "E28"}
    assert manifest["E28"]["must_keep_targets_distinct"] is True


def test_grade_stress_task_preflight_keeps_fft_and_cos_distinct():
    from trellis.agent.evals import grade_stress_task_preflight, load_stress_task_manifest
    from trellis.agent.task_runtime import load_tasks

    task = next(task for task in load_tasks("E28", "E28", status=None) if task["id"] == "E28")
    report = grade_stress_task_preflight(
        task,
        load_stress_task_manifest()["E28"],
    )

    assert report["market_capability_alignment"].passed
    assert report["comparison_target_inventory"].passed
    assert report["comparison_target_separation"].passed


def test_grade_stress_task_result_flags_forbidden_market_data_failures():
    from trellis.agent.evals import grade_stress_task_result, load_stress_task_manifest
    from trellis.agent.task_runtime import load_tasks

    task = next(task for task in load_tasks("E21", "E21", status=None) if task["id"] == "E21")
    expectation = load_stress_task_manifest()["E21"]
    report = grade_stress_task_result(
        task,
        expectation,
        {
            "task_id": "E21",
            "success": False,
            "error": "MissingCapabilityError: missing market data black_vol_surface",
        },
    )

    assert not report["forbidden_failure_patterns"].passed
    assert not report["outcome_class_alignment"].passed


def test_summarize_stress_tranche_reports_gate_failures_and_buckets():
    from trellis.agent.evals import load_stress_task_manifest, summarize_stress_tranche
    from trellis.agent.task_runtime import load_tasks

    tasks = {
        task["id"]: task
        for task in load_tasks("E21", "E22", status=None)
        if task["id"] in {"E21", "E22"}
    }
    summary = summarize_stress_tranche(
        tasks,
        [
            {"task_id": "E21", "success": True, "cross_validation": {"status": "passed"}},
            {
                "task_id": "E22",
                "success": False,
                "error": "MissingCapabilityError: missing market data black_vol_surface",
            },
        ],
        manifest={task_id: load_stress_task_manifest()[task_id] for task_id in tasks},
    )

    assert summary["totals"]["tasks"] == 2
    assert summary["totals"]["compare_ready"] == 2
    assert summary["totals"]["failed_gate"] == 1
    assert summary["by_task"]["E21"]["passed_gate"] is True
    assert summary["by_task"]["E22"]["passed_gate"] is False
    assert summary["by_task"]["E22"]["failure_bucket"] == "missing_market_data"


def test_classify_task_result_buckets_market_data_blocked_and_comparison_failures():
    from trellis.agent.evals import classify_task_result

    assert classify_task_result(
        {
            "task_id": "T1",
            "success": False,
            "error": "MissingCapabilityError: missing market data black_vol_surface",
        }
    ) == "missing_market_data"
    assert classify_task_result(
        {
            "task_id": "T2",
            "success": False,
            "blocker_details": {"blocker_codes": ["missing_symbol:demo"]},
            "failures": ["blocked for missing primitive"],
        }
    ) == "blocked"
    assert classify_task_result(
        {
            "task_id": "T3",
            "success": False,
            "comparison_task": True,
            "cross_validation": {"status": "failed", "failed_targets": ["fft"]},
        }
    ) == "comparison_failed"
    assert classify_task_result(
        {
            "task_id": "T4",
            "success": False,
            "comparison_task": True,
            "task_diagnosis_failure_bucket": "comparator_build_failure",
        }
    ) == "comparator_build_failure"


def test_summarize_task_results_reports_retry_recovery_and_reviewer_signals():
    from trellis.agent.evals import summarize_task_results

    results = [
        {
            "task_id": "T1",
            "success": True,
            "attempts": 2,
            "reflection": {
                "lesson_captured": "num_010",
                "lessons_attributed": 1,
                "knowledge_trace_saved": "/tmp/t1_knowledge.yaml",
            },
            "token_usage_summary": {
                "call_count": 2,
                "calls_with_usage": 2,
                "calls_without_usage": 0,
                "prompt_tokens": 120,
                "completion_tokens": 40,
                "total_tokens": 160,
                "by_stage": {"code_generation": {"total_tokens": 120}},
                "by_provider": {"anthropic": {"total_tokens": 160}},
            },
            "agent_observations": [
                {"agent": "critic", "severity": "error", "summary": "double discounting"},
                {"agent": "arbiter", "severity": "error", "summary": "price bound failed"},
            ],
            "knowledge_summary": {"lesson_ids": ["bi_004"]},
        },
        {
            "task_id": "T2",
            "success": False,
            "blocker_details": {"blocker_codes": ["missing_symbol:demo"]},
            "token_usage_summary": {
                "call_count": 1,
                "calls_with_usage": 0,
                "calls_without_usage": 1,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "by_stage": {"decomposition": {"calls_without_usage": 1}},
                "by_provider": {"anthropic": {"calls_without_usage": 1}},
            },
            "method_results": {
                "fft": {
                    "agent_observations": [
                        {"agent": "model_validator", "severity": "high", "summary": "unsupported route"}
                    ],
                    "knowledge_summary": {"lesson_ids": ["num_007"]},
                }
            },
        },
        {
            "task_id": "T3",
            "success": False,
            "error": "MissingCapabilityError: missing market data black_vol_surface",
        },
        {
            "task_id": "T4",
            "success": False,
            "comparison_task": True,
            "task_diagnosis_failure_bucket": "comparator_build_failure",
        },
    ]

    summary = summarize_task_results(results)

    assert summary["totals"]["tasks"] == 4
    assert summary["totals"]["successes"] == 1
    assert summary["failure_buckets"]["blocked"] == 1
    assert summary["failure_buckets"]["missing_market_data"] == 1
    assert summary["failure_buckets"]["comparator_build_failure"] == 1
    assert summary["retry_recovery"]["successful_after_retry"] == 1
    assert summary["reviewer_signals"]["tasks_with_reviewer_issues"] == 2
    assert summary["reviewer_signals"]["tasks_recovered_after_review"] == 1
    assert summary["shared_knowledge"]["tasks_with_lessons"] == 2
    assert summary["promotion_discipline"]["captured_lessons"] == 1
    assert summary["promotion_discipline"]["tasks_with_attribution"] == 1
    assert summary["promotion_discipline"]["knowledge_traces"] == 1
    assert summary["promotion_discipline"]["successful_tasks_without_reusable_artifacts"] == []
    assert summary["token_usage"]["total_tokens"] == 160
    assert summary["token_usage"]["call_count"] == 3
    assert summary["token_usage"]["calls_without_usage"] == 1
    assert summary["token_usage"]["by_stage"]["code_generation"]["total_tokens"] == 120


def test_summarize_promotion_discipline_flags_success_without_reusable_artifacts():
    from trellis.agent.evals import summarize_promotion_discipline

    summary = summarize_promotion_discipline([
        {
            "task_id": "T1",
            "success": True,
            "reflection": {},
            "knowledge_summary": {},
        },
        {
            "task_id": "T2",
            "success": True,
            "reflection": {"cookbook_candidate_saved": "/tmp/candidate.yaml"},
            "knowledge_summary": {"lesson_ids": ["num_011"]},
        },
        {
            "task_id": "T3",
            "success": True,
            "reflection": {"promotion_candidate_saved": "/tmp/promotion.yaml"},
            "knowledge_summary": {},
        },
    ])

    assert summary["successful_tasks"] == 3
    assert summary["successful_tasks_with_shared_context"] == 1
    assert summary["cookbook_candidates"] == 1
    assert summary["promotion_candidates"] == 1
    assert summary["successful_tasks_without_reusable_artifacts"] == ["T1"]


def test_compare_task_runs_reports_transitions_and_bucket_deltas():
    from trellis.agent.evals import compare_task_runs

    baseline = [
        {
            "task_id": "T1",
            "success": False,
            "error": "MissingCapabilityError: missing market data black_vol_surface",
        },
        {
            "task_id": "T2",
            "success": False,
            "comparison_task": True,
            "cross_validation": {"status": "failed"},
        },
        {
            "task_id": "T3",
            "success": False,
            "blocker_details": {"blocker_codes": ["missing_symbol:demo"]},
        },
    ]
    candidate = [
        {
            "task_id": "T1",
            "success": True,
            "attempts": 2,
            "agent_observations": [{"agent": "critic", "severity": "error", "summary": "fixed after retry"}],
        },
        {
            "task_id": "T2",
            "success": False,
            "comparison_task": True,
            "cross_validation": {"status": "failed"},
        },
        {
            "task_id": "T3",
            "success": False,
            "blocker_details": {"blocker_codes": ["missing_symbol:demo"]},
        },
    ]

    report = compare_task_runs(baseline, candidate)

    assert report["task_transitions"]["improved"] == 1
    assert report["task_transitions"]["regressed"] == 0
    assert report["task_transitions"]["by_task"]["T1"]["baseline"] == "missing_market_data"
    assert report["task_transitions"]["by_task"]["T1"]["candidate"] == "success"
    assert report["failure_bucket_deltas"]["missing_market_data"] == -1
    assert report["failure_bucket_deltas"]["blocked"] == 0
    assert report["retry_recovery_delta"]["successful_after_retry"] == 1
    assert report["promotion_discipline_delta"]["successful_tasks"] == 1


def test_render_shared_memory_report_has_five_sections():
    from trellis.agent.evals import compare_task_runs, render_shared_memory_report

    report = compare_task_runs(
        [{"task_id": "T1", "success": False, "error": "MissingCapabilityError"}],
        [{"task_id": "T1", "success": True, "attempts": 2}],
    )
    text = render_shared_memory_report(report)

    assert "1. Outcome summary" in text
    assert "2. Task transitions" in text
    assert "3. Failure bucket deltas" in text
    assert "4. Reviewer signal deltas" in text
    assert "5. Retry and shared-knowledge deltas" in text
