"""Deterministic grading helpers for agent build artifacts.

These helpers are intentionally narrow. They grade generation plans and
generated code against the repo-backed guardrails introduced in Tranche 2B.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from trellis.agent.codegen_guardrails import GenerationPlan, validate_generated_imports
from trellis.agent.knowledge.decompose import decompose_to_ir
from trellis.agent.knowledge.methods import CANONICAL_METHODS, normalize_method
from trellis.agent.semantic_validation import validate_semantics
from trellis.agent.task_runtime import (
    _task_comparison_targets,
    _task_construct_methods,
    build_market_state,
    build_market_state_for_task,
)


ROOT = Path(__file__).resolve().parents[2]
STRESS_TASK_MANIFEST = ROOT / "tests" / "evals" / "stress_tasks.yaml"


@dataclass(frozen=True)
class GradeResult:
    """Single grader outcome."""

    passed: bool
    details: tuple[str, ...] = ()


def grade_generation_plan(plan: GenerationPlan) -> dict[str, GradeResult]:
    """Grade a generation plan using deterministic guardrails."""
    results: dict[str, GradeResult] = {}

    if plan.inspected_modules and plan.approved_modules:
        results["inspection_evidence_present"] = GradeResult(True)
    else:
        details = []
        if not plan.inspected_modules:
            details.append("generation plan has no inspected modules")
        if not plan.approved_modules:
            details.append("generation plan has no approved modules")
        results["inspection_evidence_present"] = GradeResult(False, tuple(details))

    expected_tests = tuple(_expected_tests(plan))
    missing = tuple(test for test in expected_tests if test not in plan.proposed_tests)
    if missing:
        results["test_selection"] = GradeResult(
            False,
            tuple(f"missing expected test target: {target}" for target in missing),
        )
    else:
        results["test_selection"] = GradeResult(True)

    return results


def grade_generated_module(source: str, plan: GenerationPlan) -> dict[str, GradeResult]:
    """Grade generated code using deterministic import checks."""
    import_report = validate_generated_imports(source, plan)
    product_ir = None
    if plan.instrument_type:
        try:
            product_ir = decompose_to_ir(
                plan.instrument_type,
                instrument_type=plan.instrument_type,
            )
        except Exception:
            product_ir = None
    semantic_report = validate_semantics(
        source,
        product_ir=product_ir,
        generation_plan=plan,
    )
    return {
        "import_correctness": GradeResult(
            import_report.ok,
            import_report.errors,
        ),
        "semantic_validity": GradeResult(
            semantic_report.ok,
            semantic_report.errors,
        ),
    }


def grade_eval_claims(claims: list[str] | tuple[str, ...]) -> dict[str, GradeResult]:
    """Grade claimed method/capability labels against canonical knowledge."""
    unsupported = []
    for claim in claims:
        normalized = normalize_method(claim)
        if normalized not in CANONICAL_METHODS:
            unsupported.append(f"unsupported claim: {claim}")

    return {
        "unsupported_claims": GradeResult(
            not unsupported,
            tuple(unsupported),
        )
    }


def load_stress_task_manifest(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the canonical stress-task expectation manifest."""
    manifest_path = Path(path) if path is not None else STRESS_TASK_MANIFEST
    with manifest_path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    return {str(task_id): dict(spec) for task_id, spec in loaded.items()}


def grade_stress_task_preflight(
    task: dict[str, Any],
    expectation: dict[str, Any],
    *,
    market_state: Any | None = None,
) -> dict[str, GradeResult]:
    """Grade deterministic task/runtime readiness against the stress-task manifest."""
    if market_state is None:
        market_state, _ = build_market_state_for_task(task, build_market_state())

    required_capabilities = tuple(expectation.get("required_mock_capabilities") or ())
    available_capabilities = set(getattr(market_state, "available_capabilities", ()))
    missing_capabilities = tuple(
        capability for capability in required_capabilities if capability not in available_capabilities
    )

    construct_methods = _task_construct_methods(task)
    comparison_targets = tuple(
        target.target_id for target in _task_comparison_targets(task, construct_methods)
    )
    expected_targets = tuple(expectation.get("comparison_targets") or ())
    missing_targets = tuple(
        target for target in expected_targets if target not in comparison_targets
    )

    duplicate_targets = tuple(
        target for index, target in enumerate(comparison_targets)
        if target in comparison_targets[:index]
    )
    reference_target = expectation.get("reference_target")
    reference_errors = []
    if reference_target and reference_target not in comparison_targets:
        reference_errors.append(f"missing reference target: {reference_target}")

    return {
        "market_capability_alignment": GradeResult(
            not missing_capabilities,
            tuple(
                f"missing mock capability for stress task: {capability}"
                for capability in missing_capabilities
            ),
        ),
        "comparison_target_inventory": GradeResult(
            not missing_targets and bool(comparison_targets),
            tuple(
                [f"task did not produce any comparison targets"]
                if not comparison_targets
                else []
            )
            + tuple(
                f"missing comparison target in task/runtime plan: {target}"
                for target in missing_targets
            )
            + tuple(reference_errors),
        ),
        "comparison_target_separation": GradeResult(
            not duplicate_targets,
            tuple(
                f"duplicate comparison target after normalization: {target}"
                for target in duplicate_targets
            ),
        ),
    }


def grade_stress_task_result(
    task: Mapping[str, Any],
    expectation: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, GradeResult]:
    """Grade one live stress-task result against its manifest expectation."""
    del task  # reserved for future task/result cross-checks

    forbidden_patterns = tuple(str(pattern) for pattern in (expectation.get("forbidden_failure_patterns") or ()))
    haystack = _stress_result_text(result)
    forbidden_hits = tuple(
        pattern for pattern in forbidden_patterns
        if pattern.lower() in haystack
    )

    bucket = classify_task_result(result)
    outcome_class = str(expectation.get("outcome_class") or "").strip().lower()

    if outcome_class == "compare_ready":
        outcome_ok = bucket not in {"missing_market_data", "llm_response", "timeout"}
    elif outcome_class == "honest_block":
        outcome_ok = result.get("success") or bucket not in {"missing_market_data", "llm_response", "timeout"}
    else:
        outcome_ok = True

    details = []
    if not outcome_ok:
        details.append(f"unexpected stress-task outcome bucket: {bucket}")

    return {
        "forbidden_failure_patterns": GradeResult(
            not forbidden_hits,
            tuple(f"forbidden failure pattern observed: {pattern}" for pattern in forbidden_hits),
        ),
        "outcome_class_alignment": GradeResult(
            outcome_ok,
            tuple(details),
        ),
    }


def summarize_stress_tranche(
    tasks: Mapping[str, Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render a canonical summary for one connector stress-tranche batch."""
    expectations = {
        str(task_id): dict(spec)
        for task_id, spec in (manifest or load_stress_task_manifest()).items()
    }
    results_by_id = {
        str(result.get("task_id")): dict(result)
        for result in results
        if result.get("task_id")
    }

    report_by_task: dict[str, dict[str, Any]] = {}
    totals = {
        "tasks": 0,
        "passed_gate": 0,
        "failed_gate": 0,
        "compare_ready": 0,
        "honest_block": 0,
    }
    failure_buckets: dict[str, int] = {}

    for task_id, task in tasks.items():
        expectation = expectations.get(task_id, {})
        result = results_by_id.get(task_id, {})
        preflight = grade_stress_task_preflight(task, expectation) if expectation else {}
        live_report = grade_stress_task_result(task, expectation, result) if expectation and result else {}

        outcome_class = str(expectation.get("outcome_class") or "unknown")
        bucket = classify_task_result(result) if result else "missing"
        failure_buckets[bucket] = failure_buckets.get(bucket, 0) + 1

        passed_gate = all(
            item.passed for item in (
                *preflight.values(),
                *live_report.values(),
            )
        ) if preflight and live_report else False

        totals["tasks"] += 1
        if outcome_class in totals:
            totals[outcome_class] += 1
        if passed_gate:
            totals["passed_gate"] += 1
        else:
            totals["failed_gate"] += 1

        report_by_task[task_id] = {
            "title": task.get("title"),
            "outcome_class": outcome_class,
            "success": bool(result.get("success")),
            "failure_bucket": bucket,
            "run_id": result.get("task_run_history_path") or result.get("start_time"),
            "market_context": dict(result.get("market_context") or {}),
            "comparison_status": (result.get("cross_validation") or {}).get("status"),
            "preflight": {
                key: {"passed": value.passed, "details": list(value.details)}
                for key, value in preflight.items()
            },
            "live_checks": {
                key: {"passed": value.passed, "details": list(value.details)}
                for key, value in live_report.items()
            },
            "passed_gate": passed_gate,
        }

    return {
        "totals": totals,
        "failure_buckets": failure_buckets,
        "by_task": report_by_task,
    }


def classify_task_result(result: Mapping[str, Any]) -> str:
    """Bucket one task result into a stable outcome/failure class."""
    if result.get("success"):
        return "success"

    blocker_details = result.get("blocker_details") or {}
    if blocker_details:
        return "blocked"

    cross_validation = result.get("cross_validation") or {}
    cross_status = str(cross_validation.get("status") or "").strip().lower()
    if cross_status and cross_status != "passed":
        return f"comparison_{cross_status}"

    text = "\n".join(
        str(value)
        for value in (
            result.get("error"),
            *tuple(result.get("failures") or ()),
            json.dumps(blocker_details, default=str) if blocker_details else "",
        )
        if value
    ).lower()

    if any(pattern in text for pattern in ("missingcapabilityerror", "missing market data", "unknown forecast curve", "unknown fx rate")):
        return "missing_market_data"
    if any(pattern in text for pattern in ("semantic validation", "semantic_validation", "semantic_validity")):
        return "semantic_validation"
    if any(pattern in text for pattern in ("import validation", "import_validation")):
        return "import_validation"
    if any(pattern in text for pattern in ("expecting value: line 1 column 1", "empty response", "invalid json")):
        return "llm_response"
    if "timeout" in text:
        return "timeout"
    if result.get("comparison_task"):
        return "comparison_failed"
    return "build_failure"


def summarize_task_results(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one task-result tranche with shared-memory-focused metrics."""
    failure_buckets: dict[str, int] = {}
    attempts: list[int] = []
    token_usage = _empty_token_usage_summary()
    shared_knowledge_tasks = 0
    shared_knowledge_lessons = 0
    tasks_with_reviewer_issues = 0
    tasks_recovered_after_review = 0
    tasks_with_multi_reviewer_issues = 0
    reviewer_agent_counts: dict[str, int] = {}

    successes = 0
    successful_after_retry = 0
    first_attempt_successes = 0

    for result in results:
        bucket = classify_task_result(result)
        failure_buckets[bucket] = failure_buckets.get(bucket, 0) + 1

        if result.get("success"):
            successes += 1
            attempt_count = int(result.get("attempts") or 0)
            if attempt_count <= 1:
                first_attempt_successes += 1
            elif attempt_count > 1:
                successful_after_retry += 1
        attempts.append(int(result.get("attempts") or 0))

        knowledge_summaries = list(_iter_knowledge_summaries(result))
        if any(summary for summary in knowledge_summaries):
            shared_knowledge_tasks += 1
        if any(summary.get("lesson_ids") for summary in knowledge_summaries):
            shared_knowledge_lessons += 1

        observations = list(_iter_agent_observations(result))
        reviewer_issue_agents = {
            observation.get("agent")
            for observation in observations
            if observation.get("agent") in {"critic", "arbiter", "model_validator"}
            and str(observation.get("severity", "")).lower() in {"error", "critical", "high"}
        }
        if reviewer_issue_agents:
            tasks_with_reviewer_issues += 1
            if result.get("success") and int(result.get("attempts") or 0) > 1:
                tasks_recovered_after_review += 1
        if len(reviewer_issue_agents) >= 2:
            tasks_with_multi_reviewer_issues += 1
        for agent in reviewer_issue_agents:
            reviewer_agent_counts[agent] = reviewer_agent_counts.get(agent, 0) + 1

        _merge_token_usage_summary(token_usage, result.get("token_usage_summary") or {})

    return {
        "totals": {
            "tasks": len(results),
            "successes": successes,
            "failures": len(results) - successes,
            "avg_attempts": round(sum(attempts) / len(attempts), 2) if attempts else 0.0,
        },
        "failure_buckets": failure_buckets,
        "retry_recovery": {
            "successful_after_retry": successful_after_retry,
            "first_attempt_successes": first_attempt_successes,
        },
        "reviewer_signals": {
            "tasks_with_reviewer_issues": tasks_with_reviewer_issues,
            "tasks_with_multi_reviewer_issues": tasks_with_multi_reviewer_issues,
            "tasks_recovered_after_review": tasks_recovered_after_review,
            "reviewer_agent_counts": reviewer_agent_counts,
        },
        "shared_knowledge": {
            "tasks_with_shared_context": shared_knowledge_tasks,
            "tasks_with_lessons": shared_knowledge_lessons,
        },
        "promotion_discipline": summarize_promotion_discipline(results),
        "token_usage": token_usage,
    }


def summarize_promotion_discipline(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize whether successful reruns left reusable learning artifacts behind."""
    successful_results = [result for result in results if result.get("success")]
    captured_lesson_ids: set[str] = set()
    cookbook_candidate_paths: set[str] = set()
    knowledge_trace_paths: set[str] = set()
    attributed_successes = 0
    successes_with_shared_context = 0
    successes_without_reusable_artifacts: list[str] = []

    for result in successful_results:
        reusable = False
        task_has_attribution = False
        if any(summary.get("lesson_ids") for summary in _iter_knowledge_summaries(result)):
            successes_with_shared_context += 1
            reusable = True
        for reflection in _iter_reflection_payloads(result):
            lesson_captured = reflection.get("lesson_captured")
            if isinstance(lesson_captured, str) and lesson_captured.strip():
                captured_lesson_ids.add(lesson_captured.strip())
                reusable = True
            elif isinstance(lesson_captured, list):
                for item in lesson_captured:
                    if isinstance(item, str) and item.strip():
                        captured_lesson_ids.add(item.strip())
                        reusable = True
            if int(reflection.get("lessons_attributed") or 0) > 0:
                task_has_attribution = True
                reusable = True
            cookbook_candidate = reflection.get("cookbook_candidate_saved")
            if isinstance(cookbook_candidate, str) and cookbook_candidate.strip():
                cookbook_candidate_paths.add(cookbook_candidate.strip())
                reusable = True
            knowledge_trace = reflection.get("knowledge_trace_saved")
            if isinstance(knowledge_trace, str) and knowledge_trace.strip():
                knowledge_trace_paths.add(knowledge_trace.strip())
                reusable = True
        if task_has_attribution:
            attributed_successes += 1
        if not reusable:
            task_id = str(result.get("task_id") or "").strip()
            if task_id:
                successes_without_reusable_artifacts.append(task_id)

    return {
        "successful_tasks": len(successful_results),
        "successful_tasks_with_shared_context": successes_with_shared_context,
        "captured_lessons": len(captured_lesson_ids),
        "tasks_with_attribution": attributed_successes,
        "cookbook_candidates": len(cookbook_candidate_paths),
        "knowledge_traces": len(knowledge_trace_paths),
        "successful_tasks_without_reusable_artifacts": successes_without_reusable_artifacts,
    }


def _empty_token_usage_summary() -> dict[str, Any]:
    """Return the zero-value token-usage summary shape used in reports."""
    return {
        "call_count": 0,
        "calls_with_usage": 0,
        "calls_without_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_stage": {},
        "by_provider": {},
    }


def _merge_token_usage_summary(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    """Accumulate one token-usage summary into an aggregate report bucket."""
    if not incoming:
        return
    for key in (
        "call_count",
        "calls_with_usage",
        "calls_without_usage",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        target[key] += int(incoming.get(key, 0) or 0)

    for group_name in ("by_stage", "by_provider"):
        incoming_group = incoming.get(group_name) or {}
        target_group = target[group_name]
        for group_key, group_summary in incoming_group.items():
            bucket = target_group.setdefault(
                group_key,
                {
                    "call_count": 0,
                    "calls_with_usage": 0,
                    "calls_without_usage": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            for metric in bucket:
                bucket[metric] += int(group_summary.get(metric, 0) or 0)


def compare_task_runs(
    baseline_results: list[Mapping[str, Any]],
    candidate_results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two task-result tranches using stable outcome buckets."""
    baseline_summary = summarize_task_results(baseline_results)
    candidate_summary = summarize_task_results(candidate_results)

    baseline_by_id = {str(result.get("task_id")): result for result in baseline_results if result.get("task_id")}
    candidate_by_id = {str(result.get("task_id")): result for result in candidate_results if result.get("task_id")}
    all_task_ids = sorted(set(baseline_by_id) | set(candidate_by_id))

    improved = 0
    regressed = 0
    unchanged = 0
    by_task: dict[str, dict[str, Any]] = {}

    for task_id in all_task_ids:
        baseline_bucket = classify_task_result(baseline_by_id.get(task_id, {})) if task_id in baseline_by_id else "missing"
        candidate_bucket = classify_task_result(candidate_by_id.get(task_id, {})) if task_id in candidate_by_id else "missing"
        transition = _transition_label(baseline_bucket, candidate_bucket)
        if transition == "improved":
            improved += 1
        elif transition == "regressed":
            regressed += 1
        else:
            unchanged += 1
        by_task[task_id] = {
            "baseline": baseline_bucket,
            "candidate": candidate_bucket,
            "transition": transition,
        }

    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "task_transitions": {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "by_task": by_task,
        },
        "failure_bucket_deltas": _dict_deltas(
            baseline_summary["failure_buckets"],
            candidate_summary["failure_buckets"],
        ),
        "reviewer_signal_delta": _dict_deltas(
            baseline_summary["reviewer_signals"],
            candidate_summary["reviewer_signals"],
            nested_keys={"reviewer_agent_counts"},
        ),
        "retry_recovery_delta": _dict_deltas(
            baseline_summary["retry_recovery"],
            candidate_summary["retry_recovery"],
        ),
        "shared_knowledge_delta": _dict_deltas(
            baseline_summary["shared_knowledge"],
            candidate_summary["shared_knowledge"],
        ),
        "promotion_discipline_delta": _dict_deltas(
            baseline_summary["promotion_discipline"],
            candidate_summary["promotion_discipline"],
            list_keys={"successful_tasks_without_reusable_artifacts"},
        ),
    }


def render_shared_memory_report(report: Mapping[str, Any]) -> str:
    """Render a compact five-part shared-memory improvement report."""
    baseline = report["baseline"]
    candidate = report["candidate"]
    transitions = report["task_transitions"]
    failure_deltas = report["failure_bucket_deltas"]
    reviewer_delta = report["reviewer_signal_delta"]
    retry_delta = report["retry_recovery_delta"]
    shared_delta = report["shared_knowledge_delta"]
    promotion_delta = report["promotion_discipline_delta"]

    return "\n".join(
        [
            "1. Outcome summary",
            f"baseline: {baseline['totals']['successes']}/{baseline['totals']['tasks']} success",
            f"candidate: {candidate['totals']['successes']}/{candidate['totals']['tasks']} success",
            "",
            "2. Task transitions",
            f"improved: {transitions['improved']}",
            f"regressed: {transitions['regressed']}",
            f"unchanged: {transitions['unchanged']}",
            "",
            "3. Failure bucket deltas",
            *(f"{bucket}: {delta:+d}" for bucket, delta in sorted(failure_deltas.items())),
            "",
            "4. Reviewer signal deltas",
            *(f"{metric}: {delta:+d}" for metric, delta in sorted(
                (item for item in reviewer_delta.items() if isinstance(item[1], (int, float))),
                key=lambda item: item[0],
            )),
            "",
            "5. Retry and shared-knowledge deltas",
            *(f"{metric}: {delta:+d}" for metric, delta in sorted(retry_delta.items())),
            *(f"{metric}: {delta:+d}" for metric, delta in sorted(shared_delta.items())),
            *(f"{metric}: {delta:+d}" for metric, delta in sorted(
                (
                    item for item in promotion_delta.items()
                    if isinstance(item[1], (int, float))
                ),
                key=lambda item: item[0],
            )),
        ]
    )


def _expected_tests(plan: GenerationPlan) -> list[str]:
    """Return the deterministic regression tests that should accompany a plan."""
    expected = ["tests/test_agent/test_build_loop.py"]
    if plan.instrument_type == "swaption":
        expected.append("tests/test_agent/test_swaption_demo.py")
    if plan.instrument_type == "callable_bond":
        expected.append("tests/test_agent/test_callable_bond.py")
    return expected


def _stress_result_text(result: Mapping[str, Any]) -> str:
    """Return a lowercase text surface used to scan live stress-task failures."""
    fragments = [
        str(result.get("error") or ""),
        *[str(item) for item in (result.get("failures") or ())],
    ]
    cross_validation = result.get("cross_validation") or {}
    if cross_validation:
        fragments.append(json.dumps(cross_validation, default=str))
    blocker_details = result.get("blocker_details") or {}
    if blocker_details:
        fragments.append(json.dumps(blocker_details, default=str))
    return "\n".join(fragment for fragment in fragments if fragment).lower()


def _iter_agent_observations(result: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield normalized agent-observation payloads from top-level and per-method results."""
    observations = result.get("agent_observations") or ()
    for observation in observations:
        if isinstance(observation, dict):
            yield observation
    for payload in (result.get("method_results") or {}).values():
        if not isinstance(payload, Mapping):
            continue
        for observation in payload.get("agent_observations") or ():
            if isinstance(observation, dict):
                yield observation


def _iter_knowledge_summaries(result: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield normalized knowledge-summary payloads from top-level and per-method results."""
    summary = result.get("knowledge_summary")
    if isinstance(summary, Mapping):
        yield dict(summary)
    for payload in (result.get("method_results") or {}).values():
        if isinstance(payload, Mapping) and isinstance(payload.get("knowledge_summary"), Mapping):
            yield dict(payload["knowledge_summary"])


def _iter_reflection_payloads(result: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield top-level and per-method reflection payloads."""
    reflection = result.get("reflection")
    if isinstance(reflection, Mapping):
        yield dict(reflection)
        method_reflections = reflection.get("method_reflections")
        if isinstance(method_reflections, Mapping):
            for payload in method_reflections.values():
                if isinstance(payload, Mapping):
                    yield dict(payload)
    for payload in (result.get("method_results") or {}).values():
        if isinstance(payload, Mapping) and isinstance(payload.get("reflection"), Mapping):
            yield dict(payload["reflection"])


def _transition_label(baseline_bucket: str, candidate_bucket: str) -> str:
    """Classify a task transition as improved, regressed, or unchanged."""
    if baseline_bucket == candidate_bucket:
        return "unchanged"
    if baseline_bucket != "success" and candidate_bucket == "success":
        return "improved"
    if baseline_bucket == "success" and candidate_bucket != "success":
        return "regressed"
    return "unchanged"


def _dict_deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    nested_keys: set[str] | None = None,
    list_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Compute numeric deltas between two metric dictionaries, recursing selectively."""
    nested_keys = nested_keys or set()
    list_keys = list_keys or set()
    keys = set(baseline) | set(candidate)
    deltas: dict[str, Any] = {}
    for key in keys:
        left = baseline.get(key, 0)
        right = candidate.get(key, 0)
        if key in nested_keys and isinstance(left, Mapping) and isinstance(right, Mapping):
            deltas[key] = _dict_deltas(left, right)
        elif key in list_keys and isinstance(left, list) and isinstance(right, list):
            deltas[key] = len(right) - len(left)
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[key] = right - left
    return deltas
