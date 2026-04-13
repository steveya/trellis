"""Deterministic grading helpers for agent build artifacts.

These helpers are intentionally narrow. They grade generation plans and
generated code against the repo-backed guardrails introduced in Tranche 2B.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import yaml

from trellis.agent.codegen_guardrails import GenerationPlan, validate_generated_imports
from trellis.agent.knowledge.decompose import decompose_to_ir
from trellis.agent.knowledge.methods import CANONICAL_METHODS, normalize_method
from trellis.agent.knowledge.import_registry import get_test_map
from trellis.agent.knowledge.schema import EvalSpec, GraderSpec
from trellis.agent.semantic_validation import validate_semantics
from trellis.agent.task_runtime import (
    _task_comparison_targets,
    _task_construct_methods,
    build_market_state,
    build_market_state_for_task,
)


ROOT = Path(__file__).resolve().parents[2]
STRESS_TASK_MANIFEST = ROOT / "tests" / "evals" / "stress_tasks.yaml"
BINDING_FIRST_EXOTIC_PROOF_MANIFEST = ROOT / "tests" / "evals" / "binding_first_exotic_proof.yaml"
STRESS_COMPARE_READY = "compare_ready"
STRESS_HONEST_BLOCK = "honest_block"
PROOF_OUTCOME_PROVED = "proved"
PROOF_OUTCOME_HONEST_BLOCK = "honest_block"


@dataclass(frozen=True)
class GradeResult:
    """Single grader outcome."""

    passed: bool
    details: tuple[str, ...] = ()
    blocking: bool = False


DEFAULT_GRADER_SPECS: tuple[GraderSpec, ...] = (
    GraderSpec(
        id="inspection_evidence_present",
        category="repo_inspection",
        description="Generated plans must show inspected and approved modules.",
        hard=True,
        applies_to=("generation_plan",),
        signals=("inspected_modules", "approved_modules"),
    ),
    GraderSpec(
        id="test_selection",
        category="test_scope",
        description="Generated plans must include the expected regression tests.",
        hard=True,
        applies_to=("generation_plan",),
        signals=("proposed_tests",),
    ),
    GraderSpec(
        id="test_scope",
        category="test_scope",
        description="Generated plans must point at real tests in the repo.",
        hard=True,
        applies_to=("generation_plan",),
        signals=("proposed_tests",),
    ),
    GraderSpec(
        id="import_correctness",
        category="imports",
        description="Generated code must only import approved Trellis modules and exported symbols.",
        hard=True,
        applies_to=("generated_module",),
        signals=("trellis_imports",),
    ),
    GraderSpec(
        id="semantic_validity",
        category="semantics",
        description="Generated code must match the intended product semantics.",
        hard=True,
        applies_to=("generated_module",),
        signals=("semantic_contracts",),
    ),
    GraderSpec(
        id="unsupported_claims",
        category="capability_claims",
        description="Free-form capability claims must stay within canonical method support.",
        hard=True,
        applies_to=("claims",),
        signals=("claimed_capabilities",),
    ),
)

DEFAULT_EVAL_SPEC = EvalSpec(
    id="agent_codegen_guardrails",
    title="Agent code-generation guardrails",
    description="Deterministic validation surface for import, symbol, claim, and test-scope failures.",
    grader_ids=tuple(spec.id for spec in DEFAULT_GRADER_SPECS),
    hard_gates=tuple(spec.id for spec in DEFAULT_GRADER_SPECS if spec.hard),
    benchmark_ids=("stress_tasks",),
)


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
        results["inspection_evidence_present"] = GradeResult(False, tuple(details), blocking=True)

    expected_tests = tuple(_expected_tests(plan))
    missing = tuple(test for test in expected_tests if test not in plan.proposed_tests)
    if missing:
        results["test_selection"] = GradeResult(
            False,
            tuple(f"missing expected test target: {target}" for target in missing),
            blocking=True,
        )
    else:
        results["test_selection"] = GradeResult(True)

    results["test_scope"] = _grade_test_scope(plan)

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
            blocking=not import_report.ok,
        ),
        "semantic_validity": GradeResult(
            semantic_report.ok,
            semantic_report.errors,
            blocking=not semantic_report.ok,
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
            blocking=bool(unsupported),
        )
    }


def load_stress_task_manifest(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the canonical stress-task expectation manifest."""
    manifest_path = Path(path) if path is not None else STRESS_TASK_MANIFEST
    with manifest_path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    return {str(task_id): dict(spec) for task_id, spec in loaded.items()}


def load_binding_first_exotic_proof_manifest(
    path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the canonical binding-first exotic proof manifest."""
    manifest_path = Path(path) if path is not None else BINDING_FIRST_EXOTIC_PROOF_MANIFEST
    with manifest_path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    return {str(task_id): dict(spec) for task_id, spec in loaded.items()}


def select_binding_first_exotic_proof_tasks(
    manifest: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    cohort: str | None = None,
    task_ids: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select one proof-cohort slice from the manifest."""
    selected_ids = {str(task_id) for task_id in (task_ids or ()) if str(task_id).strip()}
    expectations = {
        str(task_id): dict(spec)
        for task_id, spec in (manifest or load_binding_first_exotic_proof_manifest()).items()
    }
    selected: dict[str, dict[str, Any]] = {}
    for task_id, spec in expectations.items():
        if cohort is not None and str(spec.get("cohort") or "").strip() != cohort:
            continue
        if selected_ids and task_id not in selected_ids:
            continue
        selected[task_id] = spec
    if selected_ids:
        missing = tuple(sorted(selected_ids - set(selected)))
        if missing:
            raise ValueError(
                "Unknown proof task ids requested: " + ", ".join(missing)
            )
    return selected


def _task_contract_required_capabilities(task: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the market capabilities the task contract claims it needs."""
    market_assertions = task.get("market_assertions") or {}
    return tuple(str(item) for item in (market_assertions.get("requires") or ()))


def _task_contract_comparison_targets(task: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the normalized comparison targets implied by the task contract."""
    cross_validate = task.get("cross_validate") or {}
    internal = tuple(str(item) for item in (cross_validate.get("internal") or ()))
    analytical = cross_validate.get("analytical")
    if analytical:
        return internal + (str(analytical),)
    return internal


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
    contract_capabilities = _task_contract_required_capabilities(task)
    available_capabilities = set(getattr(market_state, "available_capabilities", ()))
    missing_capabilities = tuple(
        capability for capability in required_capabilities if capability not in available_capabilities
    )

    construct_methods = _task_construct_methods(task)
    comparison_targets = tuple(
        target.target_id for target in _task_comparison_targets(task, construct_methods)
    )
    expected_targets = tuple(expectation.get("comparison_targets") or ())
    contract_targets = _task_contract_comparison_targets(task)
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

    contract_drift = []
    if required_capabilities != contract_capabilities:
        contract_drift.append(
            "required mock capabilities drift: "
            f"manifest={required_capabilities} task_contract={contract_capabilities}"
        )
    if expected_targets != contract_targets:
        contract_drift.append(
            "comparison target drift: "
            f"manifest={expected_targets} task_contract={contract_targets}"
        )

    return {
        "task_contract_alignment": GradeResult(
            not contract_drift,
            tuple(contract_drift),
        ),
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
    forbidden_patterns = tuple(str(pattern) for pattern in (expectation.get("forbidden_failure_patterns") or ()))
    haystack = _stress_result_text(result)
    forbidden_hits = tuple(
        pattern for pattern in forbidden_patterns
        if pattern.lower() in haystack
    )

    bucket = classify_task_result(result)
    outcome_class = _normalize_stress_outcome_class(expectation.get("outcome_class"))
    observed_reference_target = str(
        (result.get("cross_validation") or {}).get("reference_target") or ""
    ).strip()
    expected_reference_target = str(expectation.get("reference_target") or "").strip()
    comparison_status = str(
        (result.get("cross_validation") or {}).get("status") or ""
    ).strip().lower()
    expected_blocker_categories = tuple(
        str(category) for category in (expectation.get("expected_blocker_categories") or ())
    )
    observed_blocker_categories = _stress_observed_blocker_categories(result)

    outcome_ok = True
    outcome_details: list[str] = []
    if outcome_class == STRESS_COMPARE_READY:
        if not result.get("success"):
            outcome_ok = False
            outcome_details.append(f"compare-ready task did not succeed: bucket={bucket}")
        if task.get("cross_validate") and comparison_status != "passed":
            outcome_ok = False
            outcome_details.append(
                f"compare-ready task did not finish with passed comparison status: {comparison_status or 'missing'}"
            )
        if observed_blocker_categories:
            outcome_ok = False
            outcome_details.append(
                "compare-ready task surfaced blocker categories: "
                + ", ".join(observed_blocker_categories)
            )
    elif outcome_class == STRESS_HONEST_BLOCK:
        if not result.get("success"):
            if expected_blocker_categories:
                if not observed_blocker_categories:
                    outcome_ok = False
                    outcome_details.append(
                        "honest-block task did not surface blocker categories"
                    )
                elif not set(observed_blocker_categories) & set(expected_blocker_categories):
                    outcome_ok = False
                    outcome_details.append(
                        "honest-block task surfaced unexpected blocker categories: "
                        f"observed={observed_blocker_categories} expected={expected_blocker_categories}"
                    )
            elif bucket in {"missing_market_data", "llm_response", "timeout"}:
                outcome_ok = False
                outcome_details.append(f"unexpected honest-block outcome bucket: {bucket}")

    blocker_alignment_ok = True
    blocker_alignment_details: list[str] = []
    if expected_blocker_categories and not result.get("success"):
        observed = set(observed_blocker_categories)
        expected = set(expected_blocker_categories)
        if not observed:
            blocker_alignment_ok = False
            blocker_alignment_details.append(
                "expected blocker categories were not surfaced in the result"
            )
        elif not observed & expected:
            blocker_alignment_ok = False
            blocker_alignment_details.append(
                "observed blocker categories did not match the manifest expectation: "
                f"observed={observed_blocker_categories} expected={expected_blocker_categories}"
            )
    elif outcome_class == STRESS_COMPARE_READY and observed_blocker_categories:
        blocker_alignment_ok = False
        blocker_alignment_details.append(
            "compare-ready task should not emit blocker categories: "
            + ", ".join(observed_blocker_categories)
        )

    reference_alignment_ok = True
    reference_alignment_details: list[str] = []
    if expected_reference_target and comparison_status:
        if observed_reference_target != expected_reference_target:
            reference_alignment_ok = False
            reference_alignment_details.append(
                "comparison reference target drifted from the manifest: "
                f"observed={observed_reference_target or 'missing'} expected={expected_reference_target}"
            )

    return {
        "forbidden_failure_patterns": GradeResult(
            not forbidden_hits,
            tuple(f"forbidden failure pattern observed: {pattern}" for pattern in forbidden_hits),
        ),
        "outcome_class_alignment": GradeResult(
            outcome_ok,
            tuple(outcome_details),
        ),
        "blocker_category_alignment": GradeResult(
            blocker_alignment_ok,
            tuple(blocker_alignment_details),
        ),
        "reference_target_alignment": GradeResult(
            reference_alignment_ok,
            tuple(reference_alignment_details),
        ),
    }


def grade_binding_first_exotic_proof_preflight(
    task: Mapping[str, Any],
    expectation: Mapping[str, Any],
    *,
    market_state: Any | None = None,
) -> dict[str, GradeResult]:
    """Grade deterministic readiness for one binding-first proof task."""
    return grade_stress_task_preflight(
        dict(task),
        dict(expectation),
        market_state=market_state,
    )


def grade_binding_first_exotic_proof_result(
    task: Mapping[str, Any],
    expectation: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, GradeResult]:
    """Grade one live proof-cohort result against the binding-first expectation."""
    forbidden_patterns = tuple(
        str(pattern) for pattern in (expectation.get("forbidden_failure_patterns") or ())
    )
    haystack = _stress_result_text(result)
    forbidden_hits = tuple(
        pattern for pattern in forbidden_patterns if pattern.lower() in haystack
    )

    outcome_class = _normalize_proof_outcome_class(expectation.get("outcome_class"))
    bucket = classify_task_result(result)
    cross_validation = dict(result.get("cross_validation") or {})
    comparison_status = str(cross_validation.get("status") or "").strip().lower()
    expected_blocker_categories = tuple(
        str(category) for category in (expectation.get("expected_blocker_categories") or ())
    )
    observed_blocker_categories = _stress_observed_blocker_categories(result)
    binding_summary = _proof_binding_summary(result)
    binding_ids = tuple(binding_summary.get("binding_ids") or ())
    route_ids = tuple(binding_summary.get("route_ids") or ())

    outcome_ok = True
    outcome_details: list[str] = []
    if outcome_class == PROOF_OUTCOME_PROVED:
        if not result.get("success"):
            outcome_ok = False
            outcome_details.append(f"proved task did not succeed: bucket={bucket}")
        if task.get("cross_validate") and comparison_status != "passed":
            outcome_ok = False
            outcome_details.append(
                "proved task did not finish with passed comparison status: "
                f"{comparison_status or 'missing'}"
            )
        if observed_blocker_categories:
            outcome_ok = False
            outcome_details.append(
                "proved task surfaced blocker categories: "
                + ", ".join(observed_blocker_categories)
            )
    elif outcome_class == PROOF_OUTCOME_HONEST_BLOCK:
        if result.get("success"):
            outcome_ok = False
            outcome_details.append("honest-block sentinel unexpectedly succeeded")
        if expected_blocker_categories:
            if not observed_blocker_categories:
                outcome_ok = False
                outcome_details.append(
                    "honest-block sentinel did not surface blocker categories"
                )
            elif not set(observed_blocker_categories) & set(expected_blocker_categories):
                outcome_ok = False
                outcome_details.append(
                    "honest-block sentinel surfaced unexpected blocker categories: "
                    f"observed={observed_blocker_categories} expected={expected_blocker_categories}"
                )
        elif bucket in {"missing_market_data", "llm_response", "timeout"}:
            outcome_ok = False
            outcome_details.append(f"unexpected honest-block outcome bucket: {bucket}")

    binding_ok = True
    binding_details: list[str] = []
    requires_binding_ids = expectation.get("requires_binding_ids")
    if requires_binding_ids is None:
        requires_binding_ids = outcome_class == PROOF_OUTCOME_PROVED
    if requires_binding_ids and not binding_ids:
        binding_ok = False
        binding_details.append(
            "proved task did not persist any binding ids in its task-run record"
        )

    route_identity_ok = True
    route_identity_details: list[str] = []
    unknown_route_ids = _unknown_route_ids(route_ids)
    if unknown_route_ids:
        route_identity_ok = False
        route_identity_details.append(
            "task surfaced non-canonical route ids: " + ", ".join(unknown_route_ids)
        )

    reference_alignment_ok = True
    reference_alignment_details: list[str] = []
    expected_reference_target = str(expectation.get("reference_target") or "").strip()
    observed_reference_target = str(cross_validation.get("reference_target") or "").strip()
    if expected_reference_target and comparison_status:
        if observed_reference_target != expected_reference_target:
            reference_alignment_ok = False
            reference_alignment_details.append(
                "comparison reference target drifted from the proof manifest: "
                f"observed={observed_reference_target or 'missing'} expected={expected_reference_target}"
            )

    blocker_alignment_ok = True
    blocker_alignment_details: list[str] = []
    if expected_blocker_categories and not result.get("success"):
        observed = set(observed_blocker_categories)
        expected = set(expected_blocker_categories)
        if not observed:
            blocker_alignment_ok = False
            blocker_alignment_details.append(
                "expected blocker categories were not surfaced in the result"
            )
        elif not observed & expected:
            blocker_alignment_ok = False
            blocker_alignment_details.append(
                "observed blocker categories did not match the proof expectation: "
                f"observed={observed_blocker_categories} expected={expected_blocker_categories}"
            )
    elif outcome_class == PROOF_OUTCOME_PROVED and observed_blocker_categories:
        blocker_alignment_ok = False
        blocker_alignment_details.append(
            "proved task should not emit blocker categories: "
            + ", ".join(observed_blocker_categories)
        )

    return {
        "forbidden_failure_patterns": GradeResult(
            not forbidden_hits,
            tuple(f"forbidden failure pattern observed: {pattern}" for pattern in forbidden_hits),
        ),
        "outcome_class_alignment": GradeResult(
            outcome_ok,
            tuple(outcome_details),
        ),
        "binding_identity_alignment": GradeResult(
            binding_ok,
            tuple(binding_details),
        ),
        "route_identity_alignment": GradeResult(
            route_identity_ok,
            tuple(route_identity_details),
        ),
        "blocker_category_alignment": GradeResult(
            blocker_alignment_ok,
            tuple(blocker_alignment_details),
        ),
        "reference_target_alignment": GradeResult(
            reference_alignment_ok,
            tuple(reference_alignment_details),
        ),
    }


def summarize_stress_preflight(
    tasks: Mapping[str, Mapping[str, Any]],
    *,
    manifest: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render a canonical summary for one deterministic stress-task preflight pass."""
    expectations = {
        str(task_id): dict(spec)
        for task_id, spec in (manifest or load_stress_task_manifest()).items()
    }
    by_task: dict[str, dict[str, Any]] = {}
    failed_tasks: list[str] = []
    totals = {
        "tasks": 0,
        "passed": 0,
        "failed": 0,
    }

    for task_id, task in tasks.items():
        expectation = expectations.get(task_id, {})
        report = grade_stress_task_preflight(task, expectation)
        blocked = not all(item.passed for item in report.values())
        by_task[task_id] = {
            "title": task.get("title"),
            "blocked": blocked,
            "checks": {
                key: {"passed": value.passed, "details": list(value.details)}
                for key, value in report.items()
            },
        }
        totals["tasks"] += 1
        if blocked:
            totals["failed"] += 1
            failed_tasks.append(task_id)
        else:
            totals["passed"] += 1

    return {
        "totals": totals,
        "failed_tasks": failed_tasks,
        "by_task": by_task,
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

        outcome_class = _normalize_stress_outcome_class(expectation.get("outcome_class")) or "unknown"
        bucket = classify_task_result(result) if result else "missing"
        failure_buckets[bucket] = failure_buckets.get(bucket, 0) + 1
        blocker_categories = _stress_observed_blocker_categories(result) if result else ()
        follow_on = _stress_follow_on_candidate(
            task_id,
            task,
            expectation,
            result,
            blocker_categories=blocker_categories,
        )

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
            "reference_target": expectation.get("reference_target"),
            "observed_reference_target": (result.get("cross_validation") or {}).get("reference_target"),
            "expected_blocker_categories": list(expectation.get("expected_blocker_categories") or []),
            "observed_blocker_categories": list(blocker_categories),
            "task_run_latest_path": result.get("task_run_latest_path"),
            "task_run_history_path": result.get("task_run_history_path"),
            "diagnosis_packet_path": result.get("task_diagnosis_packet_path"),
            "diagnosis_dossier_path": result.get("task_diagnosis_dossier_path"),
            "diagnosis_latest_packet_path": result.get("task_diagnosis_latest_packet_path"),
            "diagnosis_latest_dossier_path": result.get("task_diagnosis_latest_dossier_path"),
            "preflight": {
                key: {"passed": value.passed, "details": list(value.details)}
                for key, value in preflight.items()
            },
            "live_checks": {
                key: {"passed": value.passed, "details": list(value.details)}
                for key, value in live_report.items()
            },
            "passed_gate": passed_gate,
            "follow_on": follow_on,
        }

    return {
        "totals": totals,
        "failure_buckets": failure_buckets,
        "by_task": report_by_task,
        "preflight_summary": summarize_stress_preflight(tasks, manifest=expectations),
        "artifact_inventory": _stress_artifact_inventory(results_by_id),
        "follow_on_candidates": [
            report["follow_on"]
            for report in report_by_task.values()
            if isinstance(report.get("follow_on"), Mapping)
            and report["follow_on"].get("action") != "no_action"
        ],
    }


def summarize_binding_first_exotic_proof(
    tasks: Mapping[str, Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render a canonical summary for one binding-first exotic proof cohort."""
    expectations = {
        str(task_id): dict(spec)
        for task_id, spec in (manifest or load_binding_first_exotic_proof_manifest()).items()
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
        PROOF_OUTCOME_PROVED: 0,
        PROOF_OUTCOME_HONEST_BLOCK: 0,
    }
    failure_buckets: dict[str, int] = {}

    for task_id, task in tasks.items():
        expectation = expectations.get(task_id, {})
        result = results_by_id.get(task_id, {})
        preflight = (
            grade_binding_first_exotic_proof_preflight(task, expectation)
            if expectation else {}
        )
        live_report = (
            grade_binding_first_exotic_proof_result(task, expectation, result)
            if expectation and result else {}
        )
        outcome_class = _normalize_proof_outcome_class(expectation.get("outcome_class")) or "unknown"
        bucket = classify_task_result(result) if result else "missing"
        failure_buckets[bucket] = failure_buckets.get(bucket, 0) + 1
        binding_summary = _proof_binding_summary(result) if result else _empty_proof_binding_summary()
        passed_gate = all(
            item.passed for item in (*preflight.values(), *live_report.values())
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
            "cohort": str(expectation.get("cohort") or "").strip(),
            "outcome_class": outcome_class,
            "success": bool(result.get("success")),
            "failure_bucket": bucket,
            "comparison_status": (result.get("cross_validation") or {}).get("status"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "attempts_to_success": _result_attempts_to_success(result) if result else 0,
            "first_pass": _result_attempts_to_success(result) <= 1 if result else False,
            "retry_taxonomy": list(_result_retry_taxonomy_reasons(result)) if result else [],
            "token_usage": dict(result.get("token_usage_summary") or {}),
            "binding_ids": list(binding_summary["binding_ids"]),
            "binding_families": list(binding_summary["binding_families"]),
            "route_ids": list(binding_summary["route_ids"]),
            "task_run_latest_path": result.get("task_run_latest_path"),
            "task_run_history_path": result.get("task_run_history_path"),
            "diagnosis_packet_path": result.get("task_diagnosis_packet_path"),
            "diagnosis_dossier_path": result.get("task_diagnosis_dossier_path"),
            "diagnosis_latest_packet_path": result.get("task_diagnosis_latest_packet_path"),
            "diagnosis_latest_dossier_path": result.get("task_diagnosis_latest_dossier_path"),
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
        "preflight_summary": summarize_stress_preflight(tasks, manifest=expectations),
        "task_summary": summarize_task_results(list(results_by_id.values())),
    }


def render_binding_first_exotic_proof_report(report: Mapping[str, Any]) -> str:
    """Render one binding-first exotic proof batch as operator-facing Markdown."""
    preflight = dict(report.get("preflight_summary") or {})
    proof_summary = dict(report.get("proof_summary") or report.get("summary") or {})
    task_summary = dict(report.get("task_summary") or {})
    lines = [
        "# Binding-First Exotic Proof Cohort",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Cohort: `{report.get('cohort', '')}`",
        f"- Model: `{report.get('model', '')}`",
        f"- Validation: `{report.get('validation', '')}`",
        f"- Fresh build: `{report.get('fresh_build', '')}`",
        f"- Tasks: `{', '.join(report.get('task_ids') or [])}`",
        f"- Raw results: `{report.get('raw_results_path', '')}`",
        f"- Report JSON: `{report.get('report_json_path', '')}`",
        f"- Report Markdown: `{report.get('report_md_path', '')}`",
        "",
        "## Deterministic Preflight",
        f"- Passed: `{dict(preflight.get('totals') or {}).get('passed', 0)}`",
        f"- Failed: `{dict(preflight.get('totals') or {}).get('failed', 0)}`",
    ]
    failed_tasks = list(preflight.get("failed_tasks") or [])
    if failed_tasks:
        lines.append(f"- Blocked tasks: `{', '.join(failed_tasks)}`")
        return "\n".join(lines).rstrip() + "\n"

    totals = dict(proof_summary.get("totals") or {})
    lines.extend(
        [
            "",
            "## Proof Summary",
            f"- Passed gate: `{totals.get('passed_gate', 0)}`",
            f"- Failed gate: `{totals.get('failed_gate', 0)}`",
            f"- Proved tasks: `{totals.get(PROOF_OUTCOME_PROVED, 0)}`",
            f"- Honest-block tasks: `{totals.get(PROOF_OUTCOME_HONEST_BLOCK, 0)}`",
            f"- First-pass rate: `{dict(task_summary.get('first_pass') or {}).get('rate', 0.0)}`",
            f"- Avg attempts to success: `{dict(task_summary.get('attempts_to_success') or {}).get('average', 0.0)}`",
            "",
            "## Task View",
        ]
    )
    for task_id, task_report in sorted((proof_summary.get("by_task") or {}).items()):
        task_report = dict(task_report or {})
        lines.extend(
            [
                f"### {task_id} - {task_report.get('title', '')}",
                f"- Outcome class: `{task_report.get('outcome_class', '')}`",
                f"- Gate passed: `{task_report.get('passed_gate', '')}`",
                f"- Success: `{task_report.get('success', '')}`",
                f"- Failure bucket: `{task_report.get('failure_bucket', '')}`",
                f"- Comparison status: `{task_report.get('comparison_status', '') or 'missing'}`",
                f"- Binding ids: `{', '.join(task_report.get('binding_ids') or []) or 'none'}`",
                f"- Route ids: `{', '.join(task_report.get('route_ids') or []) or 'none'}`",
                f"- First pass: `{task_report.get('first_pass', False)}`",
                f"- Attempts to success: `{task_report.get('attempts_to_success', 0)}`",
                f"- Retry taxonomy: `{', '.join(task_report.get('retry_taxonomy') or []) or 'none'}`",
                f"- Latest diagnosis dossier: `{task_report.get('diagnosis_latest_dossier_path', '') or task_report.get('diagnosis_dossier_path', '') or 'missing'}`",
            ]
        )
        failing_checks = [
            (check_name, check)
            for check_name, check in dict(task_report.get("live_checks") or {}).items()
            if not check.get("passed")
        ]
        if failing_checks:
            lines.append("- Failed live checks:")
            for check_name, check in failing_checks:
                lines.append(f"  - `{check_name}`")
                for detail in check.get("details") or []:
                    lines.append(f"    - {detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def summarize_binding_first_exotic_program_closeout(
    reports: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate multiple proof-cohort reports into one program closeout summary."""
    cohort_summaries: dict[str, dict[str, Any]] = {}
    by_task: dict[str, dict[str, Any]] = {}
    failure_buckets: dict[str, int] = {}
    unknown_route_tasks: list[str] = []
    elapsed_total = 0.0
    successful_attempts: list[int] = []
    token_usage = {
        "call_count": 0,
        "calls_with_usage": 0,
        "calls_without_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_stage": {},
        "by_provider": {},
    }
    totals = {
        "tasks": 0,
        "passed_gate": 0,
        "failed_gate": 0,
        PROOF_OUTCOME_PROVED: 0,
        PROOF_OUTCOME_HONEST_BLOCK: 0,
    }

    def _display_artifact_path(value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        return Path(text).name if text.startswith("/") else text

    def _merge_token_usage(target: dict[str, Any], source: Mapping[str, Any]) -> None:
        for key in (
            "call_count",
            "calls_with_usage",
            "calls_without_usage",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            target[key] = int(target.get(key, 0) or 0) + int(source.get(key, 0) or 0)
        for bucket_name in ("by_stage", "by_provider"):
            target_bucket = dict(target.get(bucket_name) or {})
            for name, payload in dict(source.get(bucket_name) or {}).items():
                existing = dict(target_bucket.get(name) or {})
                for key in (
                    "call_count",
                    "calls_with_usage",
                    "calls_without_usage",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ):
                    existing[key] = int(existing.get(key, 0) or 0) + int(payload.get(key, 0) or 0)
                target_bucket[name] = existing
            target[bucket_name] = target_bucket

    for raw_report in reports:
        report = dict(raw_report or {})
        cohort = str(report.get("cohort") or "").strip() or "unknown"
        if cohort in cohort_summaries:
            raise ValueError(f"duplicate binding-first closeout cohort: {cohort}")
        proof_summary = dict(report.get("proof_summary") or {})
        task_summary = dict(report.get("task_summary") or {})
        cohort_summaries[cohort] = {
            "status": report.get("status"),
            "task_ids": list(report.get("task_ids") or []),
            "totals": dict(proof_summary.get("totals") or {}),
            "failure_buckets": dict(proof_summary.get("failure_buckets") or {}),
            "task_summary": task_summary,
            "report_json_path": _display_artifact_path(report.get("report_json_path")),
            "report_md_path": _display_artifact_path(report.get("report_md_path")),
            "raw_results_path": _display_artifact_path(report.get("raw_results_path")),
        }

        for bucket, count in dict(proof_summary.get("failure_buckets") or {}).items():
            failure_buckets[str(bucket)] = failure_buckets.get(str(bucket), 0) + int(count or 0)

        for task_id, task_report_raw in dict(proof_summary.get("by_task") or {}).items():
            task_report = {
                key: (
                    _display_artifact_path(value)
                    if key.endswith("_path")
                    else value
                )
                for key, value in dict(task_report_raw or {}).items()
            }
            by_task[str(task_id)] = task_report
            totals["tasks"] += 1
            if task_report.get("passed_gate"):
                totals["passed_gate"] += 1
                if str(task_report.get("outcome_class") or "").strip() != PROOF_OUTCOME_HONEST_BLOCK:
                    successful_attempts.append(int(task_report.get("attempts_to_success") or 0))
            else:
                totals["failed_gate"] += 1
            outcome_class = str(task_report.get("outcome_class") or "").strip()
            if outcome_class in totals:
                totals[outcome_class] += 1

            elapsed_total += float(task_report.get("elapsed_seconds") or 0.0)
            _merge_token_usage(token_usage, dict(task_report.get("token_usage") or {}))
            if "unknown" in set(task_report.get("route_ids") or ()):
                unknown_route_tasks.append(str(task_id))

    first_pass_successes = sum(
        1
        for task_report in by_task.values()
        if task_report.get("passed_gate") and task_report.get("first_pass")
    )
    honest_block_certified = sum(
        1
        for task_report in by_task.values()
        if task_report.get("passed_gate")
        and str(task_report.get("outcome_class") or "").strip() == PROOF_OUTCOME_HONEST_BLOCK
    )
    return {
        "totals": totals,
        "cohorts": cohort_summaries,
        "failure_buckets": failure_buckets,
        "by_task": by_task,
        "elapsed_seconds_total": round(elapsed_total, 1),
        "unknown_route_tasks": sorted(set(unknown_route_tasks)),
        "first_pass": {
            "tasks": totals["tasks"],
            "successful_tasks": totals["passed_gate"],
            "first_pass_successes": first_pass_successes,
            "rate": (first_pass_successes / totals["tasks"]) if totals["tasks"] else 0.0,
        },
        "attempts_to_success": {
            "successful_tasks": len(successful_attempts),
            "average": (
                round(sum(successful_attempts) / len(successful_attempts), 2)
                if successful_attempts
                else 0.0
            ),
            "median": float(median(successful_attempts)) if successful_attempts else 0.0,
            "max": max(successful_attempts) if successful_attempts else 0,
        },
        "token_usage": token_usage,
        "honest_block_certified": honest_block_certified,
    }


def render_binding_first_exotic_program_closeout(report: Mapping[str, Any]) -> str:
    """Render the program-level proof closeout as operator-facing Markdown."""
    totals = dict(report.get("totals") or {})
    first_pass = dict(report.get("first_pass") or {})
    attempts = dict(report.get("attempts_to_success") or {})
    token_usage = dict(report.get("token_usage") or {})
    lines = [
        "# Binding-First Exotic Program Closeout",
        "",
        f"- Tasks: `{totals.get('tasks', 0)}`",
        f"- Passed gate: `{totals.get('passed_gate', 0)}`",
        f"- Failed gate: `{totals.get('failed_gate', 0)}`",
        f"- Proved expectations: `{totals.get(PROOF_OUTCOME_PROVED, 0)}`",
        f"- Honest-block expectations: `{totals.get(PROOF_OUTCOME_HONEST_BLOCK, 0)}`",
        f"- Certified honest blocks: `{report.get('honest_block_certified', 0)}`",
        f"- First-pass success rate: `{first_pass.get('rate', 0.0)}`",
        f"- Average attempts to success (successful tasks): `{attempts.get('average', 0.0)}`",
        f"- Total elapsed seconds: `{report.get('elapsed_seconds_total', 0.0)}`",
        f"- Total tokens: `{token_usage.get('total_tokens', 0)}`",
        f"- Unknown route-id tasks: `{', '.join(report.get('unknown_route_tasks') or []) or 'none'}`",
        "",
        "## Failure Buckets",
    ]
    for bucket, count in sorted(dict(report.get("failure_buckets") or {}).items()):
        lines.append(f"- `{bucket}`: `{count}`")
    lines.extend(
        [
            "",
        "## Cohorts",
        ]
    )
    for cohort, summary_raw in sorted(dict(report.get("cohorts") or {}).items()):
        summary = dict(summary_raw or {})
        cohort_totals = dict(summary.get("totals") or {})
        lines.extend(
            [
                f"### {cohort}",
                f"- Status: `{summary.get('status', '')}`",
                f"- Passed gate: `{cohort_totals.get('passed_gate', 0)}`",
                f"- Failed gate: `{cohort_totals.get('failed_gate', 0)}`",
                f"- Tasks: `{', '.join(summary.get('task_ids') or [])}`",
                f"- Report JSON: `{summary.get('report_json_path', '')}`",
                f"- Report Markdown: `{summary.get('report_md_path', '')}`",
                "",
            ]
        )
    lines.append("## Task View")
    for task_id, task_report_raw in sorted(dict(report.get("by_task") or {}).items()):
        task_report = dict(task_report_raw or {})
        lines.extend(
            [
                f"### {task_id} - {task_report.get('title', '')}",
                f"- Cohort: `{task_report.get('cohort', '')}`",
                f"- Expected outcome: `{task_report.get('outcome_class', '')}`",
                f"- Gate passed: `{task_report.get('passed_gate', '')}`",
                f"- Failure bucket: `{task_report.get('failure_bucket', '')}`",
                f"- Comparison status: `{task_report.get('comparison_status', '') or 'missing'}`",
                f"- Binding ids: `{', '.join(task_report.get('binding_ids') or []) or 'none'}`",
                f"- Route ids: `{', '.join(task_report.get('route_ids') or []) or 'none'}`",
                f"- First pass: `{task_report.get('first_pass', False)}`",
                f"- Attempts to success: `{task_report.get('attempts_to_success', 0)}`",
                f"- Retry taxonomy: `{', '.join(task_report.get('retry_taxonomy') or []) or 'none'}`",
                f"- Latest diagnosis dossier: `{task_report.get('diagnosis_latest_dossier_path', '') or task_report.get('diagnosis_dossier_path', '') or 'missing'}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def classify_task_result(result: Mapping[str, Any]) -> str:
    """Bucket one task result into a stable outcome/failure class."""
    if result.get("success"):
        return "success"

    diagnosis_bucket = str(
        result.get("task_diagnosis_failure_bucket")
        or result.get("diagnosis_failure_bucket")
        or ""
    ).strip()
    if diagnosis_bucket:
        return diagnosis_bucket

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
    attempts_to_success: list[int] = []
    token_usage = _empty_token_usage_summary()
    shared_knowledge_tasks = 0
    shared_knowledge_lessons = 0
    tasks_with_reviewer_issues = 0
    tasks_recovered_after_review = 0
    tasks_with_multi_reviewer_issues = 0
    reviewer_agent_counts: dict[str, int] = {}
    retry_taxonomy_by_stage: dict[str, dict[str, Any]] = {}
    retry_taxonomy_by_task: dict[str, list[str]] = {}
    unattributed_recoveries = 0

    successes = 0
    successful_after_retry = 0
    first_attempt_successes = 0

    for result in results:
        bucket = classify_task_result(result)
        failure_buckets[bucket] = failure_buckets.get(bucket, 0) + 1

        if result.get("success"):
            successes += 1
            attempt_count = _result_attempts_to_success(result)
            attempts_to_success.append(attempt_count)
            if attempt_count <= 1:
                first_attempt_successes += 1
            elif attempt_count > 1:
                successful_after_retry += 1
                task_id = str(result.get("task_id") or "").strip()
                stage_reasons = _result_retry_taxonomy_reasons(result)
                if not stage_reasons:
                    unattributed_recoveries += 1
                    stage_reasons = ("unattributed",)
                if task_id:
                    retry_taxonomy_by_task[task_id] = list(stage_reasons)
                for stage_reason in stage_reasons:
                    stage_entry = retry_taxonomy_by_stage.setdefault(
                        stage_reason,
                        {"count": 0, "task_ids": []},
                    )
                    if task_id:
                        if task_id not in stage_entry["task_ids"]:
                            stage_entry["task_ids"].append(task_id)
                        stage_entry["task_ids"].sort()
                        stage_entry["count"] = len(stage_entry["task_ids"])
                    else:
                        stage_entry["count"] += 1
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
        "first_pass": {
            "tasks": len(results),
            "successful_tasks": successes,
            "first_pass_successes": first_attempt_successes,
            "rate": round(first_attempt_successes / len(results), 2) if results else 0.0,
            "success_rate": round(first_attempt_successes / successes, 2) if successes else 0.0,
        },
        "attempts_to_success": {
            "successful_tasks": len(attempts_to_success),
            "average": round(sum(attempts_to_success) / len(attempts_to_success), 2)
            if attempts_to_success
            else 0.0,
            "median": round(float(median(attempts_to_success)), 2)
            if attempts_to_success
            else 0.0,
            "max": max(attempts_to_success) if attempts_to_success else 0,
            "distribution": {
                str(attempt): attempts_to_success.count(attempt)
                for attempt in sorted(set(attempts_to_success))
            },
        },
        "retry_taxonomy": {
            "recovered_successes": successful_after_retry,
            "unattributed_recoveries": unattributed_recoveries,
            "by_stage": dict(sorted(retry_taxonomy_by_stage.items())),
            "by_task": dict(sorted(retry_taxonomy_by_task.items())),
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


def _result_attempts_to_success(result: Mapping[str, Any]) -> int:
    """Return the task-level attempts-to-success count, normalizing comparison tasks."""
    method_results = result.get("method_results")
    if isinstance(method_results, Mapping):
        method_attempts = [
            max(int(payload.get("attempts") or 0), 0)
            for payload in method_results.values()
            if isinstance(payload, Mapping)
        ]
        if method_attempts:
            return max(method_attempts)
    return max(int(result.get("attempts") or 0), 0)


def _result_retry_taxonomy_reasons(result: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the stable retry-stage reasons that explain one recovered success."""
    reasons: list[str] = []
    method_results = result.get("method_results")
    if isinstance(method_results, Mapping):
        for payload in method_results.values():
            if not isinstance(payload, Mapping):
                continue
            if max(int(payload.get("attempts") or 0), 0) <= 1:
                continue
            reasons.extend(_payload_retry_taxonomy_reasons(payload))
    elif _result_attempts_to_success(result) > 1:
        reasons.extend(_payload_retry_taxonomy_reasons(result))
    return tuple(dict.fromkeys(reason for reason in reasons if reason))


def _payload_retry_taxonomy_reasons(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Load ordered distinct builder retry reasons from one payload trace."""
    trace_path = str(payload.get("platform_trace_path") or "").strip()
    if not trace_path:
        return ()

    try:
        from trellis.agent.platform_traces import load_platform_trace_events

        events = load_platform_trace_events(trace_path)
    except Exception:
        return ()

    reasons: list[str] = []
    for event in events:
        if event.event != "builder_attempt_failed":
            continue
        reason = str((event.details or {}).get("reason") or "").strip()
        if reason:
            reasons.append(reason)
    return tuple(dict.fromkeys(reasons))


def summarize_promotion_discipline(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize whether successful reruns left reusable learning artifacts behind."""
    successful_results = [result for result in results if result.get("success")]
    captured_lesson_ids: set[str] = set()
    cookbook_candidate_paths: set[str] = set()
    knowledge_trace_paths: set[str] = set()
    promotion_candidate_paths: set[str] = set()
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
            elif isinstance(cookbook_candidate, list):
                for item in cookbook_candidate:
                    if isinstance(item, str) and item.strip():
                        cookbook_candidate_paths.add(item.strip())
                        reusable = True
            promotion_candidate = reflection.get("promotion_candidate_saved")
            if isinstance(promotion_candidate, str) and promotion_candidate.strip():
                promotion_candidate_paths.add(promotion_candidate.strip())
                reusable = True
            elif isinstance(promotion_candidate, list):
                for item in promotion_candidate:
                    if isinstance(item, str) and item.strip():
                        promotion_candidate_paths.add(item.strip())
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
        "promotion_candidates": len(promotion_candidate_paths),
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


def _grade_test_scope(plan: GenerationPlan) -> GradeResult:
    """Grade whether proposed tests point at real, in-repo test targets."""
    test_map = get_test_map()
    known_tests = {
        test_path
        for tests in test_map.directory_to_tests.values()
        for test_path in tests
    }
    invalid = tuple(
        target for target in plan.proposed_tests
        if target not in known_tests
    )
    if invalid:
        return GradeResult(
            False,
            tuple(f"unknown or out-of-repo test target: {target}" for target in invalid),
            blocking=True,
        )
    return GradeResult(True)


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


def render_stress_tranche_report(report: Mapping[str, Any]) -> str:
    """Render one connector-stress batch report as operator-facing Markdown."""
    preflight = dict(report.get("preflight_summary") or {})
    stress_summary = dict(report.get("stress_summary") or {})
    lines = [
        "# Connector stress tranche",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Model: `{report.get('model', '')}`",
        f"- Validation: `{report.get('validation', '')}`",
        f"- Fresh build: `{report.get('fresh_build', '')}`",
        f"- Tasks: `{', '.join(report.get('task_ids') or [])}`",
        f"- Raw results: `{report.get('raw_results_path', '')}`",
        f"- Report JSON: `{report.get('report_json_path', '')}`",
        f"- Report Markdown: `{report.get('report_md_path', '')}`",
        "",
        "## Deterministic Preflight",
        f"- Passed: `{dict(preflight.get('totals') or {}).get('passed', 0)}`",
        f"- Failed: `{dict(preflight.get('totals') or {}).get('failed', 0)}`",
    ]
    failed_tasks = list(preflight.get("failed_tasks") or [])
    if failed_tasks:
        lines.append(f"- Blocked tasks: `{', '.join(failed_tasks)}`")
        for task_id in failed_tasks:
            task_report = dict((preflight.get("by_task") or {}).get(task_id) or {})
            lines.append(f"### Preflight block: {task_id} - {task_report.get('title', '')}")
            for check_name, check in dict(task_report.get("checks") or {}).items():
                if not check.get("passed"):
                    lines.append(f"- `{check_name}`:")
                    for detail in check.get("details") or []:
                        lines.append(f"  - {detail}")
        return "\n".join(lines).rstrip() + "\n"

    if not stress_summary:
        return "\n".join(lines).rstrip() + "\n"

    totals = dict(stress_summary.get("totals") or {})
    lines.extend(
        [
            "",
            "## Gate Summary",
            f"- Passed gate: `{totals.get('passed_gate', 0)}`",
            f"- Failed gate: `{totals.get('failed_gate', 0)}`",
            f"- Compare-ready tasks: `{totals.get('compare_ready', 0)}`",
            f"- Honest-block tasks: `{totals.get('honest_block', 0)}`",
            "",
            "## Task View",
        ]
    )
    for task_id, task_report in sorted((stress_summary.get("by_task") or {}).items()):
        task_report = dict(task_report or {})
        lines.extend(
            [
                f"### {task_id} - {task_report.get('title', '')}",
                f"- Outcome class: `{task_report.get('outcome_class', '')}`",
                f"- Gate passed: `{task_report.get('passed_gate', '')}`",
                f"- Success: `{task_report.get('success', '')}`",
                f"- Failure bucket: `{task_report.get('failure_bucket', '')}`",
                f"- Comparison status: `{task_report.get('comparison_status', '') or 'missing'}`",
                f"- Observed blocker categories: `{', '.join(task_report.get('observed_blocker_categories') or []) or 'none'}`",
                f"- Latest diagnosis dossier: `{task_report.get('diagnosis_latest_dossier_path', '') or task_report.get('diagnosis_dossier_path', '') or 'missing'}`",
                f"- Latest diagnosis packet: `{task_report.get('diagnosis_latest_packet_path', '') or task_report.get('diagnosis_packet_path', '') or 'missing'}`",
            ]
        )
        failing_checks = [
            (check_name, check)
            for check_name, check in dict(task_report.get("live_checks") or {}).items()
            if not check.get("passed")
        ]
        if failing_checks:
            lines.append("- Failed live checks:")
            for check_name, check in failing_checks:
                lines.append(f"  - `{check_name}`")
                for detail in check.get("details") or []:
                    lines.append(f"    - {detail}")
        follow_on = dict(task_report.get("follow_on") or {})
        if follow_on and follow_on.get("action") != "no_action":
            lines.append("- Follow-on:")
            lines.append(f"  - action: `{follow_on.get('action')}`")
            lines.append(f"  - repeat_count: `{follow_on.get('repeat_count')}`")
            lines.append(f"  - signature: `{follow_on.get('signature')}`")
            if follow_on.get("suggested_title"):
                lines.append(f"  - suggested_title: `{follow_on['suggested_title']}`")
        lines.append("")

    follow_ons = list(stress_summary.get("follow_on_candidates") or [])
    lines.append("## Follow-on Candidates")
    if not follow_ons:
        lines.append("- None")
    else:
        for item in follow_ons:
            lines.append(
                f"- `{item.get('task_id')}` `{item.get('action')}` "
                f"(repeat_count={item.get('repeat_count')}, signature={item.get('signature')})"
            )

    return "\n".join(lines).rstrip() + "\n"


def _normalize_stress_outcome_class(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_proof_outcome_class(value: Any) -> str:
    return str(value or "").strip().lower()


def _empty_proof_binding_summary() -> dict[str, tuple[str, ...]]:
    return {
        "binding_ids": (),
        "binding_families": (),
        "binding_aliases": (),
        "route_ids": (),
        "route_families": (),
    }


def _proof_binding_summary(result: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return the persisted binding/route identities observed for one task result."""
    latest_path = str(
        result.get("task_run_latest_path") or result.get("task_run_history_path") or ""
    ).strip()
    if not latest_path:
        return _empty_proof_binding_summary()

    path = Path(latest_path)
    if not path.exists():
        return _empty_proof_binding_summary()

    try:
        from trellis.agent.task_run_store import load_task_run_record

        record = load_task_run_record(path)
    except Exception:
        return _empty_proof_binding_summary()

    telemetry = dict(record.get("telemetry") or {})
    observations = telemetry.get("binding_observations") or telemetry.get("route_observations") or ()
    if not isinstance(observations, list):
        return _empty_proof_binding_summary()

    binding_ids: list[str] = []
    binding_families: list[str] = []
    binding_aliases: list[str] = []
    route_ids: list[str] = []
    route_families: list[str] = []
    for item in observations:
        if not isinstance(item, Mapping):
            continue
        binding_id = str(item.get("binding_id") or item.get("backend_binding_id") or "").strip()
        binding_family = str(item.get("binding_family") or item.get("route_family") or "").strip()
        binding_alias = str(item.get("binding_alias") or item.get("route_alias") or "").strip()
        route_id = str(item.get("route_id") or "").strip()
        route_family = str(item.get("route_family") or "").strip()
        if binding_id and binding_id not in binding_ids:
            binding_ids.append(binding_id)
        if binding_family and binding_family not in binding_families:
            binding_families.append(binding_family)
        if binding_alias and binding_alias not in binding_aliases:
            binding_aliases.append(binding_alias)
        if route_id and route_id not in route_ids:
            route_ids.append(route_id)
        if route_family and route_family not in route_families:
            route_families.append(route_family)
    return {
        "binding_ids": tuple(binding_ids),
        "binding_families": tuple(binding_families),
        "binding_aliases": tuple(binding_aliases),
        "route_ids": tuple(route_ids),
        "route_families": tuple(route_families),
    }


def _unknown_route_ids(route_ids: Iterable[str]) -> tuple[str, ...]:
    observed = tuple(str(route_id).strip() for route_id in route_ids if str(route_id).strip())
    if not observed:
        return ()
    try:
        from trellis.agent.route_registry import load_route_registry

        registry = load_route_registry(include_discovered=False)
        known = {str(route.id).strip() for route in getattr(registry, "routes", ())}
    except Exception:
        return ()
    return tuple(route_id for route_id in observed if route_id not in known)


def _iter_stress_blocker_details(result: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    blocker_details = result.get("blocker_details")
    if isinstance(blocker_details, Mapping) and blocker_details:
        yield blocker_details
    method_results = result.get("method_results") or {}
    if not isinstance(method_results, Mapping):
        return
    for payload in method_results.values():
        if not isinstance(payload, Mapping):
            continue
        blocker_details = payload.get("blocker_details")
        if isinstance(blocker_details, Mapping) and blocker_details:
            yield blocker_details


def _stress_observed_blocker_categories(result: Mapping[str, Any]) -> tuple[str, ...]:
    categories: list[str] = []
    for details in _iter_stress_blocker_details(result):
        blocker_report = details.get("blocker_report") or {}
        blockers = blocker_report.get("blockers") or []
        for blocker in blockers:
            if not isinstance(blocker, Mapping):
                continue
            categories.extend(
                _map_stress_blocker_categories(
                    blocker.get("category"),
                    blocker_id=blocker.get("id"),
                )
            )
        for raw_blocker in details.get("blockers") or []:
            categories.extend(
                _map_stress_blocker_categories(
                    None,
                    blocker_id=raw_blocker,
                )
            )
        if (details.get("new_primitive_workflow") or {}).get("items"):
            categories.append("missing_foundational_primitive")
        reason = str(details.get("reason") or "").strip()
        if reason == "semantic_clarification_required":
            categories.append("semantic_clarification")
        semantic_gap = details.get("semantic_gap") or {}
        if isinstance(semantic_gap, Mapping):
            if _semantic_gap_binding_helpers(semantic_gap):
                categories.append("missing_binding_surface")
            if semantic_gap.get("missing_runtime_primitives"):
                categories.append("missing_foundational_primitive")
            if semantic_gap.get("missing_contract_fields"):
                categories.append("semantic_contract_gap")
    seen: set[str] = set()
    ordered: list[str] = []
    for category in categories:
        text = str(category).strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return tuple(ordered)


def _map_stress_blocker_categories(
    category: Any,
    *,
    blocker_id: Any,
) -> tuple[str, ...]:
    raw_category = str(category or "").strip().lower()
    raw_blocker_id = str(blocker_id or "").strip().lower()
    mapped: list[str] = []
    if raw_category in {
        "unsupported_composite",
        "missing_foundational_primitive",
        "missing_binding_surface",
        "semantic_contract_gap",
        "semantic_clarification",
    }:
        mapped.append(raw_category)
    if raw_category in {
        "numerical_substrate_gap",
        "implementation_gap",
        "export_or_registry_gap",
        "unknown_gap",
    }:
        mapped.append("missing_foundational_primitive")
    if raw_category in {"unsupported_route", "missing_binding_surface"}:
        mapped.append("missing_binding_surface")
    if any(
        token in raw_blocker_id
        for token in (
            "path_dependent",
            "stochastic_vol",
            "unsupported_composite",
        )
    ):
        mapped.append("unsupported_composite")
    if raw_blocker_id.startswith(("missing_module:", "missing_symbol:")):
        mapped.append("missing_foundational_primitive")
    if raw_blocker_id.startswith(("missing_route_helper:", "missing_binding_helper:")):
        mapped.append("missing_binding_surface")
    return tuple(dict.fromkeys(mapped))


def _semantic_gap_binding_helpers(semantic_gap: Mapping[str, Any]) -> tuple[str, ...]:
    values = (
        semantic_gap.get("missing_binding_helpers")
        or semantic_gap.get("missing_route_helpers")
        or ()
    )
    if isinstance(values, str):
        text = values.strip()
        return (text,) if text else ()
    if isinstance(values, (list, tuple)):
        return tuple(str(item).strip() for item in values if str(item).strip())
    return ()


def _stress_artifact_inventory(
    results_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    inventory = {
        "latest_run_records": 0,
        "latest_diagnosis_packets": 0,
        "latest_diagnosis_dossiers": 0,
    }
    for result in results_by_id.values():
        if result.get("task_run_latest_path"):
            inventory["latest_run_records"] += 1
        if result.get("task_diagnosis_latest_packet_path"):
            inventory["latest_diagnosis_packets"] += 1
        if result.get("task_diagnosis_latest_dossier_path"):
            inventory["latest_diagnosis_dossiers"] += 1
    return inventory


def _stress_follow_on_candidate(
    task_id: str,
    task: Mapping[str, Any],
    expectation: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    blocker_categories: tuple[str, ...],
) -> dict[str, Any]:
    if not result or result.get("success"):
        return {"action": "no_action"}

    signature = _stress_failure_signature(
        task_id,
        result,
        blocker_categories=blocker_categories,
    )
    repeat_count = _stress_failure_repeat_count(
        task_id,
        signature,
    )
    linked_linear_issues = _stress_linked_linear_issues(result)
    action = "no_action"
    if linked_linear_issues:
        action = "tracked_externally"
    elif repeat_count >= 2:
        action = "create_follow_on"
    elif repeat_count == 1:
        action = "watch"

    from trellis.agent.request_issue_format import (
        build_stress_follow_on_body,
        build_stress_follow_on_title,
    )

    payload = {
        "task_id": task_id,
        "task_title": task.get("title"),
        "outcome_class": _normalize_stress_outcome_class(expectation.get("outcome_class")),
        "failure_bucket": classify_task_result(result),
        "comparison_status": (result.get("cross_validation") or {}).get("status"),
        "observed_blocker_categories": list(blocker_categories),
        "repeat_count": repeat_count,
        "signature": signature,
        "diagnosis_dossier_path": result.get("task_diagnosis_latest_dossier_path") or result.get("task_diagnosis_dossier_path"),
        "diagnosis_packet_path": result.get("task_diagnosis_latest_packet_path") or result.get("task_diagnosis_packet_path"),
        "task_run_history_path": result.get("task_run_history_path"),
        "task_run_latest_path": result.get("task_run_latest_path"),
        "linked_linear_issues": linked_linear_issues,
    }
    payload["action"] = action
    if action != "no_action":
        payload["suggested_title"] = build_stress_follow_on_title(payload)
        payload["suggested_body"] = build_stress_follow_on_body(payload)
    return payload


def _stress_failure_signature(
    task_id: str,
    result: Mapping[str, Any],
    *,
    blocker_categories: tuple[str, ...],
) -> str:
    bucket = classify_task_result(result)
    comparison_status = str(
        (result.get("cross_validation") or {}).get("status") or ""
    ).strip().lower()
    parts = [task_id, bucket]
    if blocker_categories:
        parts.append(",".join(blocker_categories))
    elif comparison_status and comparison_status != "passed":
        parts.append(comparison_status)
    return "::".join(parts)


def _stress_failure_repeat_count(
    task_id: str,
    signature: str,
    *,
    root: Path = ROOT,
) -> int:
    history_root = root / "task_runs" / "history" / task_id
    if not history_root.exists():
        return 0

    matches = 0
    for path in sorted(history_root.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except Exception:
            continue
        result = record.get("result") or {}
        if not isinstance(result, Mapping):
            continue
        candidate_signature = _stress_failure_signature(
            task_id,
            result,
            blocker_categories=_stress_observed_blocker_categories(result),
        )
        if candidate_signature == signature:
            matches += 1
    return matches


def _stress_linked_linear_issues(result: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    for payload in (result.get("method_results") or {}).values():
        if not isinstance(payload, Mapping):
            continue
        trace_path = payload.get("platform_trace_path")
        if not trace_path:
            continue
        trace = _load_stress_trace_summary(trace_path)
        issue = (trace or {}).get("linear_issue") or {}
        identifier = issue.get("identifier")
        if identifier and identifier not in issues:
            issues.append(str(identifier))
    return issues


def _load_stress_trace_summary(path: Any) -> dict[str, Any] | None:
    trace_path = Path(str(path))
    if not trace_path.exists() or trace_path.suffix.lower() != ".yaml":
        return None
    try:
        loaded = yaml.safe_load(trace_path.read_text()) or {}
    except Exception:
        return None
    return dict(loaded)


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
