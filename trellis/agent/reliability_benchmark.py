"""Structured reliability benchmark reports for task-run comparisons."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from trellis.agent.evals import compare_task_runs, summarize_task_results


DEFAULT_REPORT_ROOT = Path("docs") / "benchmarks"
DEFAULT_REPORT_STEM = "analytical_support_reliability"


@dataclass(frozen=True)
class ReliabilityBenchmarkArtifacts:
    """Persisted files for one benchmark report."""

    report: dict[str, Any]
    json_path: Path
    text_path: Path


def build_reliability_benchmark_report(
    *,
    benchmark_name: str,
    tasks: Sequence[Mapping[str, Any]],
    baseline_results: Sequence[Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
    baseline_mode: str = "reuse",
    candidate_mode: str = "fresh-build",
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a structured report comparing two benchmark tranches."""
    baseline_results = list(baseline_results)
    candidate_results = list(candidate_results)
    comparison = compare_task_runs(baseline_results, candidate_results)
    baseline_by_id = _results_by_task_id(baseline_results)
    candidate_by_id = _results_by_task_id(candidate_results)

    task_rows = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        task_rows.append(
            {
                "task_id": task_id,
                "title": str(task.get("title") or task_id),
                "baseline": _result_snapshot(
                    baseline_by_id.get(task_id),
                    mode=baseline_mode,
                ),
                "candidate": _result_snapshot(
                    candidate_by_id.get(task_id),
                    mode=candidate_mode,
                ),
                "transition": dict(
                    comparison["task_transitions"]["by_task"].get(task_id, {})
                ),
            }
        )

    knowledge_capture = _knowledge_capture(comparison, candidate_results, task_rows)

    return {
        "benchmark_name": benchmark_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "task_ids": [str(task.get("id") or "") for task in tasks],
        "task_titles": {
            str(task.get("id") or ""): str(task.get("title") or "")
            for task in tasks
        },
        "baseline": summarize_task_results(baseline_results),
        "candidate": summarize_task_results(candidate_results),
        "comparison": comparison,
        "tasks": task_rows,
        "knowledge_capture": knowledge_capture,
        "notes": list(notes or ()),
    }


def render_reliability_benchmark_report(report: Mapping[str, Any]) -> str:
    """Render a markdown report from a structured benchmark payload."""
    baseline = report["baseline"]
    candidate = report["candidate"]
    comparison = report["comparison"]
    knowledge = report["knowledge_capture"]

    lines = [
        f"# Reliability Benchmark: `{report['benchmark_name']}`",
        f"- Created at: `{report.get('created_at', '')}`",
        f"- Baseline mode: `{report['baseline_mode']}`",
        f"- Candidate mode: `{report['candidate_mode']}`",
        f"- Tasks: {', '.join(f'`{task_id}`' for task_id in report.get('task_ids', []))}",
    ]

    if report.get("notes"):
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in report["notes"])

    lines.extend(
        [
            "",
            "## Outcome Summary",
            f"- Baseline success: `{baseline['totals']['successes']}/{baseline['totals']['tasks']}`",
            f"- Candidate success: `{candidate['totals']['successes']}/{candidate['totals']['tasks']}`",
            f"- Baseline avg attempts: `{baseline['totals']['avg_attempts']}`",
            f"- Candidate avg attempts: `{candidate['totals']['avg_attempts']}`",
            "",
            "## Task Transitions",
            f"- Improved: `{comparison['task_transitions']['improved']}`",
            f"- Regressed: `{comparison['task_transitions']['regressed']}`",
            f"- Unchanged: `{comparison['task_transitions']['unchanged']}`",
            "",
            "## Failure Buckets",
        ]
    )
    failure_bucket_deltas = {
        bucket: delta
        for bucket, delta in comparison["failure_bucket_deltas"].items()
        if bucket != "success"
    }
    if failure_bucket_deltas:
        lines.extend(
            f"- `{bucket}`: `{delta:+d}`"
            for bucket, delta in sorted(failure_bucket_deltas.items())
        )
    else:
        lines.append("- No failure bucket delta recorded.")

    lines.extend(
        [
            "",
            "## Shared Knowledge",
            f"- Baseline tasks with shared context: `{baseline['shared_knowledge']['tasks_with_shared_context']}`",
            f"- Candidate tasks with shared context: `{candidate['shared_knowledge']['tasks_with_shared_context']}`",
            f"- Baseline tasks with lessons: `{baseline['shared_knowledge']['tasks_with_lessons']}`",
            f"- Candidate tasks with lessons: `{candidate['shared_knowledge']['tasks_with_lessons']}`",
            "",
            "## Promotion Discipline",
            f"- Baseline successful tasks: `{baseline['promotion_discipline']['successful_tasks']}`",
            f"- Candidate successful tasks: `{candidate['promotion_discipline']['successful_tasks']}`",
            f"- Baseline successful tasks without reusable artifacts: `{len(baseline['promotion_discipline']['successful_tasks_without_reusable_artifacts'])}`",
            f"- Candidate successful tasks without reusable artifacts: `{len(candidate['promotion_discipline']['successful_tasks_without_reusable_artifacts'])}`",
            "",
            "## Knowledge Capture",
            f"- Tasks with lessons: {', '.join(_format_task_ids(knowledge['tasks_with_lessons'])) or 'none'}",
            f"- Tasks with cookbook enrichment: {', '.join(_format_task_ids(knowledge['tasks_with_cookbooks'])) or 'none'}",
            f"- Tasks with promotion candidates: {', '.join(_format_task_ids(knowledge['tasks_with_promotion_candidates'])) or 'none'}",
        ]
    )

    if knowledge["follow_up_suggestions"]:
        lines.extend(["", "## Follow-Up Suggestions"])
        for item in knowledge["follow_up_suggestions"]:
            lines.append(
                f"- `{item['bucket']}` (`{item['count']}`): {item['suggestion']}"
            )

    if report["tasks"]:
        lines.extend(["", "## Per-Task Details"])
        for task in report["tasks"]:
            lines.extend(
                [
                    "",
                    f"### `{task['task_id']}` {task['title']}",
                    f"- Transition: `{task['transition'].get('transition', 'unknown')}`",
                    f"- Baseline: {_snapshot_summary(task['baseline'])}",
                    f"- Candidate: {_snapshot_summary(task['candidate'])}",
                ]
            )

    return "\n".join(lines) + "\n"


def save_reliability_benchmark_report(
    report: Mapping[str, Any],
    *,
    root: Path | None = None,
    stem: str = DEFAULT_REPORT_STEM,
) -> ReliabilityBenchmarkArtifacts:
    """Persist a reliability benchmark report as JSON plus Markdown."""
    root = root or DEFAULT_REPORT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{stem}.json"
    text_path = root / f"{stem}.md"
    payload = dict(report)
    payload["json_path"] = str(json_path)
    payload["text_path"] = str(text_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    text_path.write_text(render_reliability_benchmark_report(payload))
    return ReliabilityBenchmarkArtifacts(report=payload, json_path=json_path, text_path=text_path)


def _results_by_task_id(results: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapping: dict[str, Mapping[str, Any]] = {}
    for result in results:
        task_id = str(result.get("task_id") or "").strip()
        if task_id:
            mapping[task_id] = result
    return mapping


def _result_snapshot(
    result: Mapping[str, Any] | None,
    *,
    mode: str,
) -> dict[str, Any]:
    if not result:
        return {
            "present": False,
            "mode": mode,
        }

    reflection = dict(result.get("reflection") or {})
    artifacts = dict(result.get("artifacts") or {})
    knowledge_summary = dict(result.get("knowledge_summary") or {})
    return {
        "present": True,
        "mode": mode,
        "success": bool(result.get("success")),
        "attempts": int(result.get("attempts") or 0),
        "gap_confidence": result.get("gap_confidence"),
        "failures": list(result.get("failures") or []),
        "knowledge_gaps": list(result.get("knowledge_gaps") or []),
        "lesson_ids": _lesson_ids_from_result(result),
        "cookbook_enriched": bool(reflection.get("cookbook_enriched")),
        "promotion_candidate_saved": _normalize_strings(reflection.get("promotion_candidate_saved")),
        "knowledge_trace_saved": _normalize_strings(reflection.get("knowledge_trace_saved")),
        "artifacts": {key: len(value or []) for key, value in artifacts.items()},
        "token_usage_total": (
            result.get("token_usage_summary") or {}
        ).get("total_tokens"),
        "preferred_method": result.get("preferred_method"),
        "reference_target": result.get("reference_target"),
        "knowledge_summary_keys": sorted(knowledge_summary.keys()),
    }


def _knowledge_capture(
    comparison: Mapping[str, Any],
    candidate_results: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_summary = comparison["candidate"]
    baseline_summary = comparison["baseline"]
    task_ids_with_lessons = set()
    task_ids_with_cookbooks = set()
    task_ids_with_promotions = set()
    knowledge_gap_counts: Counter[str] = Counter()

    for result in candidate_results:
        task_id = str(result.get("task_id") or "").strip()
        if not task_id:
            continue
        reflection = dict(result.get("reflection") or {})
        if _lesson_ids_from_result(result):
            task_ids_with_lessons.add(task_id)
        if reflection.get("cookbook_enriched"):
            task_ids_with_cookbooks.add(task_id)
        if _normalize_strings(reflection.get("promotion_candidate_saved")):
            task_ids_with_promotions.add(task_id)
        for gap in result.get("knowledge_gaps") or []:
            knowledge_gap_counts[str(gap)] += 1

    successful_without_artifacts = list(
        candidate_summary["promotion_discipline"]["successful_tasks_without_reusable_artifacts"]
    )
    combined_failure_buckets = Counter(
        {
            bucket: count
            for bucket, count in baseline_summary["failure_buckets"].items()
            if bucket != "success"
        }
    )
    combined_failure_buckets.update(
        {
            bucket: count
            for bucket, count in candidate_summary["failure_buckets"].items()
            if bucket != "success"
        }
    )
    follow_up_suggestions = _follow_up_suggestions(
        failure_buckets=combined_failure_buckets,
        knowledge_gap_counts=knowledge_gap_counts,
        successful_without_artifacts=successful_without_artifacts,
    )

    return {
        "tasks_with_lessons": sorted(task_ids_with_lessons),
        "tasks_with_cookbooks": sorted(task_ids_with_cookbooks),
        "tasks_with_promotion_candidates": sorted(task_ids_with_promotions),
        "successful_tasks_without_reusable_artifacts": successful_without_artifacts,
        "knowledge_gap_counts": dict(knowledge_gap_counts),
        "follow_up_suggestions": follow_up_suggestions,
        "task_rows": list(task_rows),
    }


def _follow_up_suggestions(
    *,
    failure_buckets: Mapping[str, int],
    knowledge_gap_counts: Counter[str],
    successful_without_artifacts: Sequence[str],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []

    bucket_suggestions = {
        "import_hallucination": "Update the import registry and cookbook examples.",
        "missing_cookbook": "Add or refine a cookbook so the route reuses the shared path.",
        "missing_decomposition": "Split the semantic contract or route guidance into a smaller reusable slice.",
        "implementation_gap": "Open a concrete follow-up issue for the missing primitive or helper.",
        "validation_failure": "Add or tighten validation and benchmark coverage for the failure mode.",
        "comparison_insufficient_results": "Check the comparison-task cross-validation wiring for the fresh-build path.",
        "other": "Review the trace and decide whether this needs a new implementation or a knowledge note.",
    }
    for bucket, count in sorted(failure_buckets.items()):
        if count <= 0:
            continue
        suggestions.append(
            {
                "bucket": bucket,
                "count": count,
                "suggestion": bucket_suggestions.get(bucket, "Review the trace and capture the next follow-up."),
            }
        )

    for gap, count in knowledge_gap_counts.most_common(5):
        suggestions.append(
            {
                "bucket": "knowledge_gap",
                "count": count,
                "suggestion": f"Capture or escalate repeated gap: {gap}.",
            }
        )

    if successful_without_artifacts:
        suggestions.append(
            {
                "bucket": "promotion_discipline",
                "count": len(successful_without_artifacts),
                "suggestion": (
                    "Review successful tasks that did not leave reusable artifacts: "
                    + ", ".join(_format_task_ids(successful_without_artifacts))
                ),
            }
        )

    return suggestions


def _lesson_ids_from_result(result: Mapping[str, Any]) -> list[str]:
    lesson_ids: list[str] = []
    knowledge_summary = result.get("knowledge_summary") or {}
    if isinstance(knowledge_summary, Mapping):
        lesson_ids.extend(_normalize_strings(knowledge_summary.get("lesson_ids")))
    elif isinstance(knowledge_summary, Sequence) and not isinstance(
        knowledge_summary, (str, bytes)
    ):
        for summary in knowledge_summary:
            if isinstance(summary, Mapping):
                lesson_ids.extend(_normalize_strings(summary.get("lesson_ids")))
    reflection = dict(result.get("reflection") or {})
    lesson_ids.extend(_normalize_strings(reflection.get("lesson_captured")))
    return _normalize_strings(lesson_ids)


def _normalize_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        return _normalize_strings(list(value.values()))
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_normalize_strings(item))
        seen: set[str] = set()
        unique: list[str] = []
        for item in result:
            if item and item not in seen:
                seen.add(item)
                unique.append(item)
        return unique
    text = str(value).strip()
    return [text] if text else []


def _snapshot_summary(snapshot: Mapping[str, Any]) -> str:
    if not snapshot.get("present"):
        return "missing"
    parts = [
        snapshot.get("mode", "unknown"),
        "success" if snapshot.get("success") else "fail",
        f"attempts={snapshot.get('attempts', 0)}",
    ]
    if snapshot.get("lesson_ids"):
        parts.append(f"lessons={len(snapshot['lesson_ids'])}")
    if snapshot.get("cookbook_enriched"):
        parts.append("cookbook")
    if snapshot.get("promotion_candidate_saved"):
        parts.append("promotion")
    if snapshot.get("knowledge_gaps"):
        parts.append(f"gaps={len(snapshot['knowledge_gaps'])}")
    return ", ".join(parts)


def _format_task_ids(task_ids: Sequence[str]) -> list[str]:
    return [str(task_id) for task_id in task_ids if str(task_id).strip()]
