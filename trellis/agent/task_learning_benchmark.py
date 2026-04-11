"""Repeated-pass task-learning benchmark helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trellis.agent.evals import classify_task_result, compare_task_runs, summarize_task_results


ROOT = Path(__file__).resolve().parents[2]
CANARY_TASKS_MANIFEST = ROOT / "CANARY_TASKS.yaml"
DEFAULT_REPORT_ROOT = ROOT / "task_runs" / "learning_benchmarks" / "reports"


@dataclass(frozen=True)
class TaskLearningBenchmarkArtifacts:
    """Persisted files for one task-learning benchmark report."""

    report: dict[str, Any]
    json_path: Path
    text_path: Path


def load_canary_task_ids(path: Path | None = None) -> set[str]:
    """Load the curated canary task ids used to exclude benchmark tasks."""
    manifest_path = path or CANARY_TASKS_MANIFEST
    if not manifest_path.exists():
        return set()
    payload = yaml.safe_load(manifest_path.read_text()) or {}
    return {
        str(entry.get("id") or "").strip()
        for entry in payload.get("canary_set") or ()
        if isinstance(entry, Mapping) and str(entry.get("id") or "").strip()
    }


def select_task_learning_cohort(
    tasks: Sequence[Mapping[str, Any]],
    *,
    canary_task_ids: set[str],
    allowed_statuses: Sequence[str] = ("pending",),
    requested_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select the short-term learning cohort from the pricing-task inventory."""
    allowed = {str(status).strip().lower() for status in allowed_statuses if str(status).strip()}
    requested = {
        str(task_id).strip()
        for task_id in (requested_ids or ())
        if str(task_id).strip()
    }

    selected: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        if not task_id or task_id in canary_task_ids:
            continue
        status = str(task.get("status") or "").strip().lower()
        if allowed and status not in allowed:
            continue
        if requested and task_id not in requested:
            continue
        selected.append(dict(task))
        if limit is not None and len(selected) >= max(limit, 0):
            break
    return selected


def build_task_learning_benchmark_report(
    *,
    benchmark_name: str,
    cohort_name: str,
    git_revision: str,
    tasks: Sequence[Mapping[str, Any]],
    pass_runs: Sequence[Mapping[str, Any]],
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a repeated-pass learning report for one non-canary cohort."""
    normalized_passes: list[dict[str, Any]] = []
    raw_results_by_pass: list[list[Mapping[str, Any]]] = []

    for raw_pass in sorted(pass_runs, key=lambda item: int(item.get("pass_number") or 0)):
        results = list(raw_pass.get("results") or [])
        raw_results_by_pass.append(results)
        elapsed_total = round(
            sum(float(result.get("elapsed_seconds") or 0.0) for result in results),
            2,
        )
        summary = summarize_task_results(results)
        normalized_passes.append(
            {
                "pass_number": int(raw_pass.get("pass_number") or 0),
                "label": str(raw_pass.get("label") or f"pass_{len(normalized_passes) + 1}"),
                "fresh_build": bool(raw_pass.get("fresh_build")),
                "knowledge_profile": str(raw_pass.get("knowledge_profile") or "default"),
                "model": str(raw_pass.get("model") or ""),
                "validation": str(raw_pass.get("validation") or ""),
                "results_path": str(raw_pass.get("results_path") or ""),
                "summary_path": str(raw_pass.get("summary_path") or ""),
                "result_count": len(results),
                "task_ids": [
                    str(result.get("task_id") or "").strip()
                    for result in results
                    if str(result.get("task_id") or "").strip()
                ],
                "elapsed_seconds_total": elapsed_total,
                "elapsed_seconds_average": round(elapsed_total / len(results), 2) if results else 0.0,
                "summary": summary,
            }
        )

    pairwise_comparisons: list[dict[str, Any]] = []
    for previous, current, previous_results, current_results in zip(
        normalized_passes,
        normalized_passes[1:],
        raw_results_by_pass,
        raw_results_by_pass[1:],
    ):
        pairwise_comparisons.append(
            _build_pass_comparison(
                previous=previous,
                current=current,
                previous_results=previous_results,
                current_results=current_results,
            )
        )

    overall_comparison = (
        _build_pass_comparison(
            previous=normalized_passes[0],
            current=normalized_passes[-1],
            previous_results=raw_results_by_pass[0],
            current_results=raw_results_by_pass[-1],
        )
        if len(normalized_passes) >= 2
        else {}
    )
    attribution = (
        _build_learning_attribution(
            baseline_results=raw_results_by_pass[0],
            latest_results=raw_results_by_pass[-1],
        )
        if raw_results_by_pass
        else _empty_attribution()
    )

    return {
        "benchmark_name": benchmark_name,
        "cohort_name": cohort_name,
        "git_revision": git_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_ids": [str(task.get("id") or "") for task in tasks],
        "task_titles": {
            str(task.get("id") or ""): str(task.get("title") or "")
            for task in tasks
        },
        "pass_count": len(normalized_passes),
        "passes": normalized_passes,
        "pairwise_comparisons": pairwise_comparisons,
        "overall_comparison": overall_comparison,
        "attribution": attribution,
        "notes": list(notes or ()),
    }


def render_task_learning_benchmark_report(report: Mapping[str, Any]) -> str:
    """Render one repeated-pass learning scorecard as Markdown."""
    lines = [
        f"# Task Learning Benchmark: `{report['benchmark_name']}`",
        f"- Cohort: `{report.get('cohort_name', '')}`",
        f"- Git revision: `{report.get('git_revision', '')}`",
        f"- Passes: `{report.get('pass_count', 0)}`",
        f"- Tasks: {', '.join(f'`{task_id}`' for task_id in report.get('task_ids', []))}",
    ]
    if report.get("notes"):
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in report["notes"])

    lines.extend(["", "## Pass Summaries"])
    for learning_pass in report.get("passes") or []:
        summary = learning_pass["summary"]
        lines.extend(
            [
                "",
                f"### Pass {learning_pass['pass_number']}: `{learning_pass['label']}`",
                f"- Success: `{summary['totals']['successes']}/{summary['totals']['tasks']}`",
                f"- First-pass rate: `{summary['first_pass']['rate']:.0%}`",
                f"- Attempts-to-success average: `{summary['attempts_to_success']['average']}`",
                f"- Retry recoveries: `{summary['retry_taxonomy']['recovered_successes']}`",
                f"- Elapsed seconds: `{learning_pass['elapsed_seconds_total']}`",
                f"- Token usage: `{summary['token_usage']['total_tokens']}`",
                f"- Shared-knowledge tasks: `{summary['shared_knowledge']['tasks_with_shared_context']}`",
                f"- Results path: `{learning_pass.get('results_path', '')}`",
            ]
        )

    if report.get("pairwise_comparisons"):
        lines.extend(["", "## Pass Deltas"])
        for comparison in report["pairwise_comparisons"]:
            deltas = comparison["deltas"]
            transition = comparison["comparison"]["task_transitions"]
            lines.extend(
                [
                    "",
                    f"### Pass {comparison['from_pass']} -> Pass {comparison['to_pass']}",
                    f"- Improved: `{transition['improved']}`",
                    f"- Regressed: `{transition['regressed']}`",
                    f"- Unchanged: `{transition['unchanged']}`",
                    f"- Success delta: `{deltas['successes']:+d}`",
                    f"- First-pass rate delta: `{deltas['first_pass_rate']:+.2f}`",
                    f"- Attempts-to-success delta: `{deltas['attempts_to_success_average']:+.2f}`",
                    f"- Elapsed seconds delta: `{deltas['elapsed_seconds_total']:+.2f}`",
                    f"- Token usage delta: `{deltas['token_usage_total']:+d}`",
                ]
            )

    attribution = dict(report.get("attribution") or {})
    lines.extend(
        [
            "",
            "## Attribution",
            f"- Knowledge-assisted improvements: {', '.join(_format_task_ids(attribution.get('knowledge_assisted_improvements', {}).get('task_ids') or [])) or 'none'}",
            f"- Residual knowledge gaps: {', '.join(_format_task_ids(attribution.get('residual_knowledge_gaps', {}).get('task_ids') or [])) or 'none'}",
            f"- Residual implementation gaps: {', '.join(_format_task_ids(attribution.get('residual_implementation_gaps', {}).get('task_ids') or [])) or 'none'}",
            f"- Residual market/provider noise: {', '.join(_format_task_ids(attribution.get('residual_market_or_provider_noise', {}).get('task_ids') or [])) or 'none'}",
            f"- Unexplained improvements: {', '.join(_format_task_ids(attribution.get('unexplained_improvements', {}).get('task_ids') or [])) or 'none'}",
        ]
    )

    return "\n".join(lines) + "\n"


def save_task_learning_benchmark_report(
    report: Mapping[str, Any],
    *,
    root: Path | None = None,
    stem: str,
) -> TaskLearningBenchmarkArtifacts:
    """Persist one task-learning report as JSON plus Markdown."""
    output_root = root or DEFAULT_REPORT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{stem}.json"
    text_path = output_root / f"{stem}.md"
    payload = dict(report)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    text_path.write_text(render_task_learning_benchmark_report(payload))
    return TaskLearningBenchmarkArtifacts(report=payload, json_path=json_path, text_path=text_path)


def _build_pass_comparison(
    *,
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    previous_results: Sequence[Mapping[str, Any]],
    current_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparison = compare_task_runs(list(previous_results), list(current_results))
    return {
        "from_pass": int(previous.get("pass_number") or 0),
        "to_pass": int(current.get("pass_number") or 0),
        "comparison": comparison,
        "deltas": {
            "successes": (
                int(current["summary"]["totals"]["successes"])
                - int(previous["summary"]["totals"]["successes"])
            ),
            "first_pass_rate": round(
                float(current["summary"]["first_pass"]["rate"])
                - float(previous["summary"]["first_pass"]["rate"]),
                2,
            ),
            "attempts_to_success_average": round(
                float(current["summary"]["attempts_to_success"]["average"])
                - float(previous["summary"]["attempts_to_success"]["average"]),
                2,
            ),
            "elapsed_seconds_total": round(
                float(current.get("elapsed_seconds_total") or 0.0)
                - float(previous.get("elapsed_seconds_total") or 0.0),
                2,
            ),
            "token_usage_total": (
                int(current["summary"]["token_usage"]["total_tokens"])
                - int(previous["summary"]["token_usage"]["total_tokens"])
            ),
        },
    }


def _build_learning_attribution(
    *,
    baseline_results: Sequence[Mapping[str, Any]],
    latest_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {
        str(result.get("task_id") or "").strip(): result
        for result in baseline_results
        if str(result.get("task_id") or "").strip()
    }
    latest_by_id = {
        str(result.get("task_id") or "").strip(): result
        for result in latest_results
        if str(result.get("task_id") or "").strip()
    }

    improved: list[str] = []
    knowledge_assisted: list[str] = []
    unexplained: list[str] = []
    residual_knowledge: list[str] = []
    residual_implementation: list[str] = []
    residual_market_noise: list[str] = []

    for task_id in sorted(set(baseline_by_id) | set(latest_by_id)):
        baseline = baseline_by_id.get(task_id, {})
        latest = latest_by_id.get(task_id, {})
        if _transition_rank(classify_task_result(latest)) < _transition_rank(classify_task_result(baseline)):
            improved.append(task_id)
            if _result_has_knowledge_signal(latest):
                knowledge_assisted.append(task_id)
            else:
                unexplained.append(task_id)
        if latest and not latest.get("success"):
            if _result_is_market_or_provider_noise(latest):
                residual_market_noise.append(task_id)
            elif latest.get("knowledge_gaps"):
                residual_knowledge.append(task_id)
            else:
                residual_implementation.append(task_id)

    return {
        "improved_tasks": {"count": len(improved), "task_ids": improved},
        "knowledge_assisted_improvements": {
            "count": len(knowledge_assisted),
            "task_ids": knowledge_assisted,
        },
        "unexplained_improvements": {"count": len(unexplained), "task_ids": unexplained},
        "residual_knowledge_gaps": {
            "count": len(residual_knowledge),
            "task_ids": residual_knowledge,
        },
        "residual_implementation_gaps": {
            "count": len(residual_implementation),
            "task_ids": residual_implementation,
        },
        "residual_market_or_provider_noise": {
            "count": len(residual_market_noise),
            "task_ids": residual_market_noise,
        },
    }


def _empty_attribution() -> dict[str, Any]:
    return {
        "improved_tasks": {"count": 0, "task_ids": []},
        "knowledge_assisted_improvements": {"count": 0, "task_ids": []},
        "unexplained_improvements": {"count": 0, "task_ids": []},
        "residual_knowledge_gaps": {"count": 0, "task_ids": []},
        "residual_implementation_gaps": {"count": 0, "task_ids": []},
        "residual_market_or_provider_noise": {"count": 0, "task_ids": []},
    }


def _result_has_knowledge_signal(result: Mapping[str, Any]) -> bool:
    knowledge_summary = result.get("knowledge_summary") or {}
    if isinstance(knowledge_summary, Mapping):
        if any(_normalize_strings(value) for value in knowledge_summary.values()):
            return True
    reflection = result.get("reflection") or {}
    if not isinstance(reflection, Mapping):
        return False
    if reflection.get("cookbook_enriched"):
        return True
    if int(reflection.get("lessons_attributed") or 0) > 0:
        return True
    for key in (
        "lesson_captured",
        "cookbook_candidate_saved",
        "promotion_candidate_saved",
        "knowledge_trace_saved",
    ):
        if _normalize_strings(reflection.get(key)):
            return True
    return False


def _result_is_market_or_provider_noise(result: Mapping[str, Any]) -> bool:
    bucket = classify_task_result(result)
    if bucket in {"missing_market_data", "timeout", "llm_response"}:
        return True

    fragments = [
        str(result.get("error") or ""),
        *[str(item) for item in (result.get("failures") or ())],
    ]
    text = "\n".join(fragment for fragment in fragments if fragment).lower()
    return any(pattern in text for pattern in ("rate limit", "quota", "provider", "missing market data"))


def _transition_rank(bucket: str) -> int:
    return {
        "success": 0,
        "missing_market_data": 5,
        "timeout": 6,
        "llm_response": 6,
        "semantic_validation": 4,
        "import_validation": 4,
        "comparison_failed": 4,
        "comparison_insufficient_results": 4,
        "build_failure": 4,
        "blocked": 4,
        "missing": 7,
    }.get(str(bucket or "").strip(), 4)


def _normalize_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        return _normalize_strings(list(value.values()))
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_normalize_strings(item))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item and item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped
    text = str(value).strip()
    return [text] if text else []


def _format_task_ids(task_ids: Sequence[str]) -> list[str]:
    return [str(task_id) for task_id in task_ids if str(task_id).strip()]
