"""Negative-task benchmark selection, grading, persistence, and reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trellis.agent.evals import classify_task_result
from trellis.agent.task_manifests import load_negative_tasks


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEGATIVE_BENCHMARK_ROOT = ROOT / "task_runs" / "negative_benchmarks"


@dataclass(frozen=True)
class NegativeTaskBenchmarkArtifacts:
    report: dict[str, Any]
    json_path: Path
    text_path: Path


def load_negative_benchmark_tasks(*, root: Path = ROOT) -> list[dict[str, Any]]:
    """Load the clarification / honest-block benchmark corpus."""
    return load_negative_tasks(root=root)


def select_negative_benchmark_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    requested_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    requested = {
        str(task_id).strip()
        for task_id in (requested_ids or ())
        if str(task_id).strip()
    }
    selected: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        if not task_id:
            continue
        if requested and task_id not in requested:
            continue
        selected.append(dict(task))
        if limit is not None and len(selected) >= max(limit, 0):
            break
    return selected


def persist_negative_benchmark_record(
    record: Mapping[str, Any],
    *,
    root: Path = DEFAULT_NEGATIVE_BENCHMARK_ROOT,
) -> dict[str, str]:
    """Persist one negative-task benchmark record to append-only history and latest views."""
    task_id = str(record.get("task_id") or "unknown")
    run_id = str(record.get("run_id") or "")
    history_dir = root / "history" / task_id
    latest_dir = root / "latest"
    history_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    history_path = history_dir / f"{run_id}.json"
    latest_path = latest_dir / f"{task_id}.json"
    payload = json.dumps(dict(record), indent=2, default=str)
    history_path.write_text(payload)
    latest_path.write_text(payload)
    return {
        "history_path": str(history_path),
        "latest_path": str(latest_path),
    }


def _iter_blocker_details(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    details: list[Mapping[str, Any]] = []
    blocker_details = result.get("blocker_details")
    if isinstance(blocker_details, Mapping) and blocker_details:
        details.append(blocker_details)
    method_results = result.get("method_results") or {}
    if isinstance(method_results, Mapping):
        for payload in method_results.values():
            if not isinstance(payload, Mapping):
                continue
            blocker_details = payload.get("blocker_details")
            if isinstance(blocker_details, Mapping) and blocker_details:
                details.append(blocker_details)
    return details


def observed_negative_blocker_categories(result: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect the normalized blocker categories surfaced by a failed run."""
    observed: list[str] = []
    for details in _iter_blocker_details(result):
        blocker_report = details.get("blocker_report") or {}
        for blocker in blocker_report.get("blockers") or ():
            if not isinstance(blocker, Mapping):
                continue
            category = str(blocker.get("category") or "").strip()
            if category:
                observed.append(category)
        for blocker in details.get("blockers") or ():
            text = str(blocker or "").strip()
            if text:
                observed.append(text)
        reason = str(details.get("reason") or "").strip()
        if reason == "semantic_clarification_required":
            observed.append("semantic_clarification")
        semantic_gap = details.get("semantic_gap") or {}
        if isinstance(semantic_gap, Mapping):
            if semantic_gap.get("requires_clarification"):
                observed.append("semantic_clarification")
            if semantic_gap.get("missing_contract_fields"):
                observed.append("semantic_contract_gap")
    return tuple(dict.fromkeys(item for item in observed if item))


def observed_negative_missing_fields(result: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect missing semantic fields surfaced during clarification failures."""
    observed: list[str] = []
    for details in _iter_blocker_details(result):
        semantic_gap = details.get("semantic_gap") or {}
        if not isinstance(semantic_gap, Mapping):
            continue
        for field_name in semantic_gap.get("missing_contract_fields") or ():
            text = str(field_name or "").strip()
            if text:
                observed.append(text)
    return tuple(dict.fromkeys(observed))


def classify_negative_task_outcome(result: Mapping[str, Any]) -> str:
    """Classify the observed negative-task outcome into a stable benchmark label."""
    if result.get("success"):
        return "priced"
    categories = observed_negative_blocker_categories(result)
    if "semantic_clarification" in categories:
        return "clarification_requested"
    if categories or _iter_blocker_details(result):
        return "honest_block"
    return "failed"


def evaluate_negative_task_result(
    task: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one negative-task result against the manifest expectation."""
    expected_outcome = str(task.get("expected_outcome") or "").strip()
    clarification_contract = dict(task.get("clarification_contract") or {})
    expected_missing_fields = tuple(
        str(field).strip()
        for field in (clarification_contract.get("missing_fields") or ())
        if str(field).strip()
    )
    expected_blockers = tuple(
        str(blocker).strip()
        for blocker in (clarification_contract.get("expected_blockers") or ())
        if str(blocker).strip()
    )
    observed_outcome = classify_negative_task_outcome(result)
    observed_categories = observed_negative_blocker_categories(result)
    observed_missing = observed_negative_missing_fields(result)

    passed = observed_outcome == expected_outcome
    details: list[str] = []
    if observed_outcome != expected_outcome:
        details.append(
            f"expected outcome `{expected_outcome or 'missing'}` but observed `{observed_outcome}`"
        )
    if expected_outcome == "clarification_requested" and expected_missing_fields:
        if not set(observed_missing) & set(expected_missing_fields):
            passed = False
            details.append(
                "expected clarification fields were not surfaced: "
                f"observed={observed_missing or ('none',)} expected={expected_missing_fields}"
            )
    if expected_outcome == "honest_block" and expected_blockers:
        if not set(observed_categories) & set(expected_blockers):
            passed = False
            details.append(
                "expected honest-block categories were not surfaced: "
                f"observed={observed_categories or ('none',)} expected={expected_blockers}"
            )

    return {
        "expected_outcome": expected_outcome,
        "observed_outcome": observed_outcome,
        "passed": passed,
        "details": tuple(details),
        "result_bucket": classify_task_result(dict(result)),
        "observed_blocker_categories": observed_categories,
        "observed_missing_fields": observed_missing,
        "expected_missing_fields": expected_missing_fields,
        "expected_blockers": expected_blockers,
    }


def build_negative_benchmark_report(
    *,
    benchmark_name: str,
    git_revision: str,
    benchmark_runs: Sequence[Mapping[str, Any]],
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a run-level negative-task benchmark report."""
    expected_counts: dict[str, int] = {}
    observed_counts: dict[str, int] = {}
    passed_count = 0
    elapsed_total = 0.0
    token_total = 0
    for run in benchmark_runs:
        expected = str(run.get("expected_outcome") or "").strip()
        observed = str(run.get("observed_outcome") or "").strip()
        if expected:
            expected_counts[expected] = expected_counts.get(expected, 0) + 1
        if observed:
            observed_counts[observed] = observed_counts.get(observed, 0) + 1
        if run.get("passed_expectation"):
            passed_count += 1
        elapsed_total += float(run.get("elapsed_seconds") or 0.0)
        token_total += int(dict(run.get("token_usage_summary") or {}).get("total_tokens") or 0)
    return {
        "benchmark_name": benchmark_name,
        "git_revision": git_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(benchmark_runs),
        "passed_count": passed_count,
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "elapsed_seconds_total": round(elapsed_total, 6),
        "token_usage_total": token_total,
        "tasks": [dict(run) for run in benchmark_runs],
        "notes": list(notes or ()),
    }


def render_negative_benchmark_report(report: Mapping[str, Any]) -> str:
    """Render the negative-task benchmark report as Markdown."""
    lines = [
        f"# Negative Task Benchmark: `{report['benchmark_name']}`",
        f"- Git revision: `{report.get('git_revision', '')}`",
        f"- Tasks: `{report.get('task_count', 0)}`",
        f"- Passed expectation: `{report.get('passed_count', 0)}`",
        f"- Total elapsed: `{report.get('elapsed_seconds_total', 0)}`",
        f"- Total tokens: `{report.get('token_usage_total', 0)}`",
    ]
    if report.get("notes"):
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in report["notes"])
    lines.extend(["", "## Task Runs"])
    for task in report.get("tasks") or []:
        lines.extend(
            [
                "",
                f"### `{task.get('task_id', '')}` {task.get('title', '')}",
                f"- Expected: `{task.get('expected_outcome', '')}`",
                f"- Observed: `{task.get('observed_outcome', '')}`",
                f"- Passed: `{bool(task.get('passed_expectation'))}`",
                f"- Started: `{task.get('run_started_at', '')}`",
                f"- Completed: `{task.get('run_completed_at', '')}`",
                f"- Elapsed: `{task.get('elapsed_seconds', 0)}`",
                f"- Result bucket: `{task.get('result_bucket', '')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def save_negative_benchmark_report(
    report: Mapping[str, Any],
    *,
    root: Path = DEFAULT_NEGATIVE_BENCHMARK_ROOT,
    stem: str,
) -> NegativeTaskBenchmarkArtifacts:
    """Persist a run-level negative-task benchmark report."""
    reports_root = root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    json_path = reports_root / f"{stem}.json"
    text_path = reports_root / f"{stem}.md"
    json_path.write_text(json.dumps(dict(report), indent=2, default=str))
    text_path.write_text(render_negative_benchmark_report(report))
    return NegativeTaskBenchmarkArtifacts(
        report=dict(report),
        json_path=json_path,
        text_path=text_path,
    )
