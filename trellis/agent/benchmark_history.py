"""Repeated-run benchmark history loading and scorecard rendering."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORECARD_ROOT = ROOT / "task_runs" / "benchmark_scorecards"


@dataclass(frozen=True)
class BenchmarkHistoryArtifacts:
    report: dict[str, Any]
    json_path: Path
    text_path: Path


def load_benchmark_history_records(
    *,
    benchmark_root: Path,
    task_ids: Sequence[str] | None = None,
    campaign_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load append-only benchmark history records from one benchmark root."""
    requested = {
        str(task_id).strip()
        for task_id in (task_ids or ())
        if str(task_id).strip()
    }
    history_root = benchmark_root / "history"
    if not history_root.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(history_root.glob("*/*.json")):
        payload = json.loads(path.read_text())
        task_id = str(payload.get("task_id") or "").strip()
        if requested and task_id not in requested:
            continue
        if campaign_id is not None and (
            str(payload.get("benchmark_campaign_id") or "").strip()
            != str(campaign_id).strip()
        ):
            continue
        records.append(dict(payload))
    return sorted(records, key=_history_sort_key)


def build_benchmark_history_scorecard(
    *,
    scorecard_name: str,
    benchmark_kind: str,
    benchmark_runs: Sequence[Mapping[str, Any]],
    campaign_id: str | None = None,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a repeated-run scorecard from append-only benchmark history."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in benchmark_runs:
        task_id = str(record.get("task_id") or "").strip()
        if task_id:
            grouped[task_id].append(dict(record))

    task_summaries: list[dict[str, Any]] = []
    improved_count = 0
    regressed_count = 0
    latest_pass_count = 0
    for task_id in sorted(grouped):
        runs = sorted(grouped[task_id], key=_history_sort_key)
        summary = build_task_history_summary(task_id, runs, benchmark_kind=benchmark_kind)
        if summary["transition"] == "improved":
            improved_count += 1
        elif summary["transition"] == "regressed":
            regressed_count += 1
        if summary["latest"]["passed"]:
            latest_pass_count += 1
        task_summaries.append(summary)

    return {
        "scorecard_name": scorecard_name,
        "benchmark_kind": benchmark_kind,
        "benchmark_campaign_id": campaign_id or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(task_summaries),
        "run_count": len(benchmark_runs),
        "improved_count": improved_count,
        "regressed_count": regressed_count,
        "unchanged_count": len(task_summaries) - improved_count - regressed_count,
        "latest_pass_count": latest_pass_count,
        "tasks": task_summaries,
        "notes": list(notes or ()),
    }


def render_benchmark_history_scorecard(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Benchmark History Scorecard: `{report['scorecard_name']}`",
        f"- Benchmark kind: `{report.get('benchmark_kind', '')}`",
        f"- Campaign: `{report.get('benchmark_campaign_id', '') or 'all'}`",
        f"- Tasks: `{report.get('task_count', 0)}`",
        f"- Runs: `{report.get('run_count', 0)}`",
        f"- Latest pass count: `{report.get('latest_pass_count', 0)}`",
        f"- Improved: `{report.get('improved_count', 0)}`",
        f"- Regressed: `{report.get('regressed_count', 0)}`",
    ]
    if report.get("notes"):
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in report["notes"])

    lines.extend(["", "## Task History"])
    for task in report.get("tasks") or ():
        first = dict(task.get("first") or {})
        latest = dict(task.get("latest") or {})
        lines.extend(
            [
                "",
                f"### `{task.get('task_id', '')}` {task.get('title', '')}",
                f"- Runs: `{task.get('run_count', 0)}`",
                f"- Transition: `{task.get('transition', 'unchanged')}`",
                f"- First run: `{first.get('run_started_at', '')}` `{first.get('git_sha', '')}` `{first.get('knowledge_revision', '')}` `{first.get('result_label', '')}`",
                f"- Latest run: `{latest.get('run_started_at', '')}` `{latest.get('git_sha', '')}` `{latest.get('knowledge_revision', '')}` `{latest.get('result_label', '')}`",
                f"- Elapsed delta: `{task.get('elapsed_seconds_delta', 0.0):+.6f}`",
                f"- Token delta: `{task.get('token_total_delta', 0):+d}`",
                f"- Execution modes: `{', '.join(task.get('execution_modes', ())) or 'none'}`",
            ]
        )
    return "\n".join(lines) + "\n"


def save_benchmark_history_scorecard(
    report: Mapping[str, Any],
    *,
    reports_root: Path = DEFAULT_SCORECARD_ROOT,
    stem: str,
) -> BenchmarkHistoryArtifacts:
    reports_root.mkdir(parents=True, exist_ok=True)
    json_path = reports_root / f"{stem}.json"
    text_path = reports_root / f"{stem}.md"
    payload = dict(report)
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    text_path.write_text(render_benchmark_history_scorecard(payload))
    return BenchmarkHistoryArtifacts(report=payload, json_path=json_path, text_path=text_path)


def history_sort_key(record: Mapping[str, Any]) -> tuple[str, str]:
    """Stable sort key for benchmark history records.

    Public so per-corpus scorecard generators (`pilot_parity_scorecard` and
    future siblings) can sort records without reaching into a private helper
    in this module.  (PR #590 round-3 Copilot review.)
    """
    return (
        str(record.get("run_started_at") or record.get("run_completed_at") or ""),
        str(record.get("run_id") or ""),
    )


# Underscored alias preserved for callers that already imported the private
# helper; new code should use the public name above.
_history_sort_key = history_sort_key


def build_task_history_summary(
    task_id: str,
    runs: Sequence[Mapping[str, Any]],
    *,
    benchmark_kind: str,
) -> dict[str, Any]:
    first = dict(runs[0])
    latest = dict(runs[-1])
    first_snapshot = _task_run_snapshot(first, benchmark_kind=benchmark_kind)
    latest_snapshot = _task_run_snapshot(latest, benchmark_kind=benchmark_kind)
    transition = _transition_label(first_snapshot["passed"], latest_snapshot["passed"])
    return {
        "task_id": task_id,
        "title": str(latest.get("title") or first.get("title") or ""),
        "task_corpus": str(latest.get("task_corpus") or first.get("task_corpus") or ""),
        "run_count": len(runs),
        "transition": transition,
        "execution_modes": tuple(
            dict.fromkeys(
                str(run.get("execution_mode") or "").strip()
                for run in runs
                if str(run.get("execution_mode") or "").strip()
            )
        ),
        "elapsed_seconds_delta": round(
            float(latest_snapshot["elapsed_seconds"]) - float(first_snapshot["elapsed_seconds"]),
            6,
        ),
        "token_total_delta": int(latest_snapshot["token_total"]) - int(first_snapshot["token_total"]),
        "first": first_snapshot,
        "latest": latest_snapshot,
    }


# Underscored alias preserved for callers that already imported the private
# helper; new code should use the public name above.
_build_task_history_summary = build_task_history_summary


def _task_run_snapshot(record: Mapping[str, Any], *, benchmark_kind: str) -> dict[str, Any]:
    return {
        "run_id": str(record.get("run_id") or ""),
        "run_started_at": str(record.get("run_started_at") or ""),
        "run_completed_at": str(record.get("run_completed_at") or ""),
        "execution_mode": str(record.get("execution_mode") or ""),
        "git_sha": str(record.get("git_sha") or ""),
        "knowledge_revision": str(record.get("knowledge_revision") or ""),
        "result_label": _result_label(record, benchmark_kind=benchmark_kind),
        "passed": _record_passed(record, benchmark_kind=benchmark_kind),
        "elapsed_seconds": _elapsed_seconds(record),
        "token_total": _token_total(record),
    }


def _result_label(record: Mapping[str, Any], *, benchmark_kind: str) -> str:
    if benchmark_kind == "negative":
        if bool(record.get("passed_expectation")):
            return "passed_expectation"
        return str(record.get("observed_outcome") or record.get("status") or "").strip()
    comparison = dict(record.get("comparison_summary") or {})
    return str(comparison.get("status") or record.get("status") or "").strip()


def _record_passed(record: Mapping[str, Any], *, benchmark_kind: str) -> bool:
    if benchmark_kind == "negative":
        return bool(record.get("passed_expectation"))
    return str(dict(record.get("comparison_summary") or {}).get("status") or "").strip() == "passed"


def _elapsed_seconds(record: Mapping[str, Any]) -> float:
    if record.get("cold_agent_elapsed_seconds") is not None:
        return float(record.get("cold_agent_elapsed_seconds") or 0.0)
    return float(record.get("elapsed_seconds") or 0.0)


def _token_total(record: Mapping[str, Any]) -> int:
    if isinstance(record.get("cold_agent_token_usage"), Mapping):
        return int(dict(record.get("cold_agent_token_usage") or {}).get("total_tokens") or 0)
    return int(dict(record.get("token_usage_summary") or {}).get("total_tokens") or 0)


def _transition_label(first_passed: bool, latest_passed: bool) -> str:
    if first_passed and not latest_passed:
        return "regressed"
    if not first_passed and latest_passed:
        return "improved"
    return "unchanged"
