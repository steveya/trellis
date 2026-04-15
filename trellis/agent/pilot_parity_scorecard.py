"""Fresh-generated FinancePy pilot parity scorecard.

Consumes append-only history records from
``task_runs/financepy_benchmarks/history`` for the pilot subset
(F001/F002/F003/F007/F009/F012), verifies each run's provenance against the
QUA-866 fresh-generated boundary, and emits a timestamped scorecard summary
that flags residual misses as shared root-cause follow-ons rather than
task-specific patches.  Built to close QUA-868 under epic QUA-864.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trellis.agent.benchmark_history import (
    _build_task_history_summary,
    _history_sort_key,
)
from trellis.agent.benchmark_pilots import get_pilot_task_ids
from trellis.agent.financepy_benchmark import DEFAULT_FINANCEPY_BENCHMARK_ROOT


PILOT_SCORECARD_TASK_IDS: tuple[str, ...] = get_pilot_task_ids("financepy")


@dataclass(frozen=True)
class PilotScorecardArtifacts:
    report: dict[str, Any]
    json_path: Path
    text_path: Path


def load_pilot_benchmark_records(
    *,
    benchmark_root: Path = DEFAULT_FINANCEPY_BENCHMARK_ROOT,
    campaign_id: str | None = None,
    task_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Load append-only history records restricted to the pilot subset."""
    requested = {
        str(task_id).strip()
        for task_id in (task_ids or PILOT_SCORECARD_TASK_IDS)
        if str(task_id).strip()
    }
    history_root = Path(benchmark_root) / "history"
    if not history_root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(history_root.glob("*/*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, Mapping):
            continue
        task_id = str(payload.get("task_id") or "").strip()
        if task_id not in requested:
            continue
        if campaign_id is not None and (
            str(payload.get("benchmark_campaign_id") or "").strip()
            != str(campaign_id).strip()
        ):
            continue
        records.append(dict(payload))
    return sorted(records, key=_history_sort_key)


def build_pilot_parity_scorecard(
    *,
    scorecard_name: str,
    benchmark_runs: Sequence[Mapping[str, Any]],
    campaign_id: str | None = None,
    pilot_task_ids: Sequence[str] | None = None,
    notes: Sequence[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a timestamped pilot parity scorecard from append-only history."""
    expected = tuple(sorted(pilot_task_ids or PILOT_SCORECARD_TASK_IDS))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in benchmark_runs:
        task_id = str(record.get("task_id") or "").strip()
        if task_id in expected:
            grouped[task_id].append(dict(record))

    task_summaries: list[dict[str, Any]] = []
    residual_misses: list[dict[str, Any]] = []
    enforced = 0
    violations = 0
    missing = 0
    latest_pass = 0

    for task_id in expected:
        runs = sorted(grouped.get(task_id) or (), key=_history_sort_key)
        if not runs:
            missing += 1
            residual_misses.append(
                {
                    "task_id": task_id,
                    "category": "missing_run",
                    "reason": "no fresh-generated benchmark run recorded for this pilot task",
                    "run_id": "",
                    "run_started_at": "",
                    "git_sha": "",
                    "knowledge_revision": "",
                }
            )
            continue
        summary = _build_task_history_summary(
            task_id, runs, benchmark_kind="financepy"
        )
        latest_record = dict(runs[-1])
        boundary = latest_record.get("fresh_generated_boundary") or {}
        if not isinstance(boundary, Mapping):
            boundary = {}
        boundary_status = str(boundary.get("status") or "").strip().lower()
        if boundary_status == "enforced":
            enforced += 1
        elif boundary_status == "violated":
            violations += 1
        summary["fresh_generated_boundary_status"] = boundary_status or "unknown"
        summary["fresh_generated_boundary_reason"] = str(boundary.get("reason") or "")
        summary["fresh_generated_boundary_violations"] = list(
            boundary.get("violations") or ()
        )
        summary["fresh_generated_module"] = str(boundary.get("generated_module") or "")
        latest_snapshot = summary.get("latest") or {}
        summary["latest_comparison_status"] = str(
            dict(latest_record.get("comparison_summary") or {}).get("status") or ""
        )
        summary["latest_run_id"] = str(latest_snapshot.get("run_id") or "")
        if latest_snapshot.get("passed"):
            latest_pass += 1
        else:
            if boundary_status == "violated":
                miss_category = "boundary_violation"
                miss_reason = (
                    str(boundary.get("reason") or "")
                    or "fresh-generated boundary violation on pilot run"
                )
            else:
                miss_category = "parity_failure"
                miss_reason = (
                    summary["latest_comparison_status"]
                    or str(latest_record.get("status") or "")
                    or "latest pilot run did not pass FinancePy parity"
                )
            residual_misses.append(
                {
                    "task_id": task_id,
                    "category": miss_category,
                    "reason": miss_reason,
                    "run_id": str(latest_snapshot.get("run_id") or ""),
                    "run_started_at": str(latest_snapshot.get("run_started_at") or ""),
                    "git_sha": str(latest_snapshot.get("git_sha") or ""),
                    "knowledge_revision": str(
                        latest_snapshot.get("knowledge_revision") or ""
                    ),
                }
            )
        task_summaries.append(summary)

    pilot_summary = {
        "fresh_generated_enforced_count": enforced,
        "boundary_violation_count": violations,
        "missing_run_count": missing,
        "latest_pass_count": latest_pass,
    }
    return {
        "scorecard_name": scorecard_name,
        "benchmark_kind": "financepy",
        "benchmark_campaign_id": campaign_id or "",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "pilot_task_ids": list(expected),
        "task_count": len(task_summaries) + missing,
        "run_count": sum(len(runs) for runs in grouped.values()),
        "pilot_summary": pilot_summary,
        "residual_misses": residual_misses,
        "tasks": task_summaries,
        "notes": list(notes or ()),
    }


def render_pilot_parity_scorecard(scorecard: Mapping[str, Any]) -> str:
    """Render the pilot scorecard as Markdown."""
    summary = dict(scorecard.get("pilot_summary") or {})
    lines = [
        f"# Fresh-Generated FinancePy Pilot Scorecard: `{scorecard.get('scorecard_name', '')}`",
        f"- Created at: `{scorecard.get('created_at', '')}`",
        f"- Campaign: `{scorecard.get('benchmark_campaign_id', '') or 'all'}`",
        f"- Pilot task ids: `{', '.join(scorecard.get('pilot_task_ids') or ())}`",
        f"- Task count: `{scorecard.get('task_count', 0)}`",
        f"- Run count: `{scorecard.get('run_count', 0)}`",
        "",
        "## Pilot Summary",
        f"- Fresh-generated enforced: `{summary.get('fresh_generated_enforced_count', 0)}`",
        f"- Boundary violations: `{summary.get('boundary_violation_count', 0)}`",
        f"- Missing runs: `{summary.get('missing_run_count', 0)}`",
        f"- Latest pass count: `{summary.get('latest_pass_count', 0)}`",
    ]
    if scorecard.get("notes"):
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in scorecard["notes"])

    residual = scorecard.get("residual_misses") or ()
    lines.extend(["", "## Residual Misses"])
    if not residual:
        lines.append("- None. Every pilot task reached FinancePy parity under the fresh-generated path.")
    else:
        for miss in residual:
            lines.append(
                f"- `{miss.get('task_id', '')}` [`{miss.get('category', '')}`] "
                f"{miss.get('reason', '')}"
            )
            context_bits: list[str] = []
            if miss.get("run_id"):
                context_bits.append(f"run_id=`{miss['run_id']}`")
            if miss.get("git_sha"):
                context_bits.append(f"git_sha=`{miss['git_sha']}`")
            if miss.get("knowledge_revision"):
                context_bits.append(f"knowledge_revision=`{miss['knowledge_revision']}`")
            if context_bits:
                lines.append(f"  - {', '.join(context_bits)}")

    lines.extend(["", "## Task History"])
    for task in scorecard.get("tasks") or ():
        latest = dict(task.get("latest") or {})
        lines.extend(
            [
                "",
                f"### `{task.get('task_id', '')}` {task.get('title', '')}",
                f"- Runs: `{task.get('run_count', 0)}`",
                f"- Transition: `{task.get('transition', 'unchanged')}`",
                f"- Fresh-generated boundary: `{task.get('fresh_generated_boundary_status', '')}`",
                f"- Latest comparison: `{task.get('latest_comparison_status', '')}`",
                f"- Latest run: `{latest.get('run_started_at', '')}` "
                f"`{latest.get('git_sha', '')}` "
                f"`{latest.get('knowledge_revision', '')}`",
                f"- Latest passed: `{latest.get('passed', False)}`",
            ]
        )
        if task.get("fresh_generated_boundary_status") == "violated":
            reason = task.get("fresh_generated_boundary_reason") or "boundary violation"
            lines.append(f"- Boundary reason: `{reason}`")
    return "\n".join(lines) + "\n"


def save_pilot_parity_scorecard(
    scorecard: Mapping[str, Any],
    *,
    reports_root: Path,
    stem: str,
    timestamp: str | None = None,
) -> PilotScorecardArtifacts:
    """Persist a timestamped pilot scorecard under ``reports_root``."""
    resolved_timestamp = (
        timestamp
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    stamped_stem = f"{stem}_{resolved_timestamp}"
    json_path = reports_root / f"{stamped_stem}.json"
    text_path = reports_root / f"{stamped_stem}.md"
    payload = dict(scorecard)
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    text_path.write_text(render_pilot_parity_scorecard(payload))
    return PilotScorecardArtifacts(
        report=payload,
        json_path=json_path,
        text_path=text_path,
    )


__all__ = (
    "PILOT_SCORECARD_TASK_IDS",
    "PilotScorecardArtifacts",
    "build_pilot_parity_scorecard",
    "load_pilot_benchmark_records",
    "render_pilot_parity_scorecard",
    "save_pilot_parity_scorecard",
)
