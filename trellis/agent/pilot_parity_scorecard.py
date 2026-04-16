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
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Floor below which we suppress outlier flags so a near-zero median (numerical
# noise) cannot turn every other passing task into an outlier.  The 5% number
# I floated when the ticket was written assumed real pilot deviations were on
# the order of percent; in practice the FinancePy pilot data sat at <1% with
# F007 at 0.93%, so the floor has to be much lower or the F007-shape signal
# is suppressed entirely.  0.05% catches F007 against a near-zero median while
# still suppressing noise around legitimately tiny deviations.
_DEVIATION_OUTLIER_FLOOR_PCT = 0.05
_DEVIATION_OUTLIER_RATIO = 10.0


def _max_output_deviation_pct(record: Mapping[str, Any]) -> float | None:
    """Return the largest output_deviation_pct in a benchmark record, or None."""
    summary = record.get("comparison_summary") or {}
    if not isinstance(summary, Mapping):
        return None
    deviations = summary.get("output_deviation_pct") or {}
    if not isinstance(deviations, Mapping):
        return None
    values: list[float] = []
    for raw in deviations.values():
        try:
            values.append(abs(float(raw)))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return max(values)


def _classify_deviation_outliers(
    *,
    task_summaries: list[dict[str, Any]],
    latest_records_by_task: Mapping[str, Mapping[str, Any]],
    floor_pct: float = _DEVIATION_OUTLIER_FLOOR_PCT,
    ratio: float = _DEVIATION_OUTLIER_RATIO,
) -> tuple[float, int]:
    """Annotate task summaries with deviation outlier flags and return (median, count)."""
    deviations: dict[str, float] = {}
    for summary in task_summaries:
        task_id = summary.get("task_id")
        record = latest_records_by_task.get(task_id) if task_id else None
        if record is None:
            continue
        max_dev = _max_output_deviation_pct(record)
        if max_dev is None:
            continue
        deviations[task_id] = max_dev

    median = statistics.median(deviations.values()) if deviations else 0.0
    threshold = max(floor_pct, ratio * median)
    outlier_count = 0
    for summary in task_summaries:
        task_id = summary.get("task_id")
        max_dev = deviations.get(task_id) if task_id else None
        summary["max_output_deviation_pct"] = (
            round(float(max_dev), 6) if max_dev is not None else None
        )
        if max_dev is None or median <= 0.0:
            ratio_to_median = None
        else:
            ratio_to_median = round(max_dev / median, 6)
        summary["deviation_vs_pilot_median"] = ratio_to_median
        flagged = max_dev is not None and max_dev > threshold
        summary["outlier_flag"] = bool(flagged)
        if flagged:
            outlier_count += 1
    return round(median, 6), outlier_count

from trellis.agent.benchmark_history import (
    build_task_history_summary,
    history_sort_key,
)
from trellis.agent.benchmark_pilots import get_pilot_task_ids
from trellis.agent.financepy_benchmark import DEFAULT_FINANCEPY_BENCHMARK_ROOT
from trellis.agent.financepy_output_comparison import is_greek_output


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
    return sorted(records, key=history_sort_key)


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
    latest_records_by_task: dict[str, Mapping[str, Any]] = {}
    enforced = 0
    violations = 0
    missing = 0
    latest_pass = 0

    for task_id in expected:
        runs = sorted(grouped.get(task_id) or (), key=history_sort_key)
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
        summary = build_task_history_summary(
            task_id, runs, benchmark_kind="financepy"
        )
        latest_record = dict(runs[-1])
        latest_records_by_task[task_id] = latest_record
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
        # Surface Greek coverage honestly -- the comparison summary carries
        # `missing_trellis_outputs`, `greek_parity`, and `greek_coverage` under
        # QUA-861, but the pilot scorecard used to hide all of it behind the
        # aggregate price-parity status.
        comparison_summary = latest_record.get("comparison_summary") or {}
        if not isinstance(comparison_summary, Mapping):
            comparison_summary = {}
        summary["missing_trellis_outputs"] = list(
            comparison_summary.get("missing_trellis_outputs") or ()
        )
        summary["missing_financepy_outputs"] = list(
            comparison_summary.get("missing_financepy_outputs") or ()
        )
        greek_coverage = comparison_summary.get("greek_coverage") or {}
        if not isinstance(greek_coverage, Mapping):
            greek_coverage = {}
        summary["greek_coverage"] = dict(greek_coverage)
        summary["greek_parity"] = str(
            comparison_summary.get("greek_parity") or "not_applicable"
        )
        summary["greek_failures"] = list(comparison_summary.get("greek_failures") or ())
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

    median_deviation_pct, outlier_count = _classify_deviation_outliers(
        task_summaries=task_summaries,
        latest_records_by_task=latest_records_by_task,
    )
    already_missed = {miss["task_id"] for miss in residual_misses}
    for summary in task_summaries:
        if not summary.get("outlier_flag"):
            continue
        task_id = summary.get("task_id") or ""
        if task_id in already_missed:
            # Already escalated for a more serious reason (parity failure,
            # boundary violation, missing run); do not double-list.
            continue
        latest_snapshot = summary.get("latest") or {}
        residual_misses.append(
            {
                "task_id": task_id,
                "category": "deviation_outlier",
                "reason": (
                    f"max output deviation {summary.get('max_output_deviation_pct')}% "
                    f"exceeds {_DEVIATION_OUTLIER_RATIO}x pilot median "
                    f"{median_deviation_pct}% "
                    f"(floor {_DEVIATION_OUTLIER_FLOOR_PCT}%)"
                ),
                "run_id": str(latest_snapshot.get("run_id") or ""),
                "run_started_at": str(latest_snapshot.get("run_started_at") or ""),
                "git_sha": str(latest_snapshot.get("git_sha") or ""),
                "knowledge_revision": str(
                    latest_snapshot.get("knowledge_revision") or ""
                ),
            }
        )

    # Aggregate Greek coverage across the pilot.  A task is counted as
    # having Trellis Greek coverage when its latest comparison reports
    # `trellis_greek_count > 0` (at least one Trellis-emitted Greek),
    # regardless of whether the other side compared it.
    # `tasks_with_greek_overlap` is the stricter bar: both sides emitted
    # the same Greek.  Tasks whose binding declared Greek overlap AND
    # whose FinancePy side emitted those Greeks AND whose Trellis side
    # emitted none are flagged `missing_greek_coverage` -- price parity
    # can pass while the Greek contract is silently uncovered.
    # (QUA-861 items #4/#5; Copilot review on PR #593 round 1.)
    tasks_with_trellis_greeks = 0
    tasks_with_greek_overlap = 0
    tasks_with_greek_parity_passed = 0
    tasks_with_greek_parity_failed = 0
    already_missed_ids = {miss["task_id"] for miss in residual_misses}
    for summary in task_summaries:
        coverage = summary.get("greek_coverage") or {}
        trellis_greek_count = int(coverage.get("trellis_greek_count") or 0)
        financepy_greek_count = int(coverage.get("financepy_greek_count") or 0)
        compared_greek_count = int(coverage.get("compared_greek_count") or 0)
        if trellis_greek_count > 0:
            tasks_with_trellis_greeks += 1
        if compared_greek_count > 0:
            tasks_with_greek_overlap += 1
        greek_parity = str(summary.get("greek_parity") or "not_applicable")
        if greek_parity == "passed":
            tasks_with_greek_parity_passed += 1
        elif greek_parity == "failed":
            tasks_with_greek_parity_failed += 1
        # Missing-Greek residual miss: derive the expected-but-missing set
        # from the intersection of "declared Greek overlap" (canonical Greek
        # names in `missing_trellis_outputs`) and "FinancePy actually emitted
        # Greeks".  Without the financepy-count guard we'd mis-attribute
        # coverage gaps to Trellis when neither side emitted anything --
        # that's a binding-reference gap, not a Trellis-coverage gap.
        # (PR #593 round 1 Copilot review.)
        task_id = summary.get("task_id") or ""
        expected_greeks = [
            name
            for name in summary.get("missing_trellis_outputs") or ()
            if is_greek_output(name)
        ]
        if (
            task_id
            and expected_greeks
            and financepy_greek_count > 0
            and trellis_greek_count == 0
            and task_id not in already_missed_ids
        ):
            latest_snapshot = summary.get("latest") or {}
            residual_misses.append(
                {
                    "task_id": task_id,
                    "category": "missing_greek_coverage",
                    "reason": (
                        f"Trellis emits no Greek outputs; binding expected "
                        f"{expected_greeks}"
                    ),
                    "run_id": str(latest_snapshot.get("run_id") or ""),
                    "run_started_at": str(latest_snapshot.get("run_started_at") or ""),
                    "git_sha": str(latest_snapshot.get("git_sha") or ""),
                    "knowledge_revision": str(
                        latest_snapshot.get("knowledge_revision") or ""
                    ),
                }
            )
            already_missed_ids.add(task_id)

    pilot_summary = {
        "fresh_generated_enforced_count": enforced,
        "boundary_violation_count": violations,
        "missing_run_count": missing,
        "latest_pass_count": latest_pass,
        "deviation_median_pct": median_deviation_pct,
        "deviation_outlier_count": outlier_count,
        "tasks_with_trellis_greek_coverage": tasks_with_trellis_greeks,
        "tasks_with_greek_overlap": tasks_with_greek_overlap,
        "greek_parity_passed_count": tasks_with_greek_parity_passed,
        "greek_parity_failed_count": tasks_with_greek_parity_failed,
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
        f"- Deviation median: `{summary.get('deviation_median_pct', 0.0)}%`",
        f"- Deviation outliers: `{summary.get('deviation_outlier_count', 0)}`",
        f"- Tasks with Trellis Greek coverage: `{summary.get('tasks_with_trellis_greek_coverage', 0)}`",
        f"- Tasks with Greek overlap (both sides): `{summary.get('tasks_with_greek_overlap', 0)}`",
        f"- Greek parity passed: `{summary.get('greek_parity_passed_count', 0)}`",
        f"- Greek parity failed: `{summary.get('greek_parity_failed_count', 0)}`",
    ]
    if scorecard.get("notes"):
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in scorecard["notes"])

    outlier_tasks = [
        task
        for task in (scorecard.get("tasks") or ())
        if task.get("outlier_flag")
    ]
    if outlier_tasks:
        lines.extend(["", "## Deviation Outliers"])
        for task in outlier_tasks:
            lines.append(
                f"- `{task.get('task_id', '')}` "
                f"max_deviation=`{task.get('max_output_deviation_pct')}%` "
                f"vs_pilot_median=`{task.get('deviation_vs_pilot_median')}x`"
            )

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
                f"- Max output deviation: `{task.get('max_output_deviation_pct')}%`",
                f"- Vs pilot median: `{task.get('deviation_vs_pilot_median')}x`",
                f"- Outlier: `{task.get('outlier_flag', False)}`",
                f"- Greek parity: `{task.get('greek_parity', 'not_applicable')}`",
            ]
        )
        coverage = task.get("greek_coverage") or {}
        if coverage:
            lines.append(
                f"- Greek coverage: trellis=`{coverage.get('trellis_greek_count', 0)}` "
                f"financepy=`{coverage.get('financepy_greek_count', 0)}` "
                f"compared=`{coverage.get('compared_greek_count', 0)}`"
            )
        missing_trellis = task.get("missing_trellis_outputs") or ()
        if missing_trellis:
            lines.append(f"- Missing Trellis outputs: `{', '.join(missing_trellis)}`")
        greek_failures = task.get("greek_failures") or ()
        if greek_failures:
            lines.append(f"- Greek failures: `{', '.join(greek_failures)}`")
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
