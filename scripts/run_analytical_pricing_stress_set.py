"""Run the analytical pricing stress set with trace-rich reporting.

This is the explicit task collection that was reviewed with the user before
execution. It runs CLI-only through the pricing task runner, keeps detailed
task-run traces on failure, and emits a summary report that points at the
persisted trace artifacts for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

from trellis.agent.config import load_env
from trellis.agent.evals import classify_task_result, summarize_task_results
from trellis.agent.task_run_store import load_task_run_record
from trellis.agent.task_runtime import build_market_state, load_tasks, run_task
from trellis.cli_paths import resolve_repo_path

load_env()


# Core analytical pricing tasks reviewed with the user.
CORE_ANALYTICAL_PRICING_TASK_IDS = (
    "T13",
    "T14",
    "T16",
    "T19",
    "T21",
    "T23",
    "T24",
    "T25",
    "T26",
    "T27",
    "T28",
    "T31",
    "T32",
    "T39",
    "T41",
    "T42",
    "T45",
    "T46",
    "T48",
    "T56",
    "T57",
    "T58",
    "T61",
    "T73",
    "T74",
    "T75",
    "T76",
    "T77",
    "T78",
    "T94",
    "T96",
    "T97",
    "T98",
    "T99",
    "T100",
    "T101",
    "T102",
    "T103",
    "T104",
    "T105",
    "T106",
    "T108",
    "T109",
    "T110",
    "T111",
    "T112",
    "T115",
    "T116",
    "T118",
    "T119",
    "T120",
    "T122",
    "T123",
    "T124",
    "T126",
    "E21",
    "E22",
    "E25",
    "E28",
)


# Adjacent analytics tasks that still exercise the analytical stack.
ADJACENT_ANALYTICS_TASK_IDS = (
    "T34",
    "T51",
    "T52",
    "T71",
    "T81",
    "T83",
    "T84",
    "T86",
    "T87",
    "T88",
    "T90",
    "T95",
)


ANALYTICAL_PRICING_STRESS_TASK_IDS = (
    *CORE_ANALYTICAL_PRICING_TASK_IDS,
    *ADJACENT_ANALYTICS_TASK_IDS,
)

TASK_SET_NAME = "Analytical Pricing Stress Set"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model to use for all task-run stages.",
    )
    parser.add_argument(
        "--validation",
        default="standard",
        help="Validation profile passed through to the pricing task runner.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing generated modules when present instead of forcing rebuilds.",
    )
    parser.add_argument(
        "--output",
        help="Path for the raw task results JSON output.",
    )
    parser.add_argument(
        "--report-json",
        help="Path for the machine-readable summary report JSON.",
    )
    parser.add_argument(
        "--report-md",
        help="Path for the human-readable summary report Markdown.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help=(
            "Override the default task collection with one or more explicit task ids. "
            "May be repeated."
        ),
    )
    return parser.parse_args(argv)


def _load_tasks_by_ids(task_ids: list[str]) -> list[dict[str, Any]]:
    requested = [task_id.strip() for task_id in task_ids if task_id and task_id.strip()]
    if not requested:
        return []
    tasks_by_id = {task["id"]: task for task in load_tasks(status=None)}
    missing = [task_id for task_id in requested if task_id not in tasks_by_id]
    if missing:
        raise ValueError(f"Unknown task ids: {', '.join(sorted(missing))}")
    return [tasks_by_id[task_id] for task_id in requested]


def _task_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """Return the trace-rich diagnostics payload for one task result."""
    history_path = result.get("task_run_history_path")
    try:
        record = load_task_run_record(history_path) if history_path else {}
    except Exception:
        record = {}
    trace_summaries = list(record.get("trace_summaries") or [])
    return {
        "task_id": result.get("task_id"),
        "title": result.get("title"),
        "success": bool(result.get("success")),
        "status_bucket": classify_task_result(result),
        "attempts": result.get("attempts"),
        "gap_confidence": result.get("gap_confidence"),
        "error": result.get("error"),
        "failures": list(result.get("failures") or []),
        "knowledge_gaps": list(result.get("knowledge_gaps") or []),
        "task_run_history_path": result.get("task_run_history_path"),
        "task_run_latest_path": result.get("task_run_latest_path"),
        "task_run_latest_index_path": result.get("task_run_latest_index_path"),
        "platform_trace_path": result.get("platform_trace_path"),
        "analytical_trace_path": result.get("analytical_trace_path"),
        "trace_summaries": trace_summaries,
        "workflow": dict(record.get("workflow") or {}),
        "summary": dict(record.get("summary") or {}),
    }


def _render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.extend([
        f"# {report['collection_name']}",
        "",
        f"- Model: `{report['model']}`",
        f"- Validation: `{report['validation']}`",
        f"- Fresh build: `{report['fresh_build']}`",
        f"- CLI-only runner: `trellis-ui` not invoked",
        f"- Tasks: `{report['summary']['totals']['tasks']}`",
        f"- Successes: `{report['summary']['totals']['successes']}`",
        f"- Failures: `{report['summary']['totals']['failures']}`",
        "",
        "## Failure Buckets",
    ])
    failure_buckets = report["summary"]["failure_buckets"]
    if failure_buckets:
        for bucket, count in sorted(failure_buckets.items()):
            lines.append(f"- `{bucket}`: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Failed Tasks"])
    failed_tasks = report["failed_tasks"]
    if not failed_tasks:
        lines.append("- None")
    else:
        for item in failed_tasks:
            lines.extend([
                f"### {item['task_id']} - {item['title']}",
                f"- Bucket: `{item['status_bucket']}`",
                f"- Attempts: `{item['attempts']}`",
                f"- Error: `{item.get('error')}`",
            ])
            failures = item.get("failures") or []
            if failures:
                lines.append("- Failures:")
                for failure in failures:
                    lines.append(f"  - {failure}")
            knowledge_gaps = item.get("knowledge_gaps") or []
            if knowledge_gaps:
                lines.append("- Knowledge gaps:")
                for gap in knowledge_gaps:
                    lines.append(f"  - {gap}")
            lines.append("- Trace summaries:")
            trace_summaries = item.get("trace_summaries") or []
            if trace_summaries:
                for trace in trace_summaries:
                    trace_path = trace.get("path")
                    trace_kind = trace.get("trace_kind")
                    trace_status = trace.get("status")
                    latest_event = trace.get("latest_event")
                    lines.append(
                        f"  - `{trace_path}` ({trace_kind}, {trace_status}, latest={latest_event})"
                    )
            else:
                lines.append("  - None")
            lines.append("")

    lines.extend([
        "## Successful Tasks",
    ])
    successful_tasks = report["successful_tasks"]
    if not successful_tasks:
        lines.append("- None")
    else:
        for item in successful_tasks:
            lines.append(
                f"- `{item['task_id']}` {item['title']} "
                f"(attempts={item['attempts']}, trace={item.get('task_run_history_path')})"
            )

    return "\n".join(lines).rstrip() + "\n"


def run_analytical_pricing_stress_set(
    *,
    model: str,
    validation: str,
    force_rebuild: bool,
    task_ids: list[str],
    output_file: Path,
    report_json_file: Path,
    report_md_file: Path,
) -> dict[str, Any]:
    tasks = _load_tasks_by_ids(task_ids)
    task_map = {task["id"]: task for task in tasks}

    market_state = build_market_state()
    results: list[dict[str, Any]] = []

    print(f"\n{'#' * 72}")
    print(f"# Running {TASK_SET_NAME} → {output_file}")
    print(f"# Model: {model}")
    print(f"# Validation: {validation}")
    print(f"# Force rebuild: {force_rebuild}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 72}")

    for index, task_id in enumerate(task_ids, start=1):
        task = task_map[task_id]
        print(f"\n[{index}/{len(task_ids)}] {task['id']}: {task['title']}")
        result = run_task(
            task,
            market_state,
            model=model,
            force_rebuild=force_rebuild,
            fresh_build=force_rebuild,
            validation=validation,
        )
        results.append(result)
        output_file.write_text(json.dumps(results, indent=2, default=str))

    summary = summarize_task_results(results)
    failed_tasks = []
    successful_tasks = []
    for result in results:
        diagnostics = _task_diagnostics(result)
        if result.get("success"):
            successful_tasks.append(diagnostics)
        else:
            failed_tasks.append(diagnostics)

    report = {
        "collection_name": TASK_SET_NAME,
        "model": model,
        "validation": validation,
        "fresh_build": force_rebuild,
        "task_ids": list(task_ids),
        "summary": summary,
        "failed_tasks": failed_tasks,
        "successful_tasks": successful_tasks,
        "raw_results_path": str(output_file),
        "report_json_path": str(report_json_file),
        "report_md_path": str(report_md_file),
        "ui_mode": "off",
    }

    report_json_file.write_text(json.dumps(report, indent=2, default=str))
    report_md_file.write_text(_render_markdown_report(report))

    print(f"\n{'=' * 72}")
    print(
        f"SUMMARY: {summary['totals']['successes']}/{summary['totals']['tasks']} "
        f"succeeded"
    )
    print(f"  Failure buckets: {summary['failure_buckets']}")
    print(f"  Retry recovery: {summary['retry_recovery']}")
    print(f"  Reviewer signals: {summary['reviewer_signals']}")
    print(f"  Shared knowledge: {summary['shared_knowledge']}")
    print(f"  Promotion discipline: {summary['promotion_discipline']}")
    print(f"  Results saved to: {output_file}")
    print(f"  Report JSON saved to: {report_json_file}")
    print(f"  Report Markdown saved to: {report_md_file}")
    print(f"{'=' * 72}")

    return report


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    task_ids = list(args.task_ids or ANALYTICAL_PRICING_STRESS_TASK_IDS)

    output_file = resolve_repo_path(
        args.output,
        ROOT / "task_results_analytical_pricing_stress_set.json",
    )
    report_json_file = resolve_repo_path(
        args.report_json,
        ROOT / "task_results_analytical_pricing_stress_set_report.json",
    )
    report_md_file = resolve_repo_path(
        args.report_md,
        ROOT / "task_results_analytical_pricing_stress_set_report.md",
    )

    run_analytical_pricing_stress_set(
        model=args.model,
        validation=args.validation,
        force_rebuild=not args.reuse,
        task_ids=task_ids,
        output_file=output_file,
        report_json_file=report_json_file,
        report_md_file=report_md_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
