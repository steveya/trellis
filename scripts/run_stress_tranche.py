"""Run the connector stress tranche as a standing regression gate.

Usage:
    python scripts/run_stress_tranche.py
    python scripts/run_stress_tranche.py --model claude-sonnet-4-6
    python scripts/run_stress_tranche.py --fresh
    python scripts/run_stress_tranche.py --task-id E21 --task-id E28
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

from trellis.agent.config import get_default_model, load_env
from trellis.agent.evals import (
    GradeResult,
    grade_stress_task_preflight,
    load_stress_task_manifest,
    render_stress_tranche_report,
    summarize_stress_preflight,
    summarize_stress_tranche,
    summarize_task_results,
)
from trellis.agent.task_runtime import build_market_state, load_tasks, run_task
from trellis.cli_paths import resolve_repo_path

load_env()

STRESS_TASK_IDS = ("E21", "E22", "E23", "E24", "E25", "E26", "E27", "E28")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--validation", default="standard")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Run a subset of the stress tranche. May be repeated.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run deterministic preflight only and skip live execution.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force rebuilds instead of the cheaper default reuse mode.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Explicitly request the cheaper reuse mode (default).",
    )
    parser.add_argument("--output")
    parser.add_argument("--report-json")
    parser.add_argument("--report-md")
    return parser.parse_args(argv)


def _serialize_grade_report(report: dict[str, GradeResult]) -> dict[str, dict[str, object]]:
    return {
        key: {"passed": value.passed, "details": list(value.details)}
        for key, value in report.items()
    }


def _load_stress_tasks(task_ids: list[str] | None = None) -> dict[str, dict[str, object]]:
    selected = tuple(task_ids or STRESS_TASK_IDS)
    tasks = {
        task["id"]: task
        for task in load_tasks(status=None)
        if task["id"] in selected
    }
    missing = sorted(set(selected) - set(tasks))
    if missing:
        raise ValueError(f"Missing stress tasks in pricing manifest: {missing}")
    return {task_id: tasks[task_id] for task_id in selected}


def run_stress_tranche(
    *,
    model: str | None,
    validation: str,
    task_ids: list[str] | None,
    force_rebuild: bool,
    preflight_only: bool,
    output_path: Path,
    report_json_path: Path,
    report_md_path: Path,
) -> int:
    manifest = load_stress_task_manifest()
    selected_task_ids = list(task_ids or STRESS_TASK_IDS)
    try:
        tasks = _load_stress_tasks(selected_task_ids)
    except ValueError as exc:
        print(str(exc))
        return 1

    preflight: dict[str, dict[str, dict[str, object]]] = {}
    preflight_failed = False
    for task_id in selected_task_ids:
        report = grade_stress_task_preflight(tasks[task_id], manifest[task_id])
        preflight[task_id] = _serialize_grade_report(report)
        if not all(item.passed for item in report.values()):
            preflight_failed = True

    preflight_summary = summarize_stress_preflight(tasks, manifest=manifest)
    effective_model = model or get_default_model()
    report: dict[str, object] = {
        "collection_name": "Connector stress tranche",
        "status": "preflight_failed" if preflight_failed else (
            "preflight_only" if preflight_only else "completed"
        ),
        "model": effective_model,
        "validation": validation,
        "fresh_build": force_rebuild,
        "task_ids": selected_task_ids,
        "preflight_summary": preflight_summary,
        "preflight": preflight,
        "raw_results_path": str(output_path),
        "report_json_path": str(report_json_path),
        "report_md_path": str(report_md_path),
    }

    if preflight_failed:
        report_json_path.write_text(json.dumps(report, indent=2, default=str))
        report_md_path.write_text(render_stress_tranche_report(report))
        print(f"Preflight failed. Report saved to: {report_json_path}")
        print(f"Preflight report saved to: {report_md_path}")
        return 1

    if preflight_only:
        report_json_path.write_text(json.dumps(report, indent=2, default=str))
        report_md_path.write_text(render_stress_tranche_report(report))
        print(f"Preflight passed. Report saved to: {report_json_path}")
        print(f"Preflight report saved to: {report_md_path}")
        return 0

    market_state = build_market_state()
    results = []

    for task_id in selected_task_ids:
        task = tasks[task_id]
        print(f"{task_id}: {task['title']}", flush=True)
        result = run_task(
            task,
            market_state,
            model=effective_model,
            force_rebuild=force_rebuild,
            fresh_build=force_rebuild,
            validation=validation,
        )
        results.append(result)
        output_path.write_text(json.dumps(results, indent=2, default=str))

    report["task_summary"] = summarize_task_results(results)
    report["stress_summary"] = summarize_stress_tranche(tasks, results, manifest=manifest)
    report_json_path.write_text(json.dumps(report, indent=2, default=str))
    report_md_path.write_text(render_stress_tranche_report(report))
    print(f"Results saved to: {output_path}")
    print(f"Report saved to: {report_json_path}")
    print(f"Report Markdown saved to: {report_md_path}")
    return 0


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = resolve_repo_path(
        args.output,
        ROOT / f"task_results_stress_connector_{timestamp}.json",
    )
    report_json_path = resolve_repo_path(
        args.report_json,
        ROOT / f"task_results_stress_connector_{timestamp}_report.json",
    )
    report_md_path = resolve_repo_path(
        args.report_md,
        ROOT / f"task_results_stress_connector_{timestamp}_report.md",
    )
    return run_stress_tranche(
        model=args.model,
        validation=args.validation,
        task_ids=list(args.task_ids or []),
        force_rebuild=bool(args.fresh and not args.reuse),
        preflight_only=args.preflight_only,
        output_path=output_path,
        report_json_path=report_json_path,
        report_md_path=report_md_path,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
