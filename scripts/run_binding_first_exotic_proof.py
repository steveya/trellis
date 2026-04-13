"""Run one binding-first exotic proof cohort.

Usage:
    /Users/steveyang/miniforge3/bin/python3 scripts/run_binding_first_exotic_proof.py
    /Users/steveyang/miniforge3/bin/python3 scripts/run_binding_first_exotic_proof.py --cohort event_control_schedule
    /Users/steveyang/miniforge3/bin/python3 scripts/run_binding_first_exotic_proof.py --task-id T17 --task-id E27
    /Users/steveyang/miniforge3/bin/python3 scripts/run_binding_first_exotic_proof.py --preflight-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

from trellis.agent.config import get_default_model, load_env
from trellis.agent.evals import (
    grade_binding_first_exotic_proof_preflight,
    load_binding_first_exotic_proof_manifest,
    render_binding_first_exotic_proof_report,
    select_binding_first_exotic_proof_tasks,
    summarize_binding_first_exotic_proof,
)
from trellis.agent.task_runtime import load_tasks, run_task
from trellis.cli_paths import resolve_repo_path

load_env()


@contextmanager
def _frozen_proof_learning_surface():
    """Freeze post-build learning while evaluating the proof cohort.

    Proof runs should measure the current binding/runtime surface, not mutate it
    mid-cohort through reflection or consolidation side effects.
    """
    keys = (
        "TRELLIS_SKIP_POST_BUILD_REFLECTION",
        "TRELLIS_SKIP_POST_BUILD_CONSOLIDATION",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["TRELLIS_SKIP_POST_BUILD_REFLECTION"] = "1"
        os.environ["TRELLIS_SKIP_POST_BUILD_CONSOLIDATION"] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="event_control_schedule")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Run a subset of the proof cohort. May be repeated.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--validation", default="standard")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run deterministic preflight only and skip live execution.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Use reuse mode instead of the default fresh-build proof run.",
    )
    parser.add_argument("--output")
    parser.add_argument("--report-json")
    parser.add_argument("--report-md")
    return parser.parse_args(argv)


def _load_selected_tasks(task_ids: list[str]) -> dict[str, dict[str, object]]:
    tasks = {
        task["id"]: task
        for task in load_tasks(status=None)
        if task["id"] in task_ids
    }
    missing = sorted(set(task_ids) - set(tasks))
    if missing:
        raise ValueError(f"Missing proof tasks in pricing manifest: {missing}")
    return {task_id: tasks[task_id] for task_id in task_ids}


def run_binding_first_exotic_proof(
    *,
    cohort: str,
    task_ids: list[str] | None,
    model: str | None,
    validation: str,
    fresh_build: bool,
    preflight_only: bool,
    output_path: Path,
    report_json_path: Path,
    report_md_path: Path,
) -> int:
    manifest = load_binding_first_exotic_proof_manifest()
    try:
        selected_manifest = select_binding_first_exotic_proof_tasks(
            manifest,
            cohort=cohort,
            task_ids=task_ids,
        )
    except ValueError as exc:
        print(str(exc))
        return 1
    if not selected_manifest:
        print(f"No proof tasks selected for cohort={cohort!r} task_ids={task_ids or []}")
        return 1

    ordered_task_ids = list(selected_manifest)
    try:
        tasks = _load_selected_tasks(ordered_task_ids)
    except ValueError as exc:
        print(str(exc))
        return 1

    preflight_by_task: dict[str, dict[str, dict[str, object]]] = {}
    preflight_failed = False
    for task_id in ordered_task_ids:
        report = grade_binding_first_exotic_proof_preflight(
            tasks[task_id],
            selected_manifest[task_id],
        )
        preflight_by_task[task_id] = {
            key: {"passed": value.passed, "details": list(value.details)}
            for key, value in report.items()
        }
        if not all(item.passed for item in report.values()):
            preflight_failed = True

    effective_model = model or get_default_model()
    report: dict[str, object] = {
        "collection_name": "Binding-first exotic proof cohort",
        "status": "preflight_failed" if preflight_failed else ("preflight_only" if preflight_only else "completed"),
        "cohort": cohort,
        "model": effective_model,
        "validation": validation,
        "fresh_build": fresh_build,
        "task_ids": ordered_task_ids,
        "preflight": preflight_by_task,
        "preflight_summary": {
            "totals": {
                "tasks": len(ordered_task_ids),
                "passed": sum(
                    1 for task_report in preflight_by_task.values()
                    if all(check["passed"] for check in task_report.values())
                ),
                "failed": sum(
                    1 for task_report in preflight_by_task.values()
                    if not all(check["passed"] for check in task_report.values())
                ),
            },
            "failed_tasks": [
                task_id
                for task_id, task_report in preflight_by_task.items()
                if not all(check["passed"] for check in task_report.values())
            ],
            "by_task": {
                task_id: {
                    "title": tasks[task_id].get("title"),
                    "blocked": not all(check["passed"] for check in task_report.values()),
                    "checks": task_report,
                }
                for task_id, task_report in preflight_by_task.items()
            },
        },
        "raw_results_path": str(output_path),
        "report_json_path": str(report_json_path),
        "report_md_path": str(report_md_path),
    }

    if preflight_failed or preflight_only:
        report_json_path.write_text(json.dumps(report, indent=2, default=str))
        report_md_path.write_text(render_binding_first_exotic_proof_report(report))
        return 1 if preflight_failed else 0

    results = []
    market_state = None
    with _frozen_proof_learning_surface():
        for task_id in ordered_task_ids:
            task = tasks[task_id]
            print(f"{task_id}: {task['title']}", flush=True)
            if market_state is None:
                from trellis.agent.task_runtime import build_market_state

                market_state = build_market_state()
            result = run_task(
                task,
                market_state,
                model=effective_model,
                force_rebuild=fresh_build,
                fresh_build=fresh_build,
                validation=validation,
            )
            results.append(result)
            output_path.write_text(json.dumps(results, indent=2, default=str))

    proof_summary = summarize_binding_first_exotic_proof(
        tasks,
        results,
        manifest=selected_manifest,
    )
    totals = dict(proof_summary.get("totals") or {})
    failed_gate = int(totals.get("failed_gate") or 0)
    report["status"] = "failed_gate" if failed_gate else "completed"
    report["proof_summary"] = proof_summary
    report["task_summary"] = proof_summary["task_summary"]
    report_json_path.write_text(json.dumps(report, indent=2, default=str))
    report_md_path.write_text(render_binding_first_exotic_proof_report(report))
    print(f"Results saved to: {output_path}")
    print(f"Report saved to: {report_json_path}")
    print(f"Report Markdown saved to: {report_md_path}")
    return 1 if failed_gate else 0


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = resolve_repo_path(
        args.output,
        ROOT / f"task_results_binding_first_proof_{args.cohort}_{timestamp}.json",
    )
    report_json_path = resolve_repo_path(
        args.report_json,
        ROOT / f"task_results_binding_first_proof_{args.cohort}_{timestamp}_report.json",
    )
    report_md_path = resolve_repo_path(
        args.report_md,
        ROOT / f"task_results_binding_first_proof_{args.cohort}_{timestamp}_report.md",
    )
    return run_binding_first_exotic_proof(
        cohort=args.cohort,
        task_ids=list(args.task_ids or []),
        model=args.model,
        validation=args.validation,
        fresh_build=not args.reuse,
        preflight_only=args.preflight_only,
        output_path=output_path,
        report_json_path=report_json_path,
        report_md_path=report_md_path,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
