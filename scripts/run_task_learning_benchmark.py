"""Run the repeated non-canary task-learning benchmark at a fixed revision.

Usage:
    /Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py
    /Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py --list-tasks
    /Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py T13 T14 --passes 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

from trellis.agent.config import load_env
from trellis.agent.evals import summarize_task_results
from trellis.agent.task_learning_benchmark import (
    build_task_learning_benchmark_report,
    load_canary_task_ids,
    save_task_learning_benchmark_report,
    select_task_learning_cohort,
)
from trellis.agent.task_runtime import build_market_state, load_tasks, run_task
from trellis.cli_paths import resolve_repo_path

load_env()


DEFAULT_OUTPUT_ROOT = ROOT / "task_runs" / "learning_benchmarks"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task_ids",
        nargs="*",
        help="Optional explicit task ids. Defaults to the pending non-canary cohort.",
    )
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--validation", default="standard")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Include tasks already marked done in TASKS.yaml.",
    )
    parser.add_argument(
        "--knowledge-light",
        action="store_true",
        help="Use the compiler-first minimal knowledge profile.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Allow adapter reuse. By default the benchmark uses fresh builds to isolate knowledge carry-forward.",
    )
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--report-name", default="non_canary_task_learning")
    return parser.parse_args(argv)


def build_learning_tasks(
    *,
    requested_ids: list[str] | None = None,
    include_done: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return the selected non-canary task-learning cohort."""
    all_tasks = load_tasks(status=None)
    allowed_statuses = ("pending", "done") if include_done else ("pending",)
    selected = select_task_learning_cohort(
        all_tasks,
        canary_task_ids=load_canary_task_ids(),
        allowed_statuses=allowed_statuses,
        requested_ids=requested_ids,
        limit=limit,
    )
    if requested_ids:
        requested = {
            str(task_id).strip()
            for task_id in requested_ids
            if str(task_id).strip()
        }
        selected_ids = {str(task.get("id") or "").strip() for task in selected}
        missing = sorted(requested - selected_ids)
        if missing:
            raise ValueError(
                "Requested task ids are not in the selected non-canary cohort: "
                + ", ".join(missing)
            )
    return selected


def run_learning_benchmark(
    tasks: list[dict[str, Any]],
    *,
    benchmark_name: str,
    cohort_name: str = "non_canary_pending",
    output_root: Path,
    passes: int,
    model: str,
    validation: str,
    fresh_build: bool,
    knowledge_light: bool,
) -> dict[str, Any]:
    """Run repeated passes for one non-canary task cohort and save the report."""
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root = output_root / "raw"
    report_root = output_root / "reports"
    raw_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    git_revision = _git_revision()
    pass_runs: list[dict[str, Any]] = []

    print(f"\n{'#' * 72}")
    print(f"# Task learning benchmark: {benchmark_name}")
    print(f"# Git revision: {git_revision}")
    print(f"# Tasks: {', '.join(task['id'] for task in tasks)}")
    print(f"# Passes: {passes}")
    print(f"# Fresh build: {fresh_build}")
    print(f"# Knowledge light: {knowledge_light}")
    print(f"# Validation: {validation}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 72}")

    for pass_number in range(1, passes + 1):
        label = f"pass_{pass_number}"
        output_path = raw_root / f"{benchmark_name}_{label}.json"
        summary_path = raw_root / f"{benchmark_name}_{label}_summary.json"
        print(f"\nPass {pass_number}/{passes}: {label}", flush=True)
        results = _run_learning_pass(
            tasks,
            model=model,
            validation=validation,
            benchmark_name=benchmark_name,
            git_revision=git_revision,
            pass_number=pass_number,
            fresh_build=fresh_build,
            knowledge_light=knowledge_light,
        )
        output_path.write_text(json.dumps(results, indent=2, default=str))
        summary = summarize_task_results(results)
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        pass_runs.append(
            {
                "pass_number": pass_number,
                "label": label,
                "fresh_build": fresh_build,
                "knowledge_profile": "knowledge_light" if knowledge_light else "default",
                "model": model,
                "validation": validation,
                "results": results,
                "results_path": str(output_path),
                "summary_path": str(summary_path),
            }
        )
        print(
            "  Summary: "
            f"{summary['totals']['successes']}/{summary['totals']['tasks']} success, "
            f"first-pass={summary['first_pass']['rate']:.0%}, "
            f"avg-attempts-to-success={summary['attempts_to_success']['average']}, "
            f"tokens={summary['token_usage']['total_tokens']}"
        )

    report = build_task_learning_benchmark_report(
        benchmark_name=benchmark_name,
        cohort_name=cohort_name,
        git_revision=git_revision,
        tasks=tasks,
        pass_runs=pass_runs,
        notes=[
            "Fresh-build passes isolate knowledge carry-forward from adapter reuse."
            if fresh_build
            else "Reuse mode was enabled, so adapter reuse may contribute to improvements.",
            "This benchmark measures short-term learning evidence, not autonomous code development.",
        ],
    )
    artifacts = save_task_learning_benchmark_report(
        report,
        root=report_root,
        stem=benchmark_name,
    )
    print(f"\nSaved learning benchmark report to {artifacts.json_path}")
    print(f"Saved learning benchmark markdown to {artifacts.text_path}")
    return {
        "report": report,
        "report_json_path": artifacts.json_path,
        "report_md_path": artifacts.text_path,
        "pass_runs": pass_runs,
    }


def _run_learning_pass(
    tasks: list[dict[str, Any]],
    *,
    model: str,
    validation: str,
    benchmark_name: str,
    git_revision: str,
    pass_number: int,
    fresh_build: bool,
    knowledge_light: bool,
) -> list[dict[str, Any]]:
    market_state = build_market_state()
    results: list[dict[str, Any]] = []

    for index, task in enumerate(tasks, start=1):
        print(f"  [{index}/{len(tasks)}] {task['id']}: {task['title']}", flush=True)
        result = run_task(
            task,
            market_state,
            model=model,
            force_rebuild=fresh_build,
            fresh_build=fresh_build,
            knowledge_profile="knowledge_light" if knowledge_light else None,
            validation=validation,
        )
        payload = dict(result)
        payload["learning_benchmark_name"] = benchmark_name
        payload["learning_benchmark_pass"] = pass_number
        payload["learning_benchmark_git_revision"] = git_revision
        payload["learning_benchmark_fresh_build"] = fresh_build
        payload["learning_benchmark_knowledge_profile"] = (
            "knowledge_light" if knowledge_light else "default"
        )
        results.append(payload)
        status = "OK" if payload.get("success") else "FAIL"
        print(
            f"    [{status}] attempts={payload.get('attempts', 0)} "
            f"elapsed={(payload.get('elapsed_seconds') or 0):.1f}s"
        )

    return results


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = completed.stdout.strip()
    return revision or "unknown"


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        tasks = build_learning_tasks(
            requested_ids=args.task_ids or None,
            include_done=args.include_done,
            limit=args.limit,
        )
    except ValueError as exc:
        print(str(exc))
        return 2

    if args.list_tasks:
        for task in tasks:
            print(f"{task['id']}: {task['title']}")
        return 0
    if not tasks:
        print("No tasks selected for the non-canary learning benchmark.")
        return 1
    if args.dry_run:
        print(json.dumps(tasks, indent=2))
        return 0
    if args.passes < 2:
        print("Pass count must be at least 2 for a learning benchmark.")
        return 2

    cohort_name = (
        "explicit_non_canary_selection"
        if args.task_ids
        else ("non_canary_pending_plus_done" if args.include_done else "non_canary_pending")
    )

    output_root = Path(
        resolve_repo_path(args.output_root, DEFAULT_OUTPUT_ROOT)
    )
    run_learning_benchmark(
        tasks,
        benchmark_name=args.report_name,
        cohort_name=cohort_name,
        output_root=output_root,
        passes=args.passes,
        model=args.model,
        validation=args.validation,
        fresh_build=not args.reuse,
        knowledge_light=args.knowledge_light,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
