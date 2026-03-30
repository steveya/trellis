"""Benchmark the analytical support subtree in reuse and fresh-build modes."""

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

from trellis.agent.config import load_env
from trellis.agent.reliability_benchmark import (
    build_reliability_benchmark_report,
    save_reliability_benchmark_report,
)
from trellis.agent.task_runtime import build_market_state, load_tasks, run_task
from trellis.cli_paths import resolve_repo_path

load_env()


DEFAULT_TASK_IDS = ("T97", "E22", "E25")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task_ids",
        nargs="*",
        help="Representative task ids to benchmark. Defaults to the analytical support tranche.",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--output-root", default=str(ROOT / "docs" / "benchmarks"))
    parser.add_argument("--report-name", default="analytical_support_reliability")
    parser.add_argument(
        "--candidate-mode",
        default="fresh-build",
        choices=("fresh-build",),
        help="Mode used for the candidate tranche.",
    )
    parser.add_argument(
        "--baseline-mode",
        default="reuse",
        choices=("reuse",),
        help="Mode used for the baseline tranche.",
    )
    return parser.parse_args(argv)


def _load_tasks_by_ids(task_ids: list[str]) -> list[dict]:
    requested = {task_id.strip() for task_id in task_ids if task_id.strip()}
    tasks = [task for task in load_tasks(status=None) if task["id"] in requested]
    missing = sorted(requested - {task["id"] for task in tasks})
    if missing:
        raise ValueError(f"Unknown task ids: {', '.join(missing)}")
    return tasks


def _run_tasks(tasks: list[dict], *, mode: str, model: str) -> list[dict]:
    market_state = build_market_state()
    results: list[dict] = []
    print(f"\n#{'=' * 72}")
    print(f"# {mode.upper()} benchmark run")
    print(f"# Model: {model}")
    print(f"# Tasks: {', '.join(task['id'] for task in tasks)}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 74}")
    for index, task in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}] {task['id']}: {task['title']}")
        fresh_build = mode == "fresh-build"
        result = run_task(
            task,
            market_state,
            model=model,
            force_rebuild=fresh_build,
            fresh_build=fresh_build,
        )
        payload = dict(result)
        payload["benchmark_mode"] = mode
        results.append(payload)
        status = "OK" if payload.get("success") else "FAIL"
        print(
            f"  [{status}] attempts={payload.get('attempts', 0)} "
            f"gap_confidence={(payload.get('gap_confidence') or 0):.0%}"
        )
    return results


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    task_ids = args.task_ids or list(DEFAULT_TASK_IDS)
    tasks = _load_tasks_by_ids(task_ids)

    baseline_results = _run_tasks(tasks, mode=args.baseline_mode, model=args.model)
    candidate_results = _run_tasks(tasks, mode=args.candidate_mode, model=args.model)

    report = build_reliability_benchmark_report(
        benchmark_name="analytical_support_reliability",
        tasks=tasks,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        baseline_mode=args.baseline_mode,
        candidate_mode=args.candidate_mode,
        notes=[
            "Fresh-build benchmarking uses the same representative tranche as the reuse baseline.",
            "The report records lesson capture, cookbook enrichment, promotion candidates, and failure buckets.",
        ],
    )

    output_root = resolve_repo_path(args.output_root, ROOT / "docs" / "benchmarks")
    artifacts = save_reliability_benchmark_report(
        report,
        root=output_root,
        stem=args.report_name,
    )
    summary = report["comparison"]

    print(f"\nSaved benchmark report to {artifacts.json_path}")
    print(f"Saved markdown report to {artifacts.text_path}")
    print(
        f"Outcome delta: improved={summary['task_transitions']['improved']} "
        f"regressed={summary['task_transitions']['regressed']} "
        f"unchanged={summary['task_transitions']['unchanged']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
