"""Run ``FRAMEWORK_TASKS.yaml`` tasks through the dedicated framework runner."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trellis.agent.framework_runtime import run_framework_task
from trellis.agent.task_runtime import load_framework_tasks


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", help="'all' or a start/end task id pair")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def run_block(tasks: list[dict], output_file: str) -> None:
    results: list[dict] = []

    print(f"\n{'#' * 60}")
    print(f"# Running {len(tasks)} framework/meta tasks → {output_file}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    for index, task in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}] {task['id']}: {task['title']}")
        result = run_framework_task(task)
        results.append(result)
        with open(output_file, "w") as handle:
            json.dump(results, handle, indent=2, default=str)

    outcomes = Counter(
        str((result.get("framework_result") or {}).get("outcome_type") or "unknown")
        for result in results
    )
    successes = sum(1 for result in results if result.get("success"))

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {successes}/{len(results)} produced actionable framework outputs")
    print(f"  Outcome types: {dict(outcomes)}")
    print(f"  Results saved to: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    if not args.ids or args.ids[0] == "all":
        tasks = load_framework_tasks(status=None)
        output_file = args.output or str(ROOT / "framework_task_results_all.json")
    elif len(args.ids) == 2:
        tasks = load_framework_tasks(args.ids[0], args.ids[1], status=None)
        safe_name = f"{args.ids[0]}_{args.ids[1]}".lower()
        output_file = args.output or str(ROOT / f"framework_task_results_{safe_name}.json")
    else:
        print(f"Usage: {sys.argv[0]} [start_id end_id | all]")
        raise SystemExit(1)

    run_block(tasks, output_file)
