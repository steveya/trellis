"""Benchmark existing generated task payoffs without rebuilding them."""

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
from trellis.agent.task_runtime import benchmark_existing_task, build_market_state, load_tasks
from trellis.cli_paths import resolve_repo_path

load_env()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="+", help="Specific task ids to benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    requested = set(args.task_ids)
    tasks = [task for task in load_tasks(status=None) if task["id"] in requested]
    if not tasks:
        print("No matching tasks found.")
        return 1

    market_state = build_market_state()
    output_file = resolve_repo_path(
        args.output,
        ROOT / f"task_benchmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    results = []
    print(f"\n{'#' * 60}")
    print(f"# Benchmarking {len(tasks)} cached task payoffs → {output_file}")
    print(f"# Repeats: {args.repeats}")
    print(f"# Warmups: {args.warmups}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    for index, task in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}] {task['id']}: {task['title']}")
        try:
            result = benchmark_existing_task(
                task,
                market_state=market_state,
                repeats=args.repeats,
                warmups=args.warmups,
                model=args.model,
            )
            result["success"] = True
            print(
                f"  [OK] mean={result['mean_seconds']:.4f}s "
                f"min={result['min_seconds']:.4f}s max={result['max_seconds']:.4f}s "
                f"instantiate={result['instantiate_seconds']:.4f}s"
            )
        except Exception as exc:
            result = {
                "task_id": task["id"],
                "title": task["title"],
                "success": False,
                "error": str(exc),
            }
            print(f"  [ERROR] {type(exc).__name__}: {exc}")

        results.append(result)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    successes = sum(1 for result in results if result.get("success"))
    print(f"\n{'=' * 60}")
    print(f"DONE: {successes}/{len(results)} benchmarked successfully")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")
    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
