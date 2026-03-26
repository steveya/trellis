"""Compare two task-result tranches and emit a shared-memory improvement report.

Usage:
    /Users/steveyang/miniforge3/bin/python3 scripts/evaluate_shared_memory.py \
        baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trellis.agent.evals import compare_task_runs, render_shared_memory_report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline task_results JSON file")
    parser.add_argument("candidate", help="Candidate task_results JSON file")
    parser.add_argument("--output", help="Optional JSON output path for the full comparison report")
    return parser.parse_args(argv)


def _load_results(path: str) -> list[dict]:
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a task-results list")
    return data


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    baseline = _load_results(args.baseline)
    candidate = _load_results(args.candidate)
    report = compare_task_runs(baseline, candidate)

    print(render_shared_memory_report(report))

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nFull report saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
