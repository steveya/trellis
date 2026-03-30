"""Run the connector stress tranche as a standing regression gate.

Usage:
    python scripts/run_stress_tranche.py
    python scripts/run_stress_tranche.py --model claude-sonnet-4-6
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
    parser.add_argument("--output")
    return parser.parse_args(argv)


def _serialize_grade_report(report: dict[str, GradeResult]) -> dict[str, dict[str, object]]:
    return {
        key: {"passed": value.passed, "details": list(value.details)}
        for key, value in report.items()
    }


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    manifest = load_stress_task_manifest()
    tasks = {
        task["id"]: task
        for task in load_tasks("E21", "E28", status=None)
        if task["id"] in STRESS_TASK_IDS
    }

    if set(tasks) != set(STRESS_TASK_IDS):
        missing = sorted(set(STRESS_TASK_IDS) - set(tasks))
        print(f"Missing stress tasks in pricing manifest: {missing}")
        return 1

    preflight: dict[str, dict[str, dict[str, object]]] = {}
    preflight_failed = False
    for task_id in STRESS_TASK_IDS:
        report = grade_stress_task_preflight(tasks[task_id], manifest[task_id])
        preflight[task_id] = _serialize_grade_report(report)
        if not all(item.passed for item in report.values()):
            preflight_failed = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = resolve_repo_path(
        args.output,
        ROOT / f"task_results_stress_connector_{timestamp}.json",
    )
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    if preflight_failed:
        summary = {
            "status": "preflight_failed",
            "preflight": preflight,
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"Preflight failed. Summary saved to: {summary_path}")
        return 1

    market_state = build_market_state()
    results = []
    effective_model = args.model or get_default_model()

    for task_id in STRESS_TASK_IDS:
        task = tasks[task_id]
        print(f"{task_id}: {task['title']}", flush=True)
        result = run_task(
            task,
            market_state,
            model=effective_model,
            validation=args.validation,
        )
        results.append(result)
        output_path.write_text(json.dumps(results, indent=2, default=str))

    summary = {
        "status": "completed",
        "preflight": preflight,
        "task_summary": summarize_task_results(results),
        "stress_summary": summarize_stress_tranche(tasks, results, manifest=manifest),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
