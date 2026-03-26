"""Run the FX proving-ground pricing tranche and persist a rerun report.

Default task set:
- E25
- T105
- T108

Optional:
- T94 via --include-t94
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

from trellis.agent.config import load_env
from trellis.agent.evals import compare_task_runs, render_shared_memory_report, summarize_task_results
from trellis.agent.task_run_store import load_latest_task_run
from trellis.agent.task_runtime import build_market_state, load_tasks, run_task


DEFAULT_FX_TASK_IDS = ("E25", "T105", "T108")

load_env()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--validation", default="standard")
    parser.add_argument("--include-t94", action="store_true")
    parser.add_argument("--output", help="Path for candidate task results JSON")
    parser.add_argument("--report-output", help="Path for the tranche report JSON")
    return parser.parse_args(argv)


def _load_fx_tasks(include_t94: bool) -> list[dict]:
    selected = set(DEFAULT_FX_TASK_IDS)
    if include_t94:
        selected.add("T94")
    tasks = load_tasks(status=None)
    return [task for task in tasks if task["id"] in selected]


def _load_baseline_results(task_ids: list[str]) -> list[dict]:
    baseline: list[dict] = []
    for task_id in task_ids:
        record = load_latest_task_run(task_id)
        if record and isinstance(record.get("result"), dict):
            payload = dict(record["result"])
            payload.setdefault("task_id", task_id)
            baseline.append(payload)
    return baseline


def run_fx_tranche(
    *,
    model: str,
    validation: str,
    include_t94: bool,
    output_file: str,
    report_output_file: str,
) -> dict[str, object]:
    tasks = _load_fx_tasks(include_t94)
    task_ids = [task["id"] for task in tasks]
    baseline_results = _load_baseline_results(task_ids)
    market_state = build_market_state()
    candidate_results: list[dict] = []

    print(f"\n{'#' * 60}")
    print(f"# Running FX proving-ground tranche → {output_file}")
    print(f"# Tasks: {task_ids}")
    print(f"# Model: {model}")
    print(f"# Validation: {validation}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    for index, task in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}] {task['id']}: {task['title']}")
        result = run_task(
            task,
            market_state,
            model=model,
            validation=validation,
        )
        candidate_results.append(result)
        with open(output_file, "w") as handle:
            json.dump(candidate_results, handle, indent=2, default=str)

    task_summary = summarize_task_results(candidate_results)
    comparison_report = compare_task_runs(baseline_results, candidate_results)
    report = {
        "task_ids": task_ids,
        "baseline_count": len(baseline_results),
        "candidate_count": len(candidate_results),
        "task_summary": task_summary,
        "shared_memory_report": comparison_report,
        "rendered_shared_memory_report": render_shared_memory_report(comparison_report),
    }
    with open(report_output_file, "w") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(render_shared_memory_report(comparison_report))
    print("")
    print(f"Promotion discipline: {task_summary['promotion_discipline']}")
    print(f"Results saved to: {output_file}")
    print(f"Report saved to: {report_output_file}")
    print(f"{'=' * 60}")
    return report


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    output_file = args.output or str(ROOT / "task_results_fx_tranche.json")
    report_output = args.report_output or str(ROOT / "task_results_fx_tranche_report.json")
    run_fx_tranche(
        model=args.model,
        validation=args.validation,
        include_t94=args.include_t94,
        output_file=output_file,
        report_output_file=report_output,
    )
