"""Run pricing tasks through the knowledge-aware build pipeline.

Usage:
    python scripts/run_tasks.py T13 T24           # run T13 through T24
    python scripts/run_tasks.py F001 F008         # run FinancePy parity tasks
    python scripts/run_tasks.py all               # run all pending
    python scripts/run_tasks.py --reuse T13 T24   # reuse existing modules when present
    python scripts/run_tasks.py --corpus benchmark_financepy all
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

from trellis.agent.config import get_batch_token_budget, load_env
from trellis.agent.evals import summarize_task_results
from trellis.agent.task_run_store import (
    TASK_RUN_LATEST_INDEX,
    TASK_RUN_LATEST_ROOT,
)
from trellis.agent.task_runtime import build_market_state, load_tasks, run_task
from trellis.cli_paths import resolve_repo_path

load_env()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", help="'all' or a start/end task id pair")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing generated modules when present instead of forcing rebuilds.",
    )
    parser.add_argument(
        "--fresh-build",
        action="store_true",
        help="Bypass deterministic supported-route reuse so tasks exercise live code generation.",
    )
    parser.add_argument(
        "--knowledge-light",
        action="store_true",
        help="Use the compiler-first minimal knowledge profile during task builds.",
    )
    parser.add_argument("--validation", default="standard")
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        default=[],
        help="Filter to one or more task corpora (benchmark_financepy, extension, market_construction, proof_legacy).",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def run_block(
    tasks: list[dict],
    output_file: str,
    *,
    model: str = "gpt-5.4-mini",
    force_rebuild: bool = True,
    fresh_build: bool = False,
    knowledge_light: bool = False,
    validation: str = "standard",
):
    """Run a block of tasks and save results incrementally."""
    market_state = build_market_state()
    results = []
    batch_token_budget = get_batch_token_budget()
    batch_token_total = 0

    print(f"\n{'#' * 60}")
    print(f"# Running {len(tasks)} tasks → {output_file}")
    print(f"# Model: {model}")
    print(f"# Force rebuild: {force_rebuild}")
    print(f"# Fresh build: {fresh_build}")
    print(f"# Knowledge light: {knowledge_light}")
    print(f"# Validation: {validation}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    for index, task in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}]", end="")
        result = run_task(
            task,
            market_state,
            model=model,
            force_rebuild=(force_rebuild or fresh_build),
            fresh_build=fresh_build,
            knowledge_profile="knowledge_light" if knowledge_light else None,
            validation=validation,
        )
        results.append(result)
        batch_token_total += int(
            ((result.get("token_usage_summary") or {}).get("total_tokens") or 0)
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if batch_token_budget and batch_token_total > batch_token_budget:
            print(
                f"  [STOP] Batch token budget exceeded: {batch_token_total} > {batch_token_budget}"
            )
            break

    successes = sum(1 for result in results if result.get("success"))
    total_time = sum(result.get("elapsed_seconds", 0) for result in results)
    lessons = sum(
        1 for result in results if result.get("reflection", {}).get("lesson_captured")
    )
    cookbooks = sum(
        1 for result in results if result.get("reflection", {}).get("cookbook_enriched")
    )
    summary = summarize_task_results(results)
    summary_path = Path(output_file).with_name(f"{Path(output_file).stem}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {successes}/{len(results)} succeeded in {total_time:.0f}s")
    print(f"  Lessons captured: {lessons}")
    print(f"  Cookbooks enriched: {cookbooks}")
    print(f"  Failure buckets: {summary['failure_buckets']}")
    print(f"  Retry recovery: {summary['retry_recovery']}")
    print(f"  Reviewer signals: {summary['reviewer_signals']}")
    print(f"  Shared knowledge: {summary['shared_knowledge']}")
    print(f"  Promotion discipline: {summary['promotion_discipline']}")
    print(f"  Token usage: {summary['token_usage']}")
    if batch_token_budget:
        print(f"  Batch token budget: {batch_token_budget}")
    print(f"  Results saved to: {output_file}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Latest task runs: {TASK_RUN_LATEST_ROOT}")
    print(f"  Latest task index: {TASK_RUN_LATEST_INDEX}")
    print(f"  Latest diagnosis packets: {TASK_RUN_LATEST_ROOT.parent / 'diagnostics' / 'latest'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    if not args.ids or args.ids[0] == "all":
        tasks = load_tasks()
        output_file = resolve_repo_path(args.output, ROOT / "task_results_all.json")
    elif len(args.ids) == 2:
        tasks = load_tasks(args.ids[0], args.ids[1])
        safe_name = f"{args.ids[0]}_{args.ids[1]}".lower()
        output_file = resolve_repo_path(args.output, ROOT / f"task_results_{safe_name}.json")
    else:
        print(f"Usage: {sys.argv[0]} [--reuse] [--model MODEL] [start_id end_id | all]")
        sys.exit(1)

    if args.corpora:
        allowed = {str(corpus).strip().lower() for corpus in args.corpora if str(corpus).strip()}
        tasks = [
            task
            for task in tasks
            if str(task.get("task_corpus") or "").strip().lower() in allowed
        ]
        if not tasks:
            print(f"No tasks matched corpora: {sorted(allowed)}")
            sys.exit(1)

    run_block(
        tasks,
        output_file,
        model=args.model,
        force_rebuild=not args.reuse,
        fresh_build=args.fresh_build,
        knowledge_light=args.knowledge_light,
        validation=args.validation,
    )
