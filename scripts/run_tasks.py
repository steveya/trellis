"""Run pricing tasks through the knowledge-aware build pipeline.

Usage:
    python scripts/run_tasks.py T13 T24           # run T13 through T24
    python scripts/run_tasks.py F001 F008         # run FinancePy parity tasks
    python scripts/run_tasks.py all               # run all pending
    python scripts/run_tasks.py --task-id F001 --task-id P004 --status all
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
from trellis.agent.task_manifests import filter_loaded_tasks
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
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        default=[],
        help="Run one exact task id. May be repeated.",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing generated modules when present instead of forcing rebuilds.",
    )
    parser.add_argument(
        "--fresh-build",
        action="store_true",
        help=(
            "Bypass admitted adapter reuse and isolate the output path. This does not "
            "by itself require model-generated source."
        ),
    )
    parser.add_argument(
        "--generation-policy",
        choices=("deterministic_allowed", "builder_synthesis_required"),
        default="deterministic_allowed",
        help="Require observed builder-agent source synthesis or allow deterministic materialization.",
    )
    parser.add_argument(
        "--knowledge-light",
        action="store_true",
        help="Use the compiler-first minimal knowledge profile during task builds.",
    )
    parser.add_argument("--validation", default="standard")
    parser.add_argument(
        "--recovery-mode",
        choices=("strict", "assisted", "remediation"),
        default="assisted",
        help="Bounded automatic recovery mode for task-script runs.",
    )
    parser.add_argument(
        "--offline-local-agents",
        action="store_true",
        help=(
            "Forbid live LLM API calls during task execution. Deterministic quant, "
            "validation, exact bindings, and local/cassette paths may run; any "
            "attempted text or JSON LLM call fails the task."
        ),
    )
    parser.add_argument(
        "--status",
        choices=("pending", "all"),
        default="pending",
        help="Manifest status filter. Use `all` for reruns of already-completed tasks.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        default=[],
        help=(
            "Filter to one or more task corpora (benchmark_financepy, extension, "
            "market_construction, proof_legacy, fpml_conformance)."
        ),
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _selection_output_path(
    args: argparse.Namespace,
    *,
    task_ids: list[str],
) -> Path:
    if args.output:
        return resolve_repo_path(args.output, ROOT / "task_results_all.json")
    if task_ids:
        first = task_ids[0].lower()
        return ROOT / f"task_results_ids_{first}_{len(task_ids)}.json"
    if not args.ids or args.ids[0] == "all":
        return ROOT / "task_results_all.json"
    safe_name = f"{args.ids[0]}_{args.ids[1]}".lower()
    return ROOT / f"task_results_{safe_name}.json"


def run_block(
    tasks: list[dict],
    output_file: str,
    *,
    model: str = "gpt-5.4-mini",
    force_rebuild: bool = True,
    fresh_build: bool = False,
    generation_policy: str = "deterministic_allowed",
    knowledge_light: bool = False,
    validation: str = "standard",
    recovery_mode: str = "assisted",
    offline_local_agents: bool = False,
):
    """Run a block of tasks and save results incrementally."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    market_state = build_market_state()
    results = []
    batch_token_budget = get_batch_token_budget()
    batch_token_total = 0

    print(f"\n{'#' * 60}")
    print(f"# Running {len(tasks)} tasks → {output_file}")
    print(f"# Model: {model}")
    print(f"# Force rebuild: {force_rebuild}")
    print(f"# Fresh build: {fresh_build}")
    print(f"# Generation policy: {generation_policy}")
    print(f"# Knowledge light: {knowledge_light}")
    print(f"# Validation: {validation}")
    print(f"# Recovery mode: {recovery_mode}")
    print(f"# Offline local agents: {offline_local_agents}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    for index, task in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}]", end="")
        if offline_local_agents:
            from trellis.agent.offline_agents import offline_local_agent_run_scope

            with offline_local_agent_run_scope():
                result = run_task(
                    task,
                    market_state,
                    model=model,
                    force_rebuild=(force_rebuild or fresh_build),
                    fresh_build=fresh_build,
                    generation_policy=generation_policy,
                    knowledge_profile="knowledge_light" if knowledge_light else None,
                    validation=validation,
                    recovery_mode=recovery_mode,
                    execution_mode_override="deterministic_replay",
                )
            result["offline_local_agents"] = True
        else:
            result = run_task(
                task,
                market_state,
                model=model,
                force_rebuild=(force_rebuild or fresh_build),
                fresh_build=fresh_build,
                generation_policy=generation_policy,
                knowledge_profile="knowledge_light" if knowledge_light else None,
                validation=validation,
                recovery_mode=recovery_mode,
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

    total_time = sum(result.get("elapsed_seconds", 0) for result in results)
    lessons = sum(
        1 for result in results if result.get("reflection", {}).get("lesson_captured")
    )
    cookbooks = sum(
        1 for result in results if result.get("reflection", {}).get("cookbook_enriched")
    )
    summary = summarize_task_results(results)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    totals = summary["totals"]

    print(f"\n{'=' * 60}")
    print(
        "SUMMARY: "
        f"{totals['expectation_passes']}/{len(results)} passed expectations "
        f"in {total_time:.0f}s"
    )
    print(f"  Pricing successes: {totals['successes']}")
    print(f"  Honest blocks: {totals['honest_blocks']}")
    print(f"  Actionable failures: {totals['actionable_failures']}")
    print(f"  Lessons captured: {lessons}")
    print(f"  Cookbooks enriched: {cookbooks}")
    print(f"  Failure buckets: {summary['failure_buckets']}")
    print(f"  Actionable failure buckets: {summary['actionable_failure_buckets']}")
    print(f"  Retry recovery: {summary['retry_recovery']}")
    print(f"  Reviewer signals: {summary['reviewer_signals']}")
    print(f"  Shared knowledge: {summary['shared_knowledge']}")
    print(f"  Generation proving: {summary['generation_proving']}")
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
    if args.ids and args.task_ids:
        print("Choose either positional range/all selection or --task-id, not both.")
        sys.exit(1)
    if args.ids and args.ids[0] == "all" and len(args.ids) > 1:
        print("The `all` selector does not accept additional positional ids.")
        sys.exit(1)
    if args.ids and args.ids[0] != "all" and len(args.ids) != 2:
        print(
            f"Usage: {sys.argv[0]} [--reuse] [--model MODEL] [--task-id TASK_ID ...] [start_id end_id | all]"
        )
        sys.exit(1)

    requested_task_ids = _dedupe_preserve_order(list(args.task_ids or ()))
    status = None if args.status == "all" else args.status
    selection_root = load_tasks(status=None)

    if requested_task_ids:
        tasks = filter_loaded_tasks(
            selection_root,
            status=status,
            corpora=args.corpora,
            task_ids=requested_task_ids,
        )
        matched_ids = {str(task.get("id") or "").strip() for task in tasks}
        missing = [task_id for task_id in requested_task_ids if task_id not in matched_ids]
        if missing:
            print(
                "Requested task ids were not available under the current filters: "
                + ", ".join(missing)
            )
            sys.exit(1)
    elif not args.ids or args.ids[0] == "all":
        tasks = filter_loaded_tasks(
            selection_root,
            status=status,
            corpora=args.corpora,
        )
    else:
        tasks = filter_loaded_tasks(
            selection_root,
            args.ids[0],
            args.ids[1],
            status=status,
            corpora=args.corpora,
        )

    if args.corpora and not tasks:
        allowed = sorted(
            {
                str(corpus).strip().lower()
                for corpus in args.corpora
                if str(corpus).strip()
            }
        )
        print(f"No tasks matched corpora: {allowed}")
        sys.exit(1)
    if not tasks:
        print("No tasks matched the requested selection.")
        sys.exit(1)

    output_file = resolve_repo_path(
        args.output,
        _selection_output_path(args, task_ids=requested_task_ids),
    )

    run_block(
        tasks,
        output_file,
        model=args.model,
        force_rebuild=not args.reuse,
        fresh_build=args.fresh_build,
        generation_policy=args.generation_policy,
        knowledge_light=args.knowledge_light,
        validation=args.validation,
        recovery_mode=args.recovery_mode,
        offline_local_agents=args.offline_local_agents,
    )
