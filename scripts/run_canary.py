"""Run the curated canary task set for regression detection.

Usage:
    python scripts/run_canary.py                     # run all canaries
    python scripts/run_canary.py --task T38          # run single canary
    python scripts/run_canary.py --task T38 --replay # replay single canary from cassette
    python scripts/run_canary.py --dry-run           # show plan, no execution
    python scripts/run_canary.py --budget 1.50       # override budget limit
    python scripts/run_canary.py --subset core       # run core subset (lattice + MC + PDE)
    python scripts/run_canary.py --check-drift       # compare against golden traces
    python scripts/run_canary.py --update-golden     # promote passing results to golden

See CANARY_TASKS.yaml for the curated set and QUA-424 for design context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

import yaml

from trellis.agent.config import load_env

load_env()

from trellis.agent.checkpoints import load_latest_checkpoint
from trellis.agent.golden_traces import (
    detect_drift_for_canary,
    format_drift_report,
    update_golden_from_results,
)

CANARY_FILE = ROOT / "CANARY_TASKS.yaml"
FULL_TASK_CASSETTES_DIR = ROOT / "cassettes" / "full_task"

# Engine families considered "core" for the --subset=core option
CORE_FAMILIES = {"lattice", "monte_carlo", "pde", "credit"}
CANARY_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Canary set loading
# ---------------------------------------------------------------------------

def load_canary_set(
    path: Path = CANARY_FILE,
) -> tuple[list[dict], dict]:
    """Load CANARY_TASKS.yaml.  Returns (canary_list, meta)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    meta = {
        "version": raw.get("version", 1),
        "total_budget_usd": raw.get("total_budget_usd", 3.0),
        "refresh_cadence": raw.get("refresh_cadence", "weekly"),
    }
    return raw.get("canary_set", []), meta


def filter_canaries(
    canaries: list[dict],
    *,
    task_id: str | None = None,
    subset: str | None = None,
) -> list[dict]:
    """Filter the canary set by task ID or subset name."""
    if task_id:
        matches = [c for c in canaries if c["id"] == task_id]
        if not matches:
            available = [c["id"] for c in canaries]
            raise ValueError(
                f"Task {task_id} is not in the canary set. Available: {available}"
            )
        return matches
    if subset == "core":
        return [c for c in canaries if c.get("engine_family") in CORE_FAMILIES]
    return canaries


def merge_canary_task_payload(task: dict, canary: dict) -> dict:
    """Overlay curated canary fields onto the live task registry payload."""
    merged = dict(task)
    for key in (
        "description",
        "market",
        "market_assertions",
        "construct",
        "cross_validate",
        "new_component",
    ):
        value = canary.get(key)
        if value is not None:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Dry-run display
# ---------------------------------------------------------------------------

def display_dry_run(canaries: list[dict], meta: dict) -> None:
    """Print what would run without executing anything."""
    total_cost = sum(c.get("estimated_cost_usd", 0) for c in canaries)
    print(f"\n{'=' * 65}")
    print(f"  CANARY DRY RUN — {len(canaries)} tasks, ~${total_cost:.2f} estimated")
    print(f"  Budget: ${meta['total_budget_usd']:.2f}")
    print(f"{'=' * 65}")
    for c in canaries:
        cost = c.get("estimated_cost_usd", 0)
        family = c.get("engine_family", "?")
        complexity = c.get("complexity", "?")
        print(f"  {c['id']:6s}  {family:14s}  {complexity:8s}  ~${cost:.2f}")
        covers = c.get("covers", [])
        if covers:
            print(f"         covers: {', '.join(covers[:5])}")
    print(f"{'=' * 65}")
    print(f"  Total estimated cost: ${total_cost:.2f}")
    print(f"  Budget limit: ${meta['total_budget_usd']:.2f}")
    print()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_canaries(
    canaries: list[dict],
    meta: dict,
    *,
    model: str = "gpt-5.4-mini",
    budget_override: float | None = None,
    validation: str = "standard",
    knowledge_light: bool = False,
    output_file: str | None = None,
    replay: bool = False,
    cassette_dir: str | Path | None = None,
    cassette_stale_policy: str = "error",
) -> list[dict]:
    """Run canary tasks and return results.

    Returns list of result dicts with canary metadata attached.
    """
    from trellis.agent.task_runtime import build_market_state, load_tasks, run_task

    budget = budget_override or meta.get("total_budget_usd", 3.0)
    cassette_root = (
        Path(cassette_dir)
        if cassette_dir is not None
        else FULL_TASK_CASSETTES_DIR
    )
    # Canary coverage is curated independently of task lifecycle state, so the
    # runner must look across the full task registry rather than only "pending"
    # entries.
    all_tasks = load_tasks(status=None)
    task_lookup = {t["id"]: t for t in all_tasks}

    market_state = build_market_state()
    results: list[dict] = []
    total_tokens = 0
    total_time = 0.0
    pass_count = 0

    print(f"\n{'=' * 65}")
    print(f"  CANARY RUN — {len(canaries)} tasks")
    print(f"  Model: {model}")
    print(f"  Budget: ${budget:.2f}")
    if replay:
        print(f"  Replay: cassette-backed ({cassette_root})")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'=' * 65}")

    for idx, canary in enumerate(canaries, 1):
        task_id = canary["id"]
        task = task_lookup.get(task_id)
        if task is None:
            print(f"\n  [{idx}/{len(canaries)}] {task_id:6s}  SKIP — not found in TASKS.yaml")
            results.append({
                "canary_id": task_id,
                "engine_family": canary.get("engine_family"),
                "success": False,
                "skipped": True,
                "reason": "not_in_tasks_yaml",
            })
            continue
        task = merge_canary_task_payload(task, canary)
        cassette_path = cassette_root / f"{task_id}.yaml"

        start = time.time()
        print(f"\n  [{idx}/{len(canaries)}] {task_id:6s}  {canary.get('engine_family', '?'):14s}", end="", flush=True)

        if replay and not cassette_path.exists():
            error = (
                f"Missing cassette for {task_id} at {cassette_path}. "
                f"Record it with scripts/record_cassettes.py --task {task_id}"
            )
            print("  SKIP — missing cassette")
            results.append({
                "canary_id": task_id,
                "engine_family": canary.get("engine_family"),
                "success": False,
                "skipped": True,
                "reason": "missing_cassette",
                "error": error,
            })
            continue

        try:
            run_kwargs = {
                "model": model,
                "force_rebuild": True,
                "validation": validation,
                "max_retries": CANARY_MAX_RETRIES,
                "knowledge_profile": "knowledge_light" if knowledge_light else None,
            }
            if replay:
                from trellis.agent.cassette import llm_cassette_session

                with llm_cassette_session(
                    cassette_path,
                    mode="replay",
                    stale_policy=cassette_stale_policy,
                    name=task_id,
                ):
                    result = run_task(
                        task,
                        market_state,
                        **run_kwargs,
                    )
            else:
                result = run_task(
                    task,
                    market_state,
                    **run_kwargs,
                )
        except Exception as exc:
            result = {
                "task_id": task_id,
                "success": False,
                "error": str(exc),
                "elapsed_seconds": time.time() - start,
            }
            if replay:
                result["execution_mode"] = "cassette_replay"
                result["llm_cassette"] = {
                    "mode": "replay",
                    "name": task_id,
                    "path": str(cassette_path),
                    "stale_policy": cassette_stale_policy,
                }

        elapsed = time.time() - start
        tokens = int((result.get("token_usage_summary") or {}).get("total_tokens") or 0)
        success = result.get("success", False)
        total_tokens += tokens
        total_time += elapsed

        status = "\u2713 PASS" if success else "\u2717 FAIL"
        print(f"  {status}  {elapsed:.1f}s  {tokens}tok")

        if success:
            pass_count += 1

        # Attach canary metadata to result
        result["canary_id"] = task_id
        result["engine_family"] = canary.get("engine_family")
        result["complexity"] = canary.get("complexity")
        results.append(result)

        # Save incrementally
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

        # Budget check (approximate: $0.01 per 1000 tokens for input+output blend)
        estimated_cost = total_tokens * 0.00001  # rough estimate
        if budget and estimated_cost > budget:
            print(f"\n  [BUDGET] Estimated cost ${estimated_cost:.2f} > budget ${budget:.2f} — stopping")
            break

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  CANARY RESULTS: {pass_count}/{len(results)} passed")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.1f}s")
    if output_file:
        print(f"  Results saved: {output_file}")
    print(f"{'=' * 65}")

    return results


# ---------------------------------------------------------------------------
# Drift detection & golden trace management
# ---------------------------------------------------------------------------

def check_drift_after_run(
    results: list[dict],
    canaries: list[dict],
) -> int:
    """Load latest checkpoints and compare against golden traces.

    Returns 2 if decision-level drift detected, 0 otherwise.
    """
    reports = []
    for canary in canaries:
        task_id = canary["id"]
        engine_family = canary.get("engine_family", "")

        # Load the latest checkpoint for this task
        checkpoint = load_latest_checkpoint(task_id)
        if checkpoint is None:
            print(f"  [drift] {task_id}: no checkpoint found, skipping drift check")
            continue

        report = detect_drift_for_canary(
            task_id, checkpoint, engine_family=engine_family,
        )
        reports.append(report)

    if reports:
        print(format_drift_report(reports))

    # Return 2 if any decision-level drift (distinguishable from test failure exit=1)
    from trellis.agent.golden_traces import DriftSummary

    summary = DriftSummary(reports=tuple(reports))
    if summary.has_blocking_drift:
        print("\n  ⚠ Decision-level drift detected — review before release.")
        return 2
    return 0


def promote_golden_after_run(
    results: list[dict],
    canaries: list[dict],
) -> None:
    """Promote passing checkpoints to golden traces."""
    checkpoints = {}
    for canary in canaries:
        task_id = canary["id"]
        checkpoint = load_latest_checkpoint(task_id)
        if checkpoint is not None:
            checkpoints[task_id] = checkpoint

    if not checkpoints:
        print("  [golden] No checkpoints found to promote.")
        return

    updated = update_golden_from_results(results, checkpoints)
    if updated:
        print(f"\n  ✓ Golden traces updated: {', '.join(updated)}")
    else:
        print("\n  ✗ Golden traces NOT updated (some canaries failed).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the curated canary task set for regression detection.",
    )
    parser.add_argument("--task", help="Run a single canary task by ID (e.g., T38)")
    parser.add_argument("--subset", choices=["core"], help="Run a subset (core = lattice+MC+PDE+credit)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--budget", type=float, help="Override budget limit in USD")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--validation", default="standard")
    parser.add_argument(
        "--knowledge-light",
        action="store_true",
        help="Use the compiler-first minimal knowledge profile during canary builds",
    )
    parser.add_argument("--output", help="Output file path for results JSON")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay canaries from recorded LLM cassettes instead of making live model calls",
    )
    parser.add_argument(
        "--cassette-dir",
        help="Directory containing canary cassette YAML files (default: cassettes/)",
    )
    parser.add_argument(
        "--cassette-stale-policy",
        choices=["warn", "error"],
        default="error",
        help="How replay mode handles prompt-hash drift",
    )
    parser.add_argument("--check-drift", action="store_true", help="Compare results against golden traces and report drift")
    parser.add_argument("--update-golden", action="store_true", help="Promote passing results to golden traces (requires all pass)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 if all canaries pass, 1 otherwise."""
    args = _parse_args(argv or sys.argv[1:])

    canaries, meta = load_canary_set()
    canaries = filter_canaries(
        canaries,
        task_id=args.task,
        subset=args.subset,
    )

    if args.dry_run:
        display_dry_run(canaries, meta)
        return 0

    output_file = args.output or str(ROOT / f"canary_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    results = run_canaries(
        canaries,
        meta,
        model=args.model,
        budget_override=args.budget,
        validation=args.validation,
        knowledge_light=args.knowledge_light,
        output_file=output_file,
        replay=args.replay,
        cassette_dir=args.cassette_dir,
        cassette_stale_policy=args.cassette_stale_policy,
    )

    all_passed = all(r.get("success", False) for r in results if not r.get("skipped"))

    # Drift detection (post-run)
    if args.check_drift:
        drift_exit = check_drift_after_run(results, canaries)
        if drift_exit:
            return drift_exit

    # Golden trace promotion (only if all passed)
    if args.update_golden:
        promote_golden_after_run(results, canaries)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
