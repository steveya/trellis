"""Record full-task LLM cassettes for canary tasks.

Usage:
    python scripts/record_cassettes.py            # record all canary cassettes
    python scripts/record_cassettes.py --task T38  # record single task
    python scripts/record_cassettes.py --dry-run   # show what would be recorded

This is a one-time token spend (~$2-3 for all canary tasks).
Cassettes are saved to cassettes/full_task/ and should be committed to git.

See QUA-458 for the full-task replay design context.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

from canary_common import (
    FULL_TASK_CASSETTES_DIR,
    load_curated_canary_set,
    merge_canary_task_payload,
)
from trellis.agent.config import load_env

load_env()

CANARY_MAX_RETRIES = 3


def load_canary_set() -> list[dict]:
    canaries, _ = load_curated_canary_set()
    return canaries


def record_cassette_for_task(
    task: dict,
    canary: dict,
    *,
    model: str = "gpt-5.4-mini",
) -> tuple[Path, dict]:
    """Record a cassette for a single canary task.

    Runs the full ``run_task(...)`` pipeline with cassette recording enabled.
    Returns the cassette file path plus the recorded task result.
    """
    from trellis.agent.cassette import llm_cassette_session
    from trellis.agent.task_runtime import build_market_state, run_task

    task_id = task["id"]
    cassette_path = FULL_TASK_CASSETTES_DIR / f"{task_id}.yaml"
    merged_task = merge_canary_task_payload(task, canary)
    market_state = build_market_state()
    with llm_cassette_session(
        cassette_path,
        mode="record",
        name=task_id,
        store_prompts=True,
    ):
        result = run_task(
            merged_task,
            market_state,
            model=model,
            force_rebuild=True,
            validation="standard",
            max_retries=CANARY_MAX_RETRIES,
        )
    return cassette_path, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record LLM cassettes for canary tasks.")
    parser.add_argument("--task", help="Record a single task by ID (e.g., T38)")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without recording")
    args = parser.parse_args(argv or sys.argv[1:])

    canaries = load_canary_set()

    from trellis.agent.task_runtime import load_tasks

    # Use the same task-loading path as the canary runner so record and replay
    # see the same normalized task payload.
    tasks = load_tasks(status=None)
    task_lookup = {t["id"]: t for t in tasks}

    if args.task:
        canaries = [c for c in canaries if c["id"] == args.task]
        if not canaries:
            print(f"Task {args.task} not in CANARY_TASKS.yaml")
            return 1
        if not canaries[0].get("record_cassette", True):
            print(f"Task {args.task} is marked record_cassette=false and is not replay-backed.")
            return 1
    else:
        canaries = [c for c in canaries if c.get("record_cassette", True)]

    if args.dry_run:
        print(f"\nWould record {len(canaries)} cassettes:")
        for c in canaries:
            task = task_lookup.get(c["id"], {})
            print(f"  {c['id']:6s}  {c.get('engine_family', '?'):14s}  {task.get('title', '?')[:50]}")
        print(f"\nEstimated cost: ~${sum(c.get('estimated_cost_usd', 0.12) for c in canaries):.2f}")
        return 0

    FULL_TASK_CASSETTES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  CASSETTE RECORDING — {len(canaries)} tasks")
    print(f"  Model: {args.model}")
    print(f"  Output: {FULL_TASK_CASSETTES_DIR}")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'=' * 65}")

    recorded = 0
    for idx, canary in enumerate(canaries, 1):
        task_id = canary["id"]
        task = task_lookup.get(task_id)
        if task is None:
            print(f"\n  [{idx}/{len(canaries)}] {task_id}  SKIP — not in active task manifests")
            continue

        print(f"\n  [{idx}/{len(canaries)}] {task_id}  {canary.get('engine_family', '?'):14s}", end="", flush=True)
        start = time.time()

        try:
            path, result = record_cassette_for_task(
                task,
                canary,
                model=args.model,
            )
            elapsed = time.time() - start
            status = "recorded" if result.get("success") else "recorded with task failure"
            print(f"  {status} ({elapsed:.1f}s) → {path.name}")
            recorded += 1
        except Exception as exc:
            elapsed = time.time() - start
            print(f"  FAILED ({elapsed:.1f}s): {exc}")

    print(f"\n{'=' * 65}")
    print(f"  Recorded: {recorded}/{len(canaries)} cassettes")
    print(f"  Location: {FULL_TASK_CASSETTES_DIR}")
    print(f"{'=' * 65}")

    return 0 if recorded == len(canaries) else 1


if __name__ == "__main__":
    sys.exit(main())
