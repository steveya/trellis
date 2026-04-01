"""Record LLM cassettes for all canary tasks.

Usage:
    python scripts/record_cassettes.py            # record all canary cassettes
    python scripts/record_cassettes.py --task T38  # record single task
    python scripts/record_cassettes.py --dry-run   # show what would be recorded

This is a one-time token spend (~$2-3 for all canary tasks).
Cassettes are saved to cassettes/ and should be committed to git.

See QUA-427 for design context.
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

import yaml

from trellis.agent.config import load_env

load_env()

CANARY_FILE = ROOT / "CANARY_TASKS.yaml"
CASSETTES_DIR = ROOT / "cassettes"


def load_canary_set() -> list[dict]:
    raw = yaml.safe_load(CANARY_FILE.read_text(encoding="utf-8"))
    return raw.get("canary_set", [])


def record_cassette_for_task(
    task_id: str,
    task_title: str,
    construct: str,
    *,
    model: str = "gpt-5.4-mini",
) -> Path:
    """Record a cassette for a single canary task.

    Runs the build pipeline with cassette recording enabled.
    Returns the cassette file path.
    """
    from trellis.agent.cassette import CassetteRecorder
    from trellis.agent import config as agent_config
    from trellis.agent.executor import build_payoff

    cassette_path = CASSETTES_DIR / f"{task_id}.yaml"
    recorder = CassetteRecorder(cassette_path, name=task_id, store_prompts=True)

    # Wrap LLM functions
    real_generate = agent_config.llm_generate
    real_generate_json = agent_config.llm_generate_json

    agent_config.llm_generate = recorder.wrap_generate(real_generate)
    agent_config.llm_generate_json = recorder.wrap_generate_json(real_generate_json)

    try:
        build_payoff(
            task_title,
            instrument_type=construct if isinstance(construct, str) else None,
            model=model,
            force_rebuild=True,
        )
    except Exception as exc:
        print(f"  WARNING: Build failed for {task_id}: {exc}")
        print(f"  Cassette still saved with {len(recorder)} calls recorded.")
    finally:
        agent_config.llm_generate = real_generate
        agent_config.llm_generate_json = real_generate_json

    recorder.flush(
        provider=agent_config.get_provider(),
        model=model,
    )
    return cassette_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record LLM cassettes for canary tasks.")
    parser.add_argument("--task", help="Record a single task by ID (e.g., T38)")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without recording")
    args = parser.parse_args(argv or sys.argv[1:])

    canaries = load_canary_set()

    # Load TASKS.yaml for titles
    tasks = yaml.safe_load((ROOT / "TASKS.yaml").read_text(encoding="utf-8"))
    task_lookup = {t["id"]: t for t in tasks}

    if args.task:
        canaries = [c for c in canaries if c["id"] == args.task]
        if not canaries:
            print(f"Task {args.task} not in CANARY_TASKS.yaml")
            return 1

    if args.dry_run:
        print(f"\nWould record {len(canaries)} cassettes:")
        for c in canaries:
            task = task_lookup.get(c["id"], {})
            print(f"  {c['id']:6s}  {c.get('engine_family', '?'):14s}  {task.get('title', '?')[:50]}")
        print(f"\nEstimated cost: ~${sum(c.get('estimated_cost_usd', 0.12) for c in canaries):.2f}")
        return 0

    CASSETTES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  CASSETTE RECORDING — {len(canaries)} tasks")
    print(f"  Model: {args.model}")
    print(f"  Output: {CASSETTES_DIR}")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'=' * 65}")

    recorded = 0
    for idx, canary in enumerate(canaries, 1):
        task_id = canary["id"]
        task = task_lookup.get(task_id)
        if task is None:
            print(f"\n  [{idx}/{len(canaries)}] {task_id}  SKIP — not in TASKS.yaml")
            continue

        print(f"\n  [{idx}/{len(canaries)}] {task_id}  {canary.get('engine_family', '?'):14s}", end="", flush=True)
        start = time.time()

        try:
            path = record_cassette_for_task(
                task_id,
                task["title"],
                task.get("construct", ""),
                model=args.model,
            )
            elapsed = time.time() - start
            print(f"  recorded ({elapsed:.1f}s) → {path.name}")
            recorded += 1
        except Exception as exc:
            elapsed = time.time() - start
            print(f"  FAILED ({elapsed:.1f}s): {exc}")

    print(f"\n{'=' * 65}")
    print(f"  Recorded: {recorded}/{len(canaries)} cassettes")
    print(f"  Location: {CASSETTES_DIR}")
    print(f"{'=' * 65}")

    return 0 if recorded == len(canaries) else 1


if __name__ == "__main__":
    sys.exit(main())
