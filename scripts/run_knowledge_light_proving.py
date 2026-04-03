"""Run the tranche-2 knowledge-light simple-derivative proving set.

Usage:
    /Users/steveyang/miniforge3/bin/python3 scripts/run_knowledge_light_proving.py
    /Users/steveyang/miniforge3/bin/python3 scripts/run_knowledge_light_proving.py --list-cases
    /Users/steveyang/miniforge3/bin/python3 scripts/run_knowledge_light_proving.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

from trellis.agent.config import load_env
from trellis.agent.evals import summarize_task_results
from trellis.agent.task_runtime import build_market_state, run_task
from trellis.cli_paths import resolve_repo_path

load_env()


def _fx_vanilla_compare_case() -> dict:
    return {
        "id": "KL01",
        "title": "FX vanilla option: Garman-Kohlhagen vs MC",
        "status": "pending",
        "construct": ["analytical", "monte_carlo"],
        "cross_validate": {
            "internal": ["gk_analytical", "mc_fx_option"],
            "external": ["quantlib", "financepy"],
        },
        "new_component": "garman_kohlhagen_formula",
        "market": {
            "source": "mock",
            "as_of": "2024-11-15",
            "discount_curve": "usd_ois",
            "forecast_curve": "EUR-DISC",
            "fx_rate": "EURUSD",
        },
        "market_assertions": {
            "requires": ["discount_curve", "forward_curve", "fx_rates", "spot"],
            "selected": {
                "discount_curve": "usd_ois",
                "forecast_curve": "EUR-DISC",
                "fx_rate": "EURUSD",
            },
        },
    }


def _american_put_tree_case() -> dict:
    return {
        "id": "KL02",
        "title": "American put: equity tree knowledge-light proving",
        "description": (
            "Build a thin adapter for a vanilla American put on an equity underlier. "
            "Use the checked lattice helper for the pricing engine rather than open-coding "
            "the rollback."
        ),
        "status": "pending",
        "construct": ["rate_tree"],
        "market": {
            "source": "mock",
            "as_of": "2024-11-15",
            "discount_curve": "usd_ois",
            "vol_surface": "usd_rates_smile",
            "underlier_spot": "SPX",
        },
        "market_assertions": {
            "requires": ["discount_curve", "black_vol_surface", "spot"],
            "selected": {
                "discount_curve": "usd_ois",
                "vol_surface": "usd_rates_smile",
                "underlier_spot": "SPX",
            },
        },
    }


def _cds_compare_case() -> dict:
    return {
        "id": "KL03",
        "title": "CDS pricing: hazard rate MC vs survival prob analytical",
        "status": "pending",
        "construct": ["monte_carlo", "credit"],
        "cross_validate": {
            "internal": ["mc_cds", "analytical_cds"],
            "external": ["quantlib", "financepy"],
        },
        "new_component": "cds_pricing",
        "market": {
            "source": "mock",
            "as_of": "2024-11-15",
            "discount_curve": "usd_ois",
            "credit_curve": "usd_ig",
        },
        "market_assertions": {
            "requires": ["discount_curve", "credit_curve"],
            "selected": {
                "discount_curve": "usd_ois",
                "credit_curve": "usd_ig",
            },
        },
    }


def build_proving_tasks() -> list[dict]:
    """Return the canonical tranche-2 proving set."""
    return [
        _fx_vanilla_compare_case(),
        _american_put_tree_case(),
        _cds_compare_case(),
    ]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Optional proving case IDs to run. Defaults to the full proving set.",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--validation", default="standard")
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the available proving case IDs and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected proving tasks without executing builds.",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def _select_cases(tasks: list[dict], requested: Iterable[str] | None) -> list[dict]:
    if not requested:
        return tasks
    requested_ids = {str(case_id) for case_id in requested}
    return [task for task in tasks if task["id"] in requested_ids]


def run_proving_set(
    tasks: list[dict],
    output_file: str,
    *,
    model: str,
    validation: str,
) -> dict:
    """Run the selected proving tasks and persist results plus summary."""
    market_state = build_market_state()
    results: list[dict] = []

    print(f"\n{'#' * 60}")
    print(f"# Knowledge-light tranche-2 proving set → {output_file}")
    print(f"# Model: {model}")
    print("# Fresh build: True")
    print("# Knowledge light: True")
    print(f"# Validation: {validation}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    for index, task in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}] {task['id']}: {task['title']}", flush=True)
        result = run_task(
            task,
            market_state,
            model=model,
            force_rebuild=True,
            fresh_build=True,
            knowledge_profile="knowledge_light",
            validation=validation,
        )
        results.append(result)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    summary = summarize_task_results(results)
    summary_path = Path(output_file).with_name(f"{Path(output_file).stem}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {sum(1 for item in results if item.get('success'))}/{len(results)} succeeded")
    print(f"  Failure buckets: {summary['failure_buckets']}")
    print(f"  Retry recovery: {summary['retry_recovery']}")
    print(f"  Token usage: {summary['token_usage']}")
    print(f"  Results saved to: {output_file}")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'=' * 60}")

    return summary


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    proving_tasks = build_proving_tasks()

    if args.list_cases:
        for task in proving_tasks:
            print(f"{task['id']}: {task['title']}")
        sys.exit(0)

    selected = _select_cases(proving_tasks, args.cases)
    if not selected:
        print("No proving cases selected.")
        sys.exit(1)

    if args.dry_run:
        print(json.dumps(selected, indent=2))
        sys.exit(0)

    default_output = ROOT / "task_results_tranche2_proving_knowledge_light.json"
    output_file = resolve_repo_path(args.output, default_output)
    run_proving_set(
        selected,
        output_file,
        model=args.model,
        validation=args.validation,
    )
