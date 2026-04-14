"""Audit market-scenario coverage across benchmark, extension, negative, and canary corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trellis.agent.market_scenarios import (
    build_market_scenario_coverage_report,
    load_market_scenario_contracts,
    render_market_scenario_coverage_report,
)
from trellis.agent.task_manifests import (
    load_canary_manifest,
    load_negative_tasks,
    load_pricing_tasks,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    pricing_tasks = load_pricing_tasks(root=ROOT)
    negative_tasks = load_negative_tasks(root=ROOT)
    canaries, _ = load_canary_manifest(root=ROOT)
    scenarios = load_market_scenario_contracts(root=ROOT)

    report = build_market_scenario_coverage_report(
        pricing_tasks=pricing_tasks,
        negative_tasks=negative_tasks,
        canaries=canaries,
        scenario_contracts=scenarios,
    )
    markdown = render_market_scenario_coverage_report(report)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
