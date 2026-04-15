"""Emit a timestamped FinancePy pilot parity scorecard from history records.

Reads append-only records from ``task_runs/financepy_benchmarks/history``,
restricts them to the pilot subset (F001/F002/F003/F007/F009/F012), verifies
each run's provenance against the QUA-866 fresh-generated boundary, and
writes a timestamped JSON+Markdown scorecard.  Residual misses are
escalated in the scorecard as shared root-cause follow-ons rather than
task-specific patches.

To rerun the pilot subset fresh-generated before building the scorecard, use
``scripts/run_financepy_benchmark.py`` with ``--execution-policy=fresh_generated``
and the pilot task ids.  This script does not trigger new benchmark runs; it
only consumes already-persisted history.

Refs: QUA-868 (epic QUA-864).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        default="",
        help=(
            "Root of the benchmark history tree (defaults to "
            "task_runs/financepy_benchmarks)."
        ),
    )
    parser.add_argument(
        "--reports-root",
        default="",
        help=(
            "Directory for the emitted scorecard artifacts (defaults to "
            "task_runs/financepy_benchmarks/scorecards)."
        ),
    )
    parser.add_argument(
        "--scorecard-name",
        default="financepy_pilot",
        help="Stem used for the emitted scorecard JSON and Markdown files.",
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="Optional benchmark_campaign_id filter for history records.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional scorecard note; repeat to add multiple notes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from trellis.agent.financepy_benchmark import DEFAULT_FINANCEPY_BENCHMARK_ROOT
    from trellis.agent.pilot_parity_scorecard import (
        build_pilot_parity_scorecard,
        load_pilot_benchmark_records,
        save_pilot_parity_scorecard,
    )

    args = _parse_args(argv if argv is not None else sys.argv[1:])
    benchmark_root = (
        Path(args.benchmark_root).expanduser()
        if args.benchmark_root
        else DEFAULT_FINANCEPY_BENCHMARK_ROOT
    )
    reports_root = (
        Path(args.reports_root).expanduser()
        if args.reports_root
        else benchmark_root / "scorecards"
    )

    records = load_pilot_benchmark_records(
        benchmark_root=benchmark_root,
        campaign_id=args.campaign_id,
    )
    scorecard = build_pilot_parity_scorecard(
        scorecard_name=args.scorecard_name,
        benchmark_runs=records,
        campaign_id=args.campaign_id,
        notes=list(args.note or ()),
    )
    artifacts = save_pilot_parity_scorecard(
        scorecard,
        reports_root=reports_root,
        stem=args.scorecard_name,
    )

    payload = {
        "scorecard_name": args.scorecard_name,
        "scorecard_json": str(artifacts.json_path),
        "scorecard_md": str(artifacts.text_path),
        "pilot_task_ids": list(scorecard["pilot_task_ids"]),
        "pilot_summary": dict(scorecard["pilot_summary"]),
        "residual_misses": list(scorecard.get("residual_misses") or ()),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
