"""Build a repeated-run scorecard from persisted benchmark history."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

from trellis.agent.benchmark_history import (
    build_benchmark_history_scorecard,
    load_benchmark_history_records,
    save_benchmark_history_scorecard,
)
from trellis.agent.financepy_benchmark import DEFAULT_FINANCEPY_BENCHMARK_ROOT
from trellis.agent.negative_task_benchmark import DEFAULT_NEGATIVE_BENCHMARK_ROOT


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_kind", choices=("financepy", "negative"))
    parser.add_argument("task_ids", nargs="*")
    parser.add_argument("--campaign-id")
    parser.add_argument("--history-root")
    parser.add_argument("--report-name")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    history_root = (
        Path(args.history_root)
        if args.history_root
        else (
            DEFAULT_FINANCEPY_BENCHMARK_ROOT
            if args.benchmark_kind == "financepy"
            else DEFAULT_NEGATIVE_BENCHMARK_ROOT
        )
    )
    scorecard_name = str(
        args.report_name
        or f"{args.benchmark_kind}_{args.campaign_id or 'history'}_scorecard"
    ).strip()
    records = load_benchmark_history_records(
        benchmark_root=history_root,
        task_ids=args.task_ids or None,
        campaign_id=args.campaign_id,
    )
    if not records:
        print("No benchmark history records matched the selection.")
        return 1
    scorecard = build_benchmark_history_scorecard(
        scorecard_name=scorecard_name,
        benchmark_kind=args.benchmark_kind,
        benchmark_runs=records,
        campaign_id=args.campaign_id,
    )
    artifacts = save_benchmark_history_scorecard(
        scorecard,
        reports_root=history_root / "reports",
        stem=scorecard_name,
    )
    print(
        json.dumps(
            {
                "scorecard_json": str(artifacts.json_path),
                "scorecard_md": str(artifacts.text_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
