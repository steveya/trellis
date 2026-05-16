"""Run the bounded portfolio-AAD benchmark gate locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trellis.analytics.benchmarking import (
    DEFAULT_PORTFOLIO_AAD_REPORT_ROOT,
    DEFAULT_PORTFOLIO_AAD_REPORT_STEM,
    build_supported_portfolio_aad_benchmark_report,
    save_portfolio_aad_benchmark_report,
)
from trellis.cli_paths import resolve_repo_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / DEFAULT_PORTFOLIO_AAD_REPORT_ROOT),
        help="Directory for local JSON/Markdown benchmark reports.",
    )
    parser.add_argument("--report-name", default=DEFAULT_PORTFOLIO_AAD_REPORT_STEM)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    report = build_supported_portfolio_aad_benchmark_report(
        repeats=args.repeats,
        warmups=args.warmups,
    )
    output_root = resolve_repo_path(
        args.output_root,
        ROOT / DEFAULT_PORTFOLIO_AAD_REPORT_ROOT,
    )
    artifacts = save_portfolio_aad_benchmark_report(
        report,
        root=output_root,
        stem=args.report_name,
    )
    summary = report["summary"]
    print(f"Saved portfolio-AAD benchmark JSON to {artifacts.json_path}")
    print(f"Saved portfolio-AAD benchmark Markdown to {artifacts.text_path}")
    print(
        "Cases={case_count} total_book_size={book_size} "
        "total_factor_count={factor_count} avg_speedup={speedup}x".format(
            case_count=summary["case_count"],
            book_size=summary["total_book_size"],
            factor_count=summary["total_factor_count"],
            speedup=summary["average_relative_speedup"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
