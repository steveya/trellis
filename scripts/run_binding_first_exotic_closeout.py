"""Summarize the binding-first exotic proof program from cohort reports.

Usage:
    /Users/steveyang/miniforge3/bin/python3 scripts/run_binding_first_exotic_closeout.py \
        --report-json /tmp/qua808_report_live.json \
        --report-json /tmp/qua809_report_live.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trellis.agent.evals import (
    render_binding_first_exotic_program_closeout,
    summarize_binding_first_exotic_program_closeout,
)
from trellis.cli_paths import resolve_repo_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json",
        action="append",
        dest="report_json_paths",
        required=True,
        help="Path to one cohort report JSON. May be repeated.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args(argv)


def run_binding_first_exotic_closeout(
    *,
    report_json_paths: list[Path],
    output_json_path: Path,
    output_md_path: Path,
) -> int:
    reports = [json.loads(path.read_text()) for path in report_json_paths]
    summary = summarize_binding_first_exotic_program_closeout(reports)
    summary["source_reports"] = [path.name for path in report_json_paths]
    output_json_path.write_text(json.dumps(summary, indent=2, default=str))
    output_md_path.write_text(render_binding_first_exotic_program_closeout(summary))
    print(f"Closeout JSON saved to: {output_json_path}")
    print(f"Closeout Markdown saved to: {output_md_path}")
    return 0


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = resolve_repo_path(
        args.output_json,
        ROOT / f"task_results_binding_first_exotic_closeout_{timestamp}.json",
    )
    output_md_path = resolve_repo_path(
        args.output_md,
        ROOT / f"task_results_binding_first_exotic_closeout_{timestamp}.md",
    )
    return run_binding_first_exotic_closeout(
        report_json_paths=[resolve_repo_path(path) for path in args.report_json_paths],
        output_json_path=output_json_path,
        output_md_path=output_md_path,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
