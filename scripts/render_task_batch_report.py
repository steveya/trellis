"""Render a portable Markdown and JSON report from one task-results tranche.

Usage:
    /Users/steveyang/miniforge3/bin/python3 scripts/render_task_batch_report.py \
        task_results_wide_pricing_batch.json \
        --collection-name "Wide pricing batch" \
        --selection-mode all \
        --status all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trellis.agent.task_batch_reports import (
    build_task_batch_report,
    render_task_batch_markdown,
)
from trellis.cli_paths import resolve_repo_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", help="Path to a task_results_*.json tranche.")
    parser.add_argument("--collection-name", default="Task batch report")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--validation", default="standard")
    parser.add_argument(
        "--selection-mode",
        choices=("all", "range", "ids"),
        default="all",
    )
    parser.add_argument(
        "--status",
        choices=("pending", "all"),
        default="pending",
    )
    parser.add_argument("--start-id")
    parser.add_argument("--end-id")
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        default=[],
        help="Selection corpora used for the run. May be repeated.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        default=[],
        help="Exact task ids used for the run. May be repeated.",
    )
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--fresh-build", action="store_true")
    parser.add_argument("--knowledge-light", action="store_true")
    parser.add_argument("--summary-json", help="Optional precomputed summary JSON path.")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument(
        "--step-summary",
        help="Optional compact Markdown path for GitHub step summary use.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    results_path = resolve_repo_path(args.results_json)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError("results_json must contain a JSON list")

    default_json = results_path.with_name(f"{results_path.stem}_report.json")
    default_md = results_path.with_name(f"{results_path.stem}_report.md")
    summary_path = (
        resolve_repo_path(args.summary_json)
        if args.summary_json
        else results_path.with_name(f"{results_path.stem}_summary.json")
    )

    report = build_task_batch_report(
        results,
        collection_name=args.collection_name,
        model=args.model,
        validation=args.validation,
        force_rebuild=not args.reuse,
        fresh_build=args.fresh_build,
        knowledge_light=args.knowledge_light,
        selection={
            "selection_mode": args.selection_mode,
            "status": args.status,
            "corpora": list(args.corpora or ()),
            "requested_task_ids": list(args.task_ids or ()),
            "start_id": args.start_id,
            "end_id": args.end_id,
        },
        raw_results_path=results_path,
        summary_path=summary_path if summary_path.exists() else None,
    )

    output_json = resolve_repo_path(args.output_json, default_json)
    output_md = resolve_repo_path(args.output_md, default_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    output_md.write_text(render_task_batch_markdown(report), encoding="utf-8")

    if args.step_summary:
        step_summary = resolve_repo_path(args.step_summary)
        step_summary.parent.mkdir(parents=True, exist_ok=True)
        step_summary.write_text(
            render_task_batch_markdown(report, max_task_rows=40),
            encoding="utf-8",
        )
        print(f"Step summary saved to: {step_summary}")

    print(f"Report JSON saved to: {output_json}")
    print(f"Report Markdown saved to: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
