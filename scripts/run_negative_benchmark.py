"""Run the negative clarification / honest-block benchmark with timestamped history persistence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

from trellis.agent.config import load_env
from trellis.agent.benchmark_history import (
    build_benchmark_history_scorecard,
    load_benchmark_history_records,
    save_benchmark_history_scorecard,
)
from trellis.agent.negative_task_benchmark import (
    DEFAULT_NEGATIVE_BENCHMARK_ROOT,
    build_negative_benchmark_report,
    evaluate_negative_task_result,
    load_negative_benchmark_tasks,
    persist_negative_benchmark_record,
    save_negative_benchmark_report,
    select_negative_benchmark_tasks,
)
from trellis.agent.runtime_revisions import runtime_revision_metadata
from trellis.agent.task_runtime import build_market_state, run_task


load_env()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="*", help="Optional negative benchmark task ids.")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--validation", default="standard")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--report-name", default="negative_task_benchmark")
    parser.add_argument("--campaign-id")
    return parser.parse_args(argv)


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    tasks = select_negative_benchmark_tasks(
        load_negative_benchmark_tasks(root=ROOT),
        requested_ids=args.task_ids or None,
        limit=args.limit,
    )
    if args.list_tasks:
        for task in tasks:
            print(
                f"{task['id']:4s}  {task.get('expected_outcome',''):24s}  "
                f"{task.get('title','')}"
            )
        return 0
    if not tasks:
        print("No negative benchmark tasks selected.")
        return 1

    if args.dry_run:
        print(f"Would run {len(tasks)} negative benchmark tasks.")
        for task in tasks:
            print(f"  {task['id']:4s}  {task.get('title','')}")
        return 0

    output_root = Path(args.output_root) if args.output_root else DEFAULT_NEGATIVE_BENCHMARK_ROOT
    revisions = runtime_revision_metadata()
    git_revision = revisions["git_sha"] or _git_revision()
    knowledge_revision = revisions["knowledge_revision"]
    campaign_id = str(args.campaign_id or args.report_name).strip()
    market_state = build_market_state()
    benchmark_runs: list[dict[str, object]] = []

    for task in tasks:
        run_started_at = datetime.now(timezone.utc).isoformat()
        result = run_task(
            task,
            market_state,
            model=args.model,
            force_rebuild=True,
            validation=args.validation,
        )
        evaluation = evaluate_negative_task_result(task, result)
        run_completed_at = datetime.now(timezone.utc).isoformat()
        record = {
            "task_id": task["id"],
            "title": task["title"],
            "task_corpus": task.get("task_corpus"),
            "task_definition_version": task.get("task_definition_version"),
            "task_definition_manifest": task.get("task_definition_manifest"),
            "market_scenario_id": task.get("market_scenario_id"),
            "market_scenario_digest": dict(task.get("market") or {}).get("scenario_digest"),
            "benchmark_campaign_id": campaign_id,
            "git_sha": git_revision,
            "knowledge_revision": knowledge_revision,
            "run_id": f"{task['id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
            "run_started_at": run_started_at,
            "run_completed_at": run_completed_at,
            "execution_mode": "cold_agent_negative",
            "status": evaluation["observed_outcome"],
            "success_status": evaluation["observed_outcome"],
            "expected_outcome": evaluation["expected_outcome"],
            "observed_outcome": evaluation["observed_outcome"],
            "passed_expectation": evaluation["passed"],
            "evaluation_details": list(evaluation["details"]),
            "result_bucket": evaluation["result_bucket"],
            "observed_blocker_categories": list(evaluation["observed_blocker_categories"]),
            "observed_missing_fields": list(evaluation["observed_missing_fields"]),
            "expected_missing_fields": list(evaluation["expected_missing_fields"]),
            "expected_blockers": list(evaluation["expected_blockers"]),
            "elapsed_seconds": round(float(result.get("elapsed_seconds") or 0.0), 6),
            "token_usage_summary": dict(result.get("token_usage_summary") or {}),
            "task_run_history_path": result.get("task_run_history_path"),
            "task_run_latest_path": result.get("task_run_latest_path"),
        }
        record.update(persist_negative_benchmark_record(record, root=output_root))
        benchmark_runs.append(record)

    report = build_negative_benchmark_report(
        benchmark_name=args.report_name,
        git_revision=git_revision,
        benchmark_runs=benchmark_runs,
        notes=[
            "Negative tasks measure whether Trellis asks for clarification or blocks honestly.",
            "Every record is append-only and timestamped for repeated rerun comparison.",
        ],
    )
    artifacts = save_negative_benchmark_report(report, root=output_root, stem=args.report_name)
    scorecard = build_benchmark_history_scorecard(
        scorecard_name=f"{args.report_name}_scorecard",
        benchmark_kind="negative",
        benchmark_runs=load_benchmark_history_records(
            benchmark_root=output_root,
            task_ids=[task["id"] for task in tasks],
            campaign_id=campaign_id,
        ),
        campaign_id=campaign_id,
        notes=[
            "Negative-task history is loaded from append-only benchmark records.",
            "A task is counted as passing when the latest run matches the expected clarification or honest-block behavior.",
        ],
    )
    scorecard_artifacts = save_benchmark_history_scorecard(
        scorecard,
        reports_root=output_root / "reports",
        stem=f"{args.report_name}_scorecard",
    )
    print(
        json.dumps(
            {
                "report_json": str(artifacts.json_path),
                "report_md": str(artifacts.text_path),
                "scorecard_json": str(scorecard_artifacts.json_path),
                "scorecard_md": str(scorecard_artifacts.text_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
