"""Run the FinancePy parity benchmark with timestamped run-history persistence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "openai")

from trellis.agent.config import load_env
from trellis.agent.benchmark_history import (
    build_benchmark_history_scorecard,
    load_benchmark_history_records,
    save_benchmark_history_scorecard,
)
from trellis.agent.financepy_parity import (
    financepy_binding_for_task,
    normalize_benchmark_outputs,
)
from trellis.agent.financepy_benchmark import (
    DEFAULT_FINANCEPY_BENCHMARK_ROOT,
    build_financepy_benchmark_report,
    extract_trellis_benchmark_outputs,
    financepy_benchmark_execution_policy,
    load_financepy_benchmark_tasks,
    persist_financepy_benchmark_record,
    save_financepy_benchmark_report,
    select_financepy_benchmark_tasks,
)
from trellis.agent.benchmark_runner_dispatch import dispatch_benchmark_tasks
from trellis.agent.financepy_output_comparison import compare_benchmark_outputs
from trellis.agent.fresh_generated_boundary import (
    FreshGeneratedBoundaryError,
    enforce_fresh_generated_boundary,
)
from trellis.agent.runtime_cleanliness import (
    DirtyWorkingTreeError,
    enforce_clean_tree,
)
from trellis.agent.financepy_reference import price_financepy_reference
from trellis.agent.knowledge.promotion import record_benchmark_promotion_candidate
from trellis.agent.runtime_revisions import runtime_revision_metadata
from trellis.agent.task_runtime import (
    benchmark_generated_artifact,
    benchmark_existing_task,
    benchmark_spec_overrides,
    build_market_state,
    build_market_state_for_task,
    run_task,
)

load_env()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="*", help="Optional FinancePy benchmark task ids.")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--validation", default="standard")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--skip-financepy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--execution-policy",
        choices=("auto", "cached_existing", "fresh_generated"),
        default="auto",
    )
    parser.add_argument("--output-root")
    parser.add_argument("--report-name", default="financepy_parity")
    parser.add_argument("--campaign-id")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Allow the run even if the working tree has uncommitted source "
            "edits; records `working_tree.clean=False` and the dirty paths "
            "so downstream admission can distinguish untrusted runs."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of concurrent task workers (default 1).  Cold builds "
            "share no state, so `--workers N` for the pilot scales wall "
            "clock with the OpenAI per-key concurrency limit."
        ),
    )
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


def _read_generated_artifact_source(generated_artifact: dict[str, Any]) -> str | None:
    """Best-effort read of the persisted fresh-build source for boundary inspection."""
    file_path_text = str(generated_artifact.get("file_path") or "").strip()
    if not file_path_text:
        return None
    file_path = Path(file_path_text)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    try:
        return file_path.read_text(encoding="utf-8")
    except OSError:
        return None


_compare_outputs = compare_benchmark_outputs


def _execute_single_benchmark_task(
    task: dict[str, Any],
    *,
    args: argparse.Namespace,
    git_revision: str,
    knowledge_revision: str,
    campaign_id: str,
    market_state,
    output_root: Path,
    working_tree_status,
) -> dict[str, Any]:
    """Run one benchmark task end-to-end and return its persisted record.

    Factored out so the per-task body can be dispatched serially or via a
    thread pool (QUA-876).  Cold builds share no state across tasks; the
    only ordering constraint is the per-key OpenAI concurrency limit.
    """
    run_started_at = datetime.now(timezone.utc).isoformat()
    execution_policy = financepy_benchmark_execution_policy(
        task,
        requested_policy=args.execution_policy,
    )
    fresh_generated = execution_policy == "fresh_generated"
    cold_result = run_task(
        task,
        market_state,
        model=args.model,
        force_rebuild=args.force_rebuild,
        fresh_build=fresh_generated,
        validation=args.validation,
    )
    warm_result: dict[str, Any] | None = None
    financepy_result: dict[str, Any] | None = None
    comparison_summary: dict[str, Any]
    status = "failed"
    task_market_state, _ = build_market_state_for_task(task, market_state)
    generated_artifact = dict(cold_result.get("generated_artifact") or {})
    # Only read the generated source when we actually need to inspect it for
    # the boundary check.  Cached-existing runs skip the I/O entirely (the
    # enforcer returns `not_applicable` early for non-`fresh_generated`
    # policies).  (PR #590 round-4 Copilot review.)
    generated_source = (
        _read_generated_artifact_source(generated_artifact)
        if fresh_generated
        else None
    )
    boundary_check = enforce_fresh_generated_boundary(
        task,
        generated_artifact if generated_artifact else None,
        execution_policy=execution_policy,
        generated_source=generated_source,
        raise_on_violation=False,
    )

    if cold_result.get("success"):
        status = "priced"
        try:
            if fresh_generated:
                if boundary_check.status == "violated":
                    raise FreshGeneratedBoundaryError(
                        f"QUA-866: fresh-generated boundary violation for task "
                        f"{task['id']}: {boundary_check.reason}"
                    )
                warm_result = benchmark_generated_artifact(
                    task,
                    generated_artifact=generated_artifact,
                    market_state=task_market_state,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    spec_overrides=benchmark_spec_overrides(task),
                )
            else:
                warm_result = benchmark_existing_task(
                    task,
                    market_state=task_market_state,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    model=args.model,
                    spec_overrides=benchmark_spec_overrides(task),
                )
        except Exception as exc:
            warm_result = {"error": str(exc)}
    if not args.skip_financepy:
        try:
            financepy_result = price_financepy_reference(task, root=ROOT)
        except Exception as exc:
            financepy_result = {"error": str(exc)}

    binding = financepy_binding_for_task(task, root=ROOT)
    trellis_outputs = extract_trellis_benchmark_outputs(cold_result, warm_result)
    trellis_outputs_normalized = normalize_benchmark_outputs(
        task,
        trellis_outputs,
        source="trellis",
        root=ROOT,
    )
    financepy_outputs = {} if financepy_result is None else dict(financepy_result.get("outputs") or {})
    financepy_outputs_normalized = normalize_benchmark_outputs(
        task,
        financepy_outputs,
        source="financepy",
        root=ROOT,
    )
    comparison_summary = _compare_outputs(
        task=task,
        binding=binding,
        trellis_outputs=trellis_outputs_normalized,
        financepy_outputs=financepy_outputs_normalized,
    )
    if boundary_check.status == "violated":
        comparison_summary["status"] = "benchmark_boundary_violation"
        comparison_summary["boundary_error"] = boundary_check.reason
        comparison_summary["boundary_violations"] = list(boundary_check.violations)
        status = "failed"
    if financepy_result and financepy_result.get("error"):
        # Preserve the boundary-violation status -- it is the more serious
        # signal and downstream admission/scorecards branch on it.  The
        # financepy error is still recorded in its own field so audits
        # keep both pieces of evidence.  (PR #590 Copilot review.)
        if comparison_summary.get("status") != "benchmark_boundary_violation":
            comparison_summary["status"] = "financepy_error"
        comparison_summary["financepy_error"] = financepy_result["error"]
    if warm_result and warm_result.get("error"):
        comparison_summary.setdefault("warnings", []).append(f"warm_benchmark_error: {warm_result['error']}")

    run_completed_at = datetime.now(timezone.utc).isoformat()
    record = {
        "task_id": task["id"],
        "title": task["title"],
        "instrument_type": task.get("instrument_type"),
        "preferred_method": task.get("construct"),
        "task_corpus": task.get("task_corpus"),
        "task_definition_version": task.get("task_definition_version"),
        "task_definition_manifest": task.get("task_definition_manifest"),
        "market_scenario_id": task.get("market_scenario_id"),
        "market_scenario_digest": dict(task.get("market") or {}).get("scenario_digest"),
        "financepy_binding_id": task.get("financepy_binding_id"),
        "benchmark_campaign_id": campaign_id,
        "git_sha": git_revision,
        "knowledge_revision": knowledge_revision,
        "run_id": f"{task['id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
        "run_started_at": run_started_at,
        "run_completed_at": run_completed_at,
        "execution_mode": "cold_agent_plus_financepy_reference",
        "benchmark_execution_policy": execution_policy,
        "status": status,
        "cold_agent_elapsed_seconds": round(float(cold_result.get("elapsed_seconds") or 0.0), 6),
        "cold_agent_token_usage": dict(cold_result.get("token_usage_summary") or {}),
        "warm_agent_mean_seconds": None if warm_result is None else warm_result.get("mean_seconds"),
        "warm_agent_last_price": None if warm_result is None else warm_result.get("last_price"),
        "warm_agent_execution_mode": (
            "fresh_generated_artifact" if fresh_generated else "cached_existing_artifact"
        ),
        "financepy_elapsed_seconds": None if financepy_result is None else financepy_result.get("elapsed_seconds"),
        "trellis_outputs": trellis_outputs,
        "trellis_outputs_normalized": trellis_outputs_normalized,
        "financepy_outputs": financepy_outputs,
        "financepy_outputs_normalized": financepy_outputs_normalized,
        "generated_artifact": generated_artifact,
        "fresh_generated_boundary": boundary_check.as_record(),
        "working_tree": working_tree_status.as_record(),
        "comparison_summary": comparison_summary,
    }
    record.update(persist_financepy_benchmark_record(record, root=output_root))
    if fresh_generated and comparison_summary.get("status") == "passed":
        try:
            promotion_candidate_path = record_benchmark_promotion_candidate(
                benchmark_record=record,
            )
        except Exception as exc:
            comparison_summary.setdefault("warnings", []).append(
                f"promotion_candidate_record_error: {exc}"
            )
            promotion_candidate_path = None
        if promotion_candidate_path:
            record["promotion_candidate_path"] = promotion_candidate_path
            record.update(persist_financepy_benchmark_record(record, root=output_root))
    return record


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        working_tree_status = enforce_clean_tree(ROOT, allow_dirty=bool(args.allow_dirty))
    except DirtyWorkingTreeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    tasks = select_financepy_benchmark_tasks(
        load_financepy_benchmark_tasks(root=ROOT),
        requested_ids=args.task_ids or None,
        limit=args.limit,
    )
    if args.list_tasks:
        for task in tasks:
            print(
                f"{task['id']:4s}  {task.get('financepy_binding_id',''):45s}  "
                f"{task.get('title','')}"
            )
        return 0
    if not tasks:
        print("No FinancePy benchmark tasks selected.")
        return 1

    if args.dry_run:
        print(f"Would run {len(tasks)} FinancePy benchmark tasks.")
        for task in tasks:
            print(f"  {task['id']:4s}  {task.get('title','')}")
        return 0

    output_root = Path(args.output_root) if args.output_root else DEFAULT_FINANCEPY_BENCHMARK_ROOT
    revisions = runtime_revision_metadata()
    git_revision = revisions["git_sha"] or _git_revision()
    knowledge_revision = revisions["knowledge_revision"]
    campaign_id = str(args.campaign_id or args.report_name).strip()
    market_state = build_market_state()

    if args.workers < 1:
        print(f"--workers must be >= 1, got {args.workers}", file=sys.stderr)
        return 2

    def _run_one(task: dict[str, Any]) -> dict[str, Any]:
        return _execute_single_benchmark_task(
            task,
            args=args,
            git_revision=git_revision,
            knowledge_revision=knowledge_revision,
            campaign_id=campaign_id,
            market_state=market_state,
            output_root=output_root,
            working_tree_status=working_tree_status,
        )

    benchmark_runs = list(
        dispatch_benchmark_tasks(tasks, _run_one, workers=args.workers)
    )

    report = build_financepy_benchmark_report(
        benchmark_name=args.report_name,
        git_revision=git_revision,
        benchmark_runs=benchmark_runs,
        notes=[
            "Cold timing measures the end-to-end agent path.",
            "Warm timing measures the selected artifact runtime only.",
            "FinancePy comparisons only use overlapping outputs.",
        ],
    )
    artifacts = save_financepy_benchmark_report(report, root=output_root, stem=args.report_name)
    scorecard = build_benchmark_history_scorecard(
        scorecard_name=f"{args.report_name}_scorecard",
        benchmark_kind="financepy",
        benchmark_runs=load_benchmark_history_records(
            benchmark_root=output_root,
            task_ids=[task["id"] for task in tasks],
            campaign_id=campaign_id,
        ),
        campaign_id=campaign_id,
        notes=[
            "Task history is loaded from append-only benchmark records.",
            "A task is counted as passing when the latest FinancePy comparison passes.",
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
