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
from trellis.agent.financepy_benchmark import (
    DEFAULT_FINANCEPY_BENCHMARK_ROOT,
    build_financepy_benchmark_report,
    extract_trellis_benchmark_outputs,
    load_financepy_benchmark_tasks,
    persist_financepy_benchmark_record,
    save_financepy_benchmark_report,
    select_financepy_benchmark_tasks,
)
from trellis.agent.financepy_reference import price_financepy_reference
from trellis.agent.runtime_revisions import runtime_revision_metadata
from trellis.agent.task_manifests import load_financepy_bindings
from trellis.agent.task_runtime import (
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
    parser.add_argument("--output-root")
    parser.add_argument("--report-name", default="financepy_parity")
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


def _output_value(outputs: dict[str, Any], key: str) -> Any:
    if key in outputs:
        return outputs.get(key)
    greeks = outputs.get("greeks") or {}
    if isinstance(greeks, dict):
        return greeks.get(key)
    return None


def _compare_outputs(
    *,
    task: dict[str, Any],
    binding: dict[str, Any],
    trellis_outputs: dict[str, Any],
    financepy_outputs: dict[str, Any] | None,
) -> dict[str, Any]:
    tolerance_pct = float(((task.get("cross_validate") or {}).get("tolerance_pct") or 5.0))
    financepy_outputs = financepy_outputs or {}
    overlapping_outputs = tuple(
        str(name).strip()
        for name in (binding.get("overlapping_outputs") or ())
        if str(name).strip()
    )
    compared_outputs: list[str] = []
    output_deltas: dict[str, float] = {}
    failures: list[str] = []
    for output_name in overlapping_outputs:
        trellis_value = _output_value(trellis_outputs, output_name)
        financepy_value = _output_value(financepy_outputs, output_name)
        if trellis_value is None or financepy_value is None:
            continue
        compared_outputs.append(output_name)
        denominator = max(abs(float(financepy_value)), 1e-12)
        deviation_pct = abs(float(trellis_value) - float(financepy_value)) / denominator * 100.0
        output_deltas[output_name] = round(deviation_pct, 6)
        if deviation_pct > tolerance_pct:
            failures.append(output_name)
    if not compared_outputs:
        return {
            "status": "insufficient_overlap",
            "tolerance_pct": tolerance_pct,
            "compared_outputs": (),
            "expected_overlapping_outputs": overlapping_outputs,
            "trellis_outputs": trellis_outputs,
            "financepy_outputs": financepy_outputs,
        }
    return {
        "status": "passed" if not failures else "failed",
        "tolerance_pct": tolerance_pct,
        "compared_outputs": tuple(compared_outputs),
        "expected_overlapping_outputs": overlapping_outputs,
        "output_deviation_pct": output_deltas,
        "trellis_outputs": trellis_outputs,
        "financepy_outputs": financepy_outputs,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
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
    financepy_bindings = load_financepy_bindings(root=ROOT)
    market_state = build_market_state()
    benchmark_runs: list[dict[str, Any]] = []

    for task in tasks:
        run_started_at = datetime.now(timezone.utc).isoformat()
        cold_result = run_task(
            task,
            market_state,
            model=args.model,
            force_rebuild=True,
            validation=args.validation,
        )
        warm_result: dict[str, Any] | None = None
        financepy_result: dict[str, Any] | None = None
        comparison_summary: dict[str, Any]
        status = "failed"
        task_market_state, _ = build_market_state_for_task(task, market_state)

        if cold_result.get("success"):
            status = "priced"
            try:
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

        trellis_outputs = extract_trellis_benchmark_outputs(cold_result, warm_result)
        comparison_summary = _compare_outputs(
            task=task,
            binding=financepy_bindings.get(str(task.get("financepy_binding_id") or ""), {}),
            trellis_outputs=trellis_outputs,
            financepy_outputs=(financepy_result or {}).get("outputs") if financepy_result else None,
        )
        if financepy_result and financepy_result.get("error"):
            comparison_summary["status"] = "financepy_error"
            comparison_summary["financepy_error"] = financepy_result["error"]
        if warm_result and warm_result.get("error"):
            comparison_summary.setdefault("warnings", []).append(f"warm_benchmark_error: {warm_result['error']}")

        run_completed_at = datetime.now(timezone.utc).isoformat()
        record = {
            "task_id": task["id"],
            "title": task["title"],
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
            "status": status,
            "cold_agent_elapsed_seconds": round(float(cold_result.get("elapsed_seconds") or 0.0), 6),
            "cold_agent_token_usage": dict(cold_result.get("token_usage_summary") or {}),
            "warm_agent_mean_seconds": None if warm_result is None else warm_result.get("mean_seconds"),
            "warm_agent_last_price": None if warm_result is None else warm_result.get("last_price"),
            "financepy_elapsed_seconds": None if financepy_result is None else financepy_result.get("elapsed_seconds"),
            "trellis_outputs": trellis_outputs,
            "financepy_outputs": {} if financepy_result is None else dict(financepy_result.get("outputs") or {}),
            "comparison_summary": comparison_summary,
        }
        record.update(persist_financepy_benchmark_record(record, root=output_root))
        benchmark_runs.append(record)

    report = build_financepy_benchmark_report(
        benchmark_name=args.report_name,
        git_revision=git_revision,
        benchmark_runs=benchmark_runs,
        notes=[
            "Cold timing measures the end-to-end agent path.",
            "Warm timing measures the checked-in/generated payoff runtime only.",
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
