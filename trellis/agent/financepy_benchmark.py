"""FinancePy benchmark task selection, persistence, and reporting helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trellis.agent.benchmark_pilots import get_pilot, get_pilot_task_ids
from trellis.agent.task_manifests import load_task_manifest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINANCEPY_BENCHMARK_ROOT = ROOT / "task_runs" / "financepy_benchmarks"
FRESH_GENERATED_FINANCEPY_PILOT_TASK_IDS = frozenset(get_pilot_task_ids("financepy"))


@dataclass(frozen=True)
class FinancePyBenchmarkArtifacts:
    report: dict[str, Any]
    json_path: Path
    text_path: Path


def financepy_benchmark_execution_policy(
    task: Mapping[str, Any],
    *,
    requested_policy: str = "auto",
) -> str:
    """Resolve the benchmark execution policy for one task."""
    policy = str(requested_policy or "auto").strip().lower()
    if policy not in {"auto", "cached_existing", "fresh_generated"}:
        raise ValueError(f"Unsupported FinancePy execution policy: {requested_policy}")
    if policy != "auto":
        return policy
    task_id = str(task.get("id") or "").strip()
    pilot = get_pilot("financepy")
    if task_id in pilot.task_ids:
        return pilot.execution_policy
    return "cached_existing"


def load_financepy_benchmark_tasks(*, root: Path = ROOT) -> list[dict[str, Any]]:
    """Load the FinancePy parity corpus."""
    return load_task_manifest("TASKS_BENCHMARK_FINANCEPY.yaml", root=root)


def select_financepy_benchmark_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    requested_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    requested = {
        str(task_id).strip()
        for task_id in (requested_ids or ())
        if str(task_id).strip()
    }
    selected: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        if not task_id:
            continue
        if requested and task_id not in requested:
            continue
        selected.append(dict(task))
        if limit is not None and len(selected) >= max(limit, 0):
            break
    return selected


def persist_financepy_benchmark_record(
    record: Mapping[str, Any],
    *,
    root: Path = DEFAULT_FINANCEPY_BENCHMARK_ROOT,
) -> dict[str, str]:
    """Persist one benchmark record to append-only history and latest pointers."""
    task_id = str(record.get("task_id") or "unknown")
    run_id = str(record.get("run_id") or "")
    history_dir = root / "history" / task_id
    latest_dir = root / "latest"
    history_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    history_path = history_dir / f"{run_id}.json"
    latest_path = latest_dir / f"{task_id}.json"
    payload = json.dumps(dict(record), indent=2, default=str)
    history_path.write_text(payload)
    latest_path.write_text(payload)
    return {
        "history_path": str(history_path),
        "latest_path": str(latest_path),
    }


def extract_trellis_benchmark_outputs(
    result: Mapping[str, Any],
    warm_benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract comparable Trellis outputs from a benchmark run.

    The cold agent run is the source of truth for parity. Warm benchmark runs
    are only a runtime probe and should only backfill missing values.
    """
    outputs: dict[str, Any] = {}
    _merge_benchmark_output_candidate(outputs, result)
    payload = dict(result.get("result") or {})
    _merge_benchmark_output_candidate(outputs, payload)
    _merge_benchmark_output_candidate(outputs, warm_benchmark)
    return outputs


def _merge_benchmark_output_candidate(
    outputs: dict[str, Any],
    payload: Mapping[str, Any] | None,
) -> None:
    if not isinstance(payload, Mapping):
        return

    for key in ("price", "fair_value", "last_price"):
        if payload.get(key) is not None:
            outputs.setdefault("price", payload[key])

    benchmark_outputs = dict(payload.get("benchmark_outputs") or {})
    for key, value in benchmark_outputs.items():
        outputs.setdefault(key, value)

    # Bump-and-reprice Greeks from the warm probe (QUA-863).  They are merged
    # top-level via ``setdefault`` so native cold-run emissions always win and
    # the fallback only fills declared Greeks the payoff didn't produce.
    benchmark_greeks = dict(payload.get("benchmark_greeks") or {})
    for key, value in benchmark_greeks.items():
        outputs.setdefault(key, value)

    greeks = dict(payload.get("greeks") or {})
    if greeks:
        outputs.setdefault("greeks", greeks)

    summary = dict(payload.get("summary") or {})
    for key, value in dict(summary.get("prices") or {}).items():
        outputs.setdefault(key, value)
    summary_greeks = dict(summary.get("greeks") or {})
    if summary_greeks:
        outputs.setdefault("greeks", summary_greeks)

    comparison = dict(payload.get("comparison") or {})
    comparison_prices = dict(comparison.get("prices") or {})
    if not comparison_prices:
        comparison_summary = dict(comparison.get("summary") or {})
        comparison_prices = dict(comparison_summary.get("prices") or {})
        reference_target = str(comparison_summary.get("reference_target") or "").strip()
        if reference_target and reference_target in comparison_prices:
            outputs.setdefault("price", comparison_prices[reference_target])
        elif len(comparison_prices) == 1:
            outputs.setdefault("price", next(iter(comparison_prices.values())))
    for key, value in comparison_prices.items():
        outputs.setdefault(key, value)
    comparison_greeks = dict(comparison.get("greeks") or {})
    if not comparison_greeks:
        comparison_greeks = dict(dict(comparison.get("summary") or {}).get("greeks") or {})
    if comparison_greeks:
        outputs.setdefault("greeks", comparison_greeks)


def build_financepy_benchmark_report(
    *,
    benchmark_name: str,
    git_revision: str,
    benchmark_runs: Sequence[Mapping[str, Any]],
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a run-level FinancePy benchmark report."""
    successes = sum(1 for run in benchmark_runs if run.get("status") == "priced")
    comparison_ok = sum(
        1
        for run in benchmark_runs
        if str(((run.get("comparison_summary") or {}).get("status") or "")).strip() == "passed"
    )
    token_total = sum(
        int(dict(run.get("cold_agent_token_usage") or {}).get("total_tokens") or 0)
        for run in benchmark_runs
    )
    cold_total = sum(float(run.get("cold_agent_elapsed_seconds") or 0.0) for run in benchmark_runs)
    warm_total = sum(float(run.get("warm_agent_mean_seconds") or 0.0) for run in benchmark_runs)
    financepy_total = sum(float(run.get("financepy_elapsed_seconds") or 0.0) for run in benchmark_runs)
    return {
        "benchmark_name": benchmark_name,
        "git_revision": git_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(benchmark_runs),
        "priced_count": successes,
        "comparison_pass_count": comparison_ok,
        "token_usage_total": token_total,
        "cold_elapsed_seconds_total": round(cold_total, 6),
        "warm_elapsed_seconds_total": round(warm_total, 6),
        "financepy_elapsed_seconds_total": round(financepy_total, 6),
        "tasks": [dict(run) for run in benchmark_runs],
        "notes": list(notes or ()),
    }


def render_financepy_benchmark_report(report: Mapping[str, Any]) -> str:
    """Render the benchmark report as Markdown."""
    lines = [
        f"# FinancePy Benchmark: `{report['benchmark_name']}`",
        f"- Git revision: `{report.get('git_revision', '')}`",
        f"- Tasks: `{report.get('task_count', 0)}`",
        f"- Priced: `{report.get('priced_count', 0)}`",
        f"- Comparison pass count: `{report.get('comparison_pass_count', 0)}`",
        f"- Cold elapsed total: `{report.get('cold_elapsed_seconds_total', 0)}`",
        f"- Warm elapsed total: `{report.get('warm_elapsed_seconds_total', 0)}`",
        f"- FinancePy elapsed total: `{report.get('financepy_elapsed_seconds_total', 0)}`",
        f"- Total tokens: `{report.get('token_usage_total', 0)}`",
    ]
    if report.get("notes"):
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in report["notes"])
    lines.extend(["", "## Task Runs"])
    for task in report.get("tasks") or []:
        comparison = dict(task.get("comparison_summary") or {})
        lines.extend(
            [
                "",
                f"### `{task.get('task_id', '')}` {task.get('title', '')}",
                f"- Status: `{task.get('status', '')}`",
                f"- Started: `{task.get('run_started_at', '')}`",
                f"- Completed: `{task.get('run_completed_at', '')}`",
                f"- Cold elapsed: `{task.get('cold_agent_elapsed_seconds', 0)}`",
                f"- Warm elapsed: `{task.get('warm_agent_mean_seconds', 0)}`",
                f"- FinancePy elapsed: `{task.get('financepy_elapsed_seconds', 0)}`",
                f"- Comparison: `{comparison.get('status', 'not_attempted')}`",
                f"- Compared outputs: `{', '.join(comparison.get('compared_outputs', ())) or 'none'}`",
            ]
        )
    return "\n".join(lines) + "\n"


def save_financepy_benchmark_report(
    report: Mapping[str, Any],
    *,
    root: Path = DEFAULT_FINANCEPY_BENCHMARK_ROOT,
    stem: str,
) -> FinancePyBenchmarkArtifacts:
    """Persist a run-level benchmark report."""
    reports_root = root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    json_path = reports_root / f"{stem}.json"
    text_path = reports_root / f"{stem}.md"
    json_path.write_text(json.dumps(dict(report), indent=2, default=str))
    text_path.write_text(render_financepy_benchmark_report(report))
    return FinancePyBenchmarkArtifacts(
        report=dict(report),
        json_path=json_path,
        text_path=text_path,
    )
