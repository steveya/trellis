"""Portable Markdown and JSON reports for one pricing-task batch."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trellis.agent.evals import classify_task_result, summarize_task_results


ROOT = Path(__file__).resolve().parents[2]


def build_task_batch_report(
    results: Sequence[Mapping[str, Any]],
    *,
    collection_name: str,
    model: str,
    validation: str,
    force_rebuild: bool,
    fresh_build: bool,
    knowledge_light: bool,
    selection: Mapping[str, Any] | None = None,
    raw_results_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    generated_at: datetime | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build a portable task-batch report from raw task results."""
    generated_at = generated_at or datetime.now(timezone.utc)
    normalized_results = [dict(result) for result in results]
    summary = summarize_task_results(normalized_results)

    task_rows = [
        _build_task_row(result, root=root)
        for result in normalized_results
    ]
    failed_tasks = [row for row in task_rows if not row["success"]]
    successful_tasks = [row for row in task_rows if row["success"]]

    corpus_summary: dict[str, dict[str, int]] = {}
    for row in task_rows:
        corpus = str(row.get("task_corpus") or "unknown").strip() or "unknown"
        bucket = str(row.get("status_bucket") or "unknown").strip() or "unknown"
        entry = corpus_summary.setdefault(
            corpus,
            {"tasks": 0, "successes": 0, "failures": 0},
        )
        entry["tasks"] += 1
        if row["success"]:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        entry[bucket] = int(entry.get(bucket) or 0) + 1

    preferred_methods = Counter(
        str(result.get("preferred_method") or "").strip()
        for result in normalized_results
        if str(result.get("preferred_method") or "").strip()
    )

    report = {
        "report_kind": "task_batch_report",
        "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
        "collection_name": str(collection_name).strip() or "Task batch report",
        "config": {
            "model": str(model).strip(),
            "validation": str(validation).strip(),
            "force_rebuild": bool(force_rebuild),
            "fresh_build": bool(fresh_build),
            "knowledge_light": bool(knowledge_light),
        },
        "selection": _normalize_selection(selection or {}),
        "summary": summary,
        "corpus_summary": dict(sorted(corpus_summary.items())),
        "preferred_methods": dict(sorted(preferred_methods.items())),
        "tasks": task_rows,
        "failed_tasks": failed_tasks,
        "successful_tasks": successful_tasks,
        "artifacts": {
            "raw_results_path": _portable_path_text(raw_results_path, root=root),
            "summary_path": _portable_path_text(summary_path, root=root),
        },
    }
    return report


def render_task_batch_markdown(
    report: Mapping[str, Any],
    *,
    max_task_rows: int | None = None,
) -> str:
    """Render one portable task-batch report as Markdown."""
    config = dict(report.get("config") or {})
    selection = dict(report.get("selection") or {})
    summary = dict(report.get("summary") or {})
    totals = dict(summary.get("totals") or {})
    corpus_summary = dict(report.get("corpus_summary") or {})
    failure_buckets = {
        str(bucket): count
        for bucket, count in dict(summary.get("failure_buckets") or {}).items()
        if str(bucket) != "success"
    }
    reviewer_signals = dict(summary.get("reviewer_signals") or {})
    shared_knowledge = dict(summary.get("shared_knowledge") or {})
    retry_recovery = dict(summary.get("retry_recovery") or {})
    token_usage = dict(summary.get("token_usage") or {})
    tasks = list(report.get("tasks") or [])
    failed_tasks = list(report.get("failed_tasks") or [])
    displayed_tasks = tasks[:max_task_rows] if max_task_rows is not None else tasks

    lines = [
        f"# {report.get('collection_name') or 'Task Batch Report'}",
        "",
        f"- Generated at: `{report.get('generated_at') or ''}`",
        f"- Model: `{config.get('model') or ''}`",
        f"- Validation: `{config.get('validation') or ''}`",
        f"- Force rebuild: `{bool(config.get('force_rebuild'))}`",
        f"- Fresh build: `{bool(config.get('fresh_build'))}`",
        f"- Knowledge light: `{bool(config.get('knowledge_light'))}`",
        "",
        "## Selection",
        "",
        f"- Mode: `{selection.get('selection_mode') or 'all'}`",
        f"- Status filter: `{selection.get('status') or 'pending'}`",
        f"- Requested task count: `{len(selection.get('requested_task_ids') or [])}`",
        f"- Matched task count: `{totals.get('tasks', 0)}`",
    ]

    corpora = tuple(selection.get("corpora") or ())
    if corpora:
        lines.append(f"- Corpora: {', '.join(f'`{corpus}`' for corpus in corpora)}")
    start_id = selection.get("start_id")
    end_id = selection.get("end_id")
    if start_id and end_id:
        lines.append(f"- Range: `{start_id}` to `{end_id}`")
    requested_task_ids = tuple(selection.get("requested_task_ids") or ())
    if requested_task_ids:
        lines.append(
            "- Task ids: " + ", ".join(f"`{task_id}`" for task_id in requested_task_ids)
        )
    artifacts = dict(report.get("artifacts") or {})
    raw_results_path = str(artifacts.get("raw_results_path") or "").strip()
    summary_path = str(artifacts.get("summary_path") or "").strip()
    if raw_results_path:
        lines.append(f"- Raw results artifact: `{raw_results_path}`")
    if summary_path:
        lines.append(f"- Summary artifact: `{summary_path}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Tasks | {totals.get('tasks', 0)} |",
            f"| Successes | {totals.get('successes', 0)} |",
            f"| Failures | {totals.get('failures', 0)} |",
            f"| Avg attempts | {totals.get('avg_attempts', 0.0)} |",
            f"| Successful after retry | {retry_recovery.get('successful_after_retry', 0)} |",
            f"| First-attempt successes | {retry_recovery.get('first_attempt_successes', 0)} |",
            f"| Tasks with reviewer issues | {reviewer_signals.get('tasks_with_reviewer_issues', 0)} |",
            f"| Tasks with shared context | {shared_knowledge.get('tasks_with_shared_context', 0)} |",
            f"| Total tokens | {token_usage.get('total_tokens', 0)} |",
        ]
    )

    lines.extend(["", "## Corpus Summary", ""])
    if corpus_summary:
        lines.extend(
            [
                "| Corpus | Tasks | Successes | Failures |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for corpus, payload in sorted(corpus_summary.items()):
            lines.append(
                f"| `{_md_inline(corpus)}` | {int(payload.get('tasks') or 0)} | "
                f"{int(payload.get('successes') or 0)} | {int(payload.get('failures') or 0)} |"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Failure Buckets", ""])
    if failure_buckets:
        for bucket, count in sorted(failure_buckets.items()):
            lines.append(f"- `{bucket}`: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "## Gap Signals", ""])
    if failed_tasks:
        for row in failed_tasks:
            headline = str(row.get("diagnosis_headline") or "").strip()
            next_action = str(row.get("diagnosis_next_action") or "").strip()
            bucket = str(row.get("status_bucket") or "").strip() or "failed"
            line = f"- `{row.get('task_id')}` `{bucket}`: {headline or row.get('title') or ''}"
            if next_action:
                line += f" Next: {next_action}"
            lines.append(line)
    else:
        lines.append("- No failing tasks in this batch.")

    lines.extend(["", "## Task Results", ""])
    if displayed_tasks:
        lines.extend(
            [
                "| Task | Corpus | Outcome | Attempts | Elapsed (s) | Tokens | Diagnosis |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in displayed_tasks:
            outcome = "success" if row["success"] else str(row.get("status_bucket") or "failed")
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_md_inline(str(row.get('task_id') or ''))}`",
                        f"`{_md_inline(str(row.get('task_corpus') or ''))}`",
                        f"`{_md_inline(outcome)}`",
                        str(int(row.get("attempts") or 0)),
                        _format_elapsed(row.get("elapsed_seconds")),
                        str(int(row.get("token_usage_total") or 0)),
                        _md_table_cell(
                            _truncate(
                                str(row.get("diagnosis_headline") or row.get("title") or ""),
                                limit=120,
                            )
                        ),
                    ]
                )
                + " |"
            )
        if max_task_rows is not None and len(tasks) > len(displayed_tasks):
            lines.extend(
                [
                    "",
                    (
                        f"_Showing the first {len(displayed_tasks)} task rows out of "
                        f"{len(tasks)}. See the JSON artifact for the full batch._"
                    ),
                ]
            )
    else:
        lines.append("- None")

    return "\n".join(lines).rstrip() + "\n"


def _build_task_row(result: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    bucket = classify_task_result(result)
    return {
        "task_id": str(result.get("task_id") or "").strip(),
        "title": str(result.get("title") or "").strip(),
        "task_corpus": str(result.get("task_corpus") or "").strip(),
        "success": bool(result.get("success")),
        "status_bucket": bucket,
        "comparison_task": bool(result.get("comparison_task")),
        "preferred_method": str(result.get("preferred_method") or "").strip(),
        "attempts": int(result.get("attempts") or 0),
        "elapsed_seconds": round(float(result.get("elapsed_seconds") or 0.0), 2),
        "token_usage_total": int(
            dict(result.get("token_usage_summary") or {}).get("total_tokens") or 0
        ),
        "diagnosis_failure_bucket": str(
            result.get("task_diagnosis_failure_bucket") or ""
        ).strip(),
        "diagnosis_headline": str(result.get("task_diagnosis_headline") or "").strip(),
        "diagnosis_next_action": str(result.get("task_diagnosis_next_action") or "").strip(),
        "history_path": _portable_path_text(result.get("task_run_history_path"), root=root),
        "diagnosis_packet_path": _portable_path_text(
            result.get("task_diagnosis_packet_path"),
            root=root,
        ),
    }


def _normalize_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    corpora = tuple(
        str(corpus).strip()
        for corpus in (selection.get("corpora") or ())
        if str(corpus).strip()
    )
    requested_task_ids = tuple(
        str(task_id).strip()
        for task_id in (selection.get("requested_task_ids") or ())
        if str(task_id).strip()
    )
    return {
        "selection_mode": str(selection.get("selection_mode") or "all").strip() or "all",
        "status": str(selection.get("status") or "pending").strip() or "pending",
        "corpora": corpora,
        "requested_task_ids": requested_task_ids,
        "start_id": str(selection.get("start_id") or "").strip() or None,
        "end_id": str(selection.get("end_id") or "").strip() or None,
    }


def _portable_path_text(path_value: str | Path | None, *, root: Path) -> str:
    if path_value is None:
        return ""
    text = str(path_value).strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        return path.as_posix()
    try:
        relative = path.resolve().relative_to(root.resolve())
    except Exception:
        return path.name
    return relative.as_posix()


def _format_elapsed(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "0.0"
    return f"{numeric:.1f}"


def _truncate(text: str, *, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


def _md_inline(text: str) -> str:
    return str(text).replace("`", "'")


def _md_table_cell(text: str) -> str:
    return _md_inline(text).replace("|", "\\|").replace("\n", " ")
