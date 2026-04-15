"""Generic parallel/serial dispatch helper for benchmark runners.

Per-corpus benchmark runners (FinancePy today, negative/extension/proof later)
all share the same shape: iterate a list of task dicts, run an end-to-end
function per task, and persist the returned record in input order.  The
dispatch logic is identical and trivially parallelizable -- cold builds share
no state across tasks, the only cap is per-key OpenAI concurrency.

This module factors the dispatch out so each runner just supplies its
per-task callable and a worker count.  ``--workers 1`` keeps the existing
serial behavior; ``--workers N`` runs concurrently, preserves input order in
the persisted record list, and keeps surfacing per-task failures rather than
aborting the whole batch.

Refs: QUA-876 (epic QUA-869).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


TaskRunner = Callable[[Mapping[str, Any]], Mapping[str, Any]]
TaskFailureHook = Callable[[Mapping[str, Any], BaseException], Mapping[str, Any]]


def _default_failure_hook(
    task: Mapping[str, Any], exc: BaseException
) -> Mapping[str, Any]:
    """Fallback record when a per-task callable raises."""
    return {
        "task_id": str(task.get("id") or ""),
        "title": str(task.get("title") or ""),
        "status": "errored",
        "error": f"{type(exc).__name__}: {exc}",
    }


def dispatch_benchmark_tasks(
    tasks: Sequence[Mapping[str, Any]],
    runner: TaskRunner,
    *,
    workers: int = 1,
    on_task_failed: TaskFailureHook | None = None,
) -> list[Mapping[str, Any]]:
    """Run ``runner`` on each task and return the records in input order.

    ``workers <= 1`` runs the tasks sequentially with no thread pool overhead
    so the existing serial cassettes / replay tooling sees no behavioural
    change.  ``workers > 1`` dispatches concurrently via a
    ``ThreadPoolExecutor`` capped at ``workers``; results are reordered to
    match input order so downstream scorecards stay deterministic regardless
    of completion order.

    A per-task exception is caught, fed through ``on_task_failed`` (default:
    a synthetic ``status='errored'`` record), and the batch continues.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers!r}")
    failure_hook = on_task_failed or _default_failure_hook
    if workers == 1:
        records: list[Mapping[str, Any]] = []
        for task in tasks:
            try:
                records.append(runner(task))
            except Exception as exc:
                records.append(failure_hook(task, exc))
        return records

    indexed_tasks = list(enumerate(tasks))
    record_by_index: dict[int, Mapping[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(runner, task): index
            for index, task in indexed_tasks
        }
        for future in as_completed(futures):
            index = futures[future]
            task = indexed_tasks[index][1]
            try:
                record_by_index[index] = future.result()
            except Exception as exc:
                record_by_index[index] = failure_hook(task, exc)
    return [record_by_index[index] for index, _ in indexed_tasks]


__all__ = (
    "TaskRunner",
    "TaskFailureHook",
    "dispatch_benchmark_tasks",
)
