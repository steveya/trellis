"""Tests for the parallel benchmark dispatch helper (QUA-876)."""

from __future__ import annotations

import threading
import time

import pytest

from trellis.agent.benchmark_runner_dispatch import dispatch_benchmark_tasks


def _stub_runner(task):
    return {"task_id": task["id"], "result": task["id"].lower()}


def _slow_runner_factory(delay_s: float):
    def runner(task):
        time.sleep(delay_s)
        return {"task_id": task["id"], "thread": threading.get_ident()}

    return runner


def test_serial_dispatch_preserves_input_order():
    tasks = [{"id": tid} for tid in ("F001", "F002", "F003")]
    records = dispatch_benchmark_tasks(tasks, _stub_runner, workers=1)
    assert [rec["task_id"] for rec in records] == ["F001", "F002", "F003"]


def test_parallel_dispatch_preserves_input_order():
    tasks = [{"id": f"F{i:03d}"} for i in range(1, 9)]
    runner = _slow_runner_factory(0.05)
    records = dispatch_benchmark_tasks(tasks, runner, workers=4)
    assert [rec["task_id"] for rec in records] == [task["id"] for task in tasks]


def test_parallel_dispatch_actually_runs_concurrently():
    tasks = [{"id": f"F{i:03d}"} for i in range(1, 7)]
    runner = _slow_runner_factory(0.10)

    serial_start = time.perf_counter()
    dispatch_benchmark_tasks(tasks, runner, workers=1)
    serial_elapsed = time.perf_counter() - serial_start

    parallel_start = time.perf_counter()
    dispatch_benchmark_tasks(tasks, runner, workers=6)
    parallel_elapsed = time.perf_counter() - parallel_start

    # 6 tasks at 100ms each: serial >= 600ms, parallel should be <= 250ms with
    # comfortable margin for thread-pool overhead.
    assert serial_elapsed >= 0.55
    assert parallel_elapsed < serial_elapsed / 2


def test_parallel_dispatch_preserves_serial_record_shape():
    tasks = [{"id": "F001"}, {"id": "F002"}]
    runner = _stub_runner

    serial = dispatch_benchmark_tasks(tasks, runner, workers=1)
    parallel = dispatch_benchmark_tasks(tasks, runner, workers=2)

    assert serial == parallel


def test_dispatch_continues_after_per_task_failure():
    tasks = [{"id": tid} for tid in ("F001", "F002", "F003")]

    def runner(task):
        if task["id"] == "F002":
            raise RuntimeError("synthetic failure")
        return {"task_id": task["id"], "ok": True}

    records = dispatch_benchmark_tasks(tasks, runner, workers=3)

    assert [rec["task_id"] for rec in records] == ["F001", "F002", "F003"]
    assert records[0]["ok"] is True
    assert records[1]["status"] == "errored"
    assert "synthetic failure" in records[1]["error"]
    assert records[2]["ok"] is True


def test_dispatch_uses_custom_failure_hook():
    tasks = [{"id": "F001", "title": "boom"}]

    def runner(task):
        raise ValueError("custom")

    def hook(task, exc):
        return {
            "task_id": task["id"],
            "status": "halted",
            "title": task["title"],
            "exc_type": type(exc).__name__,
        }

    records = dispatch_benchmark_tasks(
        tasks, runner, workers=1, on_task_failed=hook
    )

    assert records == [
        {
            "task_id": "F001",
            "status": "halted",
            "title": "boom",
            "exc_type": "ValueError",
        }
    ]


def test_dispatch_rejects_zero_or_negative_workers():
    tasks = [{"id": "F001"}]
    with pytest.raises(ValueError):
        dispatch_benchmark_tasks(tasks, _stub_runner, workers=0)
    with pytest.raises(ValueError):
        dispatch_benchmark_tasks(tasks, _stub_runner, workers=-2)
