"""Tests for the write-module path-escape guard (QUA-382).

`write_module` and `_write_generated_module` must refuse to write outside
their expected root directories.  A relative path containing `../` would
otherwise silently escape the package tree via `mkdir(parents=True)`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trellis.agent.builder import (
    TRELLIS_ROOT,
    WriteTargetEscapeError,
    validate_write_target,
    write_module,
)


def test_validate_write_target_accepts_path_inside_root(tmp_path):
    target = tmp_path / "subdir" / "module.py"
    validate_write_target(target, tmp_path, "test")


def test_validate_write_target_rejects_path_outside_root(tmp_path):
    target = tmp_path / ".." / "escape.py"
    with pytest.raises(WriteTargetEscapeError) as exc_info:
        validate_write_target(target, tmp_path, "test")
    message = str(exc_info.value)
    assert "outside" in message.lower()
    # Error message should show the resolved canonical path, not the raw
    # `..`-containing input.  (PR #592 Copilot review.)
    assert ".." not in message.split("): ")[-1]


def test_write_module_rejects_repo_escape():
    with pytest.raises(WriteTargetEscapeError):
        write_module("../../etc/evil.py", "# pwned")


def test_write_module_accepts_normal_relative_path(tmp_path, monkeypatch):
    monkeypatch.setattr("trellis.agent.builder.TRELLIS_ROOT", tmp_path)
    path = write_module("instruments/_agent/test_module.py", "# ok")
    assert path.exists()
    assert path.read_text() == "# ok"
    try:
        path.resolve().relative_to(tmp_path.resolve())
    except ValueError:
        pytest.fail("write_module wrote outside the monkeypatched root")


def test_write_generated_module_rejects_benchmark_artifact_repo_escape(tmp_path, monkeypatch):
    from trellis.agent import executor as executor_mod

    monkeypatch.setattr(executor_mod, "REPO_ROOT", tmp_path)

    escape_path = tmp_path / "task_runs" / ".." / ".." / "etc" / "evil.py"
    with pytest.raises(WriteTargetEscapeError):
        executor_mod._write_generated_module(
            escape_path,
            "task_runs/../../etc/evil.py",
            "# pwned",
        )


def test_write_generated_module_accepts_normal_benchmark_artifact(tmp_path, monkeypatch):
    from trellis.agent import executor as executor_mod

    monkeypatch.setattr(executor_mod, "REPO_ROOT", tmp_path)

    artifact_path = tmp_path / "task_runs" / "financepy_benchmarks" / "gen" / "mod.py"
    result = executor_mod._write_generated_module(
        artifact_path,
        "task_runs/financepy_benchmarks/gen/mod.py",
        "# ok",
    )
    assert result.exists()
    assert result.read_text() == "# ok"
    try:
        result.resolve().relative_to(tmp_path.resolve())
    except ValueError:
        pytest.fail("benchmark artifact escaped the repo root")
