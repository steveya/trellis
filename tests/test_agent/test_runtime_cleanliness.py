"""Tests for the benchmark-runner working-tree cleanliness gate (QUA-871)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from trellis.agent.runtime_cleanliness import (
    DirtyWorkingTreeError,
    WorkingTreeStatus,
    enforce_clean_tree,
    inspect_working_tree,
)


def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "seed.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "seed"],
        cwd=tmp_path,
        check=True,
        env={
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    )
    return tmp_path


def test_inspect_working_tree_reports_clean_repo(tmp_path):
    _init_repo(tmp_path)
    status = inspect_working_tree(tmp_path)
    assert isinstance(status, WorkingTreeStatus)
    assert status.clean is True
    assert status.dirty_paths == ()


def test_inspect_working_tree_reports_dirty_source_file(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "module.py").write_text("# dirty\n")

    status = inspect_working_tree(tmp_path)

    assert status.clean is False
    assert "module.py" in status.dirty_paths


def test_inspect_working_tree_ignores_task_runs_and_traces(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "task_runs").mkdir()
    (tmp_path / "task_runs" / "example.json").write_text("{}")
    traces = tmp_path / "trellis" / "agent" / "knowledge" / "traces"
    traces.mkdir(parents=True)
    (traces / "event.ndjson").write_text("{}\n")
    (tmp_path / "trellis.egg-info").mkdir()
    (tmp_path / "trellis.egg-info" / "PKG-INFO").write_text("metadata\n")

    status = inspect_working_tree(tmp_path)
    assert status.clean is True


def test_inspect_working_tree_on_non_git_directory_is_clean(tmp_path):
    status = inspect_working_tree(tmp_path)
    assert status.clean is True
    assert status.dirty_paths == ()


def test_enforce_clean_tree_raises_on_dirty_source(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "real_source.py").write_text("# dirty\n")

    with pytest.raises(DirtyWorkingTreeError) as exc_info:
        enforce_clean_tree(tmp_path)

    assert "real_source.py" in str(exc_info.value)


def test_enforce_clean_tree_returns_status_when_clean(tmp_path):
    _init_repo(tmp_path)
    status = enforce_clean_tree(tmp_path)
    assert status.clean is True


def test_enforce_clean_tree_allows_dirty_when_opted_in(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "real_source.py").write_text("# dirty\n")

    status = enforce_clean_tree(tmp_path, allow_dirty=True)

    assert status.clean is False
    assert "real_source.py" in status.dirty_paths


def test_status_as_record_round_trips():
    status = WorkingTreeStatus(clean=False, dirty_paths=("foo.py", "bar.py"))
    record = status.as_record()
    assert record == {"clean": False, "dirty_paths": ["foo.py", "bar.py"]}
