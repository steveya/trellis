"""Tests for repo-root path normalization used by CLI scripts."""

from __future__ import annotations

from trellis.cli_paths import REPO_ROOT, resolve_repo_path


def test_repo_root_points_at_repository_root() -> None:
    assert REPO_ROOT.is_absolute()
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_resolve_repo_path_normalizes_relative_paths() -> None:
    resolved = resolve_repo_path("task_results_demo.json")
    assert resolved == (REPO_ROOT / "task_results_demo.json").resolve()


def test_resolve_repo_path_preserves_absolute_paths(tmp_path) -> None:
    absolute = tmp_path / "nested" / "output.json"
    resolved = resolve_repo_path(absolute)
    assert resolved == absolute.resolve()


def test_resolve_repo_path_uses_default_when_path_missing() -> None:
    resolved = resolve_repo_path(None, "docs/benchmarks")
    assert resolved == (REPO_ROOT / "docs" / "benchmarks").resolve()
