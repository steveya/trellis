"""Repo-root path helpers for CLI scripts and path-sensitive tests."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Return the absolute repository root."""
    return REPO_ROOT


def resolve_repo_path(path: str | Path | None, default: str | Path | None = None) -> Path:
    """Resolve a path relative to the repo root unless it is already absolute."""
    candidate: str | Path | None
    if path is None or (isinstance(path, str) and not path.strip()):
        candidate = default
    else:
        candidate = path

    if candidate is None:
        raise ValueError("path is required")

    resolved = Path(candidate).expanduser()
    if resolved.is_absolute():
        return resolved.resolve()
    return (REPO_ROOT / resolved).resolve()
