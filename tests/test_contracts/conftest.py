"""Tier 2 contract test fixtures (QUA-427).

Provides cassette-aware fixtures and helpers for canary contract tests.
All contract tests run with cassette replay (zero tokens) by default.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CASSETTES_DIR = REPO_ROOT / "cassettes"
FULL_TASK_CASSETTES_DIR = CASSETTES_DIR / "full_task"
CANARY_FILE = REPO_ROOT / "CANARY_TASKS.yaml"


# ---------------------------------------------------------------------------
# Canary task data
# ---------------------------------------------------------------------------

def _load_canary_metadata() -> dict[str, dict]:
    """Load CANARY_TASKS.yaml and return a dict keyed by task ID."""
    if not CANARY_FILE.exists():
        return {}
    raw = yaml.safe_load(CANARY_FILE.read_text(encoding="utf-8"))
    return {c["id"]: c for c in raw.get("canary_set", [])}


def _load_task_entries() -> dict[str, dict]:
    """Load TASKS.yaml and return a dict keyed by task ID."""
    tasks_path = REPO_ROOT / "TASKS.yaml"
    if not tasks_path.exists():
        return {}
    tasks = yaml.safe_load(tasks_path.read_text(encoding="utf-8"))
    return {t["id"]: t for t in tasks}


CANARY_META = _load_canary_metadata()
TASK_ENTRIES = _load_task_entries()


@pytest.fixture
def canary_meta():
    """The full canary metadata dict, keyed by task ID."""
    return CANARY_META


@pytest.fixture
def task_entries():
    """The full TASKS.yaml dict, keyed by task ID."""
    return TASK_ENTRIES


# ---------------------------------------------------------------------------
# Cassette helpers
# ---------------------------------------------------------------------------

def cassette_path_for(task_id: str) -> Path:
    """Return the expected cassette path for a canary task."""
    return CASSETTES_DIR / f"{task_id}.yaml"


def cassette_available(task_id: str) -> bool:
    """Check whether a cassette file exists for *task_id*."""
    return cassette_path_for(task_id).exists()


def full_task_cassette_path_for(task_id: str) -> Path:
    """Return the expected full-task replay cassette path for a canary task."""
    return FULL_TASK_CASSETTES_DIR / f"{task_id}.yaml"


def full_task_cassette_available(task_id: str) -> bool:
    """Check whether a full-task cassette exists for *task_id*."""
    return full_task_cassette_path_for(task_id).exists()


def requires_cassette(task_id: str):
    """Pytest skip decorator for tests that require a recorded cassette."""
    return pytest.mark.skipif(
        not cassette_available(task_id),
        reason=f"Cassette not recorded for {task_id}. Run with TRELLIS_CASSETTE_RECORD=1 to record.",
    )


# ---------------------------------------------------------------------------
# Cassette freshness
# ---------------------------------------------------------------------------

CASSETTE_MAX_AGE_DAYS = int(os.environ.get("TRELLIS_CASSETTE_MAX_AGE_DAYS", "14"))


def cassette_age_days(cassette_file: Path) -> int | None:
    """Return the age of a cassette in days, or None if unparseable."""
    try:
        raw = yaml.safe_load(cassette_file.read_text(encoding="utf-8"))
        recorded_at = raw.get("meta", {}).get("recorded_at")
        if not recorded_at:
            return None
        dt = datetime.fromisoformat(recorded_at)
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return None
