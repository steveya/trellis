"""Shared helpers for curated canary scripts."""

from __future__ import annotations

from pathlib import Path

from trellis.agent.task_manifests import (
    load_active_task_lookup,
    load_canary_manifest,
)

ROOT = Path(__file__).resolve().parent.parent
CANARY_FILE = ROOT / "CANARY_TASKS.yaml"
FULL_TASK_CASSETTES_DIR = ROOT / "cassettes" / "full_task"


def merge_canary_task_payload(task: dict, canary: dict) -> dict:
    """Overlay curated canary fields onto the live task registry payload."""
    merged = dict(task)
    for key in (
        "description",
        "market",
        "market_assertions",
        "construct",
        "cross_validate",
        "new_component",
    ):
        value = canary.get(key)
        if value is not None:
            merged[key] = value
    return merged


def load_curated_canary_set(root: Path = ROOT) -> tuple[list[dict], dict]:
    """Load the normalized canary manifest."""
    return load_canary_manifest(root=root)


def build_canary_task_lookup(root: Path = ROOT) -> dict[str, dict]:
    """Load all active pricing and negative tasks keyed by id for canary resolution."""
    return load_active_task_lookup(root=root)
