"""Cookbook accessors backed by canonical knowledge YAML.

This module is intentionally a thin compatibility shim. Cookbook content lives
in ``trellis.agent.knowledge.canonical.cookbooks.yaml``; Python code should load
from there rather than maintaining a second hand-written copy.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from trellis.agent.knowledge.methods import normalize_method

_COOKBOOKS_PATH = (
    Path(__file__).parent / "knowledge" / "canonical" / "cookbooks.yaml"
)


def _load_cookbooks() -> dict[str, str]:
    """Load cookbook templates from canonical YAML."""
    if not _COOKBOOKS_PATH.exists():
        return {}

    data = yaml.safe_load(_COOKBOOKS_PATH.read_text()) or {}
    cookbooks: dict[str, str] = {}
    for method, entry in data.items():
        cookbooks[normalize_method(method)] = entry.get("template", "")
    return cookbooks


# Backward-compatible public constant used by tests and prompts.
COOKBOOKS: dict[str, str] = _load_cookbooks()


def get_cookbook(method: str) -> str:
    """Return the cookbook for a pricing method, or empty string if none."""
    return COOKBOOKS.get(normalize_method(method), "")


def get_all_cookbooks() -> str:
    """Return all cookbook templates concatenated for prompt/debug use."""
    return "\n\n".join(COOKBOOKS[key] for key in sorted(COOKBOOKS))
