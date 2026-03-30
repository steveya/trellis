"""Compact API navigation map for agent path resolution.

The import registry remains the authoritative symbol-to-module source. This
module provides the smaller, family-level orientation layer that agents can
use before they expand into the full registry or a full package walk.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from trellis.agent.knowledge.import_registry import get_repo_revision


_KNOWLEDGE_DIR = Path(__file__).parent
_API_MAP_CACHE: dict[str, dict[str, Any]] = {}

_FAMILY_ORDER = (
    "equity_tree",
    "rate_lattice",
    "monte_carlo",
    "qmc",
    "pde",
    "fft",
    "copulas",
    "analytical",
    "calibration",
)
_UTILITY_ORDER = (
    "black76",
    "garman_kohlhagen",
    "schedule",
    "day_count",
    "vol_surface",
    "cashflow_engine",
)


def get_api_map() -> dict[str, Any]:
    """Load the canonical compact API map for the current repository revision."""
    revision = get_repo_revision()
    cached = _API_MAP_CACHE.get(revision)
    if cached is not None:
        return cached

    path = _KNOWLEDGE_DIR / "canonical" / "api_map.yaml"
    if not path.exists():
        _API_MAP_CACHE[revision] = {}
        return {}

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        data = {}

    _API_MAP_CACHE[revision] = data
    return data


def format_api_map_for_prompt(*, compact: bool = False) -> str:
    """Render the compact API map as markdown for prompt or tool output."""
    api_map = get_api_map()
    if not api_map:
        return ""

    lines = [
        "## API Map — start here before guessing module paths",
        "",
        "Use this navigation card to choose the right module family first. "
        "Confirm exact symbols with `find_symbol` or `list_exports` before "
        "calling `read_module`.",
        "",
        "### Core Types",
    ]

    _append_market_state_block(lines, api_map.get("market_state"), compact=compact)
    _append_payoff_block(lines, api_map.get("payoff"))

    lines.append("")
    lines.append("### Model Families")
    for family in _FAMILY_ORDER:
        section = api_map.get(family)
        if isinstance(section, Mapping):
            _append_family_block(lines, family, section, compact=compact)

    utilities = api_map.get("utilities")
    if isinstance(utilities, Mapping) and utilities:
        lines.append("")
        lines.append("### Utilities")
        for utility_name in _UTILITY_ORDER:
            entry = utilities.get(utility_name)
            if isinstance(entry, Mapping):
                _append_utility_block(lines, utility_name, entry, compact=compact)

    return "\n".join(lines)


def _append_market_state_block(
    lines: list[str],
    section: Mapping[str, Any] | None,
    *,
    compact: bool,
) -> None:
    """Render the MarketState quick-reference block."""
    if not isinstance(section, Mapping):
        return

    fields = list(section.get("fields", {}) or {})
    accessors = list(section.get("accessors", {}) or {})
    capabilities = list(section.get("capabilities", []) or [])

    lines.extend(
        [
            "#### MarketState",
            f"- Module: `{section.get('module', '')}`",
            f"- Class: `{section.get('class', 'MarketState')}`",
        ]
    )
    if section.get("frozen"):
        lines.append("- Frozen: `true`")
    if fields:
        lines.append(
            "- Fields: "
            + ", ".join(f"`{name}`" for name in fields)
        )
    if accessors:
        lines.append(
            "- Accessors: "
            + ", ".join(f"`{name}`" for name in accessors)
        )
    if capabilities:
        lines.append(
            "- Capabilities: "
            + ", ".join(f"`{name}`" for name in capabilities)
        )


def _append_payoff_block(
    lines: list[str],
    section: Mapping[str, Any] | None,
) -> None:
    """Render the Payoff protocol quick-reference block."""
    if not isinstance(section, Mapping):
        return

    required_methods = list(section.get("required_methods", {}) or {})
    required_properties = list(section.get("required_properties", {}) or {})
    notes = list(section.get("notes", []) or [])

    lines.extend(
        [
            "#### Payoff",
            f"- Module: `{section.get('module', '')}`",
            f"- Protocol: `{section.get('protocol', 'Payoff')}`",
            "- Required methods: "
            + ", ".join(f"`{name}`" for name in required_methods),
        ]
    )
    if required_properties:
        lines.append(
            "- Required properties: "
            + ", ".join(f"`{name}`" for name in required_properties)
        )
    if section.get("runtime_checkable"):
        lines.append("- Runtime checkable: `true`")
    if notes:
        lines.append("- Notes:")
        for note in notes[:3]:
            lines.append(f"  - {note}")


def _append_family_block(
    lines: list[str],
    family_name: str,
    section: Mapping[str, Any],
    *,
    compact: bool,
) -> None:
    """Render a model-family block from the canonical API map."""
    module = str(section.get("module", ""))
    key_imports = list(section.get("key_imports", []) or [])
    notes = list(section.get("notes", []) or [])
    import_limit = len(key_imports)
    note_limit = 2 if compact else len(notes)

    lines.extend(
        [
            f"#### {family_name}",
            f"- Module: `{module}`",
        ]
    )
    if key_imports:
        lines.append("- Key imports:")
        for stmt in key_imports[:import_limit]:
            lines.append(f"  - `{stmt}`")
    if notes:
        lines.append("- Notes:")
        for note in notes[:note_limit]:
            lines.append(f"  - {note}")


def _append_utility_block(
    lines: list[str],
    utility_name: str,
    section: Mapping[str, Any],
    *,
    compact: bool,
) -> None:
    """Render a utility subsection from the canonical API map."""
    module = str(section.get("module", ""))
    imports = list(section.get("imports", []) or [])
    import_limit = len(imports)
    notes = list(section.get("notes", []) or [])
    note_limit = 1 if compact else len(notes)

    lines.extend(
        [
            f"#### {utility_name}",
            f"- Module: `{module}`",
        ]
    )
    if imports:
        lines.append("- Imports:")
        for stmt in imports[:import_limit]:
            lines.append(f"  - `{stmt}`")
    if notes:
        lines.append("- Notes:")
        for note in notes[:note_limit]:
            lines.append(f"  - {note}")
