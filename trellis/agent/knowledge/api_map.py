"""Compact API navigation map for agent path resolution.

The import registry remains the authoritative symbol-to-module source. This
module provides the smaller, family-level orientation layer that agents can
use before they expand into the full registry or a full package walk.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

from trellis.agent.knowledge.import_registry import get_repo_revision
from trellis.agent.knowledge.methods import normalize_method


_KNOWLEDGE_DIR = Path(__file__).parent
_API_MAP_CACHE: dict[str, dict[str, Any]] = {}
_SPECIAL_TOP_LEVEL_KEYS = frozenset({"market_state", "payoff", "utilities"})
_WORD = re.compile(r"[a-z0-9]+")
_DEFAULT_COMPACT_CHARS = 4000
_DEFAULT_EXPANDED_CHARS = 9000
_DEFAULT_SELECTION_LIMIT = 4


@dataclass(frozen=True)
class ApiMapQuery:
    """Typed semantic cues for bounded API-map family selection."""

    instrument_type: str = ""
    payoff_family: str = ""
    method: str = ""
    model_family: str = ""
    features: tuple[str, ...] = ()
    route_ids: tuple[str, ...] = ()
    route_families: tuple[str, ...] = ()
    description: str = ""
    requested_families: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return whether the query carries no selection cue."""
        return not any(
            (
                self.instrument_type.strip(),
                self.payoff_family.strip(),
                self.method.strip(),
                self.model_family.strip(),
                tuple(item for item in self.features if str(item).strip()),
                tuple(item for item in self.route_ids if str(item).strip()),
                tuple(item for item in self.route_families if str(item).strip()),
                self.description.strip(),
                tuple(
                    item
                    for item in self.requested_families
                    if str(item).strip()
                ),
            )
        )


@dataclass(frozen=True)
class ApiMapSelection:
    """Deterministic API-map selection and omission evidence."""

    available_families: tuple[str, ...]
    selected_families: tuple[str, ...]
    omitted_families: tuple[str, ...]
    available_utilities: tuple[str, ...]
    selected_utilities: tuple[str, ...]
    omitted_utilities: tuple[str, ...]
    catalog_only: bool

    def summary(self) -> dict[str, object]:
        """Return machine-readable prompt-selection evidence."""
        return {
            "available_families": list(self.available_families),
            "selected_families": list(self.selected_families),
            "omitted_families": list(self.omitted_families),
            "available_utilities": list(self.available_utilities),
            "selected_utilities": list(self.selected_utilities),
            "omitted_utilities": list(self.omitted_utilities),
            "catalog_only": self.catalog_only,
        }


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


def select_api_map_sections(
    query: ApiMapQuery | None = None,
) -> ApiMapSelection:
    """Select task-relevant cards directly from the canonical API map."""
    api_map = get_api_map()
    if not api_map:
        return ApiMapSelection(
            available_families=(),
            selected_families=(),
            omitted_families=(),
            available_utilities=(),
            selected_utilities=(),
            omitted_utilities=(),
            catalog_only=query is None or query.is_empty,
        )

    family_names = _family_names(api_map)
    utility_names = _utility_names(api_map)
    catalog_only = query is None or query.is_empty
    if catalog_only:
        return ApiMapSelection(
            available_families=family_names,
            selected_families=(),
            omitted_families=family_names,
            available_utilities=utility_names,
            selected_utilities=(),
            omitted_utilities=utility_names,
            catalog_only=True,
        )

    assert query is not None
    requested = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in query.requested_families
            if str(item).strip()
        )
    )
    available_cards = frozenset((*family_names, *utility_names))
    unknown = tuple(name for name in requested if name not in available_cards)
    if unknown:
        raise ValueError("Unknown API map cards: " + ", ".join(unknown))

    if requested:
        selected_families = tuple(
            name for name in family_names if name in requested
        )
    else:
        scored = [
            (_score_card(name, api_map[name], query), index, name)
            for index, name in enumerate(family_names)
        ]
        selected_families = tuple(
            name
            for score, _, name in sorted(
                (item for item in scored if item[0] >= 80),
                key=lambda item: (-item[0], item[1]),
            )[:_DEFAULT_SELECTION_LIMIT]
        )

    selected_card_text = "\n".join(
        _section_search_text(name, api_map[name])
        for name in selected_families
    )
    selected_utilities = tuple(
        name
        for name in utility_names
        if name in requested
        or _utility_is_relevant(
            name,
            api_map["utilities"][name],
            query,
            selected_card_text,
        )
    )
    omitted_families = tuple(
        name for name in family_names if name not in selected_families
    )
    omitted_utilities = tuple(
        name for name in utility_names if name not in selected_utilities
    )
    return ApiMapSelection(
        available_families=family_names,
        selected_families=selected_families,
        omitted_families=omitted_families,
        available_utilities=utility_names,
        selected_utilities=selected_utilities,
        omitted_utilities=omitted_utilities,
        catalog_only=False,
    )


def format_api_map_for_prompt(
    *,
    compact: bool = False,
    query: ApiMapQuery | None = None,
    max_chars: int | None = None,
) -> str:
    """Render a bounded catalog or task-relevant API navigation card."""
    api_map = get_api_map()
    if not api_map:
        return ""
    selection = select_api_map_sections(query)
    if max_chars is None:
        limit = _DEFAULT_COMPACT_CHARS if compact else _DEFAULT_EXPANDED_CHARS
    else:
        limit = max_chars
    if limit < 240:
        raise ValueError("API map max_chars must be at least 240")

    lines = [
        "## API Map — start here before guessing module paths",
        "",
        "Use this navigation card to choose the right module family first. "
        "Confirm exact symbols with `find_symbol` or `list_exports` before "
        "calling `read_module`.",
    ]

    if selection.catalog_only:
        _append_catalog(lines, api_map, selection)
        return _bounded_render(
            "\n".join(lines),
            max_chars=limit,
            selected_families=(),
        )

    lines.extend(
        [
            "",
            "- Selected cards: "
            + (", ".join(selection.selected_families) or "none"),
            "- Omitted cards: "
            + (", ".join(selection.omitted_families) or "none"),
            "",
            "### Core Types",
        ]
    )
    _append_market_state_block(lines, api_map.get("market_state"), compact=compact)
    _append_payoff_block(lines, api_map.get("payoff"))

    if selection.selected_families:
        lines.append("")
        lines.append("### Selected Model Families")
    for family in selection.selected_families:
        section = api_map.get(family)
        if isinstance(section, Mapping):
            _append_family_block(lines, family, section, compact=compact)

    utilities = api_map.get("utilities")
    if (
        isinstance(utilities, Mapping)
        and utilities
        and selection.selected_utilities
    ):
        lines.append("")
        lines.append("### Selected Utilities")
        for utility_name in selection.selected_utilities:
            entry = utilities.get(utility_name)
            if isinstance(entry, Mapping):
                _append_utility_block(lines, utility_name, entry, compact=compact)

    return _bounded_render(
        "\n".join(lines),
        max_chars=limit,
        selected_families=selection.selected_families,
    )


def _family_names(api_map: Mapping[str, Any]) -> tuple[str, ...]:
    """Return canonical family cards in declaration order."""
    return tuple(
        name
        for name, section in api_map.items()
        if name not in _SPECIAL_TOP_LEVEL_KEYS
        and isinstance(section, Mapping)
        and bool(section.get("module"))
    )


def _utility_names(api_map: Mapping[str, Any]) -> tuple[str, ...]:
    utilities = api_map.get("utilities")
    if not isinstance(utilities, Mapping):
        return ()
    return tuple(
        name
        for name, section in utilities.items()
        if isinstance(section, Mapping) and bool(section.get("module"))
    )


def _append_catalog(
    lines: list[str],
    api_map: Mapping[str, Any],
    selection: ApiMapSelection,
) -> None:
    """Render a complete low-cost index without import statements."""
    market = api_map.get("market_state")
    payoff = api_map.get("payoff")
    lines.extend(["", "### Core Types"])
    if isinstance(market, Mapping):
        lines.append(
            f"- MarketState: {market.get('module', '')}"
        )
    if isinstance(payoff, Mapping):
        lines.append(f"- Payoff: {payoff.get('module', '')}")

    lines.extend(["", "### Available Model Cards"])
    for family in selection.available_families:
        section = api_map[family]
        lines.append(f"- {family}: {section.get('module', '')}")

    utilities = api_map.get("utilities")
    if isinstance(utilities, Mapping) and selection.available_utilities:
        lines.extend(["", "### Available Utility Cards"])
        for utility_name in selection.available_utilities:
            section = utilities[utility_name]
            lines.append(
                f"- {utility_name}: {section.get('module', '')}"
            )
    lines.extend(
        [
            "",
            "Pass semantic fields or explicit families to inspect selected "
            "imports and construction notes.",
        ]
    )


def _score_card(
    name: str,
    section: Mapping[str, Any],
    query: ApiMapQuery,
) -> int:
    allowed_methods = _section_methods(section)
    if (
        query.method.strip()
        and allowed_methods
        and normalize_method(query.method) not in allowed_methods
    ):
        return 0
    aliases = _section_aliases(name, section)
    score = 0
    for cue, weight in _query_cues(query):
        normalized_cue = _normalize(cue)
        if not normalized_cue:
            continue
        cue_tokens = set(_WORD.findall(normalized_cue))
        for alias in aliases:
            if normalized_cue == alias:
                score = max(score, weight * 4)
                continue
            if alias in normalized_cue or normalized_cue in alias:
                score = max(score, weight * 3)
                continue
            alias_tokens = set(_WORD.findall(alias))
            overlap = len(cue_tokens & alias_tokens)
            if overlap:
                score = max(
                    score,
                    int(weight * overlap / max(len(alias_tokens), 1)),
                )
    return score


def _section_methods(section: Mapping[str, Any]) -> frozenset[str]:
    """Return canonical method families that are eligible for one API card."""
    raw = section.get("methods") or ()
    values = raw if isinstance(raw, (list, tuple)) else (raw,)
    return frozenset(
        method
        for value in values
        if (method := normalize_method(str(value)))
    )


def _section_aliases(
    name: str,
    section: Mapping[str, Any],
) -> tuple[str, ...]:
    values: list[str] = [
        name,
        name.removesuffix("_composition"),
    ]
    navigation = section.get("navigation")
    if isinstance(navigation, Mapping):
        for item in navigation.values():
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, (list, tuple)):
                values.extend(str(value) for value in item)
    return tuple(
        dict.fromkeys(
            normalized
            for value in values
            if (normalized := _normalize(value))
        )
    )


def _query_cues(query: ApiMapQuery) -> tuple[tuple[str, int], ...]:
    cues: list[tuple[str, int]] = [
        (query.instrument_type, 120),
        (query.payoff_family, 120),
        (query.method, 20),
        (query.model_family, 40),
        (query.description, 30),
    ]
    cues.extend((str(item), 100) for item in query.features)
    cues.extend((str(item), 80) for item in query.route_ids)
    cues.extend((str(item), 80) for item in query.route_families)
    return tuple((value, weight) for value, weight in cues if value.strip())


def _section_search_text(name: str, section: Mapping[str, Any]) -> str:
    return " ".join(
        (
            name,
            str(section.get("module", "")),
            " ".join(str(item) for item in section.get("key_imports", ()) or ()),
            " ".join(str(item) for item in section.get("notes", ()) or ()),
        )
    )


def _utility_is_relevant(
    name: str,
    section: Mapping[str, Any],
    query: ApiMapQuery,
    selected_card_text: str,
) -> bool:
    if _score_card(name, {**section, "key_imports": []}, query) >= 80:
        return True
    module = str(section.get("module", "")).strip()
    return bool(module and module in selected_card_text)


def _normalize(value: str) -> str:
    return "_".join(_WORD.findall(str(value).lower()))


def _bounded_render(
    text: str,
    *,
    max_chars: int,
    selected_families: tuple[str, ...],
) -> str:
    if len(text) <= max_chars:
        return text
    selected = ", ".join(selected_families) or "catalog"
    marker = (
        "\n\n[API map truncated to the configured character budget; "
        f"selected cards: {selected}]"
    )
    keep = max(max_chars - len(marker), 0)
    return text[:keep].rstrip() + marker


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
    import_limit = min(len(key_imports), 8) if compact else len(key_imports)
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
        if import_limit < len(key_imports):
            lines.append(
                f"  - [omitted {len(key_imports) - import_limit} additional imports]"
            )
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
    import_limit = min(len(imports), 3) if compact else len(imports)
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
