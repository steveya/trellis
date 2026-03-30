"""Legacy experience system (superseded by trellis.agent.knowledge).

Loads past build failures and lessons learned from YAML files so they can
be injected into LLM prompts to avoid repeating the same mistakes.

Two tiers:
  - index.yaml: lightweight summaries used to pick which lessons apply.
  - experience.yaml: full entries with symptoms, explanations, and fixes.

Kept for backward compatibility; new code should use the knowledge store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trellis.agent.knowledge.methods import normalize_method


_DIR = Path(__file__).parent
_INDEX_PATH = _DIR / "experience" / "index.yaml"
_ENTRIES_PATH = _DIR / "experience.yaml"  # full entries (backward compat)

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_index() -> dict:
    """Load the lightweight legacy experience index from disk."""
    if _INDEX_PATH.exists():
        with open(_INDEX_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_entries() -> list[dict]:
    """Load the backward-compatible full legacy experience entries."""
    if _ENTRIES_PATH.exists():
        with open(_ENTRIES_PATH) as f:
            return yaml.safe_load(f) or []
    return []


_INDEX: dict = _load_index()
_ENTRIES: list[dict] = _load_entries()

# Backward compat
EXPERIENCE = _ENTRIES


def reload():
    """Reload the legacy experience index and entry caches from disk."""
    global _INDEX, _ENTRIES, EXPERIENCE
    _INDEX = _load_index()
    _ENTRIES = _load_entries()
    EXPERIENCE = _ENTRIES


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def query_index(
    method: str | None = None,
    instrument: str | None = None,
    features: list[str] | None = None,
    max_entries: int | None = None,
) -> list[dict]:
    """Query the index for matching entries, ranked by severity.

    Returns lightweight index entries (id, title, severity, applies_when).
    """
    index_entries = _INDEX.get("entries", [])
    method = normalize_method(method) if method else None
    if max_entries is None:
        max_entries = _INDEX.get("settings", {}).get("max_prompt_entries", 5)

    matches = []
    for entry in index_entries:
        aw = entry.get("applies_when", {})
        if method and not _matches(aw.get("method", []), method):
            continue
        if instrument and not _matches(aw.get("instrument", []), instrument):
            continue
        if features:
            entry_features = set(aw.get("has_feature", []))
            if entry_features and not entry_features.intersection(features):
                continue
        matches.append(entry)

    # Sort by severity (critical first)
    matches.sort(key=lambda e: _SEVERITY_ORDER.get(e.get("severity", "low"), 3))
    return matches[:max_entries]


def get_full_entry(entry_id: str) -> dict | None:
    """Load the full entry (symptoms, explanation, fix) by ID."""
    # Search in the flat entries file by title match against index
    index_entry = None
    for ie in _INDEX.get("entries", []):
        if ie["id"] == entry_id:
            index_entry = ie
            break
    if index_entry is None:
        return None

    # Find matching full entry by title
    for e in _ENTRIES:
        if e.get("title") == index_entry["title"]:
            return e
    return None


def _matches(allowed: list, value: str) -> bool:
    """Return whether ``value`` is admitted by an optional method/instrument filter."""
    if not allowed:
        return True
    normalized_allowed = {
        normalize_method(item) if isinstance(item, str) else item
        for item in allowed
    }
    return "any" in normalized_allowed or normalize_method(value) in normalized_allowed


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def get_experience_for_task(
    method: str,
    instrument: str | None = None,
    features: list[str] | None = None,
) -> str:
    """Get experience for a specific task — the primary agent entry point.

    Delegates to the knowledge system for feature-based retrieval.
    Falls back to legacy implementation if the knowledge system fails.
    """
    try:
        from trellis.agent.knowledge import retrieve_for_task

        knowledge = retrieve_for_task(
            method=method,
            features=features or [],
            instrument=instrument,
        )

        sections: list[str] = []

        # Principles
        principles = knowledge.get("principles", [])
        if principles:
            lines = ["## Key Principles\n"]
            for p in principles:
                lines.append(f"- **{p.id}**: {p.rule}")
            sections.append("\n".join(lines))

        # Lessons
        lessons = knowledge.get("lessons", [])
        if lessons:
            lines = ["## Lessons (ranked by severity)\n"]
            for lesson in lessons:
                lines.append(f"### [{lesson.severity.value.upper()}] {lesson.title}")
                if lesson.symptom:
                    lines.append(f"**Symptoms:** {lesson.symptom}")
                lines.append(f"**Why:** {lesson.root_cause.strip()}")
                lines.append(f"**Fix:** {lesson.fix.strip()}")
                lines.append("")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
    except Exception:
        return _get_experience_for_task_legacy(method, instrument, features)


def _get_experience_for_task_legacy(
    method: str,
    instrument: str | None = None,
    features: list[str] | None = None,
) -> str:
    """Legacy implementation — reads from old index.yaml + experience.yaml."""
    lines = []

    principles = _INDEX.get("principles", [])
    if principles:
        lines.append("## Key Principles\n")
        for p in principles:
            lines.append(f"- **{p['id']}**: {p['rule']}")
        lines.append("")

    matches = query_index(method=method, instrument=instrument, features=features)
    if not matches:
        return "\n".join(lines) if lines else ""

    lines.append("## Lessons (ranked by severity)\n")
    for idx_entry in matches:
        full = get_full_entry(idx_entry["id"])
        severity = idx_entry.get("severity", "?")
        title = idx_entry["title"]

        if full:
            symptoms = full.get("symptoms", [])
            sym_text = ""
            if symptoms:
                sym_parts = []
                for s in symptoms:
                    if isinstance(s, dict):
                        for k, v in s.items():
                            sym_parts.append(f"{k}: {v}")
                    else:
                        sym_parts.append(str(s))
                sym_text = " | ".join(sym_parts)

            lines.append(f"### [{severity.upper()}] {title}")
            if sym_text:
                lines.append(f"**Symptoms:** {sym_text}")
            lines.append(f"**Why:** {full.get('explanation', '').strip()}")
            lines.append(f"**Fix:** {full.get('fix', '').strip()}")
        else:
            lines.append(f"### [{severity.upper()}] {title}")
        lines.append("")

    return "\n".join(lines)


# Backward-compatible aliases
def format_experience_for_prompt(categories: list[str] | None = None) -> str:
    """Format legacy experience entries filtered by category for prompt injection."""
    if categories:
        entries = [e for e in _ENTRIES if e.get("category") in categories]
    else:
        entries = _ENTRIES
    return format_for_prompt(entries)


def format_for_prompt(entries: list[dict]) -> str:
    """Render legacy experience entries into the historical prompt format."""
    if not entries:
        return ""
    lines = ["## Lessons\n"]
    for e in entries:
        lines.append(f"### {e.get('title', '?')}")
        lines.append(f"**Why:** {e.get('explanation', '').strip()}")
        lines.append(f"**Fix:** {e.get('fix', '').strip()}")
        lines.append("")
    return "\n".join(lines)


def get_experience_for_method(method: str) -> str:
    """Backward-compatible helper that delegates to task-scoped experience retrieval."""
    return get_experience_for_task(method=method)


def query(method=None, instrument=None, features=None):
    """Backward compat: query full entries."""
    method = normalize_method(method) if method else None
    results = []
    for e in _ENTRIES:
        aw = e.get("applies_when", {})
        if method and not _matches(aw.get("method", []), method):
            continue
        if instrument and not _matches(aw.get("instrument", []), instrument):
            continue
        results.append(e)
    return results


def append_lesson(lesson: dict) -> None:
    """Append a new lesson to experience.yaml and index."""
    required = {"category", "title", "explanation", "fix"}
    if not required.issubset(lesson.keys()):
        raise ValueError(f"Lesson must have at least: {required}")

    for e in _ENTRIES:
        if e["title"] == lesson["title"]:
            return

    with open(_ENTRIES_PATH, "a") as f:
        f.write("\n")
        yaml.dump([lesson], f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    reload()
