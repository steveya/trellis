"""Gated promotion pipeline for the learning loop.

candidate → validated → promoted → archived

Also handles run trace recording and periodic distillation.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import yaml

from trellis.agent.knowledge.schema import LessonStatus


_KNOWLEDGE_DIR = Path(__file__).parent
_LESSONS_DIR = _KNOWLEDGE_DIR / "lessons"
_TRACES_DIR = _KNOWLEDGE_DIR / "traces"
_INDEX_PATH = _LESSONS_DIR / "index.yaml"


# ---------------------------------------------------------------------------
# Capture (Gate 1: dedup)
# ---------------------------------------------------------------------------

def capture_lesson(
    category: str,
    title: str,
    severity: str,
    symptom: str,
    root_cause: str,
    fix: str,
    validation: str = "",
    method: str | None = None,
    instrument: str | None = None,
    features: list[str] | None = None,
    error_signature: str | None = None,
    confidence: float = 0.5,
    source_trace: str | None = None,
    version: str = "",
) -> str | None:
    """Capture a new candidate lesson from a resolved failure.

    Returns the lesson ID if captured, None if duplicate detected.
    """
    index = _load_index()
    entries = index.get("entries", [])

    # Gate 1: deduplication
    for entry in entries:
        if entry["title"] == title:
            return None
        if _word_overlap(entry["title"], title) > 0.8:
            return None

    lesson_id = _generate_id(category, index)

    lesson_data = {
        "id": lesson_id,
        "title": title,
        "severity": severity,
        "category": category,
        "status": "candidate",
        "confidence": confidence,
        "created": datetime.now().isoformat(),
        "version": version,
        "source_trace": source_trace,
        "applies_when": {
            "method": [method] if method else [],
            "features": features or [],
            "instrument": [instrument] if instrument else [],
            "error_signature": error_signature,
        },
        "symptom": symptom,
        "root_cause": root_cause,
        "fix": fix,
        "validation": validation,
    }

    # Write full entry
    entry_path = _LESSONS_DIR / "entries" / f"{lesson_id}.yaml"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(entry_path, "w") as f:
        yaml.dump(lesson_data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    # Append to index
    index_entry = {
        "id": lesson_id,
        "title": title,
        "severity": severity,
        "category": category,
        "status": "candidate",
        "applies_when": lesson_data["applies_when"],
    }
    _append_to_index(index_entry)

    return lesson_id


# ---------------------------------------------------------------------------
# Validate (Gate 2)
# ---------------------------------------------------------------------------

def validate_lesson(lesson_id: str) -> bool:
    """Validate a candidate lesson.

    Criteria: non-empty fix, confidence >= 0.6.
    Returns True if promoted to 'validated' status.
    """
    entry_path = _LESSONS_DIR / "entries" / f"{lesson_id}.yaml"
    if not entry_path.exists():
        return False

    data = yaml.safe_load(entry_path.read_text())
    if not data:
        return False

    if not data.get("fix", "").strip():
        return False
    if data.get("confidence", 0) < 0.6:
        return False

    data["status"] = "validated"
    with open(entry_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    _update_index_status(lesson_id, "validated")
    return True


# ---------------------------------------------------------------------------
# Promote (Gate 3)
# ---------------------------------------------------------------------------

def promote_lesson(lesson_id: str) -> bool:
    """Promote a validated lesson to production.

    Criteria: validated status, confidence >= 0.8.
    """
    entry_path = _LESSONS_DIR / "entries" / f"{lesson_id}.yaml"
    if not entry_path.exists():
        return False

    data = yaml.safe_load(entry_path.read_text())
    if not data:
        return False
    if data.get("status") != "validated":
        return False
    if data.get("confidence", 0) < 0.8:
        return False

    data["status"] = "promoted"
    with open(entry_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    _update_index_status(lesson_id, "promoted")
    return True


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

def archive_lesson(lesson_id: str, reason: str = "") -> bool:
    """Archive a lesson — superseded, merged, or stale."""
    entry_path = _LESSONS_DIR / "entries" / f"{lesson_id}.yaml"
    if not entry_path.exists():
        return False

    data = yaml.safe_load(entry_path.read_text())
    if not data:
        return False

    data["status"] = "archived"
    if reason:
        data["archive_reason"] = reason
    with open(entry_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    _update_index_status(lesson_id, "archived")
    return True


# ---------------------------------------------------------------------------
# Confidence boost
# ---------------------------------------------------------------------------

def boost_confidence(lesson_id: str, amount: float = 0.15) -> float | None:
    """Increase a lesson's confidence (e.g., after it helped in another build).

    Returns the new confidence, or None if lesson not found.
    """
    entry_path = _LESSONS_DIR / "entries" / f"{lesson_id}.yaml"
    if not entry_path.exists():
        return None

    data = yaml.safe_load(entry_path.read_text())
    if not data:
        return None

    new_conf = min(1.0, data.get("confidence", 0.5) + amount)
    data["confidence"] = round(new_conf, 2)
    with open(entry_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    return new_conf


# ---------------------------------------------------------------------------
# Run traces (cold store)
# ---------------------------------------------------------------------------

def record_trace(
    instrument: str,
    method: str,
    description: str,
    pricing_plan: dict,
    attempt: int,
    code: str,
    validation_failures: list[str],
    diagnosis: dict | None = None,
    agent_observations: list[dict] | None = None,
    resolved: bool = False,
    lesson_id: str | None = None,
    duration_seconds: float = 0.0,
) -> str:
    """Write a run trace to the cold store. Returns the trace filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
    filename = f"{timestamp}_{instrument}_{method}.yaml"

    trace_data = {
        "timestamp": datetime.now().isoformat(),
        "instrument": instrument,
        "method": method,
        "description": description,
        "pricing_plan": pricing_plan,
        "attempt": attempt,
        "code_hash": code_hash,
        "validation_failures": validation_failures,
        "diagnosis": diagnosis,
        "agent_observations": agent_observations or [],
        "resolved": resolved,
        "lesson_id": lesson_id,
        "duration_seconds": duration_seconds,
    }

    _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = _TRACES_DIR / filename
    with open(trace_path, "w") as f:
        yaml.dump(trace_data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    return filename


# ---------------------------------------------------------------------------
# Distillation (periodic, offline)
# ---------------------------------------------------------------------------

def distill() -> dict[str, int]:
    """Periodic distillation pass.

    1. Auto-promote high-confidence validated lessons.
    2. Archive stale candidates (>30 days, low confidence).
    3. Count categories with 3+ promoted for principle candidates.

    Returns counts of actions taken.
    """
    stats = {"promoted": 0, "archived": 0, "principle_candidates": 0}

    index = _load_index()
    entries = index.get("entries", [])
    now = datetime.now()

    # Group by category for principle detection
    by_category: dict[str, list[dict]] = {}
    for e in entries:
        cat = e.get("category", "unknown")
        by_category.setdefault(cat, []).append(e)

    for _category, group in by_category.items():
        promoted = [e for e in group if e.get("status") == "promoted"]
        if len(promoted) >= 3:
            stats["principle_candidates"] += 1

    # Auto-promote validated with high confidence
    for e in entries:
        if e.get("status") == "validated":
            entry_path = _LESSONS_DIR / "entries" / f"{e['id']}.yaml"
            if entry_path.exists():
                data = yaml.safe_load(entry_path.read_text())
                if data and data.get("confidence", 0) >= 0.8:
                    if promote_lesson(e["id"]):
                        stats["promoted"] += 1

    # Archive stale candidates
    for e in entries:
        if e.get("status") == "candidate":
            entry_path = _LESSONS_DIR / "entries" / f"{e['id']}.yaml"
            if entry_path.exists():
                data = yaml.safe_load(entry_path.read_text())
                if data:
                    created = data.get("created", "")
                    if created:
                        try:
                            created_dt = datetime.fromisoformat(created)
                            age_days = (now - created_dt).days
                            if age_days > 30 and data.get("confidence", 0) < 0.6:
                                archive_lesson(
                                    e["id"],
                                    "auto-archived: low confidence after 30 days",
                                )
                                stats["archived"] += 1
                        except ValueError:
                            pass

    return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_index() -> dict:
    """Load the lesson index used by the promotion pipeline."""
    if _INDEX_PATH.exists():
        return yaml.safe_load(_INDEX_PATH.read_text()) or {}
    return {}


def _append_to_index(entry: dict) -> None:
    """Append a new lesson index entry and persist the updated index."""
    index = _load_index()
    if "entries" not in index:
        index["entries"] = []
        index["settings"] = {"max_prompt_entries": 7}
    index["entries"].append(entry)
    with open(_INDEX_PATH, "w") as f:
        yaml.dump(index, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


def _update_index_status(lesson_id: str, status: str) -> None:
    """Update the stored status for one lesson inside the index."""
    index = _load_index()
    for entry in index.get("entries", []):
        if entry["id"] == lesson_id:
            entry["status"] = status
            break
    with open(_INDEX_PATH, "w") as f:
        yaml.dump(index, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


def _generate_id(category: str, index: dict) -> str:
    """Generate the next sequential lesson id for a category-specific prefix."""
    prefix_map = {
        "volatility": "vol",
        "calibration": "cal",
        "backward_induction": "bi",
        "finite_differences": "fd",
        "monte_carlo": "mc",
        "market_data": "md",
        "testing": "tst",
        "vol_surface": "vs",
        "numerical": "num",
    }
    prefix = prefix_map.get(category, category[:3])
    existing = [
        e["id"] for e in index.get("entries", [])
        if e["id"].startswith(prefix + "_")
    ]
    max_num = 0
    for eid in existing:
        parts = eid.split("_")
        if len(parts) >= 2:
            try:
                max_num = max(max_num, int(parts[-1]))
            except ValueError:
                pass
    return f"{prefix}_{max_num + 1:03d}"


def _word_overlap(a: str, b: str) -> float:
    """Return a simple normalized token-overlap score for duplicate detection."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))
