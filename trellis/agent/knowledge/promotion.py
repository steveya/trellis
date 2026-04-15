"""Gated promotion pipeline for the learning loop.

candidate → validated → promoted → archived

Also handles run trace recording and periodic distillation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import logging
from dataclasses import dataclass, replace
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import re
from collections.abc import Mapping

import yaml

from trellis.agent.knowledge.import_registry import get_repo_revision
from trellis.agent.knowledge.schema import (
    AdapterLifecycleRecord,
    AdapterLifecycleStatus,
    LessonStatus,
)


_KNOWLEDGE_DIR = Path(__file__).parent
_LESSONS_DIR = _KNOWLEDGE_DIR / "lessons"
_TRACES_DIR = _KNOWLEDGE_DIR / "traces"
_SEMANTIC_EXTENSION_TRACES_DIR = _TRACES_DIR / "semantic_extensions"
_INDEX_PATH = _LESSONS_DIR / "index.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_INDEX_REBUILD_SUPPRESS_DEPTH = 0
_INDEX_REBUILD_PENDING = False
_ADAPTER_LIFECYCLE_STATUS_ORDER = {
    AdapterLifecycleStatus.FRESH: 0,
    AdapterLifecycleStatus.STALE: 1,
    AdapterLifecycleStatus.DEPRECATED: 2,
    AdapterLifecycleStatus.ARCHIVED: 3,
}

logger = logging.getLogger(__name__)

_LESSON_TEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "for",
    "if",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "when",
    "with",
}


@dataclass(frozen=True)
class LessonContractReport:
    """Deterministic validation report for a lesson payload."""

    contract: str
    valid: bool
    normalized_payload: dict[str, object]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the report into trace-friendly primitives."""
        return {
            "contract": self.contract,
            "valid": self.valid,
            "normalized_payload": self.normalized_payload,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


_LESSON_CONTRACT_NAME = "lesson_payload.v1"
_LESSON_STATUSES = {"candidate", "validated", "promoted", "archived"}
_LESSON_SEVERITIES = {"critical", "high", "medium", "low"}


def _loaded_store():
    """Return the already-instantiated KnowledgeStore singleton, if any."""
    try:
        import trellis.agent.knowledge as knowledge_module
    except Exception:
        return None
    return getattr(knowledge_module, "_store", None)


def _clear_loaded_store_runtime_caches() -> None:
    """Clear warm/runtime caches on the live KnowledgeStore singleton."""
    store = _loaded_store()
    if store is not None:
        store.clear_runtime_caches()


def _reload_loaded_store() -> None:
    """Reload the live KnowledgeStore singleton after hot-tier mutation."""
    store = _loaded_store()
    if store is not None:
        store.reload()


@contextmanager
def defer_index_rebuilds():
    """Batch lesson mutations so the generated index is rebuilt once."""
    global _INDEX_REBUILD_SUPPRESS_DEPTH, _INDEX_REBUILD_PENDING
    _INDEX_REBUILD_SUPPRESS_DEPTH += 1
    try:
        yield
    finally:
        _INDEX_REBUILD_SUPPRESS_DEPTH -= 1
        if _INDEX_REBUILD_SUPPRESS_DEPTH == 0 and _INDEX_REBUILD_PENDING:
            _INDEX_REBUILD_PENDING = False
            rebuild_index()


def _request_index_rebuild() -> None:
    """Rebuild the lesson index now, or defer until the current batch ends."""
    global _INDEX_REBUILD_PENDING
    if _INDEX_REBUILD_SUPPRESS_DEPTH > 0:
        _INDEX_REBUILD_PENDING = True
        return
    rebuild_index()


def _adapter_lifecycle_record_key(record: AdapterLifecycleRecord) -> tuple[str, str]:
    """Return the stable key used to merge lifecycle records."""
    return (record.adapter_id, record.module_path)


def _adapter_lifecycle_status_rank(status: AdapterLifecycleStatus) -> int:
    """Return a precedence rank for lifecycle states."""
    return _ADAPTER_LIFECYCLE_STATUS_ORDER.get(status, 0)


def _serialize_adapter_lifecycle_record(record: AdapterLifecycleRecord) -> dict[str, object]:
    """Convert one lifecycle record into YAML-friendly primitives."""
    return {
        "adapter_id": record.adapter_id,
        "status": record.status.value,
        "module_path": record.module_path,
        "validated_against_repo_revision": record.validated_against_repo_revision,
        "supersedes": list(record.supersedes),
        "replacement": record.replacement,
        "reason": record.reason,
        "code_hash": record.code_hash,
    }


def _deserialize_adapter_lifecycle_record(data: Mapping[str, object]) -> AdapterLifecycleRecord | None:
    """Restore one lifecycle record from serialized artifact payload."""
    adapter_id = str(data.get("adapter_id") or "").strip()
    module_path = str(data.get("module_path") or "").strip()
    if not adapter_id or not module_path:
        return None
    status_text = str(data.get("status") or AdapterLifecycleStatus.FRESH.value).strip().lower()
    try:
        status = AdapterLifecycleStatus(status_text)
    except ValueError:
        status = AdapterLifecycleStatus.FRESH
    return AdapterLifecycleRecord(
        adapter_id=adapter_id,
        status=status,
        module_path=module_path,
        validated_against_repo_revision=str(data.get("validated_against_repo_revision") or ""),
        supersedes=tuple(str(item) for item in data.get("supersedes", ()) if str(item).strip()),
        replacement=str(data.get("replacement") or ""),
        reason=str(data.get("reason") or ""),
        code_hash=str(data.get("code_hash") or ""),
    )


def _serialize_adapter_lifecycle_summary(records: list[AdapterLifecycleRecord]) -> dict[str, object]:
    """Summarize lifecycle records in a trace-friendly shape."""
    counts = {
        "fresh": 0,
        "stale": 0,
        "deprecated": 0,
        "archived": 0,
    }
    stale_adapter_ids: list[str] = []
    fresh_replacements: list[str] = []
    deprecated_adapter_ids: list[str] = []
    archived_adapter_ids: list[str] = []
    fresh_adapter_ids: list[str] = []
    serialized_records = [_serialize_adapter_lifecycle_record(record) for record in records]

    for record in records:
        status = record.status.value
        counts[status] = counts.get(status, 0) + 1
        if record.status in {
            AdapterLifecycleStatus.STALE,
            AdapterLifecycleStatus.DEPRECATED,
            AdapterLifecycleStatus.ARCHIVED,
        } and record.replacement:
            fresh_replacements.append(record.replacement)
        if record.status == AdapterLifecycleStatus.STALE:
            stale_adapter_ids.append(record.adapter_id)
        elif record.status == AdapterLifecycleStatus.DEPRECATED:
            deprecated_adapter_ids.append(record.adapter_id)
        elif record.status == AdapterLifecycleStatus.ARCHIVED:
            archived_adapter_ids.append(record.adapter_id)
        elif record.status == AdapterLifecycleStatus.FRESH:
            fresh_adapter_ids.append(record.adapter_id)

    return {
        "status_counts": counts,
        "stale_adapter_count": len(stale_adapter_ids),
        "stale_adapter_ids": stale_adapter_ids,
        "deprecated_adapter_count": len(deprecated_adapter_ids),
        "deprecated_adapter_ids": deprecated_adapter_ids,
        "archived_adapter_count": len(archived_adapter_ids),
        "archived_adapter_ids": archived_adapter_ids,
        "fresh_adapter_count": len(fresh_adapter_ids),
        "fresh_adapter_ids": fresh_adapter_ids,
        "fresh_replacements": fresh_replacements,
        "records": serialized_records,
    }


def _transition_adapter_lifecycle_records(
    records: list[AdapterLifecycleRecord] | tuple[AdapterLifecycleRecord, ...],
    *,
    stage: str,
) -> list[AdapterLifecycleRecord]:
    """Apply a deterministic lifecycle stage to stale adapter records."""
    normalized_stage = stage.strip().lower()
    if normalized_stage not in {"stale", "deprecated", "archived"}:
        normalized_stage = "stale"

    transitioned: list[AdapterLifecycleRecord] = []
    for record in records:
        if record.status != AdapterLifecycleStatus.STALE:
            transitioned.append(record)
            continue

        if normalized_stage == "archived":
            transitioned.append(
                replace(
                    record,
                    status=AdapterLifecycleStatus.ARCHIVED,
                    reason=record.reason
                    or "validated fresh replacement adopted; stale adapter archived",
                )
            )
        elif normalized_stage == "deprecated":
            transitioned.append(
                replace(
                    record,
                    status=AdapterLifecycleStatus.DEPRECATED,
                    reason=record.reason
                    or "validated fresh replacement is active; stale adapter deprecated",
                )
            )
        else:
            transitioned.append(record)

    return transitioned


def _adapter_lifecycle_snapshot(
    records: list[AdapterLifecycleRecord] | tuple[AdapterLifecycleRecord, ...],
    *,
    stage: str = "stale",
    adapter_id: str = "",
    replacement: str = "",
    repo_revision: str = "",
) -> dict[str, object]:
    """Return raw and resolved lifecycle views for review/adoption artifacts."""
    raw_records = list(records)
    resolved_records = _transition_adapter_lifecycle_records(raw_records, stage=stage)
    return {
        "stage": stage,
        "adapter_id": adapter_id,
        "replacement": replacement,
        "validated_against_repo_revision": repo_revision,
        "raw": {
            "summary": _serialize_adapter_lifecycle_summary(raw_records),
            "records": [_serialize_adapter_lifecycle_record(record) for record in raw_records],
        },
        "resolved": {
            "summary": _serialize_adapter_lifecycle_summary(resolved_records),
            "records": [_serialize_adapter_lifecycle_record(record) for record in resolved_records],
        },
    }


def _normalize_text(value: object, *, lower: bool = False) -> str:
    """Normalize a free-form scalar into a stripped string."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.lower() if lower else text


def _normalize_text_list(value: object) -> list[str]:
    """Normalize a scalar or sequence into a de-duplicated list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (tuple, set, frozenset, list)):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        text = _normalize_text(item)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_confidence(value: object) -> tuple[float, bool]:
    """Normalize a confidence score into a finite float."""
    if isinstance(value, bool):
        return 0.0, False
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not math.isfinite(confidence):
        return 0.0, False
    return round(confidence, 4), True


def _normalize_lesson_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Normalize a lesson payload into canonical trace-friendly primitives."""
    applies_when = payload.get("applies_when") or {}
    if not isinstance(applies_when, Mapping):
        applies_when = {}

    normalized: dict[str, object] = {
        "category": _normalize_text(payload.get("category")),
        "title": _normalize_text(payload.get("title")),
        "severity": _normalize_text(payload.get("severity"), lower=True),
        "status": _normalize_text(payload.get("status"), lower=True) or "candidate",
        "confidence": _normalize_confidence(payload.get("confidence"))[0],
        "created": _normalize_text(payload.get("created")),
        "version": _normalize_text(payload.get("version")),
        "source_trace": _normalize_text(payload.get("source_trace")) or None,
        "applies_when": {
            "method": _normalize_text_list(applies_when.get("method")),
            "features": _normalize_text_list(applies_when.get("features")),
            "instrument": _normalize_text_list(applies_when.get("instrument")),
            "error_signature": _normalize_text(applies_when.get("error_signature")) or None,
        },
        "symptom": _normalize_text(payload.get("symptom")),
        "root_cause": _normalize_text(payload.get("root_cause")),
        "fix": _normalize_text(payload.get("fix")),
        "validation": _normalize_text(payload.get("validation")),
        "supersedes": _normalize_text_list(payload.get("supersedes")),
    }

    lesson_id = _normalize_text(payload.get("id"))
    if lesson_id:
        normalized["id"] = lesson_id
    return normalized


def build_lesson_payload(
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
    *,
    status: str = "candidate",
    created: str | None = None,
    lesson_id: str | None = None,
) -> dict[str, object]:
    """Build a normalized lesson payload from explicit fields."""
    payload: dict[str, object] = {
        "category": category,
        "title": title,
        "severity": severity,
        "status": status,
        "confidence": confidence,
        "created": created or "",
        "version": version,
        "source_trace": source_trace,
        "applies_when": {
            "method": _normalize_text_list(method) if method is not None else [],
            "features": list(features or []),
            "instrument": _normalize_text_list(instrument) if instrument is not None else [],
            "error_signature": error_signature,
        },
        "symptom": symptom,
        "root_cause": root_cause,
        "fix": fix,
        "validation": validation,
    }
    if lesson_id is not None:
        payload["id"] = lesson_id
    return _normalize_lesson_payload(payload)


def validate_lesson_payload(
    payload: Mapping[str, object] | None,
    *,
    contract: str = _LESSON_CONTRACT_NAME,
) -> LessonContractReport:
    """Validate and normalize one lesson payload against the lesson contract."""
    if not isinstance(payload, Mapping):
        return LessonContractReport(
            contract=contract,
            valid=False,
            normalized_payload={},
            errors=("lesson payload must be a mapping",),
        )

    normalized = _normalize_lesson_payload(payload)
    errors: list[str] = []
    warnings: list[str] = []

    for field in ("category", "title", "severity", "symptom", "root_cause", "fix"):
        if not str(normalized.get(field) or "").strip():
            errors.append(f"{field} is required")

    if str(normalized.get("severity") or "") not in _LESSON_SEVERITIES:
        errors.append(
            "severity must be one of critical, high, medium, or low"
        )

    if str(normalized.get("status") or "") not in _LESSON_STATUSES:
        errors.append(
            "status must be one of candidate, validated, promoted, or archived"
        )

    raw_confidence = payload.get("confidence")
    confidence, confidence_valid = _normalize_confidence(raw_confidence)
    normalized["confidence"] = confidence
    if raw_confidence is None:
        warnings.append("confidence missing; normalized to 0.0")
    elif not confidence_valid:
        warnings.append("confidence was invalid; normalized to 0.0")
    elif not 0.0 <= confidence <= 1.0:
        errors.append("confidence must be between 0.0 and 1.0")

    if normalized.get("status") in {"validated", "promoted"} and not normalized.get("id"):
        warnings.append("validated lesson payload is missing an id")

    return LessonContractReport(
        contract=contract,
        valid=not errors,
        normalized_payload=normalized,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


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
    lesson_data = build_lesson_payload(
        category=category,
        title=title,
        severity=severity,
        symptom=symptom,
        root_cause=root_cause,
        fix=fix,
        validation=validation,
        method=method,
        instrument=instrument,
        features=features,
        error_signature=error_signature,
        confidence=confidence,
        source_trace=source_trace,
        version=version,
        created=datetime.now().isoformat(),
    )
    lesson_report = validate_lesson_payload(lesson_data)
    if not lesson_report.valid:
        logger.warning(
            "Rejected lesson payload for %r: %s",
            lesson_report.normalized_payload.get("title") or title,
            "; ".join(lesson_report.errors),
        )
        return None

    entries = _scan_entry_metadata()

    # Gate 1: deduplication
    normalized_title = str(lesson_report.normalized_payload.get("title") or "").strip()
    normalized_context = _lesson_context_key(lesson_report.normalized_payload)
    normalized_signature = _lesson_signature_text(lesson_report.normalized_payload)
    for entry in entries:
        if entry["title"] == normalized_title:
            return None
        if _word_overlap(entry["title"], normalized_title) > 0.8:
            return None
        if _lesson_context_key(entry) != normalized_context:
            continue
        existing_entry = _load_lesson_entry(str(entry.get("id") or "").strip())
        if not existing_entry:
            continue
        if (
            _semantic_text_overlap(
                normalized_signature,
                _lesson_signature_text(existing_entry),
            )
            >= 0.30
        ):
            return None

    lesson_id = _generate_id(
        str(lesson_report.normalized_payload.get("category") or category).strip(),
        entries,
    )
    lesson_data = dict(lesson_report.normalized_payload)
    lesson_data["id"] = lesson_id

    # Write full entry
    entry_path = _LESSONS_DIR / "entries" / f"{lesson_id}.yaml"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(entry_path, "w") as f:
        yaml.dump(lesson_data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    _request_index_rebuild()

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
    if not isinstance(data, Mapping):
        return False

    report = validate_lesson_payload(data)
    if not report.valid:
        return False

    if not report.normalized_payload.get("fix"):
        return False
    if float(report.normalized_payload.get("confidence") or 0.0) < 0.6:
        return False

    data = dict(data)
    data.update(report.normalized_payload)
    data["status"] = "validated"
    with open(entry_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    _request_index_rebuild()
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
    if not isinstance(data, Mapping):
        return False

    report = validate_lesson_payload(data)
    if not report.valid:
        return False
    if report.normalized_payload.get("status") != "validated":
        return False
    if float(report.normalized_payload.get("confidence") or 0.0) < 0.8:
        return False

    data = dict(data)
    data.update(report.normalized_payload)
    data["status"] = "promoted"
    with open(entry_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    # Auto-detect supersedes relationship
    new_category = str(data.get("category") or "").strip()
    new_fix = str(data.get("fix") or "").strip()
    entries_dir = _LESSONS_DIR / "entries"
    superseded_id = _detect_supersedes(
        lesson_id, new_category, new_fix, entries_dir,
    )
    if superseded_id:
        # Mark the new lesson as superseding the old one
        existing_supersedes = list(data.get("supersedes") or [])
        if superseded_id not in existing_supersedes:
            existing_supersedes.append(superseded_id)
        data["supersedes"] = existing_supersedes
        with open(entry_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)
        # Archive the superseded lesson
        archive_lesson(superseded_id, reason=f"superseded_by_{lesson_id}")

    _request_index_rebuild()
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

    _request_index_rebuild()
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

    _clear_loaded_store_runtime_caches()
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
# Cold-tier trace compaction (QUA-405)
# ---------------------------------------------------------------------------

_TRACE_FILENAME_RE = re.compile(
    r"^(\d{8}_\d{6})_(.+)_([a-z][a-z0-9_]*)\.yaml$"
)

_SUMMARIES_DIR = _TRACES_DIR / "summaries"
_GAP_REGISTRY_PATH = _TRACES_DIR / "gap_registry.yaml"


def compact_traces(older_than_days: int = 30) -> dict[str, int]:
    """Compact cold-tier traces older than *older_than_days*.

    Groups trace files by (instrument, method) cohort, writes a summary
    YAML per cohort under ``traces/summaries/``, removes the original
    legacy flat trace files, and inlines old platform event sidecars back
    into their summary YAML. Returns
    ``{cohorts_summarized, traces_compacted, traces_kept}``.
    """
    from collections import defaultdict

    cutoff = datetime.now()
    cohorts: dict[tuple[str, str], list[tuple[Path, dict]]] = defaultdict(list)
    kept = 0

    if not _TRACES_DIR.is_dir():
        return {"cohorts_summarized": 0, "traces_compacted": 0, "traces_kept": 0}

    for path in sorted(_TRACES_DIR.iterdir()):
        if not path.is_file() or path.suffix != ".yaml":
            continue
        m = _TRACE_FILENAME_RE.match(path.name)
        if not m:
            kept += 1
            continue
        ts_str, instrument, method = m.group(1), m.group(2), m.group(3)
        try:
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except ValueError:
            kept += 1
            continue
        age_days = (cutoff - ts).total_seconds() / 86400
        if age_days < older_than_days:
            kept += 1
            continue
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception:
            data = {}
        cohorts[(instrument, method)].append((path, data))

    compacted = 0
    _SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    for (instrument, method), entries in cohorts.items():
        total = len(entries)
        successes = sum(1 for _, d in entries if d.get("resolved"))
        failures = total - successes
        lesson_ids = sorted(
            {str(d["lesson_id"]) for _, d in entries if d.get("lesson_id")}
        )

        summary_path = _SUMMARIES_DIR / f"{instrument}_{method}.yaml"
        existing: dict = {}
        if summary_path.exists():
            try:
                existing = yaml.safe_load(summary_path.read_text()) or {}
            except Exception:
                existing = {}

        existing_total = int(existing.get("total_attempts", 0))
        existing_successes = int(existing.get("success_count", 0))
        existing_failures = int(existing.get("failure_count", 0))
        existing_lesson_ids = list(existing.get("lesson_ids", []))

        merged_lesson_ids = sorted(set(existing_lesson_ids) | set(lesson_ids))
        summary_data = {
            "instrument": instrument,
            "method": method,
            "total_attempts": existing_total + total,
            "success_count": existing_successes + successes,
            "failure_count": existing_failures + failures,
            "lesson_ids": merged_lesson_ids,
            "last_compacted": datetime.now().isoformat(),
        }
        with open(summary_path, "w") as f:
            yaml.dump(summary_data, f, default_flow_style=False, sort_keys=False)

        for path, _ in entries:
            path.unlink(missing_ok=True)
            compacted += 1

    platform_compacted, platform_kept = _compact_platform_trace_sidecars(
        older_than_days=older_than_days,
        cutoff=cutoff,
    )

    return {
        "cohorts_summarized": len(cohorts),
        "traces_compacted": compacted + platform_compacted,
        "traces_kept": kept + platform_kept,
    }


def _compact_platform_trace_sidecars(
    *,
    older_than_days: int,
    cutoff: datetime,
) -> tuple[int, int]:
    """Inline old platform event logs back into summary YAML and drop the sidecar."""
    try:
        from trellis.agent.platform_traces import (
            _load_trace_event_dicts,
            _load_trace_summary_dict,
            _write_trace_summary_dict,
        )
    except Exception:
        return 0, 0

    platform_dir = _TRACES_DIR / "platform"
    if not platform_dir.is_dir():
        return 0, 0

    compacted = 0
    kept = 0
    for summary_path in sorted(platform_dir.glob("*.yaml")):
        events_path = summary_path.with_suffix(".events.ndjson")
        if not events_path.exists():
            continue
        latest_mtime = max(summary_path.stat().st_mtime, events_path.stat().st_mtime)
        age_days = (cutoff.timestamp() - latest_mtime) / 86400
        if age_days < older_than_days:
            kept += 1
            continue

        summary = _load_trace_summary_dict(summary_path)
        if not summary:
            kept += 1
            continue

        events = _load_trace_event_dicts(summary_path, trace=summary)
        summary["events"] = [
            _normalize_trace_event_payload(event)
            for event in events
            if isinstance(event, Mapping)
        ]
        _write_trace_summary_dict(summary_path, summary)
        events_path.unlink(missing_ok=True)
        compacted += 1

    return compacted, kept


def _normalize_trace_event_payload(event: Mapping[str, object]) -> dict[str, object]:
    """Normalize one persisted trace event into stable YAML-safe primitives."""
    details = event.get("details") or {}
    if not isinstance(details, Mapping):
        details = {}
    return {
        "event": str(event.get("event") or ""),
        "status": str(event.get("status") or "info"),
        "timestamp": str(event.get("timestamp") or ""),
        "details": _normalize_trace_yaml_value(details),
    }


def _normalize_trace_yaml_value(value: object) -> object:
    """Convert nested trace payloads into YAML-friendly primitives."""
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_trace_yaml_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_normalize_trace_yaml_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_trace_yaml_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _record_gap_aggregated(
    description: str,
    method: str,
    features: list[str] | tuple[str, ...],
) -> str:
    """Record or update a knowledge gap in the aggregated gap registry.

    Returns the ``gap_id`` of the recorded gap.
    """
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)

    registry: list[dict] = []
    if _GAP_REGISTRY_PATH.exists():
        try:
            loaded = yaml.safe_load(_GAP_REGISTRY_PATH.read_text())
            if isinstance(loaded, list):
                registry = loaded
        except Exception:
            registry = []

    now = datetime.now().isoformat()
    for entry in registry:
        if entry.get("description") == description:
            entry["occurrences"] = int(entry.get("occurrences", 1)) + 1
            entry["last_seen"] = now
            with open(_GAP_REGISTRY_PATH, "w") as f:
                yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
            return str(entry["gap_id"])

    gap_id = f"gap_{hashlib.sha256(description.encode()).hexdigest()[:8]}"
    registry.append({
        "gap_id": gap_id,
        "description": description,
        "method": method,
        "features": list(features),
        "first_seen": now,
        "last_seen": now,
        "occurrences": 1,
        "status": "open",
    })
    with open(_GAP_REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
    return gap_id


def record_semantic_extension_trace(
    *,
    request_id: str | None,
    request_text: str,
    instrument_type: str | None,
    semantic_gap: dict[str, object],
    semantic_extension: dict[str, object],
    route_method: str | None = None,
    semantic_role_ownership: dict[str, object] | None = None,
) -> str:
    """Persist a deterministic semantic-extension trace and optional lesson."""
    _SEMANTIC_EXTENSION_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_key = str(semantic_extension.get("trace_key") or _semantic_extension_trace_key(
        request_text=request_text,
        instrument_type=instrument_type,
        decision=str(semantic_extension.get("decision") or "clarification"),
        semantic_gap=semantic_gap,
    ))
    path = _SEMANTIC_EXTENSION_TRACES_DIR / f"{trace_key}.yaml"
    existing = yaml.safe_load(path.read_text()) if path.exists() else {}
    if not isinstance(existing, dict):
        existing = {}

    occurrences = int(existing.get("occurrences", 0) or 0) + 1
    timestamp = datetime.now().isoformat()
    lesson_id = existing.get("lesson_id")
    decision = str(semantic_extension.get("decision") or "clarification").strip()
    title = _semantic_extension_title(semantic_gap, semantic_extension)
    lesson_payload = build_lesson_payload(
        category="semantic",
        title=title,
        severity="medium" if decision != "clarification" else "low",
        symptom=str(semantic_gap.get("summary") or title),
        root_cause=str(semantic_gap.get("summary") or "semantic DSL gap"),
        fix=str(semantic_extension.get("recommended_next_step") or "Extend the internal DSL"),
        validation=f"Repeated semantic extension trace `{trace_key}`",
        method=route_method or decision,
        instrument=instrument_type,
        features=_semantic_extension_features(semantic_gap, semantic_extension),
        confidence=semantic_extension.get("confidence", 0.0) or 0.0,
        source_trace=str(path),
    )
    lesson_contract = validate_lesson_payload(lesson_payload)
    proposal_confidence = float(lesson_contract.normalized_payload.get("confidence") or 0.0)
    lesson_promotion_outcome = "skipped"

    if lesson_id:
        lesson_promotion_outcome = "existing"
    elif occurrences >= 2 and proposal_confidence >= 0.6:
        if lesson_contract.valid:
            with defer_index_rebuilds():
                lesson_id = capture_lesson(
                    category="semantic",
                    title=title,
                    severity="medium" if decision != "clarification" else "low",
                    symptom=str(semantic_gap.get("summary") or title),
                    root_cause=str(semantic_gap.get("summary") or "semantic DSL gap"),
                    fix=str(semantic_extension.get("recommended_next_step") or "Extend the internal DSL"),
                    validation=f"Repeated semantic extension trace `{trace_key}`",
                    method=route_method or decision,
                    instrument=instrument_type,
                    features=lesson_payload["applies_when"]["features"],
                    confidence=proposal_confidence,
                    source_trace=str(path),
                )
                if lesson_id:
                    validated = validate_lesson(lesson_id) if proposal_confidence >= 0.6 else False
                    promoted = (
                        proposal_confidence >= 0.8
                        and occurrences >= 2
                        and promote_lesson(lesson_id)
                    )
                    if promoted:
                        lesson_promotion_outcome = "promoted"
                    elif validated:
                        lesson_promotion_outcome = "validated"
                    else:
                        lesson_promotion_outcome = "captured"
                else:
                    lesson_promotion_outcome = "duplicate"
        else:
            lesson_promotion_outcome = "invalid_contract"

    trace_data = {
        "trace_key": trace_key,
        "timestamp": timestamp,
        "request_id": request_id,
        "request_text": request_text,
        "instrument_type": instrument_type,
        "route_method": route_method,
        "occurrences": occurrences,
        "semantic_gap": semantic_gap,
        "semantic_extension": semantic_extension,
        "semantic_role_ownership": semantic_role_ownership or {},
        "lesson_contract": lesson_contract.to_dict(),
        "lesson_promotion_outcome": lesson_promotion_outcome,
        "lesson_id": lesson_id,
    }
    with open(path, "w") as f:
        yaml.safe_dump(
            trace_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return str(path)


def detect_adapter_lifecycle_records(*, repo_root: Path | None = None) -> list[AdapterLifecycleRecord]:
    """Compare fresh-build adapter snapshots against checked-in adapters.

    The first pass is warning-only: when a `_fresh` module differs from the
    checked-in module it maps to, we return both the stale checked-in record
    and the newer fresh-build snapshot. Prompt formatting can then surface the
    stale adapter before another codegen pass layers on top of it.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    fresh_root = root / "trellis" / "instruments" / "_agent" / "_fresh"
    if not fresh_root.exists():
        return []

    try:
        repo_revision = get_repo_revision()
    except Exception:
        repo_revision = ""

    records: list[AdapterLifecycleRecord] = []
    for fresh_file in sorted(fresh_root.glob("*.py")):
        fresh_module_path = _module_path_from_file_path(fresh_file, root)
        checked_in_module_path = _recommended_module_path(fresh_module_path)
        checked_in_file = _recommended_file_path(checked_in_module_path)
        if not checked_in_file.exists():
            continue

        fresh_source = _normalized_source_text(fresh_file.read_text())
        checked_in_source = _normalized_source_text(checked_in_file.read_text())
        if fresh_source == checked_in_source:
            continue

        stale_hash = _source_hash(checked_in_source)
        fresh_hash = _source_hash(fresh_source)
        records.append(
            AdapterLifecycleRecord(
                adapter_id=checked_in_module_path,
                status=AdapterLifecycleStatus.STALE,
                module_path=checked_in_module_path,
                validated_against_repo_revision=repo_revision,
                replacement=fresh_module_path,
                reason="checked-in adapter differs from the validated fresh-build snapshot",
                code_hash=stale_hash,
            )
        )
        records.append(
            AdapterLifecycleRecord(
                adapter_id=checked_in_module_path,
                status=AdapterLifecycleStatus.FRESH,
                module_path=fresh_module_path,
                validated_against_repo_revision=repo_revision,
                supersedes=(checked_in_module_path,),
                replacement=checked_in_module_path,
                reason="fresh-build snapshot differs from the checked-in adapter and is awaiting runtime validation",
                code_hash=fresh_hash,
            )
        )

    return records


def _load_adapter_lifecycle_artifact_records() -> list[AdapterLifecycleRecord]:
    """Load resolved lifecycle records persisted in review/adoption artifacts."""
    records: list[AdapterLifecycleRecord] = []
    for artifact_dir_name in ("promotion_reviews", "promotion_adoptions"):
        artifact_dir = _TRACES_DIR / artifact_dir_name
        if not artifact_dir.exists():
            continue
        for path in sorted(artifact_dir.glob("*.yaml"), key=lambda item: item.name, reverse=True):
            try:
                data = yaml.safe_load(path.read_text()) or {}
            except Exception:
                continue
            if not isinstance(data, Mapping):
                continue
            lifecycle = data.get("adapter_lifecycle") or {}
            if not isinstance(lifecycle, Mapping):
                continue
            resolved = lifecycle.get("resolved") or {}
            if not isinstance(resolved, Mapping):
                resolved = {}
            record_payloads = resolved.get("records")
            if not isinstance(record_payloads, list):
                record_payloads = lifecycle.get("records")
            if not isinstance(record_payloads, list):
                continue
            for payload in record_payloads:
                if not isinstance(payload, Mapping):
                    continue
                record = _deserialize_adapter_lifecycle_record(payload)
                if record is not None:
                    records.append(record)
    return records


def resolve_adapter_lifecycle_records(
    records: list[AdapterLifecycleRecord] | tuple[AdapterLifecycleRecord, ...],
    *,
    include_archived: bool = True,
) -> list[AdapterLifecycleRecord]:
    """Merge live freshness records with persisted review/adoption lifecycle state."""
    merged: dict[tuple[str, str], AdapterLifecycleRecord] = {}
    for record in list(records) + _load_adapter_lifecycle_artifact_records():
        key = _adapter_lifecycle_record_key(record)
        existing = merged.get(key)
        # Keep the first record we see for equal-status ties. Live records are
        # processed before artifacts, and artifacts are loaded newest-first, so
        # this preserves current live state and the newest persisted review.
        if existing is None or _adapter_lifecycle_status_rank(record.status) > _adapter_lifecycle_status_rank(existing.status):
            merged[key] = record

    resolved = list(merged.values())
    if not include_archived:
        resolved = [
            record
            for record in resolved
            if record.status != AdapterLifecycleStatus.ARCHIVED
        ]
    return sorted(resolved, key=lambda record: (
        record.adapter_id,
        _adapter_lifecycle_status_rank(record.status),
        record.module_path,
        record.replacement,
    ))


def summarize_adapter_lifecycle_records(
    records: list[AdapterLifecycleRecord] | tuple[AdapterLifecycleRecord, ...],
) -> dict[str, object]:
    """Summarize lifecycle records for traces and promotion artifacts."""
    return _serialize_adapter_lifecycle_summary(list(records))


def record_promotion_candidate(
    *,
    task_id: str,
    task_title: str,
    instrument_type: str | None,
    comparison_target: str,
    preferred_method: str | None,
    payoff_class: str | None,
    module_path: str | None,
    code: str,
    attempts: int,
    platform_request_id: str | None = None,
    platform_trace_path: str | None = None,
    market_context: dict | None = None,
    cross_validation: dict | None = None,
    reference_target: bool = False,
) -> str | None:
    """Persist a fresh-build candidate snapshot for later promotion review."""
    source = (code or "").strip()
    if not source:
        return None

    timestamp = datetime.now()
    candidate_dir = _TRACES_DIR / "promotion_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    suffix = _safe_filename_fragment(comparison_target or preferred_method or "candidate")
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{task_id.lower()}_{suffix}.yaml"
    path = candidate_dir / filename
    payload = {
        "timestamp": timestamp.isoformat(),
        "task_id": task_id,
        "task_title": task_title,
        "instrument_type": instrument_type,
        "comparison_target": comparison_target,
        "preferred_method": preferred_method,
        "reference_target": reference_target,
        "payoff_class": payoff_class,
        "module_path": module_path,
        "attempts": attempts,
        "platform_request_id": platform_request_id,
        "platform_trace_path": platform_trace_path,
        "market_context": market_context or {},
        "cross_validation": cross_validation or {},
        "code_hash": hashlib.sha256(source.encode()).hexdigest()[:12],
        "code": source,
    }
    with open(path, "w") as f:
        yaml.safe_dump(
            _yaml_safe_data(payload),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return str(path)


class PromotionAdmissionError(RuntimeError):
    """Raised when a benchmark-validated candidate fails pre-admission provenance checks."""


def record_benchmark_promotion_candidate(
    *,
    benchmark_record: Mapping[str, object],
) -> str | None:
    """Persist one benchmark-validated fresh-generated adapter for later admission review."""
    generated_artifact = dict(benchmark_record.get("generated_artifact") or {})
    generated_file_path = str(generated_artifact.get("file_path") or "").strip()
    execution_module_name = str(generated_artifact.get("module_name") or "").strip()
    if not generated_file_path or not execution_module_name:
        return None
    if not bool(generated_artifact.get("is_fresh_build")):
        return None

    source_path = Path(generated_file_path)
    if not source_path.exists():
        return None
    source = source_path.read_text()
    if not source.strip():
        return None

    timestamp = datetime.now()
    candidate_dir = _TRACES_DIR / "promotion_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    task_id = str(benchmark_record.get("task_id") or "unknown").strip() or "unknown"
    filename = (
        f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{task_id.lower()}_financepy_benchmark.yaml"
    )
    path = candidate_dir / filename
    admission_target_module_name = str(
        generated_artifact.get("admission_target_module_name") or ""
    ).strip() or _recommended_module_path(execution_module_name)
    admission_target_file_path = str(
        generated_artifact.get("admission_target_file_path") or ""
    ).strip() or str(_recommended_file_path(admission_target_module_name))
    payload = {
        "timestamp": timestamp.isoformat(),
        "validation_source": "financepy_benchmark",
        "task_id": task_id,
        "task_title": str(benchmark_record.get("title") or ""),
        "instrument_type": str(benchmark_record.get("instrument_type") or ""),
        "comparison_target": str(benchmark_record.get("benchmark_execution_policy") or "benchmark"),
        "preferred_method": str(benchmark_record.get("preferred_method") or ""),
        "reference_target": True,
        "payoff_class": str(generated_artifact.get("class_name") or ""),
        "module_path": execution_module_name,
        "admission_target_module_name": admission_target_module_name,
        "admission_target_file_path": admission_target_file_path,
        "attempts": 0,
        "platform_request_id": None,
        "platform_trace_path": None,
        "market_context": {
            "market_scenario_id": str(benchmark_record.get("market_scenario_id") or ""),
        },
        "generated_artifact": generated_artifact,
        "benchmark_provenance": {
            "benchmark_kind": "financepy",
            "benchmark_campaign_id": str(benchmark_record.get("benchmark_campaign_id") or ""),
            "benchmark_run_id": str(benchmark_record.get("run_id") or ""),
            "benchmark_record_path": str(benchmark_record.get("history_path") or ""),
            "benchmark_latest_path": str(benchmark_record.get("latest_path") or ""),
            "git_sha": str(benchmark_record.get("git_sha") or ""),
            "knowledge_revision": str(benchmark_record.get("knowledge_revision") or ""),
            "status": str(benchmark_record.get("status") or ""),
            "comparison_summary": dict(benchmark_record.get("comparison_summary") or {}),
            "generated_artifact": generated_artifact,
        },
        "code_hash": hashlib.sha256(source.strip().encode()).hexdigest()[:12],
        "code": source,
    }
    with open(path, "w") as f:
        yaml.safe_dump(
            _yaml_safe_data(payload),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return str(path)


def promote_benchmark_candidate(
    candidate_path: str | Path,
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
    promoted_by: str = "",
) -> dict[str, object]:
    """Admit a benchmark-validated fresh-build candidate into ``_agent``.

    The caller selects a promotion-candidate YAML that was emitted by
    :func:`record_benchmark_promotion_candidate`.  This function refuses
    (``PromotionAdmissionError``) unless every provenance link is intact:

    * deterministic review (:func:`review_promotion_candidate`) is ``approved``
    * the candidate's recorded benchmark history file still exists
    * ``run_id``, ``git_sha``, ``knowledge_revision``, module name, and
      short code hash in the history record all match the candidate
    * the candidate's embedded source hashes to the recorded hash

    On approval the fresh-build source is written to the admission target
    under ``trellis/instruments/_agent/`` and a structured admission log is
    persisted next to the candidate under
    ``trellis/agent/knowledge/traces/promotion_admissions/``.  ``dry_run``
    performs the checks without touching the adapter tree.

    Refs: QUA-867 (epic QUA-864).
    """
    path = Path(candidate_path)
    if not path.is_file():
        raise PromotionAdmissionError(
            f"QUA-867: promotion candidate not found: {path}"
        )
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - surfaced via error path
        raise PromotionAdmissionError(
            f"QUA-867: candidate snapshot could not be parsed: {exc}"
        ) from exc
    if not isinstance(data, Mapping):
        raise PromotionAdmissionError(
            "QUA-867: candidate snapshot is not a mapping"
        )

    review = review_promotion_candidate(path, persist=False)
    if str(review.get("status") or "").strip().lower() != "approved":
        failed = sorted(
            {
                str(check.get("name") or "")
                for check in review.get("checks") or ()
                if isinstance(check, Mapping)
                and not bool(check.get("passed"))
                and bool(check.get("blocking", True))
            }
        )
        raise PromotionAdmissionError(
            "QUA-867: review rejected candidate; "
            f"failing blocking checks: {', '.join(failed) or 'unknown'}"
        )

    benchmark_provenance = data.get("benchmark_provenance") or {}
    if not isinstance(benchmark_provenance, Mapping):
        benchmark_provenance = {}
    benchmark_record_path_text = str(
        benchmark_provenance.get("benchmark_record_path") or ""
    ).strip()
    if not benchmark_record_path_text:
        raise PromotionAdmissionError(
            "QUA-867: candidate does not reference a benchmark record path"
        )
    benchmark_record_path = Path(benchmark_record_path_text)
    if not benchmark_record_path.is_file():
        raise PromotionAdmissionError(
            f"QUA-867: benchmark record is missing or unreadable: {benchmark_record_path}"
        )
    try:
        benchmark_record = json.loads(benchmark_record_path.read_text())
    except Exception as exc:
        raise PromotionAdmissionError(
            f"QUA-867: benchmark record could not be parsed: {exc}"
        ) from exc
    if not isinstance(benchmark_record, Mapping):
        raise PromotionAdmissionError(
            "QUA-867: benchmark record is not a mapping"
        )

    candidate_run_id = str(benchmark_provenance.get("benchmark_run_id") or "").strip()
    record_run_id = str(benchmark_record.get("run_id") or "").strip()
    if candidate_run_id != record_run_id or not candidate_run_id:
        raise PromotionAdmissionError(
            "QUA-867: benchmark run_id mismatch "
            f"(candidate={candidate_run_id!r}, record={record_run_id!r})"
        )

    candidate_git_sha = str(benchmark_provenance.get("git_sha") or "").strip()
    record_git_sha = str(benchmark_record.get("git_sha") or "").strip()
    if candidate_git_sha != record_git_sha or not candidate_git_sha:
        raise PromotionAdmissionError(
            "QUA-867: benchmark git_sha mismatch "
            f"(candidate={candidate_git_sha!r}, record={record_git_sha!r})"
        )

    candidate_knowledge_revision = str(
        benchmark_provenance.get("knowledge_revision") or ""
    ).strip()
    record_knowledge_revision = str(
        benchmark_record.get("knowledge_revision") or ""
    ).strip()
    if candidate_knowledge_revision != record_knowledge_revision or not candidate_knowledge_revision:
        raise PromotionAdmissionError(
            "QUA-867: benchmark knowledge_revision mismatch "
            f"(candidate={candidate_knowledge_revision!r}, record={record_knowledge_revision!r})"
        )

    record_artifact = benchmark_record.get("generated_artifact") or {}
    if not isinstance(record_artifact, Mapping):
        record_artifact = {}
    candidate_module_name = str(data.get("module_path") or "").strip()
    record_module_name = str(record_artifact.get("module_name") or "").strip()
    if candidate_module_name != record_module_name or not candidate_module_name:
        raise PromotionAdmissionError(
            "QUA-867: candidate module does not match benchmark generated_artifact "
            f"(candidate={candidate_module_name!r}, record={record_module_name!r})"
        )

    candidate_hash = str(data.get("code_hash") or "").strip()
    record_hash = str(record_artifact.get("code_hash") or "").strip()
    if not _hashes_equivalent(candidate_hash, record_hash):
        raise PromotionAdmissionError(
            "QUA-867: candidate code hash does not match benchmark artifact hash "
            f"(candidate={candidate_hash!r}, record={record_hash!r})"
        )

    candidate_record_summary = benchmark_provenance.get("comparison_summary") or {}
    if not isinstance(candidate_record_summary, Mapping):
        candidate_record_summary = {}
    live_record_summary = benchmark_record.get("comparison_summary") or {}
    if not isinstance(live_record_summary, Mapping):
        live_record_summary = {}
    live_status = str(live_record_summary.get("status") or "").strip().lower()
    if live_status != "passed":
        raise PromotionAdmissionError(
            "QUA-867: live benchmark record status is not `passed` "
            f"(status={live_status!r}); benchmark provenance and validated run record "
            "do not match"
        )
    candidate_status = str(candidate_record_summary.get("status") or "").strip().lower()
    if candidate_status and candidate_status != live_status:
        raise PromotionAdmissionError(
            "QUA-867: candidate comparison_summary status differs from live record "
            f"(candidate={candidate_status!r}, record={live_status!r})"
        )

    source_text = str(data.get("code") or "")
    if not source_text.strip():
        raise PromotionAdmissionError(
            "QUA-867: candidate snapshot has no embedded source"
        )
    computed_hash = hashlib.sha256(source_text.strip().encode()).hexdigest()
    if not _hashes_equivalent(candidate_hash, computed_hash):
        raise PromotionAdmissionError(
            "QUA-867: candidate code hash does not match embedded source "
            f"(recorded={candidate_hash!r}, computed={computed_hash[:12]!r})"
        )

    admission_target_module_name = str(
        data.get("admission_target_module_name")
        or _recommended_module_path(candidate_module_name)
    ).strip()
    if not admission_target_module_name.startswith("trellis.instruments._agent"):
        raise PromotionAdmissionError(
            "QUA-867: admission target is not under trellis.instruments._agent: "
            f"{admission_target_module_name}"
        )

    resolved_repo_root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    admission_target_file_path = Path(
        str(data.get("admission_target_file_path") or "").strip()
    ) if str(data.get("admission_target_file_path") or "").strip() else (
        _recommended_file_path(admission_target_module_name)
    )
    if not admission_target_file_path.is_absolute():
        admission_target_file_path = resolved_repo_root / admission_target_file_path
    admission_target_file_path = admission_target_file_path.resolve()
    try:
        admission_target_file_path.relative_to(resolved_repo_root.resolve())
    except ValueError:
        raise PromotionAdmissionError(
            "QUA-867: admission target resolves outside repo_root: "
            f"{admission_target_file_path}"
        )

    admission_timestamp = datetime.now()
    admission_payload: dict[str, object] = {
        "status": "would_promote" if dry_run else "promoted",
        "dry_run": bool(dry_run),
        "promoted_by": str(promoted_by or "").strip(),
        "admission_timestamp": admission_timestamp.isoformat(),
        "candidate_path": str(path),
        "admission_target_module_name": admission_target_module_name,
        "admission_target_file_path": str(admission_target_file_path),
        "benchmark_run_id": candidate_run_id,
        "benchmark_record_path": str(benchmark_record_path),
        "git_sha": candidate_git_sha,
        "knowledge_revision": candidate_knowledge_revision,
        "candidate_code_hash": candidate_hash,
        "code_hash": computed_hash,
        "review_status": str(review.get("status") or ""),
        "review_path": str(review.get("review_path") or ""),
    }

    if not dry_run:
        admission_target_file_path.parent.mkdir(parents=True, exist_ok=True)
        admission_target_file_path.write_text(source_text)
        admission_payload["code_hash"] = hashlib.sha256(
            admission_target_file_path.read_text().strip().encode()
        ).hexdigest()

    admissions_dir = _TRACES_DIR / "promotion_admissions"
    admissions_dir.mkdir(parents=True, exist_ok=True)
    task_id = _safe_filename_fragment(str(data.get("task_id") or "task"))
    target_fragment = _safe_filename_fragment(
        admission_target_module_name.rsplit(".", 1)[-1] or "adapter"
    )
    status_fragment = "dry_run" if dry_run else "admitted"
    admission_log_path = admissions_dir / (
        f"{admission_timestamp.strftime('%Y%m%d_%H%M%S')}_"
        f"{task_id}_{target_fragment}_{status_fragment}.yaml"
    )
    with open(admission_log_path, "w") as handle:
        yaml.safe_dump(
            _yaml_safe_data(admission_payload),
            handle,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    admission_payload["admission_log_path"] = str(admission_log_path)
    return admission_payload


def format_adapter_lifecycle_warnings(
    records: list[AdapterLifecycleRecord] | tuple[AdapterLifecycleRecord, ...],
    *,
    compact: bool = False,
) -> str:
    """Format stale adapter lifecycle records as prompt-ready warnings."""
    if not records:
        return ""

    stale_records = [
        record
        for record in records
        if record.status in {
            AdapterLifecycleStatus.STALE,
            AdapterLifecycleStatus.DEPRECATED,
            AdapterLifecycleStatus.ARCHIVED,
        }
    ]
    if not stale_records:
        return ""

    limit = 3 if compact else None
    selected = stale_records[:limit] if limit is not None else stale_records
    lines = ["## Adapter Freshness\n"]
    for record in selected:
        if record.status == AdapterLifecycleStatus.STALE:
            lines.append(
                f"- **STALE** `{record.adapter_id}` is behind fresh build `{record.replacement}`"
            )
            if not compact and record.reason:
                lines.append(f"  - Reason: {record.reason}")
        elif record.status == AdapterLifecycleStatus.DEPRECATED:
            lines.append(
                f"- **DEPRECATED** `{record.adapter_id}` is compatibility-only"
            )
            if not compact and record.reason:
                lines.append(f"  - Reason: {record.reason}")
        else:
            lines.append(
                f"- **ARCHIVED** `{record.adapter_id}` is out of the normal retrieval path"
            )
            if not compact and record.reason:
                lines.append(f"  - Reason: {record.reason}")
        if record.validated_against_repo_revision:
            lines.append(
                f"  - Validated against repo revision `{record.validated_against_repo_revision}`"
            )
        if record.supersedes:
            lines.append(
                "  - Supersedes: "
                + ", ".join(f"`{item}`" for item in record.supersedes)
            )
        if record.code_hash:
            lines.append(f"  - Code hash: `{record.code_hash}`")
        lines.append("")

    omitted = len(stale_records) - len(selected)
    if omitted > 0:
        lines.append(f"- [omitted {omitted} additional stale adapters]")

    lines.append(
        "Stale is warning-only for now. Upgrade the checked-in adapter before layering new behavior on top."
    )
    return "\n".join(lines)


def list_promotion_candidate_paths(*, limit: int | None = None) -> list[str]:
    """List persisted promotion candidate snapshots, newest first."""
    candidate_dir = _TRACES_DIR / "promotion_candidates"
    if not candidate_dir.exists():
        return []
    paths = sorted(
        candidate_dir.glob("*.yaml"),
        key=lambda path: path.name,
        reverse=True,
    )
    if limit is not None:
        paths = paths[: max(0, limit)]
    return [str(path) for path in paths]


def list_promotion_review_paths(
    *,
    status: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """List persisted promotion review artifacts, newest first."""
    review_dir = _TRACES_DIR / "promotion_reviews"
    if not review_dir.exists():
        return []
    paths = sorted(
        review_dir.glob("*.yaml"),
        key=lambda path: path.name,
        reverse=True,
    )
    if status is not None:
        normalized = _safe_filename_fragment(status)
        paths = [path for path in paths if path.stem.endswith(f"_{normalized}")]
    if limit is not None:
        paths = paths[: max(0, limit)]
    return [str(path) for path in paths]


def review_promotion_candidate(
    candidate_path: str | Path,
    *,
    persist: bool = True,
) -> dict[str, object]:
    """Review one promotion candidate against deterministic gate criteria."""
    path = Path(candidate_path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        data = {}

    comparison_target = str(data.get("comparison_target") or "").strip()
    payoff_class = str(data.get("payoff_class") or "").strip()
    module_path = str(data.get("module_path") or "").strip()
    validation_source = str(data.get("validation_source") or "").strip().lower()
    code = str(data.get("code") or "")
    cross_validation = data.get("cross_validation") or {}
    if not isinstance(cross_validation, dict):
        cross_validation = {}
    benchmark_provenance = data.get("benchmark_provenance") or {}
    if not isinstance(benchmark_provenance, dict):
        benchmark_provenance = {}
    generated_artifact = data.get("generated_artifact") or {}
    if not isinstance(generated_artifact, dict):
        generated_artifact = {}
    successful_targets = _as_str_set(cross_validation.get("successful_targets"))
    passed_targets = _as_str_set(cross_validation.get("passed_targets"))
    price_errors = cross_validation.get("price_errors") or {}
    prices = cross_validation.get("prices") or {}
    deviations_pct = cross_validation.get("deviations_pct") or {}
    tolerance_pct = float(cross_validation.get("tolerance_pct", 0.0) or 0.0)
    recommended_module_path = (
        str(data.get("admission_target_module_name") or "").strip()
        or _recommended_module_path(module_path)
    )
    recommended_file_path = Path(
        str(data.get("admission_target_file_path") or "").strip()
    ) if str(data.get("admission_target_file_path") or "").strip() else _recommended_file_path(recommended_module_path)
    adapter_records = [
        record
        for record in detect_adapter_lifecycle_records()
        if record.adapter_id == recommended_module_path
        or record.module_path == module_path
    ]
    adapter_repo_revision = ""
    for record in adapter_records:
        if record.validated_against_repo_revision:
            adapter_repo_revision = record.validated_against_repo_revision
            break

    parsed = _parse_source(code)
    checks = [
        _review_check("candidate_exists", path.exists(), "candidate snapshot path is readable"),
        _review_check("code_present", bool(code.strip()), "candidate snapshot includes generated source"),
        _review_check(
            "source_parses",
            parsed is not None,
            "generated source parses as Python",
            failure_detail="generated source does not parse",
        ),
        _review_check(
            "payoff_class_defined",
            parsed is not None and payoff_class in _defined_class_names(parsed),
            f"generated source defines `{payoff_class}`",
            failure_detail=f"`{payoff_class}` is not defined in generated source",
        ),
        _review_check(
            "fresh_module_path",
            "._fresh." in module_path,
            "candidate module path comes from the fresh-build namespace",
            failure_detail=f"module path `{module_path}` is not a fresh-build route",
        ),
    ]
    if validation_source == "financepy_benchmark":
        benchmark_record_path_str = str(
            benchmark_provenance.get("benchmark_record_path") or ""
        ).strip()
        benchmark_record_path = (
            Path(benchmark_record_path_str) if benchmark_record_path_str else None
        )
        benchmark_record: dict[str, object] = {}
        if benchmark_record_path is not None and benchmark_record_path.is_file():
            try:
                loaded = json.loads(benchmark_record_path.read_text())
                if isinstance(loaded, dict):
                    benchmark_record = loaded
            except Exception:
                benchmark_record = {}
        benchmark_summary = benchmark_provenance.get("comparison_summary") or benchmark_record.get(
            "comparison_summary"
        ) or {}
        if not isinstance(benchmark_summary, dict):
            benchmark_summary = {}
        benchmark_artifact = benchmark_provenance.get("generated_artifact") or benchmark_record.get(
            "generated_artifact"
        ) or {}
        if not isinstance(benchmark_artifact, dict):
            benchmark_artifact = {}
        benchmark_deviations = benchmark_summary.get("output_deviation_pct") or {}
        if not isinstance(benchmark_deviations, Mapping):
            benchmark_deviations = {}
        benchmark_tolerance = float(benchmark_summary.get("tolerance_pct", 0.0) or 0.0)
        checks.extend(
            [
                _review_check(
                    "benchmark_record_exists",
                    benchmark_record_path is not None and benchmark_record_path.is_file(),
                    "benchmark record path is readable",
                    failure_detail="benchmark record path is missing or unreadable",
                ),
                _review_check(
                    "benchmark_priced",
                    str(benchmark_provenance.get("status") or benchmark_record.get("status") or "").strip()
                    == "priced",
                    "benchmark run priced successfully",
                    failure_detail=(
                        "benchmark run status was "
                        f"`{benchmark_provenance.get('status') or benchmark_record.get('status')}`"
                    ),
                ),
                _review_check(
                    "benchmark_comparison_passed",
                    str(benchmark_summary.get("status") or "").strip().lower() == "passed",
                    "benchmark comparison passed",
                    failure_detail=f"benchmark comparison status was `{benchmark_summary.get('status')}`",
                ),
                _review_check(
                    "benchmark_generated_artifact_present",
                    bool(str(benchmark_artifact.get("module_name") or "").strip())
                    and bool(str(benchmark_artifact.get("file_path") or "").strip()),
                    "benchmark recorded generated-artifact provenance",
                    failure_detail="benchmark generated-artifact provenance is missing",
                ),
                _review_check(
                    "benchmark_generated_artifact_fresh",
                    bool(benchmark_artifact.get("is_fresh_build")),
                    "benchmark executed a fresh-generated artifact",
                    failure_detail="benchmark generated artifact was not marked fresh",
                ),
                _review_check(
                    "benchmark_module_matches_candidate",
                    str(benchmark_artifact.get("module_name") or "").strip() == module_path,
                    "candidate module matches the validated benchmark artifact",
                    failure_detail=(
                        f"benchmark module `{benchmark_artifact.get('module_name')}` "
                        f"did not match candidate module `{module_path}`"
                    ),
                ),
                _review_check(
                    "benchmark_hash_matches_candidate",
                    _hashes_equivalent(
                        benchmark_artifact.get("code_hash"),
                        data.get("code_hash"),
                    ),
                    "candidate source hash matches the validated benchmark artifact",
                    failure_detail=(
                        f"benchmark code hash `{benchmark_artifact.get('code_hash')}` "
                        f"did not match candidate hash `{data.get('code_hash')}`"
                    ),
                ),
                _review_check(
                    "benchmark_within_tolerance",
                    bool(benchmark_deviations)
                    and all(float(value) <= benchmark_tolerance for value in benchmark_deviations.values()),
                    "benchmark outputs stayed within tolerance",
                    failure_detail=(
                        f"benchmark deviations {dict(benchmark_deviations)} exceeded tolerance "
                        f"{benchmark_tolerance}"
                    ),
                ),
            ]
        )
    else:
        checks.extend(
            [
                _review_check(
                    "platform_trace_recorded",
                    _path_exists(data.get("platform_trace_path")),
                    "candidate references an existing platform trace",
                    failure_detail="platform trace path is missing or unreadable",
                ),
                _review_check(
                    "cross_validation_passed",
                    str(cross_validation.get("status") or "").strip().lower() == "passed",
                    "task-level cross-validation passed",
                    failure_detail=f"cross-validation status was `{cross_validation.get('status')}`",
                ),
                _review_check(
                    "target_priced",
                    comparison_target in prices,
                    f"cross-validation recorded a price for `{comparison_target}`",
                    failure_detail=f"no price recorded for `{comparison_target}`",
                ),
                _review_check(
                    "target_has_no_price_error",
                    comparison_target not in price_errors,
                    f"`{comparison_target}` has no cross-validation pricing error",
                    failure_detail=f"price error for `{comparison_target}`: {price_errors.get(comparison_target)}",
                ),
                _review_check(
                    "target_marked_successful",
                    comparison_target in successful_targets and comparison_target in passed_targets,
                    f"`{comparison_target}` is marked successful and passed in cross-validation",
                    failure_detail=f"`{comparison_target}` missing from successful/passed targets",
                ),
                _review_check(
                    "target_within_tolerance",
                    comparison_target in deviations_pct
                    and float(deviations_pct.get(comparison_target, tolerance_pct + 1.0)) <= tolerance_pct,
                    f"`{comparison_target}` stayed within configured tolerance",
                    failure_detail=(
                        f"deviation for `{comparison_target}` was {deviations_pct.get(comparison_target)!r} "
                        f"with tolerance {tolerance_pct}"
                    ),
                ),
            ]
        )
    approved = all(check["passed"] for check in checks if check["blocking"])
    adapter_lifecycle_stage = "deprecated" if approved else "stale"
    review = {
        "timestamp": datetime.now().isoformat(),
        "candidate_path": str(path),
        "task_id": data.get("task_id"),
        "task_title": data.get("task_title"),
        "instrument_type": data.get("instrument_type"),
        "comparison_target": comparison_target,
        "preferred_method": data.get("preferred_method"),
        "payoff_class": payoff_class,
        "module_path": module_path,
        "validation_source": validation_source or "cross_validation",
        "recommended_module_path": recommended_module_path,
        "recommended_file_path": str(recommended_file_path),
        "status": "approved" if approved else "rejected",
        "approved": approved,
        "checks": checks,
        "adapter_lifecycle": _adapter_lifecycle_snapshot(
            adapter_records,
            stage=adapter_lifecycle_stage,
            adapter_id=recommended_module_path,
            replacement=module_path,
            repo_revision=adapter_repo_revision,
        ),
    }
    if persist:
        review["review_path"] = _record_promotion_review(review)
    else:
        review["review_path"] = None
    return review


def adopt_promotion_candidate(
    review_path: str | Path,
    *,
    dry_run: bool = False,
    persist: bool = True,
) -> dict[str, object]:
    """Adopt an approved promotion candidate into its checked-in route path."""
    path = Path(review_path)
    review = yaml.safe_load(path.read_text()) or {}
    if not isinstance(review, dict):
        review = {}

    candidate_path = Path(str(review.get("candidate_path") or ""))
    candidate = yaml.safe_load(candidate_path.read_text()) if candidate_path.exists() else {}
    if not isinstance(candidate, dict):
        candidate = {}

    adapter_lifecycle = review.get("adapter_lifecycle") or {}
    if not isinstance(adapter_lifecycle, Mapping):
        adapter_lifecycle = {}
    adapter_raw_records: list[AdapterLifecycleRecord] = []
    raw_block = adapter_lifecycle.get("raw") or {}
    if isinstance(raw_block, Mapping):
        raw_payloads = raw_block.get("records")
        if isinstance(raw_payloads, list):
            for payload in raw_payloads:
                if isinstance(payload, Mapping):
                    record = _deserialize_adapter_lifecycle_record(payload)
                    if record is not None:
                        adapter_raw_records.append(record)

    source = str(candidate.get("code") or "")
    recommended_file = Path(str(review.get("recommended_file_path") or ""))
    parse_ok = _parse_source(source) is not None
    source_hash = hashlib.sha256(source.strip().encode()).hexdigest()[:12] if source.strip() else ""
    snapshot_hash = str(candidate.get("code_hash") or "").strip()
    existing_text = recommended_file.read_text() if recommended_file.exists() else ""
    existing_hash = hashlib.sha256(existing_text.encode()).hexdigest()[:12] if existing_text else ""
    previous_exists = recommended_file.exists()
    checks = [
        _review_check(
            "review_exists",
            path.exists(),
            "promotion review path is readable",
            failure_detail="promotion review path is missing",
        ),
        _review_check(
            "review_approved",
            bool(review.get("approved")),
            "review approved the candidate for adoption",
            failure_detail="review is not approved",
        ),
        _review_check(
            "candidate_exists",
            candidate_path.exists(),
            "candidate snapshot referenced by the review exists",
            failure_detail="candidate snapshot is missing",
        ),
        _review_check(
            "candidate_source_parses",
            parse_ok,
            "candidate source parses as Python",
            failure_detail="candidate source failed to parse",
        ),
        _review_check(
            "candidate_hash_matches",
            _hashes_equivalent(source_hash, snapshot_hash),
            "candidate source hash matches the recorded snapshot hash",
            failure_detail=f"candidate hash mismatch: expected {snapshot_hash!r}, got {source_hash!r}",
        ),
        _review_check(
            "target_within_repo",
            _is_safe_adoption_target(recommended_file),
            "recommended target path stays inside trellis/instruments/_agent",
            failure_detail=f"unsafe adoption target `{recommended_file}`",
        ),
    ]
    blocked = any(not check["passed"] for check in checks if check["blocking"])
    normalized_source = source.rstrip() + "\n" if source.strip() else ""
    changed = existing_text != normalized_source
    status = "blocked" if blocked else ("ready" if dry_run else "adopted")
    adopted = status == "adopted"
    if not adapter_raw_records:
        adapter_raw_records = [
            record
            for record in detect_adapter_lifecycle_records()
            if record.adapter_id == str(review.get("recommended_module_path") or "")
            or record.module_path == str(review.get("module_path") or "")
        ]
    adapter_repo_revision = ""
    for record in adapter_raw_records:
        if record.validated_against_repo_revision:
            adapter_repo_revision = record.validated_against_repo_revision
            break
    if adopted:
        adapter_lifecycle_stage = "archived"
    elif bool(review.get("approved")):
        adapter_lifecycle_stage = "deprecated"
    else:
        adapter_lifecycle_stage = "stale"
    if adopted:
        recommended_file.parent.mkdir(parents=True, exist_ok=True)
        recommended_file.write_text(normalized_source)
    adoption = {
        "timestamp": datetime.now().isoformat(),
        "review_path": str(path),
        "candidate_path": str(candidate_path) if candidate_path else None,
        "status": status,
        "adopted": adopted,
        "dry_run": dry_run,
        "task_id": review.get("task_id"),
        "comparison_target": review.get("comparison_target"),
        "target_module_path": review.get("recommended_module_path"),
        "target_file_path": str(recommended_file) if str(recommended_file) else None,
        "changed": changed,
        "previous_file_exists": previous_exists,
        "previous_code_hash": existing_hash or None,
        "adopted_code_hash": snapshot_hash or None,
        "checks": checks,
        "adapter_lifecycle": _adapter_lifecycle_snapshot(
            adapter_raw_records,
            stage=adapter_lifecycle_stage,
            adapter_id=str(review.get("recommended_module_path") or ""),
            replacement=str(review.get("module_path") or ""),
            repo_revision=adapter_repo_revision,
        ),
    }
    if persist:
        adoption["adoption_path"] = _record_promotion_adoption(adoption)
    else:
        adoption["adoption_path"] = None
    return adoption


# ---------------------------------------------------------------------------
# Recalibration (trace cross-validation for stuck candidates)
# ---------------------------------------------------------------------------

def recalibrate_candidates(*, dry_run: bool = False) -> dict[str, int]:
    """Cross-validate candidate lessons against traces to unstick low-confidence candidates.

    For each candidate lesson:
    - If its source_trace exists and shows resolved=True, boost confidence by +0.15
    - If the lesson's applies_when.features overlap with features from 3+ successful
      traces, boost by +0.10
    - After boosting, re-run auto-validation gate (confidence >= 0.6 -> validated)

    Returns: {boosted: N, validated: N, promoted: N, unchanged: N, errors: N}
    """
    import random

    stats: dict[str, int] = {
        "boosted": 0,
        "validated": 0,
        "promoted": 0,
        "unchanged": 0,
        "errors": 0,
    }
    recalibration_details: list[dict[str, object]] = []

    # 1. Load all candidate lesson entries
    entries_dir = _LESSONS_DIR / "entries"
    if not entries_dir.exists():
        return stats

    candidates: list[tuple[Path, dict]] = []
    for path in sorted(entries_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text())
        except Exception:
            stats["errors"] += 1
            continue
        if not isinstance(data, dict):
            continue
        if str(data.get("status", "")).strip().lower() == "candidate":
            candidates.append((path, data))

    if not candidates:
        return stats

    # 2. Pre-load a sample of resolved traces for feature cross-matching
    trace_files = list(_TRACES_DIR.glob("*.yaml"))
    if len(trace_files) > 200:
        trace_files = random.sample(trace_files, 200)

    resolved_traces: list[dict] = []
    for tf in trace_files:
        try:
            tdata = yaml.safe_load(tf.read_text())
        except Exception:
            continue
        if isinstance(tdata, dict) and tdata.get("resolved") is True:
            resolved_traces.append(tdata)

    # 3. Process each candidate
    with defer_index_rebuilds():
        for entry_path, data in candidates:
            lesson_id = str(data.get("id", "")).strip()
            if not lesson_id:
                stats["errors"] += 1
                continue

            original_confidence = float(data.get("confidence", 0.0))
            boost_total = 0.0
            reasons: list[str] = []

            # 3a. Check source_trace for resolved=True
            source_trace = data.get("source_trace")
            if source_trace and isinstance(source_trace, str):
                trace_path = Path(source_trace)
                if not trace_path.is_absolute():
                    trace_path = _TRACES_DIR / trace_path
                if trace_path.exists():
                    try:
                        tdata = yaml.safe_load(trace_path.read_text())
                        if isinstance(tdata, dict) and tdata.get("resolved") is True:
                            boost_total += 0.15
                            reasons.append("source_trace_resolved")
                    except Exception:
                        pass

            # 3b. Feature cross-matching against resolved traces
            applies_when = data.get("applies_when") or {}
            if isinstance(applies_when, dict):
                lesson_features = set(_as_list(applies_when.get("features")))
            else:
                lesson_features = set()

            if lesson_features:
                matching_trace_count = 0
                for trace in resolved_traces:
                    plan = trace.get("pricing_plan") or {}
                    trace_features = set(_as_list(plan.get("features")))
                    if lesson_features & trace_features:
                        matching_trace_count += 1
                if matching_trace_count >= 3:
                    boost_total += 0.10
                    reasons.append(f"feature_cross_match_{matching_trace_count}")

            if boost_total <= 0:
                stats["unchanged"] += 1
                continue

            # Apply boost
            new_confidence = min(1.0, round(original_confidence + boost_total, 4))

            detail: dict[str, object] = {
                "lesson_id": lesson_id,
                "original_confidence": original_confidence,
                "boost": boost_total,
                "new_confidence": new_confidence,
                "reasons": reasons,
                "actions": [],
            }

            if not dry_run:
                boost_confidence(lesson_id, boost_total)

            stats["boosted"] += 1
            actions: list[str] = ["boosted"]

            # 3c. Auto-validation gate
            if new_confidence >= 0.6:
                if not dry_run:
                    if validate_lesson(lesson_id):
                        stats["validated"] += 1
                        actions.append("validated")

                        # 3d. Auto-promotion gate
                        if new_confidence >= 0.8:
                            if promote_lesson(lesson_id):
                                stats["promoted"] += 1
                                actions.append("promoted")
                else:
                    stats["validated"] += 1
                    actions.append("validated (dry_run)")
                    if new_confidence >= 0.8:
                        stats["promoted"] += 1
                        actions.append("promoted (dry_run)")

            detail["actions"] = actions
            recalibration_details.append(detail)

    # 4. Rebuild index once (if mutations occurred)
    if not dry_run and stats["boosted"] > 0:
        rebuild_index()

    # 5. Log results
    log_path = _TRACES_DIR / "recalibration_log.yaml"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "stats": stats,
        "details": recalibration_details,
    }
    try:
        _TRACES_DIR.mkdir(parents=True, exist_ok=True)
        existing_log: list[dict] = []
        if log_path.exists():
            try:
                content = yaml.safe_load(log_path.read_text())
                if isinstance(content, list):
                    existing_log = content
            except Exception:
                pass
        existing_log.append(log_entry)
        with open(log_path, "w") as f:
            yaml.dump(existing_log, f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
    except Exception as exc:
        logger.warning("Failed to write recalibration log: %s", exc)

    return stats


# ---------------------------------------------------------------------------
# Distillation (periodic, offline)
# ---------------------------------------------------------------------------

def distill() -> dict[str, int]:
    """Run a periodic distillation pass over the lesson index.

    Three actions, each with explicit thresholds:

    1. Auto-promote: validated lessons with confidence >= 0.8 are promoted
       to production status.
    2. Archive stale candidates: lessons still in 'candidate' status after
       30 days with confidence < 0.6 are archived automatically.
    3. Principle detection: categories with 3 or more promoted lessons are
       flagged as candidates for extracting a shared design principle.

    Returns a dict with counts: ``promoted``, ``archived``,
    ``principle_candidates``.
    """
    stats = {"promoted": 0, "archived": 0, "principle_candidates": 0}

    # Pre-pass: recalibrate stuck candidates via trace cross-validation
    index = _load_index()
    entries = index.get("entries", [])
    has_candidates = any(
        e.get("status") == "candidate" for e in entries
    )
    if has_candidates:
        recalibrate_candidates()
        # Reload index after recalibration may have changed statuses
        index = _load_index()
        entries = index.get("entries", [])
    now = datetime.now()

    # Group by category for principle detection
    by_category: dict[str, list[dict]] = {}
    for e in entries:
        cat = e.get("category", "unknown")
        by_category.setdefault(cat, []).append(e)

    with defer_index_rebuilds():
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

    # Optional: draft principle candidates via LLM
    if stats["principle_candidates"] > 0:
        try:
            from trellis.agent.config import get_default_model  # noqa: F811
            model = get_default_model()
            if model:
                draft_principle_candidates(model=model)
        except Exception:
            pass  # LLM unavailable — skip silently

    return stats


# ---------------------------------------------------------------------------
# Principle extraction
# ---------------------------------------------------------------------------


def _draft_principle_prompt(category: str, lessons: list[dict]) -> str:
    """Build prompt for principle synthesis."""
    lines = [
        f"Synthesize these {len(lessons)} lessons in the '{category}' "
        "category into ONE concise principle rule (a single sentence, "
        "imperative, actionable):\n",
    ]
    for l in lessons:  # noqa: E741
        lines.append(
            f"- [{l.get('id', '?')}] {l.get('title', '?')}: {l.get('fix', '?')}"
        )
    lines.append(
        '\nOutput ONLY a JSON object: {"rule": "...", "rationale": "..."}'
    )
    return "\n".join(lines)


def draft_principle_candidates(
    *,
    model: str | None = None,
    _llm_fn: object | None = None,
) -> list[dict]:
    """Draft principle candidates from categories with 3+ promoted lessons.

    Writes candidates to traces/principle_candidates/{category}_{timestamp}.yaml.
    Returns list of drafted candidates.  Never auto-promotes to
    canonical/principles.yaml.

    Parameters
    ----------
    model : str, optional
        LLM model identifier (ignored when *_llm_fn* is provided).
    _llm_fn : callable, optional
        ``(prompt: str) -> dict`` — injectable LLM function for testing.
        When *None*, uses ``trellis.agent.config.llm_generate_json``.
    """
    entries_dir = _LESSONS_DIR / "entries"
    if not entries_dir.exists():
        return []

    # Load full entry data for promoted lessons, grouped by category
    by_category: dict[str, list[dict]] = {}
    for path in sorted(entries_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("status") != "promoted":
            continue
        cat = data.get("category", "unknown")
        by_category.setdefault(cat, []).append(data)

    # Resolve LLM callable
    if _llm_fn is None:
        try:
            from trellis.agent.config import llm_generate_json, load_env
            load_env()
            _llm_fn = lambda prompt: llm_generate_json(prompt, model=model)  # noqa: E731
        except Exception:
            return []

    out_dir = _TRACES_DIR / "principle_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict] = []
    now_iso = datetime.now().isoformat()

    for cat, lessons in by_category.items():
        if len(lessons) < 3:
            continue
        prompt = _draft_principle_prompt(cat, lessons)
        try:
            result = _llm_fn(prompt)
        except Exception:
            logger.warning("LLM call failed for category %s", cat)
            continue

        if not isinstance(result, dict):
            continue

        rule = result.get("rule", "")
        rationale = result.get("rationale", "")
        if not rule:
            continue

        lesson_ids = [l.get("id", "?") for l in lessons]
        candidate = {
            "candidate_principle": {
                "rule": rule,
                "rationale": rationale,
                "derived_from": lesson_ids,
                "category": cat,
                "confidence": 0.7,
                "status": "candidate",
                "drafted_at": now_iso,
            }
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{cat}_{ts}.yaml"
        out_path.write_text(yaml.dump(candidate, default_flow_style=False, sort_keys=False))
        candidates.append(candidate)
        logger.info("Drafted principle candidate for category '%s' -> %s", cat, out_path)

    return candidates


def review_principle_candidate(path: str | Path) -> dict:
    """Load and return a principle candidate for review."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidate not found: {p}")
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict) or "candidate_principle" not in data:
        raise ValueError(f"Invalid candidate format in {p}")
    return data


def adopt_principle_candidate(path: str | Path) -> bool:
    """Approve a principle candidate: append to canonical/principles.yaml."""
    candidate_data = review_principle_candidate(path)
    cp = candidate_data["candidate_principle"]

    principles_path = _KNOWLEDGE_DIR / "canonical" / "principles.yaml"
    existing: list[dict] = []
    if principles_path.exists():
        existing = yaml.safe_load(principles_path.read_text()) or []
        if not isinstance(existing, list):
            existing = []

    # Determine next ID
    max_id = 0
    for p in existing:
        pid = str(p.get("id", ""))
        if pid.startswith("P") and pid[1:].isdigit():
            max_id = max(max_id, int(pid[1:]))
    next_id = f"P{max_id + 1}"

    new_principle = {
        "id": next_id,
        "rule": cp["rule"],
        "derived_from": cp.get("derived_from", []),
        "category": cp.get("category", "unknown"),
    }
    existing.append(new_principle)
    principles_path.write_text(
        yaml.dump(existing, default_flow_style=False, sort_keys=False)
    )
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_index() -> dict:
    """Load the lesson index used by the promotion pipeline."""
    if _INDEX_PATH.exists():
        return yaml.safe_load(_INDEX_PATH.read_text()) or {}
    return {}


def _scan_entry_metadata() -> list[dict[str, object]]:
    """Load lightweight metadata from canonical lesson entry files."""
    entries_dir = _LESSONS_DIR / "entries"
    if not entries_dir.exists():
        return []

    entries: list[dict[str, object]] = []
    for path in sorted(entries_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text())
        except Exception as exc:
            logger.warning("Skipping unreadable lesson entry %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            logger.warning("Skipping lesson entry %s: expected mapping", path)
            continue

        report = validate_lesson_payload(data)
        if not report.valid:
            logger.warning(
                "Skipping lesson entry %s: %s",
                path,
                "; ".join(report.errors),
            )
            continue

        data = report.normalized_payload
        lesson_id = str(data.get("id") or "").strip()
        title = str(data.get("title") or "").strip()
        severity = str(data.get("severity") or "low").strip()
        category = str(data.get("category") or "").strip()
        status = str(data.get("status") or "candidate").strip()
        applies_when = data.get("applies_when") or {}
        if not isinstance(applies_when, dict):
            applies_when = {}
        if not lesson_id or not title:
            logger.warning("Skipping lesson entry %s: missing id/title", path)
            continue
        entries.append(
            {
                "id": lesson_id,
                "title": title,
                "severity": severity,
                "category": category,
                "status": status,
                "supersedes": _as_list(data.get("supersedes")),
                "applies_when": {
                    "method": _as_list(applies_when.get("method")),
                    "features": _as_list(applies_when.get("features")),
                    "instrument": _as_list(applies_when.get("instrument")),
                    "error_signature": applies_when.get("error_signature"),
                },
            }
        )

    return entries


def rebuild_index() -> dict:
    """Rebuild ``index.yaml`` from canonical lesson entry files."""
    entries = sorted(_scan_entry_metadata(), key=lambda entry: str(entry["id"]))
    index = {
        "entries": entries,
        "settings": {"max_prompt_entries": 7},
    }
    _LESSONS_DIR.mkdir(parents=True, exist_ok=True)
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_INDEX_PATH, "w") as f:
        yaml.dump(
            index,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    _reload_loaded_store()
    return index


def _generate_id(category: str, entries: list[dict[str, object]]) -> str:
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
        str(e["id"]) for e in entries
        if str(e.get("id", "")).startswith(prefix + "_")
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


def _as_list(value: object) -> list[object]:
    """Normalize a scalar or sequence value into a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if value is None:
        return []
    return [value]


def _word_overlap(a: str, b: str) -> float:
    """Return a simple normalized token-overlap score for duplicate detection."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def _lesson_context_key(payload: Mapping[str, object]) -> tuple[object, ...]:
    """Return the semantic context key used for duplicate-candidate screening."""
    applies_when = payload.get("applies_when")
    if not isinstance(applies_when, Mapping):
        applies_when = {}

    def _normalized_tuple(value: object) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(item).strip()
                    for item in _as_list(value)
                    if str(item).strip()
                }
            )
        )

    return (
        str(payload.get("category") or "").strip(),
        _normalized_tuple(applies_when.get("method")),
        _normalized_tuple(applies_when.get("features")),
        _normalized_tuple(applies_when.get("instrument")),
        str(applies_when.get("error_signature") or "").strip(),
    )


def _lesson_signature_text(payload: Mapping[str, object]) -> str:
    """Return the text surface used for semantic duplicate screening."""
    return " ".join(
        part
        for part in (
            str(payload.get("symptom") or "").strip(),
            str(payload.get("root_cause") or "").strip(),
            str(payload.get("fix") or "").strip(),
        )
        if part
    )


def _semantic_tokens(text: str) -> set[str]:
    """Return normalized content tokens for lesson-duplicate screening."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _LESSON_TEXT_STOPWORDS
    }


def _semantic_text_overlap(text_a: str, text_b: str) -> float:
    """Return Jaccard similarity over normalized content tokens."""
    words_a = _semantic_tokens(text_a)
    words_b = _semantic_tokens(text_b)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _load_lesson_entry(lesson_id: str) -> dict[str, object]:
    """Load one lesson payload for duplicate screening."""
    if not lesson_id:
        return {}
    entry_path = _LESSONS_DIR / "entries" / f"{lesson_id}.yaml"
    if not entry_path.exists():
        return {}
    data = yaml.safe_load(entry_path.read_text())
    if not isinstance(data, Mapping):
        return {}
    return dict(data)


def _fix_word_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity over whitespace-tokenized words."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _detect_supersedes(
    new_lesson_id: str,
    new_category: str,
    new_fix: str,
    entries_dir: Path,
    *,
    threshold: float = 0.70,
) -> str | None:
    """Find an older promoted lesson in the same category with high fix overlap.

    Scans *entries_dir* for promoted lessons whose category matches
    *new_category* and whose ``fix`` field has a Jaccard word-overlap
    score above *threshold* with *new_fix*.  Returns the ID of the
    superseded lesson, or ``None``.
    """
    if not entries_dir.is_dir():
        return None

    for path in sorted(entries_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        entry_id = str(data.get("id") or "").strip()
        if not entry_id or entry_id == new_lesson_id:
            continue

        entry_status = str(data.get("status") or "").strip().lower()
        if entry_status != "promoted":
            continue

        entry_category = str(data.get("category") or "").strip().lower()
        if entry_category != new_category.strip().lower():
            continue

        entry_fix = str(data.get("fix") or "").strip()
        if not entry_fix:
            continue

        if _fix_word_overlap(new_fix, entry_fix) > threshold:
            return entry_id

    return None


def backfill_supersedes(*, dry_run: bool = False) -> dict[str, list[str]]:
    """One-time scan of promoted lessons for supersedes relationships.

    For every pair of promoted lessons in the same category, if the newer
    lesson's fix has >70 % Jaccard word-overlap with the older one, the
    newer lesson supersedes the older.

    Returns ``{new_lesson_id: [superseded_ids]}``.
    """
    entries_dir = _LESSONS_DIR / "entries"
    if not entries_dir.is_dir():
        return {}

    # Collect promoted lessons
    promoted: list[dict[str, str]] = []
    for path in sorted(entries_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if str(data.get("status") or "").strip().lower() != "promoted":
            continue
        promoted.append({
            "id": str(data.get("id") or "").strip(),
            "category": str(data.get("category") or "").strip().lower(),
            "fix": str(data.get("fix") or "").strip(),
            "created": str(data.get("created") or "").strip(),
        })

    result: dict[str, list[str]] = {}
    seen_superseded: set[str] = set()

    for i, newer in enumerate(promoted):
        if not newer["id"] or not newer["fix"]:
            continue
        for older in promoted[:i]:
            if not older["id"] or not older["fix"]:
                continue
            if older["id"] in seen_superseded:
                continue
            if older["category"] != newer["category"]:
                continue
            if _fix_word_overlap(newer["fix"], older["fix"]) > 0.70:
                result.setdefault(newer["id"], []).append(older["id"])
                seen_superseded.add(older["id"])

    if not dry_run:
        with defer_index_rebuilds():
            for new_id, old_ids in result.items():
                entry_path = entries_dir / f"{new_id}.yaml"
                if not entry_path.exists():
                    continue
                data = yaml.safe_load(entry_path.read_text())
                if not isinstance(data, dict):
                    continue
                existing = list(data.get("supersedes") or [])
                for old_id in old_ids:
                    if old_id not in existing:
                        existing.append(old_id)
                data["supersedes"] = existing
                with open(entry_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False,
                              sort_keys=False, allow_unicode=True)
                for old_id in old_ids:
                    archive_lesson(old_id, reason=f"superseded_by_{new_id}")

    return result


def _safe_filename_fragment(value: str) -> str:
    """Normalize a free-form identifier into a stable filename fragment."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return normalized or "candidate"


def _parse_source(source: str) -> ast.Module | None:
    """Parse candidate source code and return the AST when valid."""
    if not source.strip():
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _defined_class_names(module: ast.Module) -> set[str]:
    """Return top-level class names defined in parsed source."""
    return {
        node.name
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }


def _recommended_module_path(module_path: str) -> str:
    """Map a fresh-build module path onto its checked-in route path."""
    if "._fresh." in module_path:
        return module_path.replace("._fresh.", ".")
    return module_path


def _recommended_file_path(module_path: str) -> Path:
    """Map a Trellis module path to an on-disk file path inside the repo."""
    relative = Path(*module_path.split("."))
    return (_REPO_ROOT / relative).with_suffix(".py")


def _module_path_from_file_path(path: Path, repo_root: Path) -> str:
    """Convert a repository file path into a dotted module path."""
    return path.resolve().relative_to(repo_root.resolve()).with_suffix("").as_posix().replace("/", ".")


def _normalized_source_text(source: str) -> str:
    """Normalize source text before comparing fresh-build and checked-in code."""
    if not source.strip():
        return ""
    return source.rstrip() + "\n"


def _source_hash(source: str) -> str:
    """Return a short stable hash for normalized source text."""
    normalized = _normalized_source_text(source)
    return hashlib.sha256(normalized.encode()).hexdigest()[:12] if normalized else ""


def _hashes_equivalent(left: object, right: object) -> bool:
    """Treat short recorded hashes as valid prefixes of full content hashes."""
    lhs = str(left or "").strip().lower()
    rhs = str(right or "").strip().lower()
    if not lhs or not rhs:
        return False
    return lhs == rhs or lhs.startswith(rhs) or rhs.startswith(lhs)


def _yaml_safe_data(value: object) -> object:
    """Convert nested payloads into YAML-safe plain Python primitives."""
    if isinstance(value, Mapping):
        return {str(key): _yaml_safe_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_yaml_safe_data(item) for item in value]
    return value


def _review_check(
    name: str,
    passed: bool,
    success_detail: str,
    *,
    failure_detail: str | None = None,
    blocking: bool = True,
) -> dict[str, object]:
    """Build one normalized deterministic promotion-gate check."""
    return {
        "name": name,
        "passed": bool(passed),
        "blocking": blocking,
        "detail": success_detail if passed else (failure_detail or success_detail),
    }


def _record_promotion_review(review: dict[str, object]) -> str:
    """Persist one promotion review artifact next to promotion candidates."""
    review_dir = _TRACES_DIR / "promotion_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    task_id = _safe_filename_fragment(str(review.get("task_id") or "task"))
    target = _safe_filename_fragment(str(review.get("comparison_target") or "candidate"))
    status = _safe_filename_fragment(str(review.get("status") or "review"))
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_id}_{target}_{status}.yaml"
    path = review_dir / filename
    with open(path, "w") as f:
        yaml.safe_dump(
            _yaml_safe_data(review),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return str(path)


def _record_promotion_adoption(adoption: dict[str, object]) -> str:
    """Persist one promotion-adoption decision artifact."""
    adoption_dir = _TRACES_DIR / "promotion_adoptions"
    adoption_dir.mkdir(parents=True, exist_ok=True)
    task_id = _safe_filename_fragment(str(adoption.get("task_id") or "task"))
    target = _safe_filename_fragment(str(adoption.get("comparison_target") or "candidate"))
    status = _safe_filename_fragment(str(adoption.get("status") or "adoption"))
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_id}_{target}_{status}.yaml"
    path = adoption_dir / filename
    with open(path, "w") as f:
        yaml.safe_dump(
            _yaml_safe_data(adoption),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return str(path)


def _path_exists(value: object) -> bool:
    """Return True when a string path points to an existing file."""
    return isinstance(value, str) and bool(value.strip()) and Path(value).exists()


def _is_safe_adoption_target(path: Path) -> bool:
    """Allow adoption only into repo-local agent route modules."""
    if not str(path).strip():
        return False
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(_REPO_ROOT.resolve())
    except Exception:
        return False
    return relative.parts[:3] == ("trellis", "instruments", "_agent") and resolved.suffix == ".py"


def _as_str_set(values: object) -> set[str]:
    """Normalize one string or string iterable into a set."""
    if isinstance(values, str):
        return {values} if values else set()
    if isinstance(values, (list, tuple, set)):
        return {item for item in values if isinstance(item, str) and item}
    return set()


def _semantic_extension_trace_key(
    *,
    request_text: str,
    instrument_type: str | None,
    decision: str,
    semantic_gap: dict[str, object],
) -> str:
    """Build a stable trace key for one semantic extension shape."""
    concept = semantic_gap.get("semantic_concept")
    if not isinstance(concept, dict):
        concept = {}
    payload = "|".join(
        (
            request_text.strip().lower(),
            (instrument_type or "").strip().lower(),
            decision.strip().lower(),
            str(concept.get("semantic_id") or ""),
            str(concept.get("resolution_kind") or ""),
            str(concept.get("extension_kind") or ""),
            ",".join(str(item) for item in semantic_gap.get("gap_types", ())),
            ",".join(str(item) for item in semantic_gap.get("missing_contract_fields", ())),
            ",".join(str(item) for item in semantic_gap.get("missing_market_inputs", ())),
            ",".join(str(item) for item in semantic_gap.get("missing_runtime_primitives", ())),
            ",".join(
                str(item)
                for item in (
                    semantic_gap.get("missing_binding_helpers")
                    or semantic_gap.get("missing_route_helpers")
                    or ()
                )
            ),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _semantic_extension_title(
    semantic_gap: dict[str, object],
    semantic_extension: dict[str, object],
) -> str:
    """Build a stable lesson title for one semantic extension pattern."""
    summary = str(semantic_extension.get("summary") or semantic_gap.get("summary") or "").strip()
    if summary:
        return f"_semantic_extension_{_safe_filename_fragment(summary)[:48]}"
    decision = str(semantic_extension.get("decision") or "clarification").strip()
    return f"_semantic_extension_{_safe_filename_fragment(decision)[:48]}"


def _semantic_extension_features(
    semantic_gap: dict[str, object],
    semantic_extension: dict[str, object],
) -> list[str]:
    """Derive stable feature tags from a semantic extension proposal."""
    features: list[str] = []
    for key in (
        "missing_contract_fields",
        "missing_market_inputs",
        "missing_runtime_primitives",
        "missing_binding_helpers",
        "missing_knowledge_artifacts",
        "proposed_contract_fields",
        "proposed_market_inputs",
        "proposed_runtime_primitives",
        "proposed_binding_helpers",
        "proposed_knowledge_artifacts",
    ):
        values = semantic_extension.get(key) or semantic_gap.get(key) or ()
        if not values and key == "missing_binding_helpers":
            values = semantic_extension.get("missing_route_helpers") or semantic_gap.get("missing_route_helpers") or ()
        if not values and key == "proposed_binding_helpers":
            values = semantic_extension.get("proposed_route_helpers") or semantic_gap.get("proposed_route_helpers") or ()
        if isinstance(values, str):
            values = (values,)
        for value in values:
            text = str(value).strip()
            if text and text not in features:
                features.append(text)
    return features
