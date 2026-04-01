"""Generated skill layer over lessons, cookbooks, principles, and route hints.

This module does not change the authored source-of-truth artifacts. It projects
the existing knowledge surfaces into one deterministic, typed retrieval index.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml

from trellis.agent.knowledge.import_registry import get_repo_revision
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.knowledge.schema import GeneratedSkillIndex, SkillIndexManifest, SkillRecord

_KNOWLEDGE_DIR = Path(__file__).parent
_CANONICAL_DIR = _KNOWLEDGE_DIR / "canonical"
_LESSON_ENTRIES_DIR = _KNOWLEDGE_DIR / "lessons" / "entries"
_SKILL_INDEX_CACHE: dict[tuple[object, ...], GeneratedSkillIndex] = {}
_ACTIVE_SKILL_STATUSES = {"active", "validated", "promoted", "fresh"}


def clear_skill_index_cache() -> None:
    """Clear the generated skill-index cache."""
    _SKILL_INDEX_CACHE.clear()


def load_skill_index() -> GeneratedSkillIndex:
    """Return the deterministic generated skill index for the current repo."""
    key = _cache_key()
    cached = _SKILL_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    lesson_rows = _load_lesson_rows()
    lesson_by_id = {row["id"]: row for row in lesson_rows}

    records: list[SkillRecord] = []
    records.extend(_project_lessons(lesson_rows))
    records.extend(_project_principles(lesson_by_id))
    records.extend(_project_cookbooks())
    records.extend(_project_route_hints())
    records = sorted(records, key=lambda record: (record.kind, record.skill_id))

    manifest = SkillIndexManifest(
        repo_revision=get_repo_revision(),
        source_paths=tuple(str(path) for path in _source_paths()),
        source_fingerprints=tuple(_fingerprint(path) for path in _source_paths()),
        record_count=len(records),
        kind_counts=dict(sorted(Counter(record.kind for record in records).items())),
    )
    index = GeneratedSkillIndex(manifest=manifest, records=tuple(records))
    _SKILL_INDEX_CACHE[key] = index
    return index


def get_skill_record(skill_id: str) -> SkillRecord | None:
    """Return one generated skill record by identifier."""
    for record in load_skill_index().records:
        if record.skill_id == skill_id:
            return record
    return None


def query_skill_records(
    *,
    kind: str | None = None,
    kinds: Iterable[str] | None = None,
    instrument_type: str | None = None,
    method_family: str | None = None,
    route_family: str | None = None,
    failure_bucket: str | None = None,
    concept: str | None = None,
    status: str | None = None,
    include_inactive: bool = False,
) -> tuple[SkillRecord, ...]:
    """Return generated skills filtered by deterministic metadata only."""
    requested_kinds = {value for value in (kinds or ()) if value}
    if kind:
        requested_kinds.add(kind)

    normalized_method = normalize_method(method_family) if method_family else ""
    normalized_instrument = _normalize_key(instrument_type)
    normalized_route = _normalize_key(route_family)
    normalized_failure_bucket = _normalize_key(failure_bucket)
    normalized_concept = _normalize_key(concept)
    normalized_status = _normalize_key(status)

    results: list[SkillRecord] = []
    for record in load_skill_index().records:
        if requested_kinds and record.kind not in requested_kinds:
            continue
        if not include_inactive and record.status and _normalize_key(record.status) not in _ACTIVE_SKILL_STATUSES:
            continue
        if normalized_status and _normalize_key(record.status) != normalized_status:
            continue
        if normalized_instrument and normalized_instrument not in {_normalize_key(item) for item in record.instrument_types}:
            continue
        if normalized_method and normalized_method not in {normalize_method(item) for item in record.method_families}:
            continue
        if normalized_route and normalized_route not in {_normalize_key(item) for item in record.route_families}:
            continue
        if normalized_failure_bucket and normalized_failure_bucket not in {
            _normalize_key(item) for item in record.failure_buckets
        }:
            continue
        if normalized_concept and normalized_concept not in {_normalize_key(item) for item in record.concepts}:
            continue
        results.append(record)
    return tuple(results)


def _cache_key() -> tuple[object, ...]:
    return (
        get_repo_revision(),
        tuple((str(path), path.stat().st_mtime_ns) for path in _source_paths()),
    )


def _source_paths() -> tuple[Path, ...]:
    paths = [
        _CANONICAL_DIR / "principles.yaml",
        _CANONICAL_DIR / "cookbooks.yaml",
        _CANONICAL_DIR / "routes.yaml",
    ]
    if _LESSON_ENTRIES_DIR.exists():
        paths.extend(sorted(_LESSON_ENTRIES_DIR.glob("*.yaml")))
    return tuple(path for path in paths if path.exists())


def _fingerprint(path: Path) -> str:
    return f"{path}:{path.stat().st_mtime_ns}"


def _load_yaml(path: Path, default):
    if not path.exists():
        return default
    data = yaml.safe_load(path.read_text())
    return default if data is None else data


def _load_lesson_rows() -> list[dict]:
    rows: list[dict] = []
    if not _LESSON_ENTRIES_DIR.exists():
        return rows
    for path in sorted(_LESSON_ENTRIES_DIR.glob("*.yaml")):
        data = _load_yaml(path, {})
        if not isinstance(data, dict) or not data.get("id"):
            continue
        data = dict(data)
        data["_source_path"] = str(path)
        rows.append(data)
    return rows


def _project_lessons(rows: list[dict]) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    for row in rows:
        applies_when = row.get("applies_when") or {}
        if not isinstance(applies_when, dict):
            applies_when = {}
        features = _sorted_unique(applies_when.get("features", ()))
        records.append(
            SkillRecord(
                skill_id=f"lesson:{row['id']}",
                kind="lesson",
                title=str(row.get("title") or row["id"]),
                summary=str(row.get("fix") or row.get("root_cause") or row.get("symptom") or ""),
                source_artifact=str(row["id"]),
                source_path=str(row.get("_source_path") or ""),
                instrument_types=_sorted_unique(applies_when.get("instrument", ())),
                method_families=_sorted_unique(normalize_method(item) for item in applies_when.get("method", ()) if item),
                route_families=(),
                failure_buckets=(),
                concepts=features,
                tags=_sorted_unique(
                    value
                    for value in (
                        f"category:{row.get('category')}" if row.get("category") else "",
                        f"severity:{row.get('severity')}" if row.get("severity") else "",
                        f"derived_principle:{row.get('derived_principle')}" if row.get("derived_principle") else "",
                    )
                    if value
                ),
                origin="captured",
                parents=(),
                supersedes=_sorted_unique(f"lesson:{item}" for item in row.get("supersedes", ()) if item),
                status=str(row.get("status") or "promoted"),
                confidence=float(row.get("confidence") or 1.0),
                updated_at=str(row.get("created") or ""),
                source_kind="lesson_entry",
            )
        )
    return records


def _project_principles(lesson_by_id: dict[str, dict]) -> list[SkillRecord]:
    path = _CANONICAL_DIR / "principles.yaml"
    data = _load_yaml(path, [])
    records: list[SkillRecord] = []
    for row in data:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        derived_from = tuple(str(item) for item in row.get("derived_from", ()) if item)
        parent_lessons = [lesson_by_id[item] for item in derived_from if item in lesson_by_id]

        methods = _sorted_unique(
            normalize_method(method)
            for lesson in parent_lessons
            for method in ((lesson.get("applies_when") or {}).get("method", ()))
            if method
        )
        instruments = _sorted_unique(
            instrument
            for lesson in parent_lessons
            for instrument in ((lesson.get("applies_when") or {}).get("instrument", ()))
            if instrument
        )
        concepts = _sorted_unique(
            feature
            for lesson in parent_lessons
            for feature in ((lesson.get("applies_when") or {}).get("features", ()))
            if feature
        )
        updated_at = max((str(lesson.get("created") or "") for lesson in parent_lessons), default="")
        records.append(
            SkillRecord(
                skill_id=f"principle:{row['id']}",
                kind="principle",
                title=str(row.get("id") or ""),
                summary=str(row.get("rule") or ""),
                source_artifact=str(row["id"]),
                source_path=str(path),
                instrument_types=instruments,
                method_families=methods,
                route_families=(),
                failure_buckets=(),
                concepts=concepts,
                tags=_sorted_unique(
                    value
                    for value in (
                        f"category:{row.get('category')}" if row.get("category") else "",
                        "derived_from_lessons" if derived_from else "",
                    )
                    if value
                ),
                origin="derived" if derived_from else "canonical",
                parents=_sorted_unique(f"lesson:{item}" for item in derived_from),
                supersedes=(),
                status="promoted",
                confidence=1.0,
                updated_at=updated_at,
                source_kind="canonical",
            )
        )
    return records


def _project_cookbooks() -> list[SkillRecord]:
    path = _CANONICAL_DIR / "cookbooks.yaml"
    data = _load_yaml(path, {})
    records: list[SkillRecord] = []
    if not isinstance(data, dict):
        return records

    for raw_method in sorted(data):
        entry = data.get(raw_method) or {}
        if not isinstance(entry, dict):
            continue
        method = normalize_method(raw_method)
        solution_contracts = entry.get("solution_contracts", ())
        market_data = _sorted_unique(
            item
            for contract in solution_contracts
            if isinstance(contract, dict)
            for item in contract.get("market_data", ())
            if item
        )
        tags = _sorted_unique(
            value
            for value in (
                *(f"solution_contract:{contract.get('id')}" for contract in solution_contracts if isinstance(contract, dict) and contract.get("id")),
                f"version:{entry.get('version')}" if entry.get("version") else "",
            )
            if value
        )
        records.append(
            SkillRecord(
                skill_id=f"cookbook:{method}",
                kind="cookbook",
                title=method,
                summary=str(entry.get("description") or ""),
                source_artifact=method,
                source_path=str(path),
                instrument_types=_sorted_unique(entry.get("applicable_instruments", ())),
                method_families=(method,),
                route_families=(),
                failure_buckets=(),
                concepts=market_data,
                tags=tags,
                origin="canonical",
                parents=(),
                supersedes=(),
                status="promoted",
                confidence=1.0,
                updated_at=str(entry.get("version") or ""),
                source_kind="canonical",
            )
        )
    return records


def _project_route_hints() -> list[SkillRecord]:
    records: list[SkillRecord] = []
    from trellis.agent.route_registry import load_route_registry

    registry = load_route_registry()
    routes_path = _CANONICAL_DIR / "routes.yaml"
    for route in sorted(registry.routes, key=lambda item: item.id):
        source_path = str(Path(route.discovered_from)) if route.discovered_from else str(routes_path)
        route_families = _sorted_unique(
            [route.route_family, *(item.route_family for item in route.conditional_route_family or ())]
        )
        concepts = _sorted_unique(
            [
                *(route.match_payoff_traits or ()),
                *(route.match_required_market_data or ()),
                *(route.market_data_access.required.keys()),
            ]
        )
        tags = _sorted_unique(
            [
                f"engine_family:{route.engine_family}",
                f"route:{route.id}",
                *(f"adapter:{adapter}" for adapter in route.adapters),
            ]
        )

        route_helpers = [primitive for primitive in route.primitives if primitive.role == "route_helper"]
        if route_helpers:
            helper = route_helpers[0]
            records.append(
                SkillRecord(
                    skill_id=f"route_hint:{route.id}:route-helper",
                    kind="route_hint",
                    title=f"{route.id} route helper",
                    summary=(
                        "Use the selected route helper directly inside `evaluate()`; "
                        "do not rebuild the process, engine, or discount glue manually."
                    ),
                    source_artifact=route.id,
                    source_path=source_path,
                    instrument_types=_sorted_unique(route.match_instruments or ()),
                    method_families=_sorted_unique(normalize_method(item) for item in route.match_methods),
                    route_families=route_families,
                    failure_buckets=(),
                    concepts=concepts,
                    tags=_sorted_unique([*tags, f"module:{helper.module}", f"symbol:{helper.symbol}"]),
                    origin="canonical" if not route.discovered_from else "captured",
                    parents=(),
                    supersedes=(),
                    status=route.status,
                    confidence=route.confidence,
                    updated_at="",
                    precedence_rank=100,
                    instruction_type="hard_constraint",
                    source_kind="route_card",
                )
            )

        needs_schedule_builder = any(
            primitive.role == "schedule_builder" or primitive.symbol == "generate_schedule"
            for primitive in route.primitives
        )
        if needs_schedule_builder:
            for skill_id, title, summary, precedence_rank in (
                (
                    f"route_hint:{route.id}:schedule-builder",
                    f"{route.id} schedule builder",
                    "Use `trellis.core.date_utils.generate_schedule` to build ordered dates before pricing.",
                    90,
                ),
                (
                    f"route_hint:{route.id}:schedule-body",
                    f"{route.id} avoid hard-coded schedule grids",
                    "Do not hard-code observation or payment grids inside the payoff body.",
                    85,
                ),
            ):
                records.append(
                    SkillRecord(
                        skill_id=skill_id,
                        kind="route_hint",
                        title=title,
                        summary=summary,
                        source_artifact=route.id,
                        source_path=source_path,
                        instrument_types=_sorted_unique(route.match_instruments or ()),
                        method_families=_sorted_unique(normalize_method(item) for item in route.match_methods),
                        route_families=route_families,
                        failure_buckets=(),
                        concepts=concepts,
                        tags=tags,
                        origin="canonical" if not route.discovered_from else "captured",
                        parents=(),
                        supersedes=(),
                        status=route.status,
                        confidence=route.confidence,
                        updated_at="",
                        precedence_rank=precedence_rank,
                        instruction_type="route_hint",
                        source_kind="route_card",
                    )
                )

        for index, note in enumerate(route.notes, start=1):
            instruction_type = "route_hint" if "do not" in note.lower() else "historical_note"
            records.append(
                SkillRecord(
                    skill_id=f"route_hint:{route.id}:note:{index}",
                    kind="route_hint",
                    title=f"{route.id} note {index}",
                    summary=note,
                    source_artifact=route.id,
                    source_path=source_path,
                    instrument_types=_sorted_unique(route.match_instruments or ()),
                    method_families=_sorted_unique(normalize_method(item) for item in route.match_methods),
                    route_families=route_families,
                    failure_buckets=(),
                    concepts=concepts,
                    tags=_sorted_unique([*tags, f"note_index:{index}"]),
                    origin="canonical" if not route.discovered_from else "captured",
                    parents=(),
                    supersedes=(),
                    status=route.status,
                    confidence=route.confidence,
                    updated_at="",
                    precedence_rank=50 - index,
                    instruction_type=instruction_type,
                    source_kind="route_card",
                )
            )

        for index, note in enumerate(route.dynamic_notes, start=1):
            records.append(
                SkillRecord(
                    skill_id=f"route_hint:{route.id}:dynamic:{index}",
                    kind="route_hint",
                    title=f"{route.id} dynamic note {index}",
                    summary=note.template,
                    source_artifact=route.id,
                    source_path=source_path,
                    instrument_types=_sorted_unique(route.match_instruments or ()),
                    method_families=_sorted_unique(normalize_method(item) for item in route.match_methods),
                    route_families=route_families,
                    failure_buckets=(),
                    concepts=concepts,
                    tags=_sorted_unique([*tags, f"dynamic_source:{note.source}", f"dynamic_function:{note.function}"]),
                    origin="canonical" if not route.discovered_from else "captured",
                    parents=(),
                    supersedes=(),
                    status=route.status,
                    confidence=route.confidence,
                    updated_at="",
                    precedence_rank=40 - index,
                    instruction_type="historical_note",
                    source_kind="route_card",
                )
            )
    return records


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value).strip()}))


def _normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower().replace(" ", "_")
