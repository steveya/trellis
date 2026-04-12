"""Generated skill layer over lessons, cookbooks, principles, and route hints.

This module does not change the authored source-of-truth artifacts. It projects
the existing knowledge surfaces into one deterministic, typed retrieval index.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from trellis.agent.knowledge.import_registry import get_repo_revision
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.knowledge.schema import GeneratedSkillIndex, SkillIndexManifest, SkillRecord

_KNOWLEDGE_DIR = Path(__file__).parent
_CANONICAL_DIR = _KNOWLEDGE_DIR / "canonical"
_LESSON_ENTRIES_DIR = _KNOWLEDGE_DIR / "lessons" / "entries"
_SKILL_INDEX_CACHE: dict[tuple[object, ...], GeneratedSkillIndex] = {}
_SKILL_LINEAGE_CACHE: dict[tuple[object, ...], dict[str, dict[str, Any]]] = {}
_ACTIVE_SKILL_STATUSES = {"active", "validated", "promoted", "fresh"}


def clear_skill_index_cache() -> None:
    """Clear the generated skill-index cache."""
    _SKILL_INDEX_CACHE.clear()
    _SKILL_LINEAGE_CACHE.clear()


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
    cookbook_records = _project_cookbooks()
    records.extend(cookbook_records)
    records.extend(
        _project_route_hints(cookbook_ids={record.skill_id for record in cookbook_records})
    )
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


def get_skill_lineage(skill_id: str) -> dict[str, Any] | None:
    """Return surfaced lineage metadata for one generated skill record."""
    lineage = load_skill_lineage_index().get(skill_id)
    return dict(lineage) if lineage is not None else None


def load_skill_lineage_index() -> dict[str, dict[str, Any]]:
    """Return the canonical lineage index for the generated skill layer."""
    key = _cache_key()
    cached = _SKILL_LINEAGE_CACHE.get(key)
    if cached is not None:
        return cached

    index = load_skill_index()
    children_by_parent: dict[str, set[str]] = {}
    replacements_by_target: dict[str, set[str]] = {}
    same_source_groups: dict[tuple[str, str, str], set[str]] = {}

    for record in index.records:
        for parent in record.parents:
            children_by_parent.setdefault(parent, set()).add(record.skill_id)
        for target in record.supersedes:
            replacements_by_target.setdefault(target, set()).add(record.skill_id)
        same_source_groups.setdefault(
            (record.source_kind, record.source_artifact, record.source_path),
            set(),
        ).add(record.skill_id)

    lineage_index = {
        record.skill_id: {
            "skill_id": record.skill_id,
            "kind": record.kind,
            "origin": record.origin,
            "source_kind": record.source_kind,
            "source_artifact": record.source_artifact,
            "source_path": record.source_path,
            "lineage_status": record.lineage_status,
            "lineage_evidence": record.lineage_evidence,
            "parents": record.parents,
            "supersedes": record.supersedes,
            "children": _sorted_unique(children_by_parent.get(record.skill_id, ())),
            "replaced_by": _sorted_unique(replacements_by_target.get(record.skill_id, ())),
            "same_source": _sorted_unique(
                item
                for item in same_source_groups.get(
                    (record.source_kind, record.source_artifact, record.source_path),
                    (),
                )
                if item != record.skill_id
            ),
        }
        for record in index.records
    }
    _SKILL_LINEAGE_CACHE[key] = lineage_index
    return lineage_index


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


def select_prompt_skill_artifacts(
    text: str,
    *,
    audience: str,
    stage: str,
    instrument_type: str | None = None,
    pricing_method: str | None = None,
    route_ids: Iterable[str] = (),
    route_families: Iterable[str] = (),
    knowledge_surface: str = "compact",
) -> list[dict[str, Any]]:
    """Return small prompt-ready skill guidance for one agent audience/stage.

    The selector is deterministic and intentionally conservative: it keeps the
    generated skill layer as a thin guidance surface over the existing
    KnowledgeStore payloads rather than replacing them outright.
    """
    instrument_tokens = {
        token
        for token in (
            _normalize_key(instrument_type),
            "credit_default_swap" if _normalize_key(instrument_type) == "cds" else "",
            "cds" if _normalize_key(instrument_type) == "credit_default_swap" else "",
        )
        if token
    }
    method_token = normalize_method(pricing_method) if pricing_method else ""
    normalized_route_ids = tuple(
        sorted(
            {
                _normalize_key(item)
                for item in route_ids
                if _normalize_key(item)
            }
        )
    )
    normalized_route_families = tuple(
        sorted(
            {
                _normalize_key(item)
                for item in route_families
                if _normalize_key(item)
            }
        )
    )
    kind_order = _prompt_skill_kind_order(
        audience=audience,
        stage=stage,
    )

    candidates: list[SkillRecord] = []
    for record in load_skill_index().records:
        if record.kind not in kind_order:
            continue
        if record.status and _normalize_key(record.status) not in _ACTIVE_SKILL_STATUSES:
            continue
        if not _skill_record_matches_scope(
            record,
            instrument_tokens=instrument_tokens,
            method_token=method_token,
            route_ids=normalized_route_ids,
            route_families=normalized_route_families,
        ):
            continue
        candidates.append(record)

    candidates.sort(
        key=lambda record: _skill_prompt_rank(
            record,
            kind_order=kind_order,
            instrument_tokens=instrument_tokens,
            method_token=method_token,
            route_ids=normalized_route_ids,
            route_families=normalized_route_families,
        )
    )

    artifacts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in candidates:
        if record.skill_id in seen_ids:
            continue
        seen_ids.add(record.skill_id)
        summary = " ".join(str(record.summary or "").split())
        if not summary:
            continue
        if len(summary) > 220:
            summary = summary[:217].rstrip() + "..."
        artifacts.append(
            {
                "id": record.skill_id,
                "kind": record.kind,
                "title": str(record.title or "").strip(),
                "summary": summary,
                "origin": record.origin,
                "parents": list(record.parents),
                "supersedes": list(record.supersedes),
                "lineage_status": record.lineage_status,
                "lineage_summary": _lineage_summary(record),
            }
        )

    return _prune_prompt_skill_artifacts(
        text,
        artifacts,
        knowledge_surface=knowledge_surface,
    )


def append_prompt_skill_artifacts(
    text: str,
    artifacts: Iterable[dict[str, Any]],
    *,
    heading: str = "## Generated Skills",
) -> str:
    """Append one generated-skill section to prompt text when artifacts exist."""
    selected = [
        artifact
        for artifact in artifacts
        if str(artifact.get("summary") or "").strip()
    ]
    if not selected:
        return text

    section_lines = [heading]
    section_lines.extend(
        _render_prompt_skill_artifact(artifact)
        for artifact in selected
    )
    suffix = "\n".join(section_lines)
    if text:
        return f"{text}\n\n{suffix}"
    return suffix


def augment_prompt_with_skill_records(
    text: str,
    *,
    audience: str,
    stage: str,
    instrument_type: str | None = None,
    pricing_method: str | None = None,
    route_ids: Iterable[str] = (),
    route_families: Iterable[str] = (),
    knowledge_surface: str = "compact",
    heading: str = "## Generated Skills",
) -> tuple[str, list[dict[str, Any]]]:
    """Select prompt-ready skills and append them to the supplied text."""
    artifacts = select_prompt_skill_artifacts(
        text,
        audience=audience,
        stage=stage,
        instrument_type=instrument_type,
        pricing_method=pricing_method,
        route_ids=route_ids,
        route_families=route_families,
        knowledge_surface=knowledge_surface,
    )
    return append_prompt_skill_artifacts(text, artifacts, heading=heading), artifacts


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
                lineage_status="superseding" if row.get("supersedes") else "source_root",
                lineage_evidence=("lesson.supersedes",) if row.get("supersedes") else ("lesson.entry",),
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
                lineage_status="derived" if derived_from else "source_root",
                lineage_evidence=("principles.derived_from",) if derived_from else ("principles.entry",),
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
                lineage_status="source_root",
                lineage_evidence=("cookbooks.entry",),
            )
        )
    return records


def _project_route_hints(*, cookbook_ids: set[str]) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    from trellis.agent.route_registry import load_route_registry

    registry = load_route_registry()
    routes_path = _CANONICAL_DIR / "routes.yaml"
    for route in sorted(registry.routes, key=lambda item: item.id):
        source_path = str(Path(route.discovered_from)) if route.discovered_from else str(routes_path)
        route_families = _sorted_unique(
            [route.route_family, *(item.route_family for item in route.conditional_route_family or ())]
        )
        cookbook_candidates = _sorted_unique(
            f"cookbook:{normalize_method(method)}"
            for method in route.match_methods
            if f"cookbook:{normalize_method(method)}" in cookbook_ids
        )
        concepts = _sorted_unique(
            [
                *(route.match_payoff_traits or ()),
                *(route.match_required_market_data or ()),
                *(route.market_data_access.required.keys()),
            ]
        )
        if len(cookbook_candidates) == 1:
            parent_cookbooks = cookbook_candidates
            lineage_status = "derived"
            lineage_evidence = ("route.match_method_to_cookbook",)
        elif len(cookbook_candidates) > 1:
            parent_cookbooks = ()
            lineage_status = "advisory"
            lineage_evidence = ("route.match_method_to_cookbook_ambiguous",)
        else:
            parent_cookbooks = ()
            lineage_status = "source_root"
            lineage_evidence = ("route_card",)
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
                    parents=parent_cookbooks,
                    supersedes=(),
                    status=route.status,
                    confidence=route.confidence,
                    updated_at="",
                    precedence_rank=100,
                    instruction_type="hard_constraint",
                    source_kind="route_card",
                    lineage_status=lineage_status,
                    lineage_evidence=lineage_evidence,
                )
            )

        schedule_builder = next(
            (
                primitive
                for primitive in route.primitives
                if primitive.role == "schedule_builder" or primitive.symbol == "generate_schedule"
            ),
            None,
        )
        if schedule_builder is not None:
            for skill_id, title, summary, precedence_rank in (
                (
                    f"route_hint:{route.id}:schedule-builder",
                    f"{route.id} schedule builder",
                    (
                        f"Use `{schedule_builder.module}.{schedule_builder.symbol}` "
                        "to build the route schedule before pricing."
                    ),
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
                        parents=parent_cookbooks,
                        supersedes=(),
                        status=route.status,
                        confidence=route.confidence,
                        updated_at="",
                        precedence_rank=precedence_rank,
                        instruction_type="route_hint",
                        source_kind="route_card",
                        lineage_status=lineage_status,
                        lineage_evidence=lineage_evidence,
                    )
                )

        for index, note in enumerate(route.notes, start=1):
            records.append(
                SkillRecord(
                    skill_id=f"route_hint:{route.id}:note:{index}",
                    kind="historical_note",
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
                    parents=parent_cookbooks,
                    supersedes=(),
                    status=route.status,
                    confidence=route.confidence,
                    updated_at="",
                    precedence_rank=50 - index,
                    instruction_type="historical_note",
                    source_kind="route_card",
                    lineage_status=lineage_status,
                    lineage_evidence=lineage_evidence,
                )
            )

        for index, note in enumerate(route.dynamic_notes, start=1):
            records.append(
                SkillRecord(
                    skill_id=f"route_hint:{route.id}:dynamic:{index}",
                    kind="historical_note",
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
                    parents=parent_cookbooks,
                    supersedes=(),
                    status=route.status,
                    confidence=route.confidence,
                    updated_at="",
                    precedence_rank=40 - index,
                    instruction_type="historical_note",
                    source_kind="route_card",
                    lineage_status=lineage_status,
                    lineage_evidence=lineage_evidence,
                )
            )
    return records


def _render_prompt_skill_artifact(artifact: dict[str, Any]) -> str:
    """Render one prompt-ready skill artifact with compact lineage context."""
    lineage = str(artifact.get("lineage_summary") or "").strip()
    suffix = f" [lineage: {lineage}]" if lineage else ""
    return f"- [{artifact['kind']}] {artifact['title']}: {artifact['summary']}{suffix}"


def _lineage_summary(record: SkillRecord) -> str:
    """Return a compact lineage caption for one generated skill record."""
    if record.parents:
        return f"derived from {_short_lineage_list(record.parents)}"
    if record.supersedes:
        return f"supersedes {_short_lineage_list(record.supersedes)}"
    return ""


def _short_lineage_list(values: Iterable[str]) -> str:
    """Render at most two lineage ids with a count suffix when needed."""
    ordered = [str(value) for value in values if str(value).strip()]
    if not ordered:
        return ""
    head = ordered[:2]
    if len(ordered) <= 2:
        return ", ".join(head)
    return f"{', '.join(head)} +{len(ordered) - 2} more"


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value).strip()}))


def _normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower().replace(" ", "_")


def _prompt_skill_kind_order(
    *,
    audience: str,
    stage: str,
) -> tuple[str, ...]:
    """Return the preferred generated-skill kinds for one prompt stage."""
    normalized_stage = _normalize_key(stage)
    if audience == "routing":
        return ("route_hint", "principle", "lesson", "cookbook")
    if audience == "review":
        if normalized_stage == "critic_review":
            return ("principle", "lesson", "route_hint")
        return ("principle", "lesson", "route_hint", "cookbook")
    if normalized_stage in {
        "code_generation_failed",
        "import_validation_failed",
        "semantic_validation_failed",
        "lite_review_failed",
        "actual_market_smoke_failed",
        "validation_failed",
        "comparison_insufficient_results",
        "retry_build",
    }:
        return ("route_hint", "cookbook", "principle", "lesson")
    return ("cookbook", "principle", "lesson", "route_hint")


def _skill_record_matches_scope(
    record: SkillRecord,
    *,
    instrument_tokens: set[str],
    method_token: str,
    route_ids: tuple[str, ...],
    route_families: tuple[str, ...],
) -> bool:
    """Apply deterministic prompt-scope matching for one generated skill."""
    record_methods = {_normalize_key(item) for item in record.method_families}
    if method_token and record_methods and method_token not in record_methods:
        return False

    record_instruments = {_normalize_key(item) for item in record.instrument_types}
    record_route_families = {_normalize_key(item) for item in record.route_families}
    record_tags = {_normalize_key(item) for item in record.tags}
    route_id_tags = {f"route:{route_id}" for route_id in route_ids if route_id}

    if record.kind == "route_hint":
        if route_id_tags & record_tags:
            return True
        if route_families and record_route_families and set(route_families) & record_route_families:
            return True
        if instrument_tokens and record_instruments and instrument_tokens & record_instruments:
            return True
        return False

    if instrument_tokens and record_instruments:
        return bool(instrument_tokens & record_instruments)
    return True


def _skill_prompt_rank(
    record: SkillRecord,
    *,
    kind_order: tuple[str, ...],
    instrument_tokens: set[str],
    method_token: str,
    route_ids: tuple[str, ...],
    route_families: tuple[str, ...],
) -> tuple[object, ...]:
    """Order prompt-ready skills by scope fit and stability."""
    record_instruments = {_normalize_key(item) for item in record.instrument_types}
    record_methods = {_normalize_key(item) for item in record.method_families}
    record_route_families = {_normalize_key(item) for item in record.route_families}
    record_tags = {_normalize_key(item) for item in record.tags}
    route_id_score = int(bool({f"route:{route_id}" for route_id in route_ids if route_id} & record_tags))
    route_family_score = int(bool(set(route_families) & record_route_families))
    instrument_score = int(bool(instrument_tokens & record_instruments))
    method_score = int(bool(method_token and method_token in record_methods))
    hard_constraint_score = int(str(getattr(record, "instruction_type", "") or "") == "hard_constraint")
    return (
        -hard_constraint_score,
        -instrument_score,
        -method_score,
        -route_family_score,
        -route_id_score,
        kind_order.index(record.kind),
        -int(getattr(record, "precedence_rank", 0) or 0),
        -float(getattr(record, "confidence", 0.0) or 0.0),
        str(getattr(record, "skill_id", "") or ""),
    )


def _prune_prompt_skill_artifacts(
    text: str,
    artifacts: list[dict[str, Any]],
    *,
    knowledge_surface: str,
) -> list[dict[str, Any]]:
    """Drop duplicate prompt guidance and enforce a small skill budget."""
    normalized_text = " ".join(str(text or "").lower().split())
    budget = 900 if knowledge_surface == "expanded" else 420
    selected: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str, str]] = set()
    used_chars = 0

    for artifact in artifacts:
        artifact_id = str(artifact.get("id") or "").strip()
        kind = str(artifact.get("kind") or "").strip()
        title = " ".join(str(artifact.get("title") or "").split())
        summary = " ".join(str(artifact.get("summary") or "").split())
        if not summary:
            continue
        signature = (artifact_id, title.lower(), summary.lower())
        if signature in seen_signatures:
            continue
        if summary.lower() in normalized_text:
            continue
        if title and title.lower() in normalized_text:
            continue
        projected = used_chars + len(title) + len(summary) + 8
        if selected and projected > budget:
            break
        selected.append(
            {
                "id": artifact_id,
                "kind": kind,
                "title": title,
                "summary": summary,
                "origin": str(artifact.get("origin") or "").strip(),
                "parents": list(artifact.get("parents") or []),
                "supersedes": list(artifact.get("supersedes") or []),
                "lineage_status": str(artifact.get("lineage_status") or "").strip(),
                "lineage_summary": str(artifact.get("lineage_summary") or "").strip(),
            }
        )
        seen_signatures.add(signature)
        used_chars = projected
    return selected
