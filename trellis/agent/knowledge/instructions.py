"""Structured instruction lifecycle and resolution for route guidance.

This module resolves precedence-aware guidance for embedded agents. It does
not implement numerical payoffs or pricing kernels itself.
"""

from __future__ import annotations

from typing import Iterable

from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.knowledge.schema import (
    InstructionConflict,
    InstructionRecord,
    ProductIR,
    ResolvedInstructionSet,
)


_ACTIVE_STATUSES = {"active", "validated", "promoted"}
_INSTRUCTION_TYPE_ORDER = {
    "hard_constraint": 3,
    "route_hint": 2,
    "historical_note": 1,
    "deprecation_notice": 0,
}
_SOURCE_KIND_ORDER = {
    "canonical": 4,
    "cookbook": 3,
    "lesson": 3,
    "route_card": 2,
    "trace": 1,
}


def resolve_instruction_records(
    records: Iterable[InstructionRecord],
    *,
    method: str | None = None,
    instrument_type: str | None = None,
    route: str = "",
    product_ir: ProductIR | None = None,
) -> ResolvedInstructionSet:
    """Resolve instruction records into one effective, precedence-aware set.

    The returned instructions are meant to steer the embedded agent that builds
    the actual payoff or pricing path.
    """
    normalized_method = normalize_method(method) if method else None
    normalized_instrument = _normalize_key(instrument_type or getattr(product_ir, "instrument", None))
    normalized_route = _normalize_key(route)
    feature_set = set(getattr(product_ir, "payoff_traits", ()) or ())
    candidate_records = [
        record
        for record in records
        if _record_matches_scope(
            record,
            method=normalized_method,
            instrument_type=normalized_instrument,
            route=normalized_route,
            feature_set=feature_set,
        )
    ]

    dropped: list[InstructionRecord] = [
        record for record in candidate_records if record.status not in _ACTIVE_STATUSES
    ]
    active = [record for record in candidate_records if record.status in _ACTIVE_STATUSES]

    superseded_ids = {record_id for record in active for record_id in record.supersedes}
    superseded_records = [record for record in active if record.id in superseded_ids]
    active = [
        record
        for record in active
        if record.id not in superseded_ids
    ]
    dropped.extend(superseded_records)

    grouped = _group_by_scope(active)
    conflicts: list[InstructionConflict] = []

    for scope_key, scope_records in grouped.items():
        ordered = sorted(
            scope_records,
            key=_instruction_sort_key,
            reverse=True,
        )
        if len(ordered) > 1 and _conflict_exists(ordered):
            conflicts.append(
                InstructionConflict(
                    reason=_conflict_reason(scope_key, ordered),
                    conflicting_ids=tuple(record.id for record in ordered),
                    winner_id=ordered[0].id,
                )
            )

    resolved = _dedupe_records(sorted(active, key=_instruction_sort_key, reverse=True))
    resolved_ids = {record.id for record in resolved}
    dropped.extend(record for record in active if record.id not in resolved_ids)
    dropped = _dedupe_records(dropped)
    return ResolvedInstructionSet(
        route=normalized_route,
        effective_instructions=tuple(resolved),
        dropped_instructions=tuple(dropped),
        conflicts=tuple(conflicts),
    )


def _record_matches_scope(
    record: InstructionRecord,
    *,
    method: str | None,
    instrument_type: str,
    route: str,
    feature_set: set[str],
) -> bool:
    if record.scope_methods:
        if method is None or method not in {_normalize_key(item) for item in record.scope_methods}:
            return False
    if record.scope_instruments:
        if not instrument_type or instrument_type not in {_normalize_key(item) for item in record.scope_instruments}:
            return False
    if record.scope_routes:
        if not route or route not in {_normalize_key(item) for item in record.scope_routes}:
            return False
    if record.scope_features:
        if not (set(_normalize_key(item) for item in record.scope_features) & feature_set):
            return False
    return True


def _group_by_scope(records: list[InstructionRecord]) -> dict[tuple[str, ...], list[InstructionRecord]]:
    grouped: dict[tuple[str, ...], list[InstructionRecord]] = {}
    for record in records:
        scope_key = (
            tuple(sorted(_normalize_key(item) for item in record.scope_methods)),
            tuple(sorted(_normalize_key(item) for item in record.scope_instruments)),
            tuple(sorted(_normalize_key(item) for item in record.scope_routes)),
            tuple(sorted(_normalize_key(item) for item in record.scope_features)),
        )
        grouped.setdefault(scope_key, []).append(record)
    return grouped


def _instruction_sort_key(record: InstructionRecord) -> tuple[int, int, int, str]:
    return (
        _INSTRUCTION_TYPE_ORDER.get(record.instruction_type, 1),
        _SOURCE_KIND_ORDER.get(record.source_kind, 1),
        record.precedence_rank,
        record.updated_at or record.created_at or record.id,
    )


def _conflict_exists(records: list[InstructionRecord]) -> bool:
    hard_constraints = [record for record in records if record.instruction_type == "hard_constraint"]
    if len(hard_constraints) <= 1:
        return False
    statements = {record.statement.strip() for record in hard_constraints if record.statement.strip()}
    return len(statements) > 1


def _conflict_reason(scope_key: tuple[tuple[str, ...], ...], records: list[InstructionRecord]) -> str:
    scopes = []
    labels = ("methods", "instruments", "routes", "features")
    for label, values in zip(labels, scope_key):
        if values:
            scopes.append(f"{label}={','.join(values)}")
    scope_text = ", ".join(scopes) if scopes else "shared scope"
    return (
        f"Conflicting instruction records in {scope_text}; "
        f"selected `{records[0].id}` by precedence."
    )


def _dedupe_records(records: list[InstructionRecord]) -> list[InstructionRecord]:
    deduped: list[InstructionRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            continue
        deduped.append(record)
        seen.add(record.id)
    return deduped


def _normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower().replace(" ", "_")
