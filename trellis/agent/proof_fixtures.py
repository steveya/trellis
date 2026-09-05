"""Named, immutable economics fixtures for retained pricing-proof tasks."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


def _stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ProofFixture:
    """One versioned set of reusable, task-authored product terms."""

    fixture_id: str
    schema_version: int
    task_fields: Mapping[str, Any]
    fixture_digest: str

    def materialized_fields(self) -> dict[str, Any]:
        """Return an isolated mutable copy suitable for one loaded task."""
        return _thaw(self.task_fields)


def load_proof_fixtures(
    manifest_name: str,
    *,
    root: Path,
) -> dict[str, ProofFixture]:
    """Load the named proof fixtures declared by one task manifest."""
    path = root / manifest_name
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        return {}
    schema_version = raw.get("proof_fixture_version", 1)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        raise ValueError("proof_fixture_version must be a positive integer")
    raw_fixtures = raw.get("proof_fixtures") or {}
    if not isinstance(raw_fixtures, Mapping):
        raise ValueError("proof_fixtures must be a mapping")

    fixtures: dict[str, ProofFixture] = {}
    for raw_id, raw_fields in raw_fixtures.items():
        fixture_id = str(raw_id or "").strip()
        if not fixture_id or fixture_id != raw_id:
            raise ValueError("proof fixture ids must be exact non-empty strings")
        if not isinstance(raw_fields, Mapping) or not raw_fields:
            raise ValueError(f"proof fixture {fixture_id!r} must contain task fields")
        fields = {str(key): deepcopy(value) for key, value in raw_fields.items()}
        if "proof_fixture_id" in fields:
            raise ValueError(
                f"proof fixture {fixture_id!r} must not declare proof_fixture_id"
            )
        digest_payload = {
            "fixture_id": fixture_id,
            "schema_version": schema_version,
            "task_fields": fields,
        }
        fixtures[fixture_id] = ProofFixture(
            fixture_id=fixture_id,
            schema_version=schema_version,
            task_fields=_freeze(fields),
            fixture_digest=hashlib.sha256(
                _stable_json(digest_payload).encode("utf-8")
            ).hexdigest(),
        )
    return fixtures


def materialize_task_proof_fixture(
    task: Mapping[str, Any],
    *,
    fixtures: Mapping[str, ProofFixture],
) -> dict[str, Any]:
    """Hydrate one task from its named fixture without allowing local drift."""
    payload = dict(task)
    fixture_id = str(payload.get("proof_fixture_id") or "").strip()
    if not fixture_id:
        return payload
    try:
        fixture = fixtures[fixture_id]
    except KeyError as exc:
        raise ValueError(f"unknown proof fixture {fixture_id!r}") from exc

    for field, fixture_value in fixture.materialized_fields().items():
        if field in payload and payload[field] != fixture_value:
            raise ValueError(
                f"task {payload.get('id')!r} cannot override proof fixture "
                f"{fixture_id!r} field {field!r}"
            )
        payload[field] = fixture_value
    payload["proof_fixture_schema_version"] = fixture.schema_version
    payload["proof_fixture_digest"] = fixture.fixture_digest
    return payload


__all__ = [
    "ProofFixture",
    "load_proof_fixtures",
    "materialize_task_proof_fixture",
]
