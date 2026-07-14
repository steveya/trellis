"""Typed runtime orientation contracts for bounded agent roles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml


_MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "knowledge"
    / "canonical"
    / "agent_orientations.yaml"
)
_REQUIRED_ROLES = frozenset({"quant", "model_validator"})


@dataclass(frozen=True)
class OrientationResource:
    """One ordered machine-readable or documentation navigation target."""

    order: int
    resource_id: str
    kind: str
    path: str
    purpose: str


@dataclass(frozen=True)
class RoleOrientation:
    """One versioned runtime role-orientation contract."""

    role: str
    contract_id: str
    version: int
    title: str
    purpose: str
    max_render_chars: int
    max_context_chars: int
    max_resource_chars: int
    owns: tuple[str, ...]
    excludes: tuple[str, ...]
    navigation: tuple[OrientationResource, ...]

    @property
    def identity(self) -> str:
        """Return the compact contract identity used in prompts and traces."""
        return f"{self.contract_id}@{self.version}"


def _required_text(payload: Mapping[str, object], key: str, *, context: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{context} requires non-empty {key!r}")
    return value


def _string_tuple(values: object, *, context: str) -> tuple[str, ...]:
    if values is None:
        raise ValueError(f"{context} is missing")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{context} requires a non-empty list")
    normalized = tuple(str(value or "").strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{context} cannot contain empty values")
    return normalized


def _parse_orientation(role: str, payload: object) -> RoleOrientation:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Role orientation {role!r} must be a mapping")
    context = f"Role orientation {role!r}"
    navigation_payload = payload.get("navigation")
    if not isinstance(navigation_payload, list) or not navigation_payload:
        raise ValueError(f"{context} requires non-empty navigation")

    navigation: list[OrientationResource] = []
    for item in navigation_payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"{context} navigation entries must be mappings")
        order = int(item.get("order") or 0)
        navigation.append(
            OrientationResource(
                order=order,
                resource_id=_required_text(item, "resource_id", context=context),
                kind=_required_text(item, "kind", context=context),
                path=_required_text(item, "path", context=context),
                purpose=_required_text(item, "purpose", context=context),
            )
        )
    navigation.sort(key=lambda item: item.order)
    expected_order = list(range(1, len(navigation) + 1))
    if [item.order for item in navigation] != expected_order:
        raise ValueError(f"{context} navigation order must be consecutive from 1")
    resource_ids = [item.resource_id for item in navigation]
    if len(resource_ids) != len(set(resource_ids)):
        raise ValueError(f"{context} navigation resource ids must be unique")

    version = int(payload.get("version") or 0)
    max_render_chars = int(payload.get("max_render_chars") or 0)
    max_context_chars = int(payload.get("max_context_chars") or 0)
    max_resource_chars = int(payload.get("max_resource_chars") or 0)
    if (
        version <= 0
        or max_render_chars <= 0
        or max_context_chars <= 0
        or max_resource_chars <= 0
    ):
        raise ValueError(
            f"{context} requires positive version, max_render_chars, "
            "max_context_chars, and max_resource_chars"
        )
    if max_resource_chars > max_context_chars:
        raise ValueError(
            f"{context} max_resource_chars cannot exceed max_context_chars"
        )
    orientation = RoleOrientation(
        role=role,
        contract_id=_required_text(payload, "contract_id", context=context),
        version=version,
        title=_required_text(payload, "title", context=context),
        purpose=_required_text(payload, "purpose", context=context),
        max_render_chars=max_render_chars,
        max_context_chars=max_context_chars,
        max_resource_chars=max_resource_chars,
        owns=_string_tuple(payload.get("owns"), context=f"{context} owns"),
        excludes=_string_tuple(
            payload.get("excludes"),
            context=f"{context} excludes",
        ),
        navigation=tuple(navigation),
    )
    _render_orientation_card(orientation)
    return orientation


def _load_role_orientations_from_path(
    manifest_path: Path,
) -> Mapping[str, RoleOrientation]:
    payload = yaml.safe_load(manifest_path.read_text())
    if not isinstance(payload, Mapping) or int(payload.get("schema_version") or 0) != 1:
        raise ValueError("Agent orientation manifest requires schema_version 1")
    raw_orientations = payload.get("orientations")
    if not isinstance(raw_orientations, Mapping):
        raise ValueError("Agent orientation manifest requires an orientations mapping")
    roles = {str(role).strip() for role in raw_orientations}
    if roles != _REQUIRED_ROLES:
        raise ValueError(
            "Agent orientation manifest requires exactly quant and model_validator roles"
        )
    orientations = {
        role: _parse_orientation(role, raw_orientations[role])
        for role in sorted(_REQUIRED_ROLES)
    }
    return MappingProxyType(orientations)


@cache
def _load_default_role_orientations() -> Mapping[str, RoleOrientation]:
    return _load_role_orientations_from_path(_MANIFEST_PATH)


def load_role_orientations(
    path: str | Path | None = None,
) -> Mapping[str, RoleOrientation]:
    """Load and validate canonical or explicitly supplied role orientations."""
    if path is None:
        return _load_default_role_orientations()
    return _load_role_orientations_from_path(Path(path))


def get_role_orientation(role: str) -> RoleOrientation:
    """Return one supported runtime role orientation or fail closed."""
    normalized = str(role or "").strip().lower().replace("-", "_")
    if normalized not in _REQUIRED_ROLES:
        raise ValueError(f"Unsupported runtime agent role {role!r}")
    return load_role_orientations()[normalized]


def role_orientation_summary(role: str) -> dict[str, object]:
    """Return the low-cardinality orientation identity persisted in traces."""
    orientation = get_role_orientation(role)
    return {
        "role": orientation.role,
        "contract_id": orientation.contract_id,
        "version": orientation.version,
    }


def _render_orientation_card(orientation: RoleOrientation) -> str:
    lines = [
        f"## {orientation.title}",
        f"- Contract: `{orientation.identity}`",
        f"- Purpose: {orientation.purpose}",
        "### Owns",
        *(f"- {item}" for item in orientation.owns),
        "### Does Not Own",
        *(f"- {item}" for item in orientation.excludes),
        "### Navigation Order",
        *(
            f"{item.order}. [{item.kind}] `{item.path}`: {item.purpose}"
            for item in orientation.navigation
        ),
    ]
    card = "\n".join(lines)
    if len(card) > orientation.max_render_chars:
        raise ValueError(
            f"Role orientation {orientation.identity} renders to {len(card)} chars, "
            f"above its {orientation.max_render_chars}-char budget"
        )
    return card


def render_role_orientation_card(role: str) -> str:
    """Render one compact role card without loading referenced documents."""
    return _render_orientation_card(get_role_orientation(role))
