"""Shared helper-facing runtime contract types."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import trellis.core.capabilities as capability_registry


def _freeze_mapping(mapping: Mapping[str, object] | dict[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping view over ``mapping``."""
    return MappingProxyType(dict(mapping or {}))


def _normalize_requirements(requirements) -> tuple[str, ...]:
    """Return a deduplicated tuple of non-empty requirement labels."""
    result: list[str] = []
    for item in tuple(requirements or ()):
        label = str(item).strip()
        if label and label not in result:
            result.append(label)
    return tuple(result)


def _capability_field_map() -> Mapping[str, str]:
    """Map normalized capability names onto their canonical MarketState field."""
    mapping: dict[str, str] = {}
    for item in capability_registry.MARKET_DATA:
        mapping[item.name] = item.market_state_field
    for name in list(mapping):
        normalized = tuple(capability_registry.normalize_market_data_requirements((name,)))
        if normalized:
            mapping[normalized[0]] = mapping[name]
    return MappingProxyType(mapping)


def _field_requirement_map() -> Mapping[str, tuple[str, ...]]:
    """Map one MarketState field onto the capability labels that can satisfy it."""
    capability_to_field = _capability_field_map()
    result: dict[str, list[str]] = {}
    for capability, field_name in capability_to_field.items():
        result.setdefault(field_name, [])
        if capability not in result[field_name]:
            result[field_name].append(capability)
    return MappingProxyType(
        {field_name: tuple(labels) for field_name, labels in result.items()}
    )


_CAPABILITY_TO_FIELD = _capability_field_map()
_FIELD_TO_REQUIREMENTS = _field_requirement_map()
_PROXY_MAPPING_FIELDS = frozenset(
    {
        "forecast_curves",
        "fx_rates",
        "underlier_spots",
        "local_vol_surfaces",
        "jump_parameter_sets",
        "model_parameter_sets",
        "selected_curve_names",
        "market_provenance",
    }
)


@dataclass(frozen=True)
class ContractViolation(RuntimeError):
    """Structured runtime-contract failure surfaced during payoff evaluation."""

    kind: str
    field: str
    requirement: str = ""
    message: str = ""
    available_capabilities: tuple[str, ...] = ()
    available_keys: tuple[str, ...] = ()
    missing_key: str = ""
    context: str = ""

    def __post_init__(self) -> None:
        message = (self.message or "").strip()
        if not message:
            detail = f"Runtime contract violation ({self.kind}) on MarketState.{self.field}"
            if self.missing_key:
                detail += f"[{self.missing_key!r}]"
            if self.requirement:
                detail += f" for requirement {self.requirement!r}"
            if self.context:
                detail += f" while evaluating {self.context}"
            if self.available_keys:
                detail += f". Available keys: {list(self.available_keys)!r}"
            elif self.available_capabilities:
                detail += f". Available capabilities: {list(self.available_capabilities)!r}"
            else:
                detail += "."
            object.__setattr__(self, "message", detail)
        RuntimeError.__init__(self, self.message)


class ContractAwareMapping(Mapping[str, object]):
    """Read-only mapping wrapper that raises ``ContractViolation`` on missing keys."""

    def __init__(
        self,
        mapping: Mapping[str, object],
        *,
        field: str,
        requirement: str = "",
        available_capabilities: tuple[str, ...] = (),
        context: str = "",
    ) -> None:
        self._mapping = mapping
        self._field = field
        self._requirement = requirement
        self._available_capabilities = tuple(available_capabilities)
        self._context = context

    def __getitem__(self, key: str) -> object:
        if key not in self._mapping:
            raise ContractViolation(
                kind="missing_market_key",
                field=self._field,
                requirement=self._requirement,
                available_capabilities=self._available_capabilities,
                available_keys=tuple(str(item) for item in self._mapping.keys()),
                missing_key=str(key),
                context=self._context,
            )
        return self._mapping[key]

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def get(self, key: str, default=None):
        return self._mapping.get(key, default)

    def __repr__(self) -> str:
        return repr(self._mapping)


class MarketStateContractProxy:
    """Thin MarketState wrapper that surfaces contract violations deterministically."""

    def __init__(
        self,
        market_state,
        *,
        requirements=(),
        context: str = "",
    ) -> None:
        self._market_state = market_state
        self._requirements = _normalize_requirements(requirements)
        self._required_fields = frozenset(
            _CAPABILITY_TO_FIELD[label]
            for label in self._requirements
            if label in _CAPABILITY_TO_FIELD
        )
        self._context = context

    @property
    def raw_market_state(self):
        """Return the wrapped MarketState instance."""
        return self._market_state

    def __getattr__(self, name: str):
        value = getattr(self._market_state, name)
        if name in self._required_fields and value is None:
            requirement = next(
                (
                    label
                    for label in self._requirements
                    if _CAPABILITY_TO_FIELD.get(label) == name
                ),
                next(iter(_FIELD_TO_REQUIREMENTS.get(name, ())), ""),
            )
            raise ContractViolation(
                kind="missing_market_field",
                field=name,
                requirement=requirement,
                available_capabilities=tuple(
                    sorted(getattr(self._market_state, "available_capabilities", set()))
                ),
                context=self._context,
            )
        if name in _PROXY_MAPPING_FIELDS and value is not None and isinstance(value, Mapping):
            requirement = next(
                (
                    label
                    for label in self._requirements
                    if _CAPABILITY_TO_FIELD.get(label) == name
                ),
                next(iter(_FIELD_TO_REQUIREMENTS.get(name, ())), ""),
            )
            return ContractAwareMapping(
                value,
                field=name,
                requirement=requirement,
                available_capabilities=tuple(
                    sorted(getattr(self._market_state, "available_capabilities", set()))
                ),
                context=self._context,
            )
        return value

    def forecast_forward_curve(self, rate_index: str | None = None):
        """Proxy forward-curve resolution and surface structured failures."""
        try:
            return self._market_state.forecast_forward_curve(rate_index)
        except ValueError as exc:
            raise ContractViolation(
                kind="missing_market_field",
                field="forward_curve",
                requirement="forward_curve",
                available_capabilities=tuple(
                    sorted(getattr(self._market_state, "available_capabilities", set()))
                ),
                context=self._context,
                message=(
                    f"Runtime contract violation (missing_market_field) on "
                    f"MarketState.forward_curve while evaluating {self._context or 'payoff'}: {exc}"
                ),
            ) from exc

    def __repr__(self) -> str:
        return f"MarketStateContractProxy({self._market_state!r})"


def wrap_market_state_with_contract(
    market_state,
    *,
    requirements=(),
    context: str = "",
):
    """Wrap a MarketState with contract-aware access for payoff evaluation."""
    return MarketStateContractProxy(
        market_state,
        requirements=requirements,
        context=context,
    )


@dataclass(frozen=True)
class ContractState:
    """Mutable contract runtime, split into event state and contract memory."""

    event_state: Mapping[str, object] = field(default_factory=dict)
    contract_memory: Mapping[str, object] = field(default_factory=dict)
    phase: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_state", _freeze_mapping(self.event_state))
        object.__setattr__(self, "contract_memory", _freeze_mapping(self.contract_memory))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def get_event(self, name: str, default=None):
        """Return one event-state item, falling back to ``default``."""
        return self.event_state.get(str(name), default)

    def require_event(self, name: str):
        """Return one required event-state item or raise ``KeyError``."""
        key = str(name)
        if key not in self.event_state:
            raise KeyError(f"Missing event_state binding {key!r}")
        return self.event_state[key]

    def get_memory(self, name: str, default=None):
        """Return one contract-memory item, falling back to ``default``."""
        return self.contract_memory.get(str(name), default)

    def require_memory(self, name: str):
        """Return one required contract-memory item or raise ``KeyError``."""
        key = str(name)
        if key not in self.contract_memory:
            raise KeyError(f"Missing contract_memory binding {key!r}")
        return self.contract_memory[key]

    def with_event_state(self, **updates) -> "ContractState":
        """Return a copy with updated event-state bindings."""
        merged = dict(self.event_state)
        merged.update(updates)
        return replace(self, event_state=merged)

    def with_contract_memory(self, **updates) -> "ContractState":
        """Return a copy with updated contract-memory bindings."""
        merged = dict(self.contract_memory)
        merged.update(updates)
        return replace(self, contract_memory=merged)


@dataclass(frozen=True)
class ResolvedInputs:
    """Immutable resolved helper-facing bindings, separate from mutable state."""

    bindings: Mapping[str, object]
    requirements: tuple[str, ...] = ()
    source_kind: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _freeze_mapping(self.bindings))
        object.__setattr__(self, "requirements", _normalize_requirements(self.requirements))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def get(self, name: str, default=None):
        """Return one resolved binding, falling back to ``default``."""
        return self.bindings.get(str(name), default)

    def require(self, name: str):
        """Return one required resolved binding or raise ``KeyError``."""
        key = str(name)
        if key not in self.bindings:
            raise KeyError(f"Missing resolved binding {key!r}")
        return self.bindings[key]


@dataclass(frozen=True)
class RuntimeContext:
    """Thin execution context for phase-aware helper evaluation."""

    phase: str = ""
    schedule_role: str = ""
    step: int | None = None
    event_name: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


__all__ = [
    "ContractState",
    "ContractViolation",
    "ContractAwareMapping",
    "MarketStateContractProxy",
    "ResolvedInputs",
    "RuntimeContext",
    "wrap_market_state_with_contract",
]
