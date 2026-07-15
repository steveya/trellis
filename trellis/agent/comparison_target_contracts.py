"""Typed contracts for one method target in a comparison task.

Comparison target ids such as ``mc_exact`` or ``qe_heston`` are labels, not
execution evidence.  This module turns task-manifest declarations into a
small immutable contract that can be carried through planning, building, and
cross-validation without encoding product-specific dispatch in the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from trellis.agent.knowledge.methods import is_known_method, normalize_method


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ComparisonTargetContract:
    """Immutable semantic and numerical identity for one comparison target."""

    target_id: str
    method: str
    contract_id: str = ""
    route_id: str = ""
    route_family: str = ""
    backend_binding_id: str = ""
    variant_parameters: Mapping[str, Any] = field(default_factory=dict)
    spec_overrides: Mapping[str, Any] = field(default_factory=dict)
    validation_bundle_id: str = ""
    payoff_family: str = ""
    exercise_style: str = ""
    model_family: str = ""
    observation_style: str = ""
    semantic_axes: Mapping[str, Any] = field(default_factory=dict)
    equivalence_group: str = ""
    resolution_source: str = "explicit"
    explicit: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        target_id = str(self.target_id or "").strip()
        method = normalize_method(str(self.method or "").strip())
        if not target_id:
            raise ValueError("Comparison target contract requires target_id")
        if not method:
            raise ValueError(
                f"Comparison target contract {target_id!r} requires a method"
            )
        if not is_known_method(method):
            raise ValueError(
                f"Comparison target contract {target_id!r} has unknown method "
                f"family {method!r}"
            )
        if int(self.schema_version) != 1:
            raise ValueError(
                f"Unsupported comparison target contract schema: {self.schema_version}"
            )
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "contract_id", str(self.contract_id or "").strip())
        object.__setattr__(self, "route_id", str(self.route_id or "").strip())
        object.__setattr__(self, "route_family", str(self.route_family or "").strip())
        object.__setattr__(
            self,
            "backend_binding_id",
            str(self.backend_binding_id or "").strip(),
        )
        object.__setattr__(
            self,
            "validation_bundle_id",
            str(self.validation_bundle_id or "").strip(),
        )
        object.__setattr__(self, "payoff_family", str(self.payoff_family or "").strip())
        object.__setattr__(self, "exercise_style", str(self.exercise_style or "").strip())
        object.__setattr__(self, "model_family", str(self.model_family or "").strip())
        object.__setattr__(
            self,
            "observation_style",
            str(self.observation_style or "").strip(),
        )
        object.__setattr__(
            self,
            "equivalence_group",
            str(self.equivalence_group or "").strip(),
        )
        object.__setattr__(
            self,
            "resolution_source",
            str(self.resolution_source or "").strip() or "explicit",
        )
        object.__setattr__(
            self,
            "variant_parameters",
            _freeze_value(dict(self.variant_parameters or {})),
        )
        object.__setattr__(
            self,
            "spec_overrides",
            _freeze_value(dict(self.spec_overrides or {})),
        )
        object.__setattr__(
            self,
            "semantic_axes",
            _freeze_value(dict(self.semantic_axes or {})),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a YAML/JSON-safe representation for traces and task records."""
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "target_id": self.target_id,
            "method": self.method,
            "route_id": self.route_id,
            "route_family": self.route_family,
            "backend_binding_id": self.backend_binding_id,
            "variant_parameters": _thaw_value(self.variant_parameters),
            "spec_overrides": _thaw_value(self.spec_overrides),
            "validation_bundle_id": self.validation_bundle_id,
            "payoff_family": self.payoff_family,
            "exercise_style": self.exercise_style,
            "model_family": self.model_family,
            "observation_style": self.observation_style,
            "semantic_axes": _thaw_value(self.semantic_axes),
            "equivalence_group": self.equivalence_group,
            "resolution_source": self.resolution_source,
            "explicit": self.explicit,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ComparisonTargetContract":
        """Rehydrate a persisted target contract."""
        return cls(**dict(payload))


def resolve_comparison_target_contract(
    *,
    task_id: str | None,
    target_id: str,
    preferred_method: str,
    declaration: Mapping[str, Any] | None = None,
) -> ComparisonTargetContract:
    """Resolve an explicit manifest declaration or a typed legacy fallback."""
    raw = dict(declaration or {})
    explicit = bool(declaration)
    semantic_axes = dict(raw.get("semantic_axes") or {})
    for axis in (
        "derivative_family",
        "underlying_asset_class",
        "option_type",
        "path_dependence",
        "schedule_dependence",
        "state_dependence",
    ):
        if axis in raw and axis not in semantic_axes:
            semantic_axes[axis] = raw[axis]

    normalized_task_id = str(task_id or "task").strip() or "task"
    normalized_target_id = str(target_id or "").strip()
    return ComparisonTargetContract(
        target_id=normalized_target_id,
        method=raw.get("method") or preferred_method,
        contract_id=(
            raw.get("contract_id")
            or f"comparison-target:{normalized_task_id}:{normalized_target_id}:v1"
        ),
        route_id=raw.get("route_id") or "",
        route_family=raw.get("route_family") or "",
        backend_binding_id=(
            raw.get("backend_binding_id") or raw.get("backend_binding") or ""
        ),
        variant_parameters=(
            raw.get("variant_parameters") or raw.get("variants") or {}
        ),
        spec_overrides=raw.get("spec_overrides") or {},
        validation_bundle_id=(
            raw.get("validation_bundle_id") or raw.get("validation_bundle") or ""
        ),
        payoff_family=raw.get("payoff_family") or "",
        exercise_style=raw.get("exercise_style") or "",
        model_family=raw.get("model_family") or "",
        observation_style=raw.get("observation_style") or "",
        semantic_axes=semantic_axes,
        equivalence_group=raw.get("equivalence_group") or "",
        resolution_source=(
            "task.cross_validate.target_contracts"
            if explicit
            else "legacy_target_inference"
        ),
        explicit=explicit,
    )


def declared_comparison_target_contract(
    declarations: object,
    target_id: str,
) -> ComparisonTargetContract | None:
    """Return the canonical full contract declared for one executable target."""
    if not isinstance(declarations, Mapping):
        return None
    declaration = declarations.get(str(target_id or "").strip())
    if not isinstance(declaration, Mapping):
        return None
    raw_contract = declaration.get("target_contract")
    if not isinstance(raw_contract, Mapping):
        return None
    try:
        return ComparisonTargetContract.from_payload(raw_contract)
    except (TypeError, ValueError):
        return None


def comparison_target_execution_identity(
    contract: ComparisonTargetContract,
) -> dict[str, Any]:
    """Return executable semantics without request/provenance identity fields."""
    payload = contract.to_payload()
    for key in (
        "schema_version",
        "contract_id",
        "target_id",
        "equivalence_group",
        "resolution_source",
        "explicit",
    ):
        payload.pop(key, None)
    return payload


def comparison_target_contracts_compatible(
    expected: ComparisonTargetContract,
    actual: ComparisonTargetContract,
) -> bool:
    """Return whether an artifact contract proves the requested execution identity."""
    if expected.target_id != actual.target_id:
        return False
    if (
        expected.contract_id
        and actual.contract_id
        and expected.contract_id != actual.contract_id
    ):
        return False
    return comparison_target_execution_identity(
        expected
    ) == comparison_target_execution_identity(actual)


__all__ = [
    "ComparisonTargetContract",
    "comparison_target_contracts_compatible",
    "comparison_target_execution_identity",
    "declared_comparison_target_contract",
    "resolve_comparison_target_contract",
]
