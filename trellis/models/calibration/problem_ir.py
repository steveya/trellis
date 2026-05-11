"""Typed calibration problem IR for engine-backed calibration migration."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as raw_np


def _normalize_str(value: object, *, field_name: str) -> str:
    normalized = "" if value is None else str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _optional_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _normalize_str_tuple(values: Sequence[object] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    return tuple(str(value).strip() for value in values if str(value).strip())


def _normalize_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return _freeze_mapping(mapping)


def _payload_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    return dict(mapping)


def _validate_finite(value: float, *, field_name: str) -> float:
    normalized = float(value)
    if not raw_np.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


@dataclass(frozen=True)
class CalibrationVariableSpec:
    """One calibration coordinate in a problem IR."""

    name: str
    coordinate_chart: str
    initial_value: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    scaling: float = 1.0
    warm_start_source: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _normalize_str(self.name, field_name="CalibrationVariableSpec.name")
        chart = _normalize_str(self.coordinate_chart, field_name=f"CalibrationVariableSpec {name!r} coordinate_chart")
        initial_value = _validate_finite(
            self.initial_value,
            field_name=f"CalibrationVariableSpec {name!r} initial_value",
        )
        lower_bound = None if self.lower_bound is None else _validate_finite(
            self.lower_bound,
            field_name=f"CalibrationVariableSpec {name!r} lower_bound",
        )
        upper_bound = None if self.upper_bound is None else _validate_finite(
            self.upper_bound,
            field_name=f"CalibrationVariableSpec {name!r} upper_bound",
        )
        scaling = _validate_finite(self.scaling, field_name=f"CalibrationVariableSpec {name!r} scaling")
        if scaling <= 0.0:
            raise ValueError(f"CalibrationVariableSpec {name!r} scaling must be positive")
        if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
            raise ValueError(f"CalibrationVariableSpec {name!r} lower_bound must not exceed upper_bound")
        if lower_bound is not None and initial_value < lower_bound:
            raise ValueError(f"CalibrationVariableSpec {name!r} initial_value is below lower_bound")
        if upper_bound is not None and initial_value > upper_bound:
            raise ValueError(f"CalibrationVariableSpec {name!r} initial_value is above upper_bound")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coordinate_chart", chart)
        object.__setattr__(self, "initial_value", initial_value)
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)
        object.__setattr__(self, "scaling", scaling)
        object.__setattr__(self, "warm_start_source", _optional_str(self.warm_start_source))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly variable payload."""
        return {
            "name": self.name,
            "coordinate_chart": self.coordinate_chart,
            "initial_value": float(self.initial_value),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "scaling": float(self.scaling),
            "warm_start_source": self.warm_start_source,
            "metadata": _payload_mapping(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationTargetSpec:
    """One observed market target in a calibration problem IR."""

    target_id: str
    instrument_id: str
    quote_family: str
    quote_value: float
    quote_convention: str = ""
    weight: float = 1.0
    quote_map_payload: Mapping[str, object] = field(default_factory=dict)
    validation_tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target_id = _normalize_str(self.target_id, field_name="CalibrationTargetSpec.target_id")
        instrument_id = _normalize_str(
            self.instrument_id,
            field_name=f"CalibrationTargetSpec {target_id!r} instrument_id",
        )
        quote_family = _normalize_str(
            self.quote_family,
            field_name=f"CalibrationTargetSpec {target_id!r} quote_family",
        )
        quote_value = _validate_finite(
            self.quote_value,
            field_name=f"CalibrationTargetSpec {target_id!r} quote_value",
        )
        weight = _validate_finite(self.weight, field_name=f"CalibrationTargetSpec {target_id!r} weight")
        if weight <= 0.0:
            raise ValueError(f"CalibrationTargetSpec {target_id!r} weight must be positive")

        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "instrument_id", instrument_id)
        object.__setattr__(self, "quote_family", quote_family)
        object.__setattr__(self, "quote_convention", _optional_str(self.quote_convention))
        object.__setattr__(self, "quote_value", quote_value)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "quote_map_payload", _normalize_mapping(self.quote_map_payload))
        object.__setattr__(self, "validation_tags", _normalize_str_tuple(self.validation_tags))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly target payload."""
        return {
            "target_id": self.target_id,
            "instrument_id": self.instrument_id,
            "quote_family": self.quote_family,
            "quote_convention": self.quote_convention,
            "quote_value": float(self.quote_value),
            "weight": float(self.weight),
            "quote_map": _payload_mapping(self.quote_map_payload),
            "validation_tags": list(self.validation_tags),
            "metadata": _payload_mapping(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationObjectiveSpec:
    """Objective metadata for a calibration problem IR."""

    objective_kind: str
    loss_function: str
    residual_count: int
    derivative_method: str = ""
    solve_request_id: str = ""
    regularization: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        objective_kind = _normalize_str(self.objective_kind, field_name="CalibrationObjectiveSpec.objective_kind")
        loss_function = _normalize_str(self.loss_function, field_name="CalibrationObjectiveSpec.loss_function")
        residual_count = int(self.residual_count)
        if residual_count <= 0:
            raise ValueError("CalibrationObjectiveSpec residual_count must be positive")
        object.__setattr__(self, "objective_kind", objective_kind)
        object.__setattr__(self, "loss_function", loss_function)
        object.__setattr__(self, "residual_count", residual_count)
        object.__setattr__(self, "derivative_method", _optional_str(self.derivative_method))
        object.__setattr__(self, "solve_request_id", _optional_str(self.solve_request_id))
        object.__setattr__(self, "regularization", _freeze_mapping(self.regularization))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly objective payload."""
        return {
            "objective_kind": self.objective_kind,
            "loss_function": self.loss_function,
            "residual_count": int(self.residual_count),
            "derivative_method": self.derivative_method,
            "solve_request_id": self.solve_request_id,
            "regularization": _payload_mapping(self.regularization),
            "metadata": _payload_mapping(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationDependencySpec:
    """One upstream dependency consumed by a calibration problem."""

    dependency_id: str
    object_kind: str
    object_name: str
    required: bool = True
    source_ref: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dependency_id = _normalize_str(self.dependency_id, field_name="CalibrationDependencySpec.dependency_id")
        object.__setattr__(self, "dependency_id", dependency_id)
        object.__setattr__(
            self,
            "object_kind",
            _normalize_str(self.object_kind, field_name=f"CalibrationDependencySpec {dependency_id!r} object_kind"),
        )
        object.__setattr__(
            self,
            "object_name",
            _normalize_str(self.object_name, field_name=f"CalibrationDependencySpec {dependency_id!r} object_name"),
        )
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "source_ref", _optional_str(self.source_ref))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly dependency payload."""
        return {
            "dependency_id": self.dependency_id,
            "object_kind": self.object_kind,
            "object_name": self.object_name,
            "required": bool(self.required),
            "source_ref": self.source_ref,
            "metadata": _payload_mapping(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationMaterializationSpec:
    """Runtime materialization target declared by a calibration problem."""

    object_kind: str
    object_name: str
    destination_field: str
    source_ref: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_kind = _normalize_str(self.object_kind, field_name="CalibrationMaterializationSpec.object_kind")
        object.__setattr__(self, "object_kind", object_kind)
        object.__setattr__(
            self,
            "object_name",
            _normalize_str(self.object_name, field_name=f"CalibrationMaterializationSpec {object_kind!r} object_name"),
        )
        object.__setattr__(
            self,
            "destination_field",
            _normalize_str(
                self.destination_field,
                field_name=f"CalibrationMaterializationSpec {object_kind!r} destination_field",
            ),
        )
        object.__setattr__(self, "source_ref", _optional_str(self.source_ref))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly materialization payload."""
        return {
            "object_kind": self.object_kind,
            "object_name": self.object_name,
            "destination_field": self.destination_field,
            "source_ref": self.source_ref,
            "metadata": _payload_mapping(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationDiagnosticSpec:
    """One required diagnostic or tolerance for a problem IR."""

    diagnostic_id: str
    metric_name: str
    tolerance: float | None = None
    severity: str = "warning"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        diagnostic_id = _normalize_str(self.diagnostic_id, field_name="CalibrationDiagnosticSpec.diagnostic_id")
        tolerance = None if self.tolerance is None else _validate_finite(
            self.tolerance,
            field_name=f"CalibrationDiagnosticSpec {diagnostic_id!r} tolerance",
        )
        if tolerance is not None and tolerance < 0.0:
            raise ValueError(f"CalibrationDiagnosticSpec {diagnostic_id!r} tolerance must be non-negative")
        object.__setattr__(self, "diagnostic_id", diagnostic_id)
        object.__setattr__(
            self,
            "metric_name",
            _normalize_str(self.metric_name, field_name=f"CalibrationDiagnosticSpec {diagnostic_id!r} metric_name"),
        )
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "severity", _optional_str(self.severity) or "warning")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly diagnostic payload."""
        return {
            "diagnostic_id": self.diagnostic_id,
            "metric_name": self.metric_name,
            "tolerance": self.tolerance,
            "severity": self.severity,
            "metadata": _payload_mapping(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationProblemIR:
    """Typed, immutable representation of one calibration problem node."""

    problem_id: str
    workflow_id: str
    family_id: str
    variables: tuple[CalibrationVariableSpec, ...]
    targets: tuple[CalibrationTargetSpec, ...]
    objective: CalibrationObjectiveSpec
    dependencies: tuple[CalibrationDependencySpec, ...] = ()
    materialization: CalibrationMaterializationSpec | None = None
    diagnostics: tuple[CalibrationDiagnosticSpec, ...] = ()
    solve_request_payload: Mapping[str, object] = field(default_factory=dict)
    replay_metadata: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        problem_id = _normalize_str(self.problem_id, field_name="CalibrationProblemIR.problem_id")
        workflow_id = _normalize_str(self.workflow_id, field_name=f"CalibrationProblemIR {problem_id!r} workflow_id")
        family_id = _normalize_str(self.family_id, field_name=f"CalibrationProblemIR {problem_id!r} family_id")
        variables = tuple(self.variables)
        targets = tuple(self.targets)
        dependencies = tuple(self.dependencies)
        diagnostics = tuple(self.diagnostics)
        if not variables:
            raise ValueError(f"CalibrationProblemIR {problem_id!r} requires at least one variable")
        if not targets:
            raise ValueError(f"CalibrationProblemIR {problem_id!r} requires at least one target")
        self._validate_unique(
            [variable.name for variable in variables],
            problem_id=problem_id,
            kind="variable",
        )
        self._validate_unique(
            [target.target_id for target in targets],
            problem_id=problem_id,
            kind="target",
        )
        self._validate_unique(
            [dependency.dependency_id for dependency in dependencies],
            problem_id=problem_id,
            kind="dependency",
        )
        self._validate_unique(
            [diagnostic.diagnostic_id for diagnostic in diagnostics],
            problem_id=problem_id,
            kind="diagnostic",
        )
        if self.objective.residual_count != len(targets):
            raise ValueError(
                f"CalibrationProblemIR {problem_id!r} objective residual_count must match target count"
            )
        solve_request_payload = _freeze_mapping(self.solve_request_payload)
        request_id = solve_request_payload.get("request_id")
        if self.objective.solve_request_id and request_id and str(request_id) != self.objective.solve_request_id:
            raise ValueError(
                f"CalibrationProblemIR {problem_id!r} solve_request_payload request_id "
                "must match objective solve_request_id"
            )

        object.__setattr__(self, "problem_id", problem_id)
        object.__setattr__(self, "workflow_id", workflow_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "solve_request_payload", solve_request_payload)
        object.__setattr__(self, "replay_metadata", _freeze_mapping(self.replay_metadata))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def initial_guess(self) -> tuple[float, ...]:
        """Return variables' initial values in parameter order."""
        return tuple(variable.initial_value for variable in self.variables)

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly problem payload."""
        return {
            "problem_id": self.problem_id,
            "workflow_id": self.workflow_id,
            "family_id": self.family_id,
            "variables": [variable.to_payload() for variable in self.variables],
            "targets": [target.to_payload() for target in self.targets],
            "objective": self.objective.to_payload(),
            "dependencies": [dependency.to_payload() for dependency in self.dependencies],
            "materialization": None if self.materialization is None else self.materialization.to_payload(),
            "diagnostics": [diagnostic.to_payload() for diagnostic in self.diagnostics],
            "solve_request": _payload_mapping(self.solve_request_payload),
            "replay_metadata": _payload_mapping(self.replay_metadata),
            "metadata": _payload_mapping(self.metadata),
        }

    @staticmethod
    def _validate_unique(values: Sequence[str], *, problem_id: str, kind: str) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        if duplicates:
            duplicate_text = ", ".join(repr(value) for value in duplicates)
            raise ValueError(f"CalibrationProblemIR {problem_id!r} has duplicate {kind} id(s): {duplicate_text}")


__all__ = [
    "CalibrationDependencySpec",
    "CalibrationDiagnosticSpec",
    "CalibrationMaterializationSpec",
    "CalibrationObjectiveSpec",
    "CalibrationProblemIR",
    "CalibrationTargetSpec",
    "CalibrationVariableSpec",
]
