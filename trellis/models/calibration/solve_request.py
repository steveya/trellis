"""Typed solve-request substrate for calibration and inversion workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

import numpy as raw_np
from scipy.optimize import brentq, least_squares, minimize

from trellis.analytics.derivative_methods import derivative_method_payload


def _normalize_float_tuple(values: Sequence[float] | None) -> tuple[float, ...]:
    """Normalize numeric sequences onto an immutable float tuple."""
    if values is None:
        return ()
    return tuple(float(value) for value in values)


def _normalize_optional_float_tuple(
    values: Sequence[float | None] | None,
) -> tuple[float | None, ...]:
    """Normalize optional numeric sequences onto an immutable tuple."""
    if values is None:
        return ()
    normalized: list[float | None] = []
    for value in values:
        normalized.append(None if value is None else float(value))
    return tuple(normalized)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy for user metadata."""
    return MappingProxyType(dict(mapping or {}))


def _metadata_string(mapping: Mapping[str, object], key: str) -> str:
    """Return a normalized metadata string field when present."""
    value = mapping.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _solver_derivative_metadata(method_id: str, **metadata: object) -> dict[str, object]:
    """Return normalized solver derivative metadata with legacy key compatibility."""
    payload = derivative_method_payload(method_id, **metadata)
    payload["derivative_method"] = payload["resolved_derivative_method"]
    return payload


@dataclass(frozen=True)
class SolveBounds:
    """Serializable per-parameter lower/upper bounds."""

    lower: tuple[float | None, ...] = ()
    upper: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        lower = _normalize_optional_float_tuple(self.lower)
        upper = _normalize_optional_float_tuple(self.upper)
        if lower and upper and len(lower) != len(upper):
            raise ValueError("SolveBounds lower/upper sequences must have the same length")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "lower": list(self.lower),
            "upper": list(self.upper),
        }


@dataclass(frozen=True)
class ConstraintSpec:
    """Serializable constraint metadata for a solve request."""

    kind: str
    parameter_names: tuple[str, ...] = ()
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "kind": self.kind,
            "parameter_names": list(self.parameter_names),
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WarmStart:
    """Optional warm-start metadata for replayable calibration workflows."""

    parameter_values: tuple[float, ...]
    source: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_values", _normalize_float_tuple(self.parameter_values))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "parameter_values": list(self.parameter_values),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ObjectiveBundle:
    """Typed objective metadata plus optional runtime hooks."""

    objective_kind: Literal["root_scalar", "least_squares"]
    labels: tuple[str, ...] = ()
    target_values: tuple[float, ...] = ()
    weights: tuple[float, ...] = ()
    scalar_objective_fn: Callable[[object], float] | None = field(default=None, repr=False, compare=False)
    vector_objective_fn: Callable[[raw_np.ndarray], raw_np.ndarray] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    jacobian_fn: Callable[[raw_np.ndarray], raw_np.ndarray] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    hessian_fn: Callable[[raw_np.ndarray], raw_np.ndarray] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        target_values = _normalize_float_tuple(self.target_values)
        weights = _normalize_float_tuple(self.weights)
        if not self.scalar_objective_fn and not self.vector_objective_fn:
            raise ValueError("ObjectiveBundle requires a scalar_objective_fn or vector_objective_fn")
        if labels and target_values and len(labels) != len(target_values):
            raise ValueError("ObjectiveBundle labels and target_values must have the same length")
        if weights and target_values and len(weights) != len(target_values):
            raise ValueError("ObjectiveBundle weights and target_values must have the same length")
        if self.objective_kind == "root_scalar" and target_values and len(target_values) != 1:
            raise ValueError("root_scalar objectives must have at most one target value")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "target_values", target_values)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def derivative_payload(self) -> dict[str, str]:
        """Return availability metadata for derivative hooks."""
        return {
            "jacobian": "provided" if self.jacobian_fn is not None else "none",
            "hessian": "provided" if self.hessian_fn is not None else "none",
        }

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "objective_kind": self.objective_kind,
            "labels": list(self.labels),
            "target_values": list(self.target_values),
            "weights": list(self.weights),
            "derivatives": self.derivative_payload(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SolveRequest:
    """Typed solve request for scalar and vector calibration workflows."""

    request_id: str
    problem_kind: Literal["root_scalar", "least_squares"]
    parameter_names: tuple[str, ...]
    initial_guess: tuple[float, ...]
    objective: ObjectiveBundle
    bounds: SolveBounds = field(default_factory=SolveBounds)
    constraints: tuple[ConstraintSpec, ...] = ()
    solver_hint: str = ""
    warm_start: WarmStart | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameter_names = tuple(self.parameter_names)
        initial_guess = _normalize_float_tuple(self.initial_guess)
        constraints = tuple(self.constraints)
        if not parameter_names:
            raise ValueError("SolveRequest requires at least one parameter name")
        if len(initial_guess) != len(parameter_names):
            raise ValueError("SolveRequest initial_guess must align with parameter_names")
        if self.problem_kind != self.objective.objective_kind:
            raise ValueError("SolveRequest problem_kind must match ObjectiveBundle objective_kind")
        if self.problem_kind == "root_scalar" and len(parameter_names) != 1:
            raise ValueError("root_scalar solve requests require exactly one parameter")
        bound_length = len(self.bounds.lower) or len(self.bounds.upper)
        if bound_length and bound_length != len(parameter_names):
            raise ValueError("SolveRequest bounds must align with parameter_names")
        if self.warm_start is not None and len(self.warm_start.parameter_values) != len(parameter_names):
            raise ValueError("SolveRequest warm_start must align with parameter_names")
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "initial_guess", initial_guess)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "options", _freeze_mapping(self.options))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload suitable for replay/audit."""
        return {
            "request_id": self.request_id,
            "problem_kind": self.problem_kind,
            "parameter_names": list(self.parameter_names),
            "initial_guess": list(self.initial_guess),
            "objective": self.objective.to_payload(),
            "bounds": self.bounds.to_payload(),
            "constraints": [constraint.to_payload() for constraint in self.constraints],
            "solver_hint": self.solver_hint,
            "warm_start": None if self.warm_start is None else self.warm_start.to_payload(),
            "metadata": dict(self.metadata),
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class SolveResult:
    """Structured solve result for replayable calibration workflows."""

    solution: tuple[float, ...]
    objective_value: float
    residual_vector: tuple[float, ...] = ()
    success: bool = True
    method: str = ""
    iteration_count: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "solution", _normalize_float_tuple(self.solution))
        object.__setattr__(self, "residual_vector", _normalize_float_tuple(self.residual_vector))
        object.__setattr__(self, "objective_value", float(self.objective_value))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "solution": list(self.solution),
            "objective_value": self.objective_value,
            "residual_vector": list(self.residual_vector),
            "success": self.success,
            "method": self.method,
            "iteration_count": self.iteration_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SolveProvenance:
    """Structured solver provenance payload for calibration governance."""

    backend: Mapping[str, object]
    options: Mapping[str, object] = field(default_factory=dict)
    termination: Mapping[str, object] = field(default_factory=dict)
    residual_diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _freeze_mapping(self.backend))
        object.__setattr__(self, "options", _freeze_mapping(self.options))
        object.__setattr__(self, "termination", _freeze_mapping(self.termination))
        object.__setattr__(self, "residual_diagnostics", _freeze_mapping(self.residual_diagnostics))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "backend": dict(self.backend),
            "options": dict(self.options),
            "termination": dict(self.termination),
            "residual_diagnostics": dict(self.residual_diagnostics),
        }


@dataclass(frozen=True)
class SolveReplayArtifact:
    """Structured replay/review artifact for one solve path."""

    request: Mapping[str, object]
    backend: Mapping[str, object] = field(default_factory=dict)
    options: Mapping[str, object] = field(default_factory=dict)
    termination: Mapping[str, object] = field(default_factory=dict)
    residual_diagnostics: Mapping[str, object] = field(default_factory=dict)
    result: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", _freeze_mapping(self.request))
        object.__setattr__(self, "backend", _freeze_mapping(self.backend))
        object.__setattr__(self, "options", _freeze_mapping(self.options))
        object.__setattr__(self, "termination", _freeze_mapping(self.termination))
        object.__setattr__(self, "residual_diagnostics", _freeze_mapping(self.residual_diagnostics))
        object.__setattr__(self, "result", _freeze_mapping(self.result))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "request": dict(self.request),
            "backend": dict(self.backend),
            "options": dict(self.options),
            "termination": dict(self.termination),
            "residual_diagnostics": dict(self.residual_diagnostics),
            "result": dict(self.result),
        }


class UnsupportedSolveCapabilityError(ValueError):
    """Raised when a selected solve backend cannot satisfy the request contract."""

    def __init__(self, backend_id: str, missing_capabilities: Sequence[str]):
        self.backend_id = backend_id
        self.missing_capabilities = tuple(missing_capabilities)
        joined = ", ".join(self.missing_capabilities)
        super().__init__(
            f"Solve backend '{backend_id}' does not support requested capabilities: {joined}"
        )


@dataclass(frozen=True)
class SolveBackendRecord:
    """One solve backend adapter plus its advertised capabilities."""

    backend_id: str
    executor: Callable[[SolveRequest], SolveResult] = field(repr=False, compare=False)
    problem_kinds: tuple[str, ...] = ("root_scalar", "least_squares")
    supports_bounds: bool = True
    supports_constraints: bool = False
    supports_warm_start: bool = True
    supports_jacobian: bool = True
    supports_hessian: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_kinds", tuple(self.problem_kinds))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def missing_capabilities(self, request: SolveRequest) -> tuple[str, ...]:
        """Return any request capabilities that this backend cannot satisfy."""
        missing: list[str] = []
        if request.problem_kind not in self.problem_kinds:
            missing.append(f"objective_shape:{request.problem_kind}")
        if (request.bounds.lower or request.bounds.upper) and not self.supports_bounds:
            missing.append("bounds")
        if request.constraints and not self.supports_constraints:
            missing.append("constraints")
        if request.warm_start is not None and not self.supports_warm_start:
            missing.append("warm_start")
        if request.objective.jacobian_fn is not None and not self.supports_jacobian:
            missing.append("jacobian")
        if request.objective.hessian_fn is not None and not self.supports_hessian:
            missing.append("hessian")
        return tuple(missing)


class SolveBackendRegistry:
    """Registry for calibration solve backends."""

    def __init__(
        self,
        *,
        records: Sequence[SolveBackendRecord] | None = None,
        default_backend_id: str = "scipy",
    ) -> None:
        self._records: dict[str, SolveBackendRecord] = {}
        self.register(_scipy_backend_record())
        for record in records or ():
            self.register(record)
        if default_backend_id not in self._records:
            raise ValueError(f"Unknown default backend: {default_backend_id!r}")
        self.default_backend_id = default_backend_id

    def register(self, record: SolveBackendRecord) -> None:
        """Register or replace one backend record."""
        self._records[record.backend_id] = record

    def get_backend(self, backend_id: str) -> SolveBackendRecord:
        """Return one backend record by id."""
        try:
            return self._records[backend_id]
        except KeyError as exc:
            raise ValueError(f"Unknown solve backend: {backend_id!r}") from exc

    def list_backends(self) -> tuple[SolveBackendRecord, ...]:
        """Return every registered backend record."""
        return tuple(self._records.values())

    def missing_capabilities(self, request: SolveRequest, backend_id: str) -> tuple[str, ...]:
        """Return the unsupported capability list for one backend/request pair."""
        return self.get_backend(backend_id).missing_capabilities(request)


def execute_solve_request(
    request: SolveRequest,
    *,
    backend: str | None = None,
    fallback_backend: str | None = None,
    registry: SolveBackendRegistry | None = None,
) -> SolveResult:
    """Execute one solve request through the selected backend registry surface."""
    resolved_registry = registry or SolveBackendRegistry()
    requested_backend = backend or resolved_registry.default_backend_id
    requested_record = resolved_registry.get_backend(requested_backend)
    missing = requested_record.missing_capabilities(request)
    if missing:
        if fallback_backend and fallback_backend != requested_backend:
            fallback_record = resolved_registry.get_backend(fallback_backend)
            fallback_missing = fallback_record.missing_capabilities(request)
            if fallback_missing:
                raise UnsupportedSolveCapabilityError(fallback_backend, fallback_missing)
            result = fallback_record.executor(request)
            return _annotate_result(
                result,
                backend_id=fallback_backend,
                requested_backend=requested_backend,
                fallback_from=requested_backend,
                fallback_reason=tuple(missing),
            )
        raise UnsupportedSolveCapabilityError(requested_backend, missing)
    result = requested_record.executor(request)
    return _annotate_result(
        result,
        backend_id=requested_backend,
        requested_backend=requested_backend,
    )


def _execute_root_scalar(request: SolveRequest) -> SolveResult:
    """Execute a scalar root solve."""
    objective_fn = request.objective.scalar_objective_fn
    if objective_fn is None:
        raise ValueError("root_scalar solve requests require scalar_objective_fn")
    if not request.bounds.lower or not request.bounds.upper:
        raise ValueError("root_scalar solve requests require finite lower/upper bounds")
    lower = request.bounds.lower[0]
    upper = request.bounds.upper[0]
    if lower is None or upper is None:
        raise ValueError("root_scalar solve requests require finite lower/upper bounds")

    tol = float(request.options.get("tol", 1e-12))
    method = request.solver_hint or "brentq"
    if method.lower() not in {"brentq", "brent"}:
        raise ValueError(f"Unsupported root_scalar solver_hint: {request.solver_hint!r}")

    root = float(brentq(lambda value: float(objective_fn(float(value))), lower, upper, xtol=tol, rtol=tol))
    residual = float(objective_fn(root))
    return SolveResult(
        solution=(root,),
        objective_value=abs(residual),
        residual_vector=(residual,),
        success=True,
        method="brentq",
        iteration_count=None,
        metadata=_solver_derivative_metadata(
            "not_applicable_root_scalar",
            solver_family="scipy",
        ),
    )


def _execute_least_squares(request: SolveRequest) -> SolveResult:
    """Execute a vector or scalar calibration objective through SciPy."""
    method = request.solver_hint or "L-BFGS-B"
    vector_objective_fn = request.objective.vector_objective_fn
    scalar_objective_fn = request.objective.scalar_objective_fn
    jacobian_fn = request.objective.jacobian_fn
    declared_derivative_method = _metadata_string(request.objective.metadata, "derivative_method")

    target_values = raw_np.asarray(request.objective.target_values, dtype=float)
    weights = raw_np.asarray(request.objective.weights, dtype=float)
    initial_guess = raw_np.asarray(request.initial_guess, dtype=float)
    parameter_count = initial_guess.size

    def residual_vector(params: raw_np.ndarray) -> raw_np.ndarray:
        if vector_objective_fn is None:
            return raw_np.asarray((), dtype=float)
        values = raw_np.asarray(vector_objective_fn(params), dtype=float)
        if target_values.size:
            if values.size != target_values.size:
                raise ValueError("vector objective values must align with target_values")
            return values - target_values
        return values

    def weighted_residual_vector(params: raw_np.ndarray) -> raw_np.ndarray:
        residuals = residual_vector(params)
        if weights.size:
            if weights.size != residuals.size:
                raise ValueError("objective weights must align with residual_vector length")
            return raw_np.sqrt(weights) * residuals
        return residuals

    def scalar_objective(params: raw_np.ndarray) -> float:
        if scalar_objective_fn is not None:
            return float(scalar_objective_fn(params))
        residuals = residual_vector(params)
        if residuals.size == 0:
            raise ValueError("least_squares solve requests require scalar or vector objective hooks")
        if weights.size:
            if weights.size != residuals.size:
                raise ValueError("objective weights must align with residual_vector length")
            return float(raw_np.sum(weights * residuals ** 2))
        return float(raw_np.sum(residuals ** 2))

    lower_bounds = raw_np.full(parameter_count, -raw_np.inf, dtype=float)
    upper_bounds = raw_np.full(parameter_count, raw_np.inf, dtype=float)
    minimize_bounds = None
    if request.bounds.lower or request.bounds.upper:
        lower = request.bounds.lower or tuple(None for _ in request.parameter_names)
        upper = request.bounds.upper or tuple(None for _ in request.parameter_names)
        minimize_bounds = []
        for index, (lower_value, upper_value) in enumerate(zip(lower, upper)):
            lower_bound = -raw_np.inf if lower_value is None else float(lower_value)
            upper_bound = raw_np.inf if upper_value is None else float(upper_value)
            lower_bounds[index] = lower_bound
            upper_bounds[index] = upper_bound
            minimize_bounds.append((lower_bound, upper_bound))

    vector_solver_methods = {"trf", "dogbox", "lm"}
    residual_count = 0
    if vector_objective_fn is not None:
        residual_count = int(residual_vector(initial_guess).size)

    jacobian_matrix_fn = None
    if jacobian_fn is not None and vector_objective_fn is not None and method.lower() in vector_solver_methods:
        jacobian_sample = raw_np.asarray(jacobian_fn(initial_guess), dtype=float)
        if jacobian_sample.ndim == 2 and jacobian_sample.shape == (residual_count, parameter_count):
            def jacobian_matrix_fn(params: raw_np.ndarray) -> raw_np.ndarray:
                matrix = raw_np.asarray(jacobian_fn(params), dtype=float)
                if weights.size:
                    return raw_np.sqrt(weights)[:, None] * matrix
                return matrix

    if vector_objective_fn is not None and method.lower() in vector_solver_methods:
        derivative_method = (
            declared_derivative_method
            if jacobian_matrix_fn is not None and declared_derivative_method
            else "provided_vector_jacobian"
            if jacobian_matrix_fn is not None
            else "scipy_2point_residual_jacobian"
        )
        result = least_squares(
            weighted_residual_vector,
            x0=initial_guess,
            jac=jacobian_matrix_fn if jacobian_matrix_fn is not None else "2-point",
            bounds=(lower_bounds, upper_bounds),
            method=method.lower(),
            ftol=float(request.options.get("ftol", 1e-8)),
            xtol=float(request.options.get("xtol", request.options.get("tol", 1e-8))),
            gtol=float(request.options.get("gtol", request.options.get("tol", 1e-8))),
            max_nfev=(
                None
                if request.options.get("maxiter") is None
                else int(request.options.get("maxiter"))
            ),
        )

        residuals = residual_vector(result.x)
        return SolveResult(
            solution=tuple(float(value) for value in result.x),
            objective_value=float(raw_np.sum(weights * residuals ** 2)) if weights.size else float(raw_np.sum(residuals ** 2)),
            residual_vector=tuple(float(value) for value in residuals),
            success=bool(result.success),
            method=method.lower(),
            iteration_count=getattr(result, "nfev", None),
            metadata=_solver_derivative_metadata(
                derivative_method,
                solver_family="scipy",
                message=str(result.message),
            ),
        )

    derivative_method = (
        declared_derivative_method
        if jacobian_fn is not None and declared_derivative_method
        else "provided_scalar_gradient"
        if jacobian_fn is not None
        else "scipy_internal_finite_difference_gradient"
    )
    result = minimize(
        scalar_objective,
        x0=initial_guess,
        jac=jacobian_fn,
        bounds=minimize_bounds,
        method=method,
        options=dict(request.options),
    )

    residuals = residual_vector(result.x)
    return SolveResult(
        solution=tuple(float(value) for value in result.x),
        objective_value=float(result.fun),
        residual_vector=tuple(float(value) for value in residuals),
        success=bool(result.success),
        method=method,
        iteration_count=getattr(result, "nit", None),
        metadata=_solver_derivative_metadata(
            derivative_method,
            solver_family="scipy",
            message=str(result.message),
        ),
    )


def _execute_scipy_request(request: SolveRequest) -> SolveResult:
    """Dispatch one request through the built-in SciPy-backed executor."""
    if request.problem_kind == "root_scalar":
        return _execute_root_scalar(request)
    if request.problem_kind == "least_squares":
        return _execute_least_squares(request)
    raise ValueError(f"Unsupported problem_kind: {request.problem_kind!r}")


def _scipy_backend_record() -> SolveBackendRecord:
    """Return the built-in SciPy backend registration."""
    return SolveBackendRecord(
        backend_id="scipy",
        executor=_execute_scipy_request,
        problem_kinds=("root_scalar", "least_squares"),
        supports_bounds=True,
        supports_constraints=False,
        supports_warm_start=True,
        supports_jacobian=True,
        supports_hessian=False,
        metadata={"solver_family": "scipy"},
    )


def _annotate_result(result: SolveResult, **metadata: object) -> SolveResult:
    """Return ``result`` with execution metadata merged into the payload."""
    merged = dict(result.metadata)
    merged.update(metadata)
    return replace(result, metadata=merged)


def build_solve_provenance(request: SolveRequest, result: SolveResult) -> SolveProvenance:
    """Build the standardized solver provenance payload for one solve."""
    backend = {
        "backend_id": str(result.metadata.get("backend_id", "")).strip(),
        "requested_backend": str(result.metadata.get("requested_backend", "")).strip(),
        "fallback_from": result.metadata.get("fallback_from"),
        "solver_family": str(result.metadata.get("solver_family", "")).strip(),
        "derivative_method": str(result.metadata.get("derivative_method", "")).strip(),
        "resolved_derivative_method": str(result.metadata.get("resolved_derivative_method", "")).strip(),
        "derivative_method_category": str(result.metadata.get("derivative_method_category", "")).strip(),
        "derivative_method_support": str(result.metadata.get("derivative_method_support", "")).strip(),
        "backend_operator": result.metadata.get("backend_operator"),
        "fallback_derivative_method": result.metadata.get("fallback_derivative_method"),
        "method": result.method,
    }
    termination = {
        "success": bool(result.success),
        "reason": _termination_reason(result),
        "message": str(result.metadata.get("message", "")).strip(),
        "iteration_count": result.iteration_count,
        "objective_value": float(result.objective_value),
    }
    residual_diagnostics = _residual_diagnostics(result)
    return SolveProvenance(
        backend=backend,
        options=dict(request.options),
        termination=termination,
        residual_diagnostics=residual_diagnostics,
    )


def build_solve_replay_artifact(request: SolveRequest, result: SolveResult) -> SolveReplayArtifact:
    """Build the standardized replay/review artifact for one solve."""
    provenance = build_solve_provenance(request, result)
    return SolveReplayArtifact(
        request=request.to_payload(),
        backend=dict(provenance.backend),
        options=dict(provenance.options),
        termination=dict(provenance.termination),
        residual_diagnostics=dict(provenance.residual_diagnostics),
        result=result.to_payload(),
    )


def _residual_diagnostics(result: SolveResult) -> dict[str, object]:
    """Summarize the residual vector for audit and replay consumers."""
    residuals = raw_np.asarray(result.residual_vector, dtype=float)
    if residuals.size == 0:
        return {
            "residual_count": 0,
            "max_abs_residual": 0.0,
            "l2_norm": 0.0,
            "residual_vector": [],
        }
    abs_residuals = raw_np.abs(residuals)
    return {
        "residual_count": int(residuals.size),
        "max_abs_residual": float(abs_residuals.max()),
        "l2_norm": float(raw_np.linalg.norm(residuals)),
        "residual_vector": [float(value) for value in residuals],
    }


def _termination_reason(result: SolveResult) -> str:
    """Return a stable termination-reason string for one solve result."""
    message = str(result.metadata.get("message", "")).strip()
    if message:
        return message
    return "success" if result.success else "failure"


__all__ = [
    "ConstraintSpec",
    "ObjectiveBundle",
    "SolveBackendRecord",
    "SolveBackendRegistry",
    "SolveBounds",
    "SolveProvenance",
    "SolveRequest",
    "SolveReplayArtifact",
    "SolveResult",
    "UnsupportedSolveCapabilityError",
    "WarmStart",
    "build_solve_provenance",
    "build_solve_replay_artifact",
    "execute_solve_request",
]
