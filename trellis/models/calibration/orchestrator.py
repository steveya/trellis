"""Bounded public orchestrator for supported calibration problem IRs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from trellis.core.market_state import MarketState
from trellis.models.calibration.credit import (
    CreditHazardCalibrationQuote,
    fit_single_name_credit_problem_ir,
)
from trellis.models.calibration.problem_ir import CalibrationProblemIR
from trellis.models.calibration.sabr_fit import (
    SABRSmileSurface,
    fit_sabr_smile_problem_ir,
)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy."""
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class CalibrationProblemIRAdapterSpec:
    """One supported problem-IR orchestrator adapter."""

    adapter_id: str
    family_id: str
    workflow_id: str
    required_context: tuple[str, ...]
    result_type: str
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", str(self.adapter_id).strip())
        object.__setattr__(self, "family_id", str(self.family_id).strip())
        object.__setattr__(self, "workflow_id", str(self.workflow_id).strip())
        object.__setattr__(self, "required_context", tuple(str(value).strip() for value in self.required_context))
        object.__setattr__(self, "result_type", str(self.result_type).strip())
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly support-matrix payload."""
        return {
            "adapter_id": self.adapter_id,
            "family_id": self.family_id,
            "workflow_id": self.workflow_id,
            "required_context": list(self.required_context),
            "result_type": self.result_type,
            "description": self.description,
            "metadata": dict(self.metadata),
        }


class UnsupportedCalibrationProblemIRError(ValueError):
    """Raised when the public problem-IR orchestrator has no matching adapter."""

    def __init__(self, problem: CalibrationProblemIR, supported: Sequence[CalibrationProblemIRAdapterSpec]):
        self.problem = problem
        self.supported = tuple(supported)
        supported_text = ", ".join(
            f"{adapter.family_id}:{adapter.workflow_id}" for adapter in self.supported
        )
        super().__init__(
            "Unsupported calibration problem IR "
            f"{problem.family_id}:{problem.workflow_id}. "
            f"Supported workflows: {supported_text}"
        )


_SABR_ADAPTER = CalibrationProblemIRAdapterSpec(
    adapter_id="sabr_smile_problem_ir_v1",
    family_id="sabr",
    workflow_id="sabr_smile",
    required_context=("surface",),
    result_type="SABRSmileCalibrationResult",
    description="Adapter-backed SABR smile calibration through the checked direct workflow.",
    metadata={"support_boundary": "bounded_problem_ir_orchestrator"},
)

_CREDIT_ADAPTER = CalibrationProblemIRAdapterSpec(
    adapter_id="single_name_credit_problem_ir_v1",
    family_id="credit",
    workflow_id="single_name_credit_curve",
    required_context=("quotes", "market_state"),
    result_type="CreditHazardCalibrationResult",
    description="Adapter-backed single-name credit curve calibration through the checked direct workflow.",
    metadata={"support_boundary": "bounded_problem_ir_orchestrator"},
)


def supported_calibration_problem_ir_adapters() -> tuple[CalibrationProblemIRAdapterSpec, ...]:
    """Return the bounded public problem-IR support matrix."""
    return (_SABR_ADAPTER, _CREDIT_ADAPTER)


def _resolve_adapter(problem: CalibrationProblemIR) -> CalibrationProblemIRAdapterSpec:
    for adapter in supported_calibration_problem_ir_adapters():
        if adapter.family_id == problem.family_id and adapter.workflow_id == problem.workflow_id:
            return adapter
    raise UnsupportedCalibrationProblemIRError(problem, supported_calibration_problem_ir_adapters())


def _credit_curve_name(problem: CalibrationProblemIR, curve_name: str | None) -> str:
    if curve_name is not None:
        return str(curve_name)
    if problem.materialization is not None and problem.materialization.object_name:
        return problem.materialization.object_name
    return "single_name_credit"


def _credit_recovery(problem: CalibrationProblemIR, recovery: float | None) -> float:
    if recovery is not None:
        return float(recovery)
    if problem.materialization is not None:
        value = problem.materialization.metadata.get("recovery")
        if value is not None:
            return float(value)
    return 0.4


def _credit_max_hazard(problem: CalibrationProblemIR, max_hazard: float | None) -> float:
    if max_hazard is not None:
        return float(max_hazard)
    value = problem.replay_metadata.get("max_hazard")
    if value is not None:
        return float(value)
    return 5.0


def _attach_orchestrator_provenance(
    result: object,
    problem: CalibrationProblemIR,
    adapter: CalibrationProblemIRAdapterSpec,
) -> object:
    """Return a result with bounded orchestrator provenance attached."""
    provenance = dict(getattr(result, "provenance", {}) or {})
    provenance["calibration_problem_ir_orchestrator"] = {
        "adapter_id": adapter.adapter_id,
        "family_id": adapter.family_id,
        "workflow_id": adapter.workflow_id,
        "problem_id": problem.problem_id,
        "support_boundary": adapter.metadata.get("support_boundary", ""),
    }
    return replace(result, provenance=provenance)


def calibrate_problem_ir(
    problem: CalibrationProblemIR,
    *,
    surface: SABRSmileSurface | None = None,
    quotes: Sequence[CreditHazardCalibrationQuote] | None = None,
    market_state: MarketState | None = None,
    recovery: float | None = None,
    curve_name: str | None = None,
    max_hazard: float | None = None,
) -> object:
    """Execute one supported calibration problem IR through a fail-closed adapter."""
    adapter = _resolve_adapter(problem)
    if adapter is _SABR_ADAPTER:
        if surface is None:
            raise ValueError("SABR problem-IR orchestration requires `surface` context")
        result = fit_sabr_smile_problem_ir(problem, surface)
        return _attach_orchestrator_provenance(result, problem, adapter)
    if adapter is _CREDIT_ADAPTER:
        if quotes is None:
            raise ValueError("single-name credit problem-IR orchestration requires `quotes` context")
        if market_state is None:
            raise ValueError("single-name credit problem-IR orchestration requires `market_state` context")
        result = fit_single_name_credit_problem_ir(
            problem,
            quotes,
            market_state,
            recovery=_credit_recovery(problem, recovery),
            curve_name=_credit_curve_name(problem, curve_name),
            max_hazard=_credit_max_hazard(problem, max_hazard),
        )
        return _attach_orchestrator_provenance(result, problem, adapter)
    raise UnsupportedCalibrationProblemIRError(problem, supported_calibration_problem_ir_adapters())


__all__ = [
    "CalibrationProblemIRAdapterSpec",
    "UnsupportedCalibrationProblemIRError",
    "calibrate_problem_ir",
    "supported_calibration_problem_ir_adapters",
]
