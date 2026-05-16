"""Semantic admission policy for bounded graph-owned hybrid AD lanes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from trellis.agent.contract_ir import (
    Add,
    ArithmeticMean,
    CompositeUnderlying,
    Constant,
    ContractIR,
    CurveQuote,
    EquitySpot,
    Forward,
    Indicator,
    LinearBasket,
    Max,
    Min,
    Mul,
    Scaled,
    Spot,
    Strike,
    Sub,
    SurfaceQuote,
    SwapRate,
    Annuity,
    VarianceObservable,
)
from trellis.agent.dynamic_contract_ir import DynamicContractIR


_DERIVATIVE_METHODS = frozenset({"vjp", "jvp", "hvp"})
_STATE_KINDS = frozenset(
    {
        "terminal_state",
        "smooth_path_summary",
        "discontinuous_event_monitor",
        "early_exercise_control",
        "dynamic_state",
    }
)
_STATE_SUPPORT_STATUSES = frozenset({"supported", "planned", "unsupported"})
_STATE_DIFFERENTIABILITY_CLASSES = frozenset(
    {
        "smooth",
        "piecewise",
        "discontinuous",
        "projected",
        "held_fixed",
        "unsupported",
    }
)
_SCALAR_CORRELATION_STRUCTURES = frozenset(
    {
        "scalar",
        "scalar_correlation",
        "correlation_scalar",
        "hybrid_correlation",
        "graph_scalar",
        "graph_scalar_coordinates",
    }
)
_MATRIX_CORRELATION_STRUCTURES = frozenset(
    {"matrix", "correlation_matrix", "psd_matrix", "correlation_matrix_psd_policy"}
)
_SURFACE_CORRELATION_STRUCTURES = frozenset(
    {"surface", "correlation_surface", "correlation_surface_policy"}
)


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _copy_diagnostics(
    diagnostics: Iterable[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(diagnostic) for diagnostic in (diagnostics or ()))


@dataclass(frozen=True)
class HybridADStatePolicy:
    """Semantic state/event policy for bounded hybrid AD admission."""

    state_kind: str
    support_status: str
    differentiability_class: str
    reason: str
    event_policy: str = "not_applicable"
    control_policy: str = "not_applicable"
    state_variable_roles: tuple[str, ...] = field(default_factory=tuple)
    fail_closed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple, hash=False)

    def __post_init__(self) -> None:
        state_kind = _normalize(self.state_kind)
        if state_kind not in _STATE_KINDS:
            raise ValueError(f"state_kind must be one of {sorted(_STATE_KINDS)}")
        object.__setattr__(self, "state_kind", state_kind)
        support_status = _normalize(self.support_status)
        if support_status not in _STATE_SUPPORT_STATUSES:
            raise ValueError(
                f"support_status must be one of {sorted(_STATE_SUPPORT_STATUSES)}"
            )
        object.__setattr__(self, "support_status", support_status)
        differentiability_class = _normalize(self.differentiability_class)
        if differentiability_class not in _STATE_DIFFERENTIABILITY_CLASSES:
            raise ValueError(
                "differentiability_class must be one of "
                f"{sorted(_STATE_DIFFERENTIABILITY_CLASSES)}"
            )
        object.__setattr__(self, "differentiability_class", differentiability_class)
        object.__setattr__(self, "reason", _clean(self.reason, "reason"))
        object.__setattr__(
            self,
            "event_policy",
            str(self.event_policy or "not_applicable").strip() or "not_applicable",
        )
        object.__setattr__(
            self,
            "control_policy",
            str(self.control_policy or "not_applicable").strip() or "not_applicable",
        )
        object.__setattr__(
            self,
            "state_variable_roles",
            tuple(
                role
                for role in (
                    str(raw_role).strip()
                    for raw_role in self.state_variable_roles
                )
                if role
            ),
        )
        object.__setattr__(self, "fail_closed", bool(self.fail_closed))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))
        object.__setattr__(self, "diagnostics", _copy_diagnostics(self.diagnostics))

    @property
    def supported(self) -> bool:
        """Return whether this state policy admits runtime derivative execution."""
        return self.support_status == "supported"

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly policy payload."""
        return {
            "state_kind": self.state_kind,
            "support_status": self.support_status,
            "differentiability_class": self.differentiability_class,
            "reason": self.reason,
            "event_policy": self.event_policy,
            "control_policy": self.control_policy,
            "state_variable_roles": list(self.state_variable_roles),
            "fail_closed": self.fail_closed,
            "metadata": dict(self.metadata),
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "HybridADStatePolicy":
        """Build a state policy from :meth:`to_payload` output."""
        return cls(
            state_kind=str(payload["state_kind"]),
            support_status=str(payload["support_status"]),
            differentiability_class=str(payload["differentiability_class"]),
            reason=str(payload["reason"]),
            event_policy=str(payload.get("event_policy", "not_applicable")),
            control_policy=str(payload.get("control_policy", "not_applicable")),
            state_variable_roles=tuple(payload.get("state_variable_roles", ())),
            fail_closed=bool(payload.get("fail_closed", True)),
            metadata=payload.get("metadata") or {},
            diagnostics=payload.get("diagnostics") or (),
        )


@dataclass(frozen=True)
class HybridADFactorRequirement:
    """Semantic market-coordinate requirement for one graph-owned hybrid AD lane."""

    object_type: str
    coordinate_type: str
    risk_class: str
    parameterization: str
    semantic_role: str = ""
    graph_role: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_type", _clean(self.object_type, "object_type"))
        object.__setattr__(
            self,
            "coordinate_type",
            _clean(self.coordinate_type, "coordinate_type"),
        )
        object.__setattr__(self, "risk_class", _clean(self.risk_class, "risk_class"))
        object.__setattr__(
            self,
            "parameterization",
            _clean(self.parameterization, "parameterization"),
        )
        object.__setattr__(self, "semantic_role", str(self.semantic_role or "").strip())
        object.__setattr__(self, "graph_role", str(self.graph_role or "").strip())
        object.__setattr__(self, "required", bool(self.required))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "object_type": self.object_type,
            "coordinate_type": self.coordinate_type,
            "risk_class": self.risk_class,
            "parameterization": self.parameterization,
            "semantic_role": self.semantic_role,
            "graph_role": self.graph_role,
            "required": self.required,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> HybridADFactorRequirement:
        """Build a requirement from :meth:`to_payload` output."""
        return cls(
            object_type=str(payload["object_type"]),
            coordinate_type=str(payload["coordinate_type"]),
            risk_class=str(payload["risk_class"]),
            parameterization=str(payload["parameterization"]),
            semantic_role=str(payload.get("semantic_role", "")),
            graph_role=str(payload.get("graph_role", "")),
            required=bool(payload.get("required", True)),
        )


@dataclass(frozen=True)
class HybridADLaneAdmission:
    """Semantic support decision for a graph-owned hybrid AD lane."""

    admitted: bool
    lane_id: str
    support_status: str
    reason: str
    semantic_contract_type: str
    product_family: str
    contract_shape: str
    derivative_methods: tuple[str, ...] = ("vjp", "hvp")
    factor_requirements: tuple[HybridADFactorRequirement, ...] = field(
        default_factory=tuple
    )
    derivative_method_category: str = "hybrid_ad"
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "admitted", bool(self.admitted))
        object.__setattr__(self, "lane_id", _clean(self.lane_id, "lane_id"))
        object.__setattr__(self, "support_status", _clean(self.support_status, "support_status"))
        object.__setattr__(self, "reason", _clean(self.reason, "reason"))
        object.__setattr__(
            self,
            "semantic_contract_type",
            _clean(self.semantic_contract_type, "semantic_contract_type"),
        )
        object.__setattr__(self, "product_family", _clean(self.product_family, "product_family"))
        object.__setattr__(self, "contract_shape", _clean(self.contract_shape, "contract_shape"))
        methods = tuple(_normalize(method) for method in self.derivative_methods)
        if not methods:
            raise ValueError("derivative_methods must be non-empty")
        for method in methods:
            if method not in _DERIVATIVE_METHODS:
                raise ValueError("derivative_methods must contain vjp, jvp, or hvp")
        object.__setattr__(self, "derivative_methods", methods)
        requirements = tuple(self.factor_requirements)
        for requirement in requirements:
            if not isinstance(requirement, HybridADFactorRequirement):
                raise TypeError(
                    "factor_requirements must contain HybridADFactorRequirement values"
                )
        object.__setattr__(self, "factor_requirements", requirements)
        object.__setattr__(
            self,
            "derivative_method_category",
            _clean(self.derivative_method_category, "derivative_method_category"),
        )
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))
        object.__setattr__(self, "diagnostics", _copy_diagnostics(self.diagnostics))

    @property
    def supported(self) -> bool:
        """Return whether this admission can enter a hybrid AD lane today."""
        return self.admitted and self.support_status == "supported"

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly admission payload."""
        return {
            "admitted": self.admitted,
            "supported": self.supported,
            "lane_id": self.lane_id,
            "support_status": self.support_status,
            "reason": self.reason,
            "semantic_contract_type": self.semantic_contract_type,
            "product_family": self.product_family,
            "contract_shape": self.contract_shape,
            "derivative_methods": list(self.derivative_methods),
            "factor_requirements": [
                requirement.to_payload()
                for requirement in self.factor_requirements
            ],
            "derivative_method_category": self.derivative_method_category,
            "derivative_method_support": self.support_status,
            "metadata": dict(self.metadata),
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> HybridADLaneAdmission:
        """Build an admission decision from :meth:`to_payload` output."""
        return cls(
            admitted=bool(payload["admitted"]),
            lane_id=str(payload["lane_id"]),
            support_status=str(payload["support_status"]),
            reason=str(payload["reason"]),
            semantic_contract_type=str(payload["semantic_contract_type"]),
            product_family=str(payload["product_family"]),
            contract_shape=str(payload["contract_shape"]),
            derivative_methods=tuple(
                str(method)
                for method in payload.get("derivative_methods", ("vjp", "hvp"))
            ),
            factor_requirements=tuple(
                HybridADFactorRequirement.from_payload(requirement)
                for requirement in payload.get("factor_requirements", ())
            ),
            derivative_method_category=str(
                payload.get("derivative_method_category", "hybrid_ad")
            ),
            metadata=payload.get("metadata") or {},
            diagnostics=payload.get("diagnostics") or (),
        )


def admit_hybrid_ad_lane(
    semantic_contract: object,
    *,
    derivative_method: str = "vjp",
    correlation_structure: str = "scalar",
    product_family: str | None = None,
) -> HybridADLaneAdmission:
    """Return the bounded graph-owned hybrid AD admission decision.

    Admission is deliberately narrower than pricing support. It proves whether
    the current semantic contract may enter an existing graph-backed derivative
    lane and otherwise records the fail-closed reason without executing AD.
    """
    requested_method = _normalize(derivative_method) or "vjp"
    structure = _normalize(correlation_structure) or "scalar"
    if requested_method not in _DERIVATIVE_METHODS:
        return _admission(
            admitted=False,
            lane_id="unsupported_hybrid_derivative_method",
            support_status="unsupported",
            reason="hybrid_derivative_method_unsupported",
            semantic_contract_type=type(semantic_contract).__name__,
            product_family=product_family or "unknown",
            contract_shape="unknown",
            derivative_methods=("vjp", "hvp"),
            factor_requirements=(),
            metadata={"requested_derivative_method": requested_method},
            diagnostics=(
                {
                    "code": "hybrid_derivative_method_unsupported",
                    "severity": "warning",
                    "requested_derivative_method": requested_method,
                },
            ),
        )
    if isinstance(semantic_contract, DynamicContractIR):
        return _dynamic_contract_admission(
            semantic_contract,
            derivative_method=requested_method,
            product_family=product_family,
        )
    if not isinstance(semantic_contract, ContractIR):
        return _admission(
            admitted=False,
            lane_id="unsupported_semantic_contract",
            support_status="unsupported",
            reason="unsupported_semantic_contract_type",
            semantic_contract_type=type(semantic_contract).__name__,
            product_family=product_family or "unknown",
            contract_shape="unknown",
            derivative_methods=("vjp", "hvp"),
            factor_requirements=(),
            diagnostics=(
                {
                    "code": "unsupported_semantic_contract_type",
                    "severity": "warning",
                },
            ),
        )
    return _contract_ir_admission(
        semantic_contract,
        derivative_method=requested_method,
        correlation_structure=structure,
        product_family=product_family,
    )


def _dynamic_contract_admission(
    contract: DynamicContractIR,
    *,
    derivative_method: str,
    product_family: str | None,
) -> HybridADLaneAdmission:
    family = product_family or contract.semantic_family or "dynamic_hybrid_contract"
    if derivative_method == "jvp":
        return _admission(
            admitted=False,
            lane_id="dynamic_hybrid_state_jvp",
            support_status="unsupported",
            reason="hybrid_jvp_backend_unsupported",
            semantic_contract_type="DynamicContractIR",
            product_family=family,
            contract_shape="dynamic_hybrid_state",
            derivative_methods=("vjp", "hvp"),
            factor_requirements=(_scalar_correlation_requirement(),),
            metadata={
                "requested_derivative_method": derivative_method,
                "backend_operator": "jvp",
                "base_track": contract.base_track,
                "fail_closed": True,
            },
            diagnostics=(
                {
                    "code": "hybrid_jvp_backend_unsupported",
                    "severity": "warning",
                    "backend_operator": "jvp",
                },
            ),
        )
    return _admission(
        admitted=False,
        lane_id="dynamic_hybrid_state_policy",
        support_status="planned",
        reason="dynamic_hybrid_state_admission_pending",
        semantic_contract_type="DynamicContractIR",
        product_family=family,
        contract_shape="dynamic_hybrid_state",
        derivative_methods=("vjp", "hvp"),
        factor_requirements=(_scalar_correlation_requirement(),),
        metadata={
            "requested_derivative_method": derivative_method,
            "base_track": contract.base_track,
            "fail_closed": True,
        },
        diagnostics=(
            {
                "code": "dynamic_hybrid_state_admission_pending",
                "severity": "warning",
            },
        ),
    )


def _contract_ir_admission(
    contract: ContractIR,
    *,
    derivative_method: str,
    correlation_structure: str,
    product_family: str | None,
) -> HybridADLaneAdmission:
    if derivative_method == "jvp":
        return _admission(
            admitted=False,
            lane_id="quanto_scalar_graph_jvp",
            support_status="unsupported",
            reason="hybrid_jvp_backend_unsupported",
            semantic_contract_type="ContractIR",
            product_family=product_family or "unknown",
            contract_shape=_contract_shape(contract),
            derivative_methods=("vjp", "hvp"),
            factor_requirements=_quanto_scalar_requirements(),
            metadata={
                "requested_derivative_method": derivative_method,
                "backend_operator": "jvp",
                "fail_closed": True,
            },
            diagnostics=(
                {
                    "code": "hybrid_jvp_backend_unsupported",
                    "severity": "warning",
                    "backend_operator": "jvp",
                },
            ),
        )
    if (
        correlation_structure not in _SCALAR_CORRELATION_STRUCTURES
        and correlation_structure not in _MATRIX_CORRELATION_STRUCTURES
        and correlation_structure not in _SURFACE_CORRELATION_STRUCTURES
    ):
        return _admission(
            admitted=False,
            lane_id="unsupported_correlation_structure",
            support_status="unsupported",
            reason="unsupported_correlation_structure",
            semantic_contract_type="ContractIR",
            product_family=product_family or "unknown",
            contract_shape=_contract_shape(contract),
            derivative_methods=("vjp", "hvp"),
            factor_requirements=(),
            metadata={
                "requested_derivative_method": derivative_method,
                "correlation_structure": correlation_structure,
                "fail_closed": True,
            },
            diagnostics=(
                {
                    "code": "unsupported_correlation_structure",
                    "severity": "warning",
                    "correlation_structure": correlation_structure,
                },
            ),
        )
    if _contains_indicator(contract.payoff):
        return _admission(
            admitted=False,
            lane_id="path_dependent_event_policy",
            support_status="unsupported",
            reason="unsupported_discontinuous_event_monitor",
            semantic_contract_type="ContractIR",
            product_family=product_family or "path_dependent_hybrid_option",
            contract_shape="discontinuous_event_monitor",
            derivative_methods=("vjp", "hvp"),
            factor_requirements=(_scalar_correlation_requirement(),),
            metadata={"requested_derivative_method": derivative_method, "fail_closed": True},
            diagnostics=(
                {
                    "code": "unsupported_discontinuous_event_monitor",
                    "severity": "warning",
                },
            ),
        )
    if isinstance(contract.underlying.spec, CompositeUnderlying):
        return _admission(
            admitted=False,
            lane_id="hybrid_factor_graph_admission",
            support_status="planned",
            reason="hybrid_factor_graph_admission_pending",
            semantic_contract_type="ContractIR",
            product_family=product_family or "hybrid_option",
            contract_shape="hybrid_composite_underlying",
            derivative_methods=("vjp", "hvp"),
            factor_requirements=(_scalar_correlation_requirement(),),
            metadata={
                "requested_derivative_method": derivative_method,
                "composite_underlying_count": len(contract.underlying.spec.parts),
                "fail_closed": True,
            },
            diagnostics=(
                {
                    "code": "hybrid_factor_graph_admission_pending",
                    "severity": "warning",
                },
            ),
        )
    if _is_path_dependent(contract):
        return _admission(
            admitted=False,
            lane_id="path_dependent_hybrid_state",
            support_status="planned",
            reason="path_dependent_hybrid_state_pending",
            semantic_contract_type="ContractIR",
            product_family=product_family or "path_dependent_hybrid_option",
            contract_shape="path_dependent_hybrid_state",
            derivative_methods=("vjp", "hvp"),
            factor_requirements=_quanto_scalar_requirements(),
            metadata={
                "requested_derivative_method": derivative_method,
                "observation_kind": contract.observation.kind,
                "fail_closed": True,
            },
            diagnostics=(
                {
                    "code": "path_dependent_hybrid_state_pending",
                    "severity": "warning",
                },
            ),
        )
    if contract.exercise.style in {"american", "bermudan"}:
        return _admission(
            admitted=False,
            lane_id="early_exercise_hybrid_state",
            support_status="planned",
            reason="early_exercise_hybrid_state_pending",
            semantic_contract_type="ContractIR",
            product_family=product_family or "early_exercise_hybrid_option",
            contract_shape="early_exercise_hybrid_state",
            derivative_methods=("vjp", "hvp"),
            factor_requirements=_quanto_scalar_requirements(),
            metadata={
                "requested_derivative_method": derivative_method,
                "exercise_style": contract.exercise.style,
                "fail_closed": True,
            },
            diagnostics=(
                {
                    "code": "early_exercise_hybrid_state_pending",
                    "severity": "warning",
                },
            ),
        )
    if correlation_structure in _MATRIX_CORRELATION_STRUCTURES:
        return _correlation_structure_admission(
            contract,
            product_family=product_family,
            derivative_method=derivative_method,
            structure_type="correlation_matrix",
        )
    if correlation_structure in _SURFACE_CORRELATION_STRUCTURES:
        return _correlation_structure_admission(
            contract,
            product_family=product_family,
            derivative_method=derivative_method,
            structure_type="correlation_surface",
        )
    if _is_quanto_family(product_family) and _is_terminal_vanilla_option(contract):
        lane_id = f"quanto_scalar_graph_{derivative_method}"
        return _admission(
            admitted=True,
            lane_id=lane_id,
            support_status="supported",
            reason=f"supported_{lane_id}",
            semantic_contract_type="ContractIR",
            product_family=product_family or "quanto_option",
            contract_shape="quanto_terminal_vanilla_option",
            derivative_methods=("vjp", "hvp"),
            factor_requirements=_quanto_scalar_requirements(),
            metadata={
                "requested_derivative_method": derivative_method,
                "correlation_structure": "scalar_correlation",
                "coordinate_space": "constrained",
                "hybrid_derivative_policy": "bounded_quanto_scalar_graph_vjp_hvp",
                "runtime_helper": "trellis.analytics.hybrid_ad.differentiate_quanto_scalar_inputs",
                "fail_closed_near_correlation_bounds": True,
            },
        )
    return _admission(
        admitted=False,
        lane_id="unsupported_contract_ir_hybrid_ad",
        support_status="unsupported",
        reason="unsupported_contract_ir_hybrid_ad_shape",
        semantic_contract_type="ContractIR",
        product_family=product_family or "unknown",
        contract_shape=_contract_shape(contract),
        derivative_methods=("vjp", "hvp"),
        factor_requirements=(),
        metadata={"requested_derivative_method": derivative_method, "fail_closed": True},
        diagnostics=(
            {
                "code": "unsupported_contract_ir_hybrid_ad_shape",
                "severity": "warning",
            },
        ),
    )


def _correlation_structure_admission(
    contract: ContractIR,
    *,
    product_family: str | None,
    derivative_method: str,
    structure_type: str,
) -> HybridADLaneAdmission:
    if structure_type == "correlation_matrix":
        if _is_quanto_family(product_family) and _is_terminal_vanilla_option(contract):
            lane_id = f"quanto_matrix_graph_{derivative_method}"
            return _admission(
                admitted=True,
                lane_id=lane_id,
                support_status="supported",
                reason=f"supported_{lane_id}",
                semantic_contract_type="ContractIR",
                product_family=product_family or "quanto_option",
                contract_shape="quanto_terminal_vanilla_option",
                derivative_methods=("vjp", "hvp"),
                factor_requirements=_quanto_matrix_requirements(),
                metadata={
                    "requested_derivative_method": derivative_method,
                    "correlation_structure": "correlation_matrix",
                    "chart_policy_status": "validated_executable",
                    "coordinate_space": "matrix",
                    "projection_policy": "unsupported_no_smoothing_or_projection",
                    "hybrid_derivative_policy": "bounded_quanto_matrix_graph_vjp_hvp",
                    "runtime_helper": (
                        "trellis.analytics.hybrid_ad."
                        "differentiate_quanto_correlation_matrix"
                    ),
                    "fail_closed_near_psd_boundary": True,
                },
            )
        reason = "correlation_matrix_derivative_not_implemented"
        requirement = _correlation_matrix_requirement()
        chart_policy_status = "validated_fail_closed"
    else:
        reason = "correlation_surface_chart_not_implemented"
        requirement = _correlation_surface_requirement()
        chart_policy_status = "surface_chart_not_implemented"
    return _admission(
        admitted=False,
        lane_id=f"{structure_type}_hybrid_ad_policy",
        support_status="planned",
        reason=reason,
        semantic_contract_type="ContractIR",
        product_family=product_family or "quanto_option",
        contract_shape=_contract_shape(contract),
        derivative_methods=("vjp", "hvp"),
        factor_requirements=(requirement,),
        metadata={
            "requested_derivative_method": derivative_method,
            "correlation_structure": structure_type,
            "chart_policy_status": chart_policy_status,
            "fail_closed": True,
        },
        diagnostics=(
            {
                "code": reason,
                "severity": "warning",
                "correlation_structure": structure_type,
                "chart_policy_status": chart_policy_status,
            },
        ),
    )


def _admission(
    *,
    admitted: bool,
    lane_id: str,
    support_status: str,
    reason: str,
    semantic_contract_type: str,
    product_family: str,
    contract_shape: str,
    derivative_methods: tuple[str, ...],
    factor_requirements: tuple[HybridADFactorRequirement, ...],
    metadata: Mapping[str, Any] | None = None,
    diagnostics: Iterable[Mapping[str, Any]] | None = None,
) -> HybridADLaneAdmission:
    return HybridADLaneAdmission(
        admitted=admitted,
        lane_id=lane_id,
        support_status=support_status,
        reason=reason,
        semantic_contract_type=semantic_contract_type,
        product_family=product_family,
        contract_shape=contract_shape,
        derivative_methods=derivative_methods,
        factor_requirements=factor_requirements,
        metadata=metadata or {},
        diagnostics=tuple(diagnostics or ()),
    )


def _contract_shape(contract: ContractIR) -> str:
    if isinstance(contract.underlying.spec, CompositeUnderlying):
        return "hybrid_composite_underlying"
    if _contains_indicator(contract.payoff):
        return "discontinuous_event_monitor"
    if _is_path_dependent(contract):
        return "path_dependent_hybrid_state"
    if contract.exercise.style in {"american", "bermudan"}:
        return "early_exercise_hybrid_state"
    if _is_terminal_vanilla_option(contract):
        return "quanto_terminal_vanilla_option"
    return "unsupported_contract_ir"


def _is_terminal_vanilla_option(contract: ContractIR) -> bool:
    if contract.exercise.style != "european" or contract.observation.kind != "terminal":
        return False
    body = _vanilla_intrinsic_body(contract.payoff)
    if body is None:
        return False
    if not isinstance(contract.underlying.spec, EquitySpot):
        return False
    return _is_spot_strike_sub(body) or _is_strike_spot_sub(body)


def _is_path_dependent(contract: ContractIR) -> bool:
    return contract.observation.kind == "path_dependent" or _contains_arithmetic_mean(
        contract.payoff
    )


def _vanilla_intrinsic_body(expr: object) -> object | None:
    if not isinstance(expr, Max) or len(expr.args) != 2:
        return None
    left, right = expr.args
    if _is_zero_constant(left):
        return right
    if _is_zero_constant(right):
        return left
    return None


def _is_spot_strike_sub(expr: object) -> bool:
    return isinstance(expr, Sub) and isinstance(expr.lhs, Spot) and isinstance(expr.rhs, Strike)


def _is_strike_spot_sub(expr: object) -> bool:
    return isinstance(expr, Sub) and isinstance(expr.lhs, Strike) and isinstance(expr.rhs, Spot)


def _is_zero_constant(expr: object) -> bool:
    return isinstance(expr, Constant) and expr.value == 0.0


def _contains_indicator(expr: object) -> bool:
    if isinstance(expr, Indicator):
        return True
    return any(_contains_indicator(child) for child in _child_exprs(expr))


def _contains_arithmetic_mean(expr: object) -> bool:
    if isinstance(expr, ArithmeticMean):
        return True
    return any(_contains_arithmetic_mean(child) for child in _child_exprs(expr))


def _child_exprs(expr: object) -> tuple[object, ...]:
    if isinstance(expr, (Add, Max, Min, Mul)):
        return tuple(expr.args)
    if isinstance(expr, Sub):
        return (expr.lhs, expr.rhs)
    if isinstance(expr, Scaled):
        return (expr.scalar, expr.body)
    if isinstance(expr, LinearBasket):
        return tuple(child for _, child in expr.terms)
    if isinstance(expr, ArithmeticMean):
        return (expr.expr,)
    if isinstance(
        expr,
        (
            Constant,
            Spot,
            Strike,
            Forward,
            SwapRate,
            Annuity,
            CurveQuote,
            SurfaceQuote,
            VarianceObservable,
            Indicator,
        ),
    ):
        return ()
    return ()


def _quanto_scalar_requirements() -> tuple[HybridADFactorRequirement, ...]:
    return (
        HybridADFactorRequirement(
            object_type="spot",
            coordinate_type="spot",
            risk_class="equity",
            parameterization="scalar_spot",
            semantic_role="underlier_spot",
            graph_role="underlier_spot",
        ),
        HybridADFactorRequirement(
            object_type="fx_rate",
            coordinate_type="spot",
            risk_class="fx",
            parameterization="scalar_fx_spot",
            semantic_role="fx_spot",
            graph_role="fx_spot",
        ),
        HybridADFactorRequirement(
            object_type="curve",
            coordinate_type="zero_rate",
            risk_class="rates",
            parameterization="zero_curve_nodes",
            semantic_role="domestic_curve",
            graph_role="domestic_curve",
        ),
        HybridADFactorRequirement(
            object_type="curve",
            coordinate_type="zero_rate",
            risk_class="rates",
            parameterization="zero_curve_nodes",
            semantic_role="foreign_curve",
            graph_role="foreign_curve",
        ),
        HybridADFactorRequirement(
            object_type="vol_surface",
            coordinate_type="black_vol",
            risk_class="volatility",
            parameterization="flat_or_grid_vol",
            semantic_role="underlier_vol",
            graph_role="underlier_vol",
        ),
        HybridADFactorRequirement(
            object_type="vol_surface",
            coordinate_type="black_vol",
            risk_class="volatility",
            parameterization="flat_or_grid_vol",
            semantic_role="fx_vol",
            graph_role="fx_vol",
        ),
        HybridADFactorRequirement(
            object_type="model_parameter",
            coordinate_type="correlation",
            risk_class="hybrid",
            parameterization="scalar_correlation",
            semantic_role="scalar_correlation",
            graph_role="correlation",
        ),
    )


def _quanto_matrix_requirements() -> tuple[HybridADFactorRequirement, ...]:
    return _quanto_scalar_requirements()[:-1] + (_correlation_matrix_requirement(),)


def _scalar_correlation_requirement() -> HybridADFactorRequirement:
    return HybridADFactorRequirement(
        object_type="model_parameter",
        coordinate_type="correlation",
        risk_class="hybrid",
        parameterization="scalar_correlation",
        semantic_role="cross_factor_dependence",
        graph_role="correlation",
    )


def _correlation_matrix_requirement() -> HybridADFactorRequirement:
    return HybridADFactorRequirement(
        object_type="correlation_matrix",
        coordinate_type="correlation",
        risk_class="hybrid",
        parameterization="correlation_matrix_psd_policy",
        semantic_role="cross_factor_dependence",
        graph_role="correlation_matrix",
    )


def _correlation_surface_requirement() -> HybridADFactorRequirement:
    return HybridADFactorRequirement(
        object_type="correlation_surface",
        coordinate_type="correlation",
        risk_class="hybrid",
        parameterization="correlation_surface_policy",
        semantic_role="cross_factor_dependence",
        graph_role="correlation_surface",
    )


def _clean(value: object, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _normalize(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _is_quanto_family(product_family: str | None) -> bool:
    return _normalize(product_family) in {"quanto_option", "single_name_quanto_option"}


__all__ = [
    "HybridADFactorRequirement",
    "HybridADLaneAdmission",
    "HybridADStatePolicy",
    "admit_hybrid_ad_lane",
]
