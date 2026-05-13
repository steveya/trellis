"""Semantic admission policy for bounded portfolio-AAD lanes."""

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
from trellis.agent.dynamic_contract_ir import DecisionEvent, DynamicContractIR


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _copy_diagnostics(
    diagnostics: Iterable[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(diagnostic) for diagnostic in (diagnostics or ()))


@dataclass(frozen=True)
class PortfolioAADFactorRequirement:
    """Semantic market-coordinate requirement for one portfolio-AAD lane."""

    object_type: str
    coordinate_type: str
    risk_class: str
    parameterization: str
    semantic_role: str = ""
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
        object.__setattr__(self, "required", bool(self.required))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "object_type": self.object_type,
            "coordinate_type": self.coordinate_type,
            "risk_class": self.risk_class,
            "parameterization": self.parameterization,
            "semantic_role": self.semantic_role,
            "required": self.required,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> PortfolioAADFactorRequirement:
        """Build a requirement from :meth:`to_payload` output."""
        return cls(
            object_type=str(payload["object_type"]),
            coordinate_type=str(payload["coordinate_type"]),
            risk_class=str(payload["risk_class"]),
            parameterization=str(payload["parameterization"]),
            semantic_role=str(payload.get("semantic_role", "")),
            required=bool(payload.get("required", True)),
        )


@dataclass(frozen=True)
class PortfolioAADLaneAdmission:
    """Semantic support decision for a ContractIR-backed portfolio-AAD lane."""

    admitted: bool
    lane_id: str
    support_status: str
    reason: str
    semantic_contract_type: str
    product_family: str
    contract_shape: str
    factor_requirements: tuple[PortfolioAADFactorRequirement, ...] = field(default_factory=tuple)
    derivative_method_category: str = "portfolio_aad"
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
        object.__setattr__(
            self,
            "factor_requirements",
            tuple(self.factor_requirements),
        )
        for requirement in self.factor_requirements:
            if not isinstance(requirement, PortfolioAADFactorRequirement):
                raise TypeError(
                    "factor_requirements must contain PortfolioAADFactorRequirement values"
                )
        object.__setattr__(
            self,
            "derivative_method_category",
            _clean(self.derivative_method_category, "derivative_method_category"),
        )
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))
        object.__setattr__(self, "diagnostics", _copy_diagnostics(self.diagnostics))

    @property
    def supported(self) -> bool:
        """Return whether this admission can enter an AAD lane today."""
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
    ) -> PortfolioAADLaneAdmission:
        """Build an admission decision from :meth:`to_payload` output."""
        return cls(
            admitted=bool(payload["admitted"]),
            lane_id=str(payload["lane_id"]),
            support_status=str(payload["support_status"]),
            reason=str(payload["reason"]),
            semantic_contract_type=str(payload["semantic_contract_type"]),
            product_family=str(payload["product_family"]),
            contract_shape=str(payload["contract_shape"]),
            factor_requirements=tuple(
                PortfolioAADFactorRequirement.from_payload(requirement)
                for requirement in payload.get("factor_requirements", ())
            ),
            derivative_method_category=str(
                payload.get("derivative_method_category", "portfolio_aad")
            ),
            metadata=payload.get("metadata") or {},
            diagnostics=payload.get("diagnostics") or (),
        )


def admit_portfolio_aad_lane(
    semantic_contract: object,
    *,
    market_parameterization: str = "flat_vol",
    product_family: str | None = None,
) -> PortfolioAADLaneAdmission:
    """Return the bounded portfolio-AAD admission decision for a semantic contract.

    Admission is deliberately narrower than pricing support. A ``planned``
    decision records a known semantic lane that still fails closed until its
    adapter and verification evidence exist.
    """
    parameterization = _normalize(market_parameterization) or "flat_vol"
    if isinstance(semantic_contract, DynamicContractIR):
        return _admit_dynamic_contract(
            semantic_contract,
            market_parameterization=parameterization,
            product_family=product_family,
        )
    if isinstance(semantic_contract, ContractIR):
        return _admit_contract_ir(
            semantic_contract,
            market_parameterization=parameterization,
            product_family=product_family,
        )
    return _admission(
        admitted=False,
        lane_id="unsupported_semantic_contract",
        support_status="unsupported",
        reason="unsupported_semantic_contract_type",
        semantic_contract_type=type(semantic_contract).__name__,
        product_family=product_family or "unknown",
        contract_shape="unknown",
        factor_requirements=(),
    )


def _admit_dynamic_contract(
    contract: DynamicContractIR,
    *,
    market_parameterization: str,
    product_family: str | None,
) -> PortfolioAADLaneAdmission:
    decision_events = _dynamic_decision_events(contract)
    if contract.control_program is not None or decision_events:
        decision_style = (
            ""
            if contract.control_program is None
            else contract.control_program.decision_style
        )
        return _admission(
            admitted=False,
            lane_id="early_exercise_control_policy",
            support_status="planned",
            reason="early_exercise_control_aad_pending",
            semantic_contract_type="DynamicContractIR",
            product_family=product_family or contract.semantic_family or "early_exercise_option",
            contract_shape="dynamic_control_exercise",
            factor_requirements=(_vol_requirement(market_parameterization),),
            metadata={
                "base_track": contract.base_track,
                "semantic_family": contract.semantic_family,
                "decision_style": decision_style,
                "decision_event_count": len(decision_events),
                "fail_closed": True,
            },
        )
    if isinstance(contract.base_contract, ContractIR):
        base_admission = _admit_contract_ir(
            contract.base_contract,
            market_parameterization=market_parameterization,
            product_family=product_family or contract.semantic_family or None,
        )
        return _admission(
            admitted=base_admission.admitted,
            lane_id=base_admission.lane_id,
            support_status=base_admission.support_status,
            reason=base_admission.reason,
            semantic_contract_type="DynamicContractIR",
            product_family=base_admission.product_family,
            contract_shape=base_admission.contract_shape,
            factor_requirements=base_admission.factor_requirements,
            metadata={
                **dict(base_admission.metadata),
                "base_track": contract.base_track,
                "semantic_family": contract.semantic_family,
            },
            diagnostics=base_admission.diagnostics,
        )
    return _admission(
        admitted=False,
        lane_id="unsupported_dynamic_contract",
        support_status="unsupported",
        reason="unsupported_dynamic_contract_shape",
        semantic_contract_type="DynamicContractIR",
        product_family=product_family or contract.semantic_family or "dynamic_contract",
        contract_shape="dynamic_contract",
        factor_requirements=(),
    )


def _admit_contract_ir(
    contract: ContractIR,
    *,
    market_parameterization: str,
    product_family: str | None,
) -> PortfolioAADLaneAdmission:
    if _contains_indicator(contract.payoff):
        return _admission(
            admitted=False,
            lane_id="path_dependent_discontinuity_policy",
            support_status="unsupported",
            reason="unsupported_discontinuous_event_monitor",
            semantic_contract_type="ContractIR",
            product_family=product_family or "path_dependent_option",
            contract_shape="discontinuous_event_monitor",
            factor_requirements=(_vol_requirement(market_parameterization),),
            metadata={"fail_closed": True},
        )
    if isinstance(contract.underlying.spec, CompositeUnderlying):
        return _admission(
            admitted=False,
            lane_id="hybrid_factor_correlation",
            support_status="planned",
            reason="hybrid_factor_aad_pending",
            semantic_contract_type="ContractIR",
            product_family=product_family or "hybrid_option",
            contract_shape="hybrid_composite_underlying",
            factor_requirements=(_correlation_requirement(),),
            metadata={"fail_closed": True},
        )
    if contract.exercise.style in {"american", "bermudan"}:
        return _admission(
            admitted=False,
            lane_id="early_exercise_control_policy",
            support_status="planned",
            reason="early_exercise_control_aad_pending",
            semantic_contract_type="ContractIR",
            product_family=product_family or "early_exercise_option",
            contract_shape="early_exercise_option",
            factor_requirements=(_vol_requirement(market_parameterization),),
            metadata={"exercise_style": contract.exercise.style, "fail_closed": True},
        )
    if contract.observation.kind == "path_dependent" or _contains_arithmetic_mean(contract.payoff):
        return _admission(
            admitted=False,
            lane_id="path_dependent_option_policy",
            support_status="planned",
            reason="path_dependent_aad_pending",
            semantic_contract_type="ContractIR",
            product_family=product_family or "path_dependent_option",
            contract_shape="path_dependent_smooth_summary",
            factor_requirements=(_vol_requirement(market_parameterization),),
            metadata={"observation_kind": contract.observation.kind, "fail_closed": True},
        )
    if _is_terminal_vanilla_option(contract):
        if market_parameterization in {"flat_vol", "scalar_flat_vol"}:
            return _admission(
                admitted=True,
                lane_id="vanilla_equity_option_flat_vol",
                support_status="supported",
                reason="supported_terminal_vanilla_flat_vol_aad",
                semantic_contract_type="ContractIR",
                product_family=product_family or "vanilla_equity_option",
                contract_shape="terminal_vanilla_option",
                factor_requirements=(_flat_vol_requirement(),),
            )
        if market_parameterization in {"grid_vol", "grid_node_vols"}:
            return _admission(
                admitted=False,
                lane_id="vanilla_equity_option_grid_vol",
                support_status="planned",
                reason="grid_vol_option_aad_pending",
                semantic_contract_type="ContractIR",
                product_family=product_family or "vanilla_equity_option",
                contract_shape="terminal_vanilla_option",
                factor_requirements=(_grid_vol_requirement(),),
                metadata={"fail_closed": True},
            )
        return _admission(
            admitted=False,
            lane_id="vanilla_equity_option_unsupported_vol",
            support_status="unsupported",
            reason="unsupported_vol_parameterization",
            semantic_contract_type="ContractIR",
            product_family=product_family or "vanilla_equity_option",
            contract_shape="terminal_vanilla_option",
            factor_requirements=(_vol_requirement(market_parameterization),),
            metadata={"market_parameterization": market_parameterization, "fail_closed": True},
        )
    return _admission(
        admitted=False,
        lane_id="unsupported_contract_ir",
        support_status="unsupported",
        reason="unsupported_contract_ir_shape",
        semantic_contract_type="ContractIR",
        product_family=product_family or "unknown",
        contract_shape="unsupported_contract_ir",
        factor_requirements=(),
        metadata={"fail_closed": True},
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
    factor_requirements: tuple[PortfolioAADFactorRequirement, ...],
    metadata: Mapping[str, Any] | None = None,
    diagnostics: Iterable[Mapping[str, Any]] | None = None,
) -> PortfolioAADLaneAdmission:
    return PortfolioAADLaneAdmission(
        admitted=admitted,
        lane_id=lane_id,
        support_status=support_status,
        reason=reason,
        semantic_contract_type=semantic_contract_type,
        product_family=product_family,
        contract_shape=contract_shape,
        factor_requirements=factor_requirements,
        metadata=metadata or {},
        diagnostics=tuple(diagnostics or ()),
    )


def _dynamic_decision_events(contract: DynamicContractIR) -> tuple[DecisionEvent, ...]:
    events: list[DecisionEvent] = []
    for bucket in contract.event_program.buckets:
        for event in bucket.events:
            if isinstance(event, DecisionEvent):
                events.append(event)
    return tuple(events)


def _is_terminal_vanilla_option(contract: ContractIR) -> bool:
    if contract.exercise.style != "european" or contract.observation.kind != "terminal":
        return False
    body = _vanilla_intrinsic_body(contract.payoff)
    if body is None:
        return False
    if not isinstance(contract.underlying.spec, tuple(_EQUITY_UNDERLYING_TYPES)):
        return False
    return _is_spot_strike_sub(body) or _is_strike_spot_sub(body)


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


def _vol_requirement(parameterization: str) -> PortfolioAADFactorRequirement:
    if parameterization in {"grid_vol", "grid_node_vols"}:
        return _grid_vol_requirement()
    if parameterization in {"correlation_scalar", "hybrid_correlation"}:
        return _correlation_requirement()
    return _flat_vol_requirement()


def _flat_vol_requirement() -> PortfolioAADFactorRequirement:
    return PortfolioAADFactorRequirement(
        object_type="vol_surface",
        coordinate_type="flat_vol",
        risk_class="volatility",
        parameterization="scalar_flat_vol",
        semantic_role="option_volatility",
    )


def _grid_vol_requirement() -> PortfolioAADFactorRequirement:
    return PortfolioAADFactorRequirement(
        object_type="vol_surface",
        coordinate_type="black_vol",
        risk_class="volatility",
        parameterization="grid_node_vols",
        semantic_role="option_volatility",
    )


def _correlation_requirement() -> PortfolioAADFactorRequirement:
    return PortfolioAADFactorRequirement(
        object_type="model_parameter",
        coordinate_type="correlation",
        risk_class="hybrid",
        parameterization="scalar_correlation",
        semantic_role="cross_factor_dependence",
    )


def _clean(value: object, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _normalize(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


_EQUITY_UNDERLYING_TYPES = ()


def _init_equity_underlying_types() -> None:
    global _EQUITY_UNDERLYING_TYPES
    if not _EQUITY_UNDERLYING_TYPES:
        from trellis.agent.contract_ir import EquitySpot

        _EQUITY_UNDERLYING_TYPES = (EquitySpot,)


_init_equity_underlying_types()


__all__ = [
    "PortfolioAADFactorRequirement",
    "PortfolioAADLaneAdmission",
    "admit_portfolio_aad_lane",
]
