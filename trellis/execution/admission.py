"""Capability admission for route-free execution IR artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from trellis.execution.ir import ContractExecutionIR


@dataclass(frozen=True)
class ExecutionCapabilityBlocker:
    """Structured blocker emitted when an execution IR cannot use an engine."""

    blocker_id: str
    message: str
    method: str
    missing_primitive: str = ""
    severity: str = "blocked"

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocker_id", _text(self.blocker_id))
        object.__setattr__(self, "message", _text(self.message))
        object.__setattr__(self, "method", _normalize_method(self.method))
        object.__setattr__(self, "missing_primitive", _text(self.missing_primitive))
        object.__setattr__(self, "severity", _text(self.severity) or "blocked")


@dataclass(frozen=True)
class ExecutionCapabilityAdmission:
    """Capability decision for one execution IR and requested engine method."""

    method: str
    admitted: bool
    required_capabilities: tuple[str, ...]
    matched_capabilities: tuple[str, ...] = ()
    blockers: tuple[ExecutionCapabilityBlocker, ...] = ()
    engine_family: str = ""
    source_semantic_id: str = ""
    product_family: str = ""

    def __post_init__(self) -> None:
        blockers = tuple(self.blockers or ())
        object.__setattr__(self, "method", _normalize_method(self.method))
        object.__setattr__(self, "required_capabilities", _tuple_text(self.required_capabilities))
        object.__setattr__(self, "matched_capabilities", _tuple_text(self.matched_capabilities))
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "admitted", bool(self.admitted) and not blockers)
        object.__setattr__(self, "engine_family", _text(self.engine_family))
        object.__setattr__(self, "source_semantic_id", _text(self.source_semantic_id))
        object.__setattr__(self, "product_family", _text(self.product_family))


def admit_execution_capabilities(
    ir: ContractExecutionIR,
    *,
    method: str,
    available_primitives: tuple[str, ...] | list[str] = (),
) -> ExecutionCapabilityAdmission:
    """Return a route-free capability decision for one execution IR."""
    normalized_method = _normalize_method(method)
    if _is_bermudan_best_of_basket(ir):
        return _admit_bermudan_best_of_basket(
            ir,
            method=normalized_method,
            available_primitives=_tuple_text(available_primitives),
        )
    if _is_callable_bond(ir):
        return _admit_callable_bond(
            ir,
            method=normalized_method,
        )
    return ExecutionCapabilityAdmission(
        method=normalized_method,
        admitted=False,
        required_capabilities=(),
        blockers=(
            ExecutionCapabilityBlocker(
                blocker_id="unsupported_execution_ir_family",
                method=normalized_method,
                missing_primitive="execution_capability_profile",
                message=(
                    "No execution capability profile is registered for "
                    f"product family `{ir.source_track.product_family}`."
                ),
            ),
        ),
        source_semantic_id=ir.source_track.semantic_id,
        product_family=ir.source_track.product_family,
    )


def _admit_bermudan_best_of_basket(
    ir: ContractExecutionIR,
    *,
    method: str,
    available_primitives: tuple[str, ...],
) -> ExecutionCapabilityAdmission:
    if method == "monte_carlo":
        required = (
            "multi_asset_correlated_diffusion",
            "correlation_matrix",
            "path_simulation",
            "bermudan_holder_exercise",
            "best_of_basket_payoff",
        )
        blockers = _missing_shape_blockers(ir, method=method)
        return ExecutionCapabilityAdmission(
            method=method,
            admitted=not blockers,
            required_capabilities=required,
            matched_capabilities=() if blockers else required,
            blockers=blockers,
            engine_family="monte_carlo",
            source_semantic_id=ir.source_track.semantic_id,
            product_family=ir.source_track.product_family,
        )

    if method in {"lattice", "rate_tree"}:
        required = (
            "multi_asset_bermudan_state_grid",
            "bermudan_holder_exercise",
            "best_of_basket_payoff",
        )
        if (
            "multi_asset_bermudan_state_grid" in available_primitives
            or "multi_asset_product_state_lattice" in available_primitives
        ):
            return ExecutionCapabilityAdmission(
                method="lattice",
                admitted=True,
                required_capabilities=required,
                matched_capabilities=required + ("multi_asset_product_state_lattice",)
                if "multi_asset_product_state_lattice" in available_primitives
                else required,
                engine_family="lattice",
                source_semantic_id=ir.source_track.semantic_id,
                product_family=ir.source_track.product_family,
            )
        return ExecutionCapabilityAdmission(
            method="lattice",
            admitted=False,
            required_capabilities=required,
            matched_capabilities=("bermudan_holder_exercise", "best_of_basket_payoff"),
            blockers=(
                ExecutionCapabilityBlocker(
                    blocker_id="missing_multi_asset_bermudan_state_grid",
                    method="lattice",
                    missing_primitive="multi_asset_bermudan_state_grid",
                    message=(
                        "P001 lattice admission requires a multi-asset Bermudan "
                        "state grid; short-rate lattice is not compatible with "
                        "the named best-of basket execution IR."
                    ),
                ),
            ),
            engine_family="lattice",
            source_semantic_id=ir.source_track.semantic_id,
            product_family=ir.source_track.product_family,
        )

    return ExecutionCapabilityAdmission(
        method=method,
        admitted=False,
        required_capabilities=(),
        blockers=(
            ExecutionCapabilityBlocker(
                blocker_id="unsupported_execution_method",
                method=method,
                missing_primitive="execution_method_profile",
                message=(
                    "Bermudan best-of basket execution currently admits only "
                    "monte_carlo and lattice capability profiles."
                ),
            ),
        ),
        source_semantic_id=ir.source_track.semantic_id,
        product_family=ir.source_track.product_family,
    )


def _missing_shape_blockers(
    ir: ContractExecutionIR,
    *,
    method: str,
) -> tuple[ExecutionCapabilityBlocker, ...]:
    blockers: list[ExecutionCapabilityBlocker] = []
    if _observable_count(ir, "spot") < 2:
        blockers.append(
            ExecutionCapabilityBlocker(
                blocker_id="missing_multi_asset_spot_observables",
                method=method,
                missing_primitive="multi_asset_spot_observables",
                message="Monte Carlo admission requires at least two named spot observables.",
            )
        )
    if _observable_count(ir, "correlation_matrix") < 1:
        blockers.append(
            ExecutionCapabilityBlocker(
                blocker_id="missing_correlation_matrix",
                method=method,
                missing_primitive="correlation_matrix",
                message="Monte Carlo admission requires a basket correlation matrix.",
            )
        )
    if not any(
        action.action_type == "holder_max"
        for action in ir.decision_program.actions
    ):
        blockers.append(
            ExecutionCapabilityBlocker(
                blocker_id="missing_bermudan_holder_exercise",
                method=method,
                missing_primitive="bermudan_holder_exercise",
                message="Monte Carlo admission requires holder-max Bermudan decisions.",
            )
        )
    if not any(
        step.settlement_kind == "best_of_call_payoff"
        for step in ir.settlement_program.steps
    ):
        blockers.append(
            ExecutionCapabilityBlocker(
                blocker_id="missing_best_of_basket_payoff",
                method=method,
                missing_primitive="best_of_basket_payoff",
                message="Monte Carlo admission requires best-of basket payoff semantics.",
            )
        )
    return tuple(blockers)


def _is_bermudan_best_of_basket(ir: ContractExecutionIR) -> bool:
    return (
        ir.source_track.product_family == "bermudan_best_of_basket"
        and any(
            step.settlement_kind == "best_of_call_payoff"
            for step in ir.settlement_program.steps
        )
        and any(action.action_type == "holder_max" for action in ir.decision_program.actions)
    )


def _is_callable_bond(ir: ContractExecutionIR) -> bool:
    return (
        ir.source_track.product_family == "callable_bond"
        and any(
            step.settlement_kind == "embedded_fixed_income_contract"
            for step in ir.settlement_program.steps
        )
        and {"continue", "terminate"}.issubset(
            {
                action.action_type
                for action in ir.decision_program.actions
            }
        )
    )


def _admit_callable_bond(
    ir: ContractExecutionIR,
    *,
    method: str,
) -> ExecutionCapabilityAdmission:
    blockers = _callable_bond_shape_blockers(ir, method=method)
    if method in {"lattice"}:
        required = (
            "embedded_fixed_income_schedule",
            "issuer_call_discrete_control",
            "one_factor_short_rate_lattice",
        )
        return ExecutionCapabilityAdmission(
            method=method,
            admitted=not blockers,
            required_capabilities=required,
            matched_capabilities=() if blockers else required,
            blockers=blockers,
            engine_family="lattice",
            source_semantic_id=ir.source_track.semantic_id,
            product_family=ir.source_track.product_family,
        )
    if method in {"pde", "event_aware_pde", "pde_solver"}:
        required = (
            "embedded_fixed_income_schedule",
            "issuer_call_discrete_control",
            "one_factor_short_rate_event_pde",
        )
        return ExecutionCapabilityAdmission(
            method="pde",
            admitted=not blockers,
            required_capabilities=required,
            matched_capabilities=() if blockers else required,
            blockers=blockers,
            engine_family="pde",
            source_semantic_id=ir.source_track.semantic_id,
            product_family=ir.source_track.product_family,
        )
    return ExecutionCapabilityAdmission(
        method=method,
        admitted=False,
        required_capabilities=(),
        blockers=(
            ExecutionCapabilityBlocker(
                blocker_id="unsupported_execution_method",
                method=method,
                missing_primitive="execution_method_profile",
                message=(
                    "Callable-bond dynamic execution currently admits only "
                    "lattice and pde capability profiles."
                ),
            ),
        ),
        source_semantic_id=ir.source_track.semantic_id,
        product_family=ir.source_track.product_family,
    )


def _callable_bond_shape_blockers(
    ir: ContractExecutionIR,
    *,
    method: str,
) -> tuple[ExecutionCapabilityBlocker, ...]:
    blockers: list[ExecutionCapabilityBlocker] = []
    if _observable_count(ir, "curve_quote") < 1:
        blockers.append(
            ExecutionCapabilityBlocker(
                blocker_id="missing_discount_curve_observable",
                method=method,
                missing_primitive="discount_curve_observable",
                message="Callable-bond execution requires a discount-curve observable.",
            )
        )
    if _observable_count(ir, "surface_quote") < 1:
        blockers.append(
            ExecutionCapabilityBlocker(
                blocker_id="missing_rates_vol_observable",
                method=method,
                missing_primitive="surface_quote_observable",
                message="Callable-bond execution requires a bounded rate-volatility observable.",
            )
        )
    if not {"continue", "terminate"}.issubset(
        {action.action_type for action in ir.decision_program.actions}
    ):
        blockers.append(
            ExecutionCapabilityBlocker(
                blocker_id="missing_issuer_call_control",
                method=method,
                missing_primitive="issuer_call_discrete_control",
                message="Callable-bond execution requires issuer continue/terminate control semantics.",
            )
        )
    if not any(event.event_kind == "payment" for event in ir.event_plan.events):
        blockers.append(
            ExecutionCapabilityBlocker(
                blocker_id="missing_payment_timeline",
                method=method,
                missing_primitive="embedded_fixed_income_schedule",
                message="Callable-bond execution requires payment events from the base fixed-income schedule.",
            )
        )
    return tuple(blockers)


def _observable_count(ir: ContractExecutionIR, observable_kind: str) -> int:
    return sum(
        1
        for observable in ir.observables
        if observable.observable_kind == observable_kind
    )


def _normalize_method(method: object) -> str:
    text = _text(method).lower().replace("-", "_")
    if text in {"mc", "montecarlo"}:
        return "monte_carlo"
    if text == "rate_tree":
        return "lattice"
    if text in {"event_aware_pde", "pde_solver"}:
        return "pde"
    return text


def _tuple_text(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    result: list[str] = []
    for value in values or ():
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _text(value: object) -> str:
    return str(value or "").strip()


__all__ = [
    "ExecutionCapabilityAdmission",
    "ExecutionCapabilityBlocker",
    "admit_execution_capabilities",
]
