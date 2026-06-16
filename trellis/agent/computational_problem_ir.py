"""Typed computational problem IR for stochastic-volatility task diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


STOCHASTIC_VOL_TRANSFORM = "stochastic_vol_transform"
STOCHASTIC_VOL_MONTE_CARLO = "stochastic_vol_monte_carlo"
STOCHASTIC_VOL_PDE = "stochastic_vol_pde"
CALIBRATION_TO_SURFACE = "calibration_to_surface"
AFFINE_JUMP_STOCHASTIC_VOL = "affine_jump_stochastic_vol"
SLV_LSV = "slv_lsv"
UNSUPPORTED_PATH_DEPENDENT_CONTROL = "unsupported_path_dependent_control"
STOCHASTIC_VOL_MIXED = "stochastic_vol_mixed"


@dataclass(frozen=True)
class RepairPacket:
    """Machine-readable repair hint for a missing computational primitive."""

    packet_type: str
    summary: str
    missing_primitive: str | None = None
    unsupported_class: str | None = None
    suggested_action: str = "open_remediation_packet"
    evidence: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "packet_type": self.packet_type,
            "summary": self.summary,
            "missing_primitive": self.missing_primitive,
            "unsupported_class": self.unsupported_class,
            "suggested_action": self.suggested_action,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class MarketBindingSemantics:
    """Market-input requirements implied by the computational problem class."""

    requires_model_parameters: bool
    requires_black_vol_surface: bool
    requires_local_vol_surface: bool = False
    requires_jump_parameters: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "requires_model_parameters": self.requires_model_parameters,
            "requires_black_vol_surface": self.requires_black_vol_surface,
            "requires_local_vol_surface": self.requires_local_vol_surface,
            "requires_jump_parameters": self.requires_jump_parameters,
        }


@dataclass(frozen=True)
class ModelParameterSemantics:
    """How stochastic-vol model parameters relate to market vol surfaces."""

    model_family: str
    model_parameter_source: str
    black_vol_surface_role: str
    requires_calibration_bridge: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_parameter_source": self.model_parameter_source,
            "black_vol_surface_role": self.black_vol_surface_role,
            "requires_calibration_bridge": self.requires_calibration_bridge,
        }


@dataclass(frozen=True)
class CalibrationProblemSemantics:
    """How a calibration target bridges market quotes into model parameters."""

    status: str
    family_id: str
    workflow_id: str
    calibration_direction: str
    input_quote_family: str
    input_quote_convention: str
    output_parameter_source: str
    requires_recorded_calibration_step: bool
    evidence: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "family_id": self.family_id,
            "workflow_id": self.workflow_id,
            "calibration_direction": self.calibration_direction,
            "input_quote_family": self.input_quote_family,
            "input_quote_convention": self.input_quote_convention,
            "output_parameter_source": self.output_parameter_source,
            "requires_recorded_calibration_step": self.requires_recorded_calibration_step,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class AffineJumpStochasticVolSemantics:
    """Required Bates-style affine jump stochastic-volatility capabilities."""

    process_family: str
    base_process_family: str
    jump_family: str
    required_model_parameters: tuple[str, ...]
    required_jump_parameters: tuple[str, ...]
    jump_parameter_aliases: tuple[tuple[str, tuple[str, ...]], ...]
    transform_capability: str
    monte_carlo_capability: str
    validation_requirements: tuple[str, ...]
    supported_now: bool
    missing_primitives: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "process_family": self.process_family,
            "base_process_family": self.base_process_family,
            "jump_family": self.jump_family,
            "required_model_parameters": list(self.required_model_parameters),
            "required_jump_parameters": list(self.required_jump_parameters),
            "jump_parameter_aliases": {
                name: list(aliases)
                for name, aliases in self.jump_parameter_aliases
            },
            "transform_capability": self.transform_capability,
            "monte_carlo_capability": self.monte_carlo_capability,
            "validation_requirements": list(self.validation_requirements),
            "supported_now": self.supported_now,
            "missing_primitives": list(self.missing_primitives),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class LeverageFunctionSemantics:
    """Required SLV/LSV leverage-function calibration and solver contracts."""

    process_family: str
    leverage_function_kind: str
    required_market_inputs: tuple[str, ...]
    required_model_inputs: tuple[str, ...]
    calibration_requirements: tuple[str, ...]
    interpolation_domain: tuple[str, ...]
    diagnostics: tuple[str, ...]
    solver_requirements: tuple[str, ...]
    validation_requirements: tuple[str, ...]
    supported_now: bool
    missing_components: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "process_family": self.process_family,
            "leverage_function_kind": self.leverage_function_kind,
            "required_market_inputs": list(self.required_market_inputs),
            "required_model_inputs": list(self.required_model_inputs),
            "calibration_requirements": list(self.calibration_requirements),
            "interpolation_domain": list(self.interpolation_domain),
            "diagnostics": list(self.diagnostics),
            "solver_requirements": list(self.solver_requirements),
            "validation_requirements": list(self.validation_requirements),
            "supported_now": self.supported_now,
            "missing_components": list(self.missing_components),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class PathDependentControlSemantics:
    """Required path-state and early-exercise control contracts under Heston."""

    process_family: str
    composite_class: str
    state_requirements: tuple[str, ...]
    path_state_requirements: tuple[str, ...]
    event_monitor_requirements: tuple[str, ...]
    payoff_summary_requirements: tuple[str, ...]
    control_requirements: tuple[str, ...]
    stochastic_vol_coupling_requirements: tuple[str, ...]
    solver_requirements: tuple[str, ...]
    validation_requirements: tuple[str, ...]
    supported_now: bool
    expected_honest_block: bool
    model_validator_policy: str
    missing_components: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "process_family": self.process_family,
            "composite_class": self.composite_class,
            "state_requirements": list(self.state_requirements),
            "path_state_requirements": list(self.path_state_requirements),
            "event_monitor_requirements": list(self.event_monitor_requirements),
            "payoff_summary_requirements": list(self.payoff_summary_requirements),
            "control_requirements": list(self.control_requirements),
            "stochastic_vol_coupling_requirements": list(
                self.stochastic_vol_coupling_requirements
            ),
            "solver_requirements": list(self.solver_requirements),
            "validation_requirements": list(self.validation_requirements),
            "supported_now": self.supported_now,
            "expected_honest_block": self.expected_honest_block,
            "model_validator_policy": self.model_validator_policy,
            "missing_components": list(self.missing_components),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class StochasticVolTargetProblem:
    """Computational class for one concrete comparison/build target."""

    target_id: str
    bucket: str
    solver_target: str
    process_family: str
    payoff_class: str
    model_parameter_semantics: ModelParameterSemantics
    market_bindings: MarketBindingSemantics
    validation_bundle: str
    unsupported_features: tuple[str, ...] = ()
    repair_packet: RepairPacket | None = None
    calibration_problem: CalibrationProblemSemantics | None = None
    affine_jump_process: AffineJumpStochasticVolSemantics | None = None
    leverage_function_contract: LeverageFunctionSemantics | None = None
    path_dependent_control_contract: PathDependentControlSemantics | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "bucket": self.bucket,
            "solver_target": self.solver_target,
            "process_family": self.process_family,
            "payoff_class": self.payoff_class,
            "model_parameter_semantics": self.model_parameter_semantics.to_payload(),
            "market_bindings": self.market_bindings.to_payload(),
            "validation_bundle": self.validation_bundle,
            "unsupported_features": list(self.unsupported_features),
            "repair_packet": (
                self.repair_packet.to_payload()
                if self.repair_packet is not None
                else None
            ),
            "calibration_problem": (
                self.calibration_problem.to_payload()
                if self.calibration_problem is not None
                else None
            ),
            "affine_jump_process": (
                self.affine_jump_process.to_payload()
                if self.affine_jump_process is not None
                else None
            ),
            "leverage_function_contract": (
                self.leverage_function_contract.to_payload()
                if self.leverage_function_contract is not None
                else None
            ),
            "path_dependent_control_contract": (
                self.path_dependent_control_contract.to_payload()
                if self.path_dependent_control_contract is not None
                else None
            ),
        }


@dataclass(frozen=True)
class StochasticVolTaskProblemReport:
    """Computational problem report for a stochastic-vol task."""

    task_id: str
    task_title: str
    task_bucket: str
    target_problems: tuple[StochasticVolTargetProblem, ...]

    @property
    def repair_packets(self) -> tuple[dict[str, Any], ...]:
        packets: list[dict[str, Any]] = []
        for target in self.target_problems:
            if target.repair_packet is None:
                continue
            packet = target.repair_packet.to_payload()
            packet["target_id"] = target.target_id
            packet["bucket"] = target.bucket
            packets.append(packet)
        return tuple(packets)

    @property
    def is_supported_now(self) -> bool:
        return not self.repair_packets

    def target_payload(self, target_id: str) -> dict[str, Any] | None:
        for target in self.target_problems:
            if target.target_id == target_id:
                return target.to_payload()
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "task_title": self.task_title,
            "task_bucket": self.task_bucket,
            "is_supported_now": self.is_supported_now,
            "targets": [target.to_payload() for target in self.target_problems],
            "repair_packets": list(self.repair_packets),
        }


def classify_stochastic_vol_task(
    task: Mapping[str, Any],
) -> StochasticVolTaskProblemReport | None:
    """Classify a task into stochastic-vol computational buckets when applicable."""
    task_id = str(task.get("id") or "").strip()
    task_title = str(task.get("title") or task_id).strip()
    targets = _task_target_ids(task)
    if not targets:
        targets = ("task",)
    if not _looks_stochastic_vol_task(task, targets):
        return None

    target_problems = tuple(
        _classify_target(task, target_id)
        for target_id in targets
    )
    return StochasticVolTaskProblemReport(
        task_id=task_id,
        task_title=task_title,
        task_bucket=_task_bucket(target_problems),
        target_problems=target_problems,
    )


def stochastic_vol_problem_payload(task: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a JSON-stable stochastic-vol computational problem payload."""
    report = classify_stochastic_vol_task(task)
    if report is None:
        return None
    return report.to_payload()


def _classify_target(
    task: Mapping[str, Any],
    target_id: str,
) -> StochasticVolTargetProblem:
    target_text = _target_route_text(task, target_id)
    context_text = _text_blob(task, target_id)
    bucket = _target_bucket(target_text)
    solver_target = _solver_target(target_text, bucket)
    process_family = _process_family(context_text, bucket)
    semantics = _model_parameter_semantics(bucket, process_family)
    bindings = _market_bindings(bucket, process_family)
    repair_packet = _repair_packet(bucket, target_id=target_id, text=target_text)
    unsupported = _unsupported_features(bucket, repair_packet)
    calibration_problem = _calibration_problem_semantics(
        bucket,
        target_id=target_id,
        text=target_text,
        process_family=process_family,
    )
    affine_jump_process = _affine_jump_process_semantics(
        bucket,
        target_id=target_id,
        text=target_text,
        process_family=process_family,
    )
    leverage_function_contract = _leverage_function_semantics(
        bucket,
        target_id=target_id,
        text=target_text,
        process_family=process_family,
        solver_target=solver_target,
    )
    path_dependent_control_contract = _path_dependent_control_semantics(
        bucket,
        target_id=target_id,
        text=target_text,
        process_family=process_family,
        solver_target=solver_target,
    )
    return StochasticVolTargetProblem(
        target_id=target_id,
        bucket=bucket,
        solver_target=solver_target,
        process_family=process_family,
        payoff_class=_payoff_class(context_text),
        model_parameter_semantics=semantics,
        market_bindings=bindings,
        validation_bundle=_validation_bundle(bucket, process_family),
        unsupported_features=unsupported,
        repair_packet=repair_packet,
        calibration_problem=calibration_problem,
        affine_jump_process=affine_jump_process,
        leverage_function_contract=leverage_function_contract,
        path_dependent_control_contract=path_dependent_control_contract,
    )


def _target_bucket(text: str) -> str:
    if _has_any(text, ("american_pathdep", "american path", "asian barrier")):
        return UNSUPPORTED_PATH_DEPENDENT_CONTROL
    if _has_any(text, ("slv", "lsv", "stochastic local vol", "local-stochastic vol")):
        return SLV_LSV
    if "bates" in text:
        return AFFINE_JUMP_STOCHASTIC_VOL
    if _has_any(text, ("calibrat", "market_prices", "market prices")):
        return CALIBRATION_TO_SURFACE
    if _has_any(text, ("pde", "adi")):
        return STOCHASTIC_VOL_PDE
    if _has_any(text, ("mc", "monte", "euler", "qe")):
        return STOCHASTIC_VOL_MONTE_CARLO
    return STOCHASTIC_VOL_TRANSFORM


def _solver_target(text: str, bucket: str) -> str:
    if bucket == UNSUPPORTED_PATH_DEPENDENT_CONTROL:
        if "pde" in text:
            return "path_dependent_control_pde"
        if _has_any(text, ("mc", "monte")):
            return "path_dependent_control_monte_carlo"
        return "path_dependent_control_transform"
    if bucket == SLV_LSV:
        if "pde" in text:
            return "leverage_function_pde"
        if _has_any(text, ("mc", "monte")):
            return "leverage_function_monte_carlo"
        return "leverage_function_calibration"
    if bucket == AFFINE_JUMP_STOCHASTIC_VOL:
        if _has_any(text, ("mc", "monte")):
            return "affine_jump_monte_carlo"
        return "affine_jump_transform"
    if bucket == CALIBRATION_TO_SURFACE:
        return "surface_calibration"
    if bucket == STOCHASTIC_VOL_PDE:
        if "adi" in text:
            return "pde_adi"
        return "pde"
    if bucket == STOCHASTIC_VOL_MONTE_CARLO:
        if "qe" in text:
            return "monte_carlo_qe"
        if "euler" in text:
            return "monte_carlo_euler"
        return "monte_carlo"
    if "cos" in text:
        return "cos_transform"
    if "laguerre" in text:
        return "gauss_laguerre_quadrature"
    if "fft" in text:
        return "fft_transform"
    return "semi_analytical_transform"


def _process_family(text: str, bucket: str) -> str:
    if bucket == SLV_LSV:
        return "slv_lsv"
    if bucket == AFFINE_JUMP_STOCHASTIC_VOL:
        return "bates"
    if "rough" in text:
        return "rough_heston"
    return "heston"


def _payoff_class(text: str) -> str:
    if _has_any(text, ("american_pathdep", "american path", "asian barrier")):
        return "path_dependent_early_exercise"
    return "vanilla_option"


def _model_parameter_semantics(
    bucket: str,
    process_family: str,
) -> ModelParameterSemantics:
    if bucket == CALIBRATION_TO_SURFACE:
        return ModelParameterSemantics(
            model_family=process_family,
            model_parameter_source="calibration_to_market_surface",
            black_vol_surface_role="calibration_target",
            requires_calibration_bridge=True,
        )
    return ModelParameterSemantics(
        model_family=process_family,
        model_parameter_source="explicit_model_parameters",
        black_vol_surface_role="market_input_not_model_calibration",
        requires_calibration_bridge=False,
    )


def _market_bindings(
    bucket: str,
    process_family: str,
) -> MarketBindingSemantics:
    return MarketBindingSemantics(
        requires_model_parameters=bucket != CALIBRATION_TO_SURFACE,
        requires_black_vol_surface=(
            bucket in {CALIBRATION_TO_SURFACE, SLV_LSV}
            or process_family == "slv_lsv"
        ),
        requires_local_vol_surface=process_family == "slv_lsv",
        requires_jump_parameters=process_family == "bates",
    )


def _validation_bundle(bucket: str, process_family: str) -> str:
    if bucket == STOCHASTIC_VOL_TRANSFORM:
        return f"{process_family}:transform"
    if bucket == STOCHASTIC_VOL_MONTE_CARLO:
        return f"{process_family}:monte_carlo"
    if bucket == STOCHASTIC_VOL_PDE:
        return f"{process_family}:pde"
    if bucket == CALIBRATION_TO_SURFACE:
        return f"{process_family}:calibration_to_surface"
    if bucket == AFFINE_JUMP_STOCHASTIC_VOL:
        return "bates:affine_jump_stochastic_vol"
    if bucket == SLV_LSV:
        return "slv_lsv:leverage_function"
    return "heston:path_dependent_control"


def _repair_packet(bucket: str, *, target_id: str, text: str) -> RepairPacket | None:
    evidence = tuple(item for item in (target_id, _short_text_evidence(text)) if item)
    if bucket == AFFINE_JUMP_STOCHASTIC_VOL:
        return RepairPacket(
            packet_type="missing_affine_jump_stochastic_vol_kernel",
            missing_primitive="bates_affine_jump_stochastic_vol_kernel",
            summary=(
                "Bates targets need an affine Heston-plus-jump characteristic "
                "function and matching MC process contract."
            ),
            evidence=evidence,
        )
    if bucket == SLV_LSV:
        return RepairPacket(
            packet_type="missing_slv_lsv_leverage_contract",
            missing_primitive="leverage_function_contract",
            summary=(
                "SLV/LSV targets need an explicit leverage-function contract "
                "before route binding can select PDE or MC engines."
            ),
            evidence=evidence,
        )
    if bucket == UNSUPPORTED_PATH_DEPENDENT_CONTROL:
        return RepairPacket(
            packet_type="unsupported_path_dependent_control",
            missing_primitive="path_dependent_heston_control_contract",
            unsupported_class="path_dependent_early_exercise_under_stochastic_vol",
            summary=(
                "Path-dependent early-exercise under stochastic volatility is a "
                "composite control problem and should block honestly until a "
                "path-state, event-monitor, payoff-summary, control-policy, "
                "and stochastic-vol coupling contract exists."
            ),
            evidence=evidence,
        )
    if bucket == STOCHASTIC_VOL_TRANSFORM and "laguerre" in text:
        return RepairPacket(
            packet_type="missing_heston_gauss_laguerre_transform_kernel",
            missing_primitive="heston_gauss_laguerre_transform_kernel",
            unsupported_class="heston_gauss_laguerre_transform",
            summary=(
                "Heston Gauss-Laguerre transform targets need a checked "
                "quadrature kernel before route binding can admit the target."
            ),
            evidence=evidence,
        )
    return None


def _calibration_problem_semantics(
    bucket: str,
    *,
    target_id: str,
    text: str,
    process_family: str,
) -> CalibrationProblemSemantics | None:
    if bucket != CALIBRATION_TO_SURFACE:
        return None
    status = "calibration_supported"
    input_quote_family = "implied_vol"
    input_quote_convention = "black"
    if process_family != "heston":
        status = "calibration_blocked"
    elif _has_any(text, ("market prices", "market price")) or target_id == "market_prices":
        status = "calibration_needed"
        input_quote_family = "option_price"
        input_quote_convention = "model_price"
    return CalibrationProblemSemantics(
        status=status,
        family_id=process_family,
        workflow_id="heston_smile" if process_family == "heston" else f"{process_family}_calibration",
        calibration_direction="surface_to_model_parameters",
        input_quote_family=input_quote_family,
        input_quote_convention=input_quote_convention,
        output_parameter_source="calibrated_model_parameter_set",
        requires_recorded_calibration_step=True,
        evidence=tuple(item for item in (target_id, _short_text_evidence(text)) if item),
    )


def _affine_jump_process_semantics(
    bucket: str,
    *,
    target_id: str,
    text: str,
    process_family: str,
) -> AffineJumpStochasticVolSemantics | None:
    if bucket != AFFINE_JUMP_STOCHASTIC_VOL:
        return None
    missing_primitive = "bates_affine_jump_stochastic_vol_kernel"
    return AffineJumpStochasticVolSemantics(
        process_family=process_family,
        base_process_family="heston",
        jump_family="compound_poisson_lognormal",
        required_model_parameters=("kappa", "theta", "xi", "rho", "v0"),
        required_jump_parameters=(
            "jump_intensity",
            "jump_mean",
            "jump_variance",
        ),
        jump_parameter_aliases=(
            ("jump_intensity", ("lam", "lambda")),
            ("jump_variance", ("jump_var", "jump_vol^2", "jump_vol")),
        ),
        transform_capability="bates_characteristic_function",
        monte_carlo_capability="bates_jump_stochastic_vol_process",
        validation_requirements=(
            "consume_heston_model_parameters",
            "consume_jump_parameters",
            "reject_black_vol_surface_as_model_parameters",
            "cross_validate_transform_and_monte_carlo_when_admitted",
        ),
        supported_now=False,
        missing_primitives=(missing_primitive,),
        evidence=tuple(item for item in (target_id, _short_text_evidence(text)) if item),
    )


def _leverage_function_semantics(
    bucket: str,
    *,
    target_id: str,
    text: str,
    process_family: str,
    solver_target: str,
) -> LeverageFunctionSemantics | None:
    if bucket != SLV_LSV:
        return None
    solver_requirements, missing_solver = _slv_lsv_solver_requirements(solver_target)
    return LeverageFunctionSemantics(
        process_family=process_family,
        leverage_function_kind="spot_time_surface",
        required_market_inputs=(
            "local_vol_surface",
            "black_vol_surface",
            "underlier_spot",
            "discount_curve",
        ),
        required_model_inputs=(
            "heston_model_parameters",
            "leverage_function_surface",
        ),
        calibration_requirements=(
            "recorded_leverage_calibration_problem",
            "local_vol_surface_authority",
            "stochastic_vol_process_coupling",
        ),
        interpolation_domain=("time", "spot"),
        diagnostics=(
            "surface_coverage",
            "leverage_bounds",
            "calibration_residual",
            "martingale_check",
        ),
        solver_requirements=solver_requirements,
        validation_requirements=(
            "consume_local_vol_surface",
            "consume_heston_model_parameters",
            "consume_leverage_function_surface",
            "reject_black_vol_surface_as_model_parameters",
        ),
        supported_now=False,
        missing_components=(
            "leverage_function_calibration_contract",
            "stochastic_local_vol_coupling_contract",
            missing_solver,
        ),
        evidence=tuple(item for item in (target_id, _short_text_evidence(text)) if item),
    )


def _slv_lsv_solver_requirements(solver_target: str) -> tuple[tuple[str, ...], str]:
    if solver_target == "leverage_function_pde":
        return (
            (
                "two_factor_spot_variance_pde_operator",
                "leverage_function_grid_projection",
                "pde_boundary_conditions",
            ),
            "slv_lsv_pde_solver",
        )
    if solver_target == "leverage_function_monte_carlo":
        return (
            (
                "coupled_heston_local_vol_path_simulator",
                "leverage_function_interpolator",
                "variance_scheme_binding",
            ),
            "slv_lsv_monte_carlo_solver",
        )
    return (
        (
            "leverage_function_calibration_problem",
            "surface_projection_diagnostics",
        ),
        "leverage_function_calibration_contract",
    )


def _path_dependent_control_semantics(
    bucket: str,
    *,
    target_id: str,
    text: str,
    process_family: str,
    solver_target: str,
) -> PathDependentControlSemantics | None:
    if bucket != UNSUPPORTED_PATH_DEPENDENT_CONTROL:
        return None
    solver_requirements, missing_solver = _path_dependent_control_solver_requirements(
        solver_target
    )
    return PathDependentControlSemantics(
        process_family=process_family,
        composite_class="american_asian_barrier_under_stochastic_vol",
        state_requirements=(
            "spot_state",
            "variance_state",
            "path_summary_state",
            "exercise_state",
        ),
        path_state_requirements=(
            "running_average_state",
            "barrier_status_state",
            "monitoring_grid_state",
        ),
        event_monitor_requirements=(
            "barrier_monitor",
            "exercise_schedule",
            "monitoring_grid",
        ),
        payoff_summary_requirements=(
            "asian_average_summary",
            "barrier_survival_indicator",
            "exercise_intrinsic_value",
        ),
        control_requirements=(
            "early_exercise_policy",
            "continuation_value_estimator",
            "exercise_projection_policy",
        ),
        stochastic_vol_coupling_requirements=(
            "heston_path_state_coupling",
            "correlated_spot_variance_shocks",
            "variance_scheme_binding",
        ),
        solver_requirements=solver_requirements,
        validation_requirements=(
            "consume_heston_model_parameters",
            "consume_path_state_contract",
            "consume_event_monitor_contract",
            "consume_payoff_summary_contract",
            "consume_early_exercise_control_contract",
            "reject_terminal_transform_route",
        ),
        supported_now=False,
        expected_honest_block=True,
        model_validator_policy="skip_expected_honest_block",
        missing_components=(
            "path_state_simulation_contract",
            "event_monitor_contract",
            "path_payoff_summary_contract",
            "early_exercise_control_contract",
            "stochastic_vol_control_coupling_contract",
            missing_solver,
        ),
        evidence=tuple(item for item in (target_id, _short_text_evidence(text)) if item),
    )


def _path_dependent_control_solver_requirements(
    solver_target: str,
) -> tuple[tuple[str, ...], str]:
    if solver_target == "path_dependent_control_pde":
        return (
            (
                "augmented_state_pde_operator",
                "path_state_grid",
                "free_boundary_control",
            ),
            "path_dependent_heston_pde_solver",
        )
    if solver_target == "path_dependent_control_monte_carlo":
        return (
            (
                "heston_path_state_simulator",
                "pathwise_event_monitor",
                "lsm_under_stochastic_vol_path_state",
            ),
            "path_dependent_heston_monte_carlo_solver",
        )
    return (
        (
            "transform_route_exclusion",
            "terminal_only_characteristic_function_limit",
        ),
        "path_dependent_heston_transform_blocker",
    )


def _unsupported_features(
    bucket: str,
    repair_packet: RepairPacket | None,
) -> tuple[str, ...]:
    if repair_packet is not None and repair_packet.unsupported_class:
        return (repair_packet.unsupported_class,)
    if bucket in {AFFINE_JUMP_STOCHASTIC_VOL, SLV_LSV}:
        return (bucket,)
    return ()


def _task_bucket(targets: Sequence[StochasticVolTargetProblem]) -> str:
    buckets = tuple(dict.fromkeys(target.bucket for target in targets))
    if not buckets:
        return STOCHASTIC_VOL_MIXED
    if len(buckets) == 1:
        return buckets[0]
    if UNSUPPORTED_PATH_DEPENDENT_CONTROL in buckets:
        return UNSUPPORTED_PATH_DEPENDENT_CONTROL
    if SLV_LSV in buckets:
        return SLV_LSV
    if AFFINE_JUMP_STOCHASTIC_VOL in buckets:
        return AFFINE_JUMP_STOCHASTIC_VOL
    return STOCHASTIC_VOL_MIXED


def _task_target_ids(task: Mapping[str, Any]) -> tuple[str, ...]:
    cross_validate = task.get("cross_validate")
    if not isinstance(cross_validate, Mapping):
        return ()
    raw_targets: list[Any] = []
    raw_targets.extend(_as_list(cross_validate.get("internal")))
    raw_targets.extend(_as_list(cross_validate.get("analytical")))
    return tuple(
        dict.fromkeys(
            str(target).strip()
            for target in raw_targets
            if str(target).strip()
        )
    )


def _looks_stochastic_vol_task(
    task: Mapping[str, Any],
    target_ids: Sequence[str],
) -> bool:
    text = _text_blob(task, *target_ids)
    return _has_any(
        text,
        (
            "heston",
            "bates",
            "stochastic vol",
            "stochastic local vol",
            "local-stochastic vol",
            "slv",
            "lsv",
            "rough heston",
        ),
    )


def _text_blob(task: Mapping[str, Any], *target_ids: str) -> str:
    pieces: list[str] = [
        str(task.get("id") or ""),
        str(task.get("title") or ""),
        str(task.get("description") or ""),
    ]
    pieces.extend(target_ids)
    pieces.extend(str(item) for item in _as_list(task.get("construct")))
    pieces.extend(str(item) for item in _as_list(task.get("new_component")))
    market = task.get("market")
    if isinstance(market, Mapping):
        pieces.extend(str(value) for value in market.values())
    assertions = task.get("market_assertions")
    if isinstance(assertions, Mapping):
        pieces.extend(_flatten_strings(assertions))
    return " ".join(pieces).replace("_", " ").lower()


def _target_text(target_id: str) -> str:
    return str(target_id or "").replace("_", " ").lower()


def _target_route_text(task: Mapping[str, Any], target_id: str) -> str:
    text = _target_text(target_id)
    if target_id != "task":
        return text
    pieces = [text, str(task.get("title") or "")]
    pieces.extend(str(item) for item in _as_list(task.get("construct")))
    pieces.extend(str(item) for item in _as_list(task.get("new_component")))
    return " ".join(pieces).replace("_", " ").lower()


def _short_text_evidence(text: str) -> str:
    words = text.split()
    return " ".join(words[:18])


def _has_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        items: list[str] = []
        for key, inner in value.items():
            items.append(str(key))
            items.extend(_flatten_strings(inner))
        return items
    if isinstance(value, (list, tuple)):
        items = []
        for inner in value:
            items.extend(_flatten_strings(inner))
        return items
    return [str(value)]


__all__ = [
    "AFFINE_JUMP_STOCHASTIC_VOL",
    "AffineJumpStochasticVolSemantics",
    "CALIBRATION_TO_SURFACE",
    "CalibrationProblemSemantics",
    "LeverageFunctionSemantics",
    "MarketBindingSemantics",
    "ModelParameterSemantics",
    "PathDependentControlSemantics",
    "RepairPacket",
    "SLV_LSV",
    "STOCHASTIC_VOL_MIXED",
    "STOCHASTIC_VOL_MONTE_CARLO",
    "STOCHASTIC_VOL_PDE",
    "STOCHASTIC_VOL_TRANSFORM",
    "StochasticVolTargetProblem",
    "StochasticVolTaskProblemReport",
    "UNSUPPORTED_PATH_DEPENDENT_CONTROL",
    "classify_stochastic_vol_task",
    "stochastic_vol_problem_payload",
]
