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
    process_family = _process_family(context_text, bucket)
    semantics = _model_parameter_semantics(bucket, process_family)
    bindings = _market_bindings(bucket, process_family)
    repair_packet = _repair_packet(bucket, target_id=target_id, text=target_text)
    unsupported = _unsupported_features(bucket, repair_packet)
    return StochasticVolTargetProblem(
        target_id=target_id,
        bucket=bucket,
        solver_target=_solver_target(target_text, bucket),
        process_family=process_family,
        payoff_class=_payoff_class(context_text),
        model_parameter_semantics=semantics,
        market_bindings=bindings,
        validation_bundle=_validation_bundle(bucket, process_family),
        unsupported_features=unsupported,
        repair_packet=repair_packet,
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
            missing_primitive="path_dependent_heston_control_kernel",
            unsupported_class="path_dependent_early_exercise_under_stochastic_vol",
            summary=(
                "Path-dependent early-exercise under stochastic volatility is a "
                "composite control problem and should block honestly until a "
                "dedicated kernel exists."
            ),
            evidence=evidence,
        )
    if bucket == STOCHASTIC_VOL_MONTE_CARLO and "qe" in text:
        return RepairPacket(
            packet_type="missing_heston_qe_scheme",
            missing_primitive="heston_andersen_qe_scheme",
            summary=(
                "The target asks for Andersen QE Heston simulation; route binding "
                "must not silently fall back to Euler semantics."
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
    "CALIBRATION_TO_SURFACE",
    "MarketBindingSemantics",
    "ModelParameterSemantics",
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
