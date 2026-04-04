"""Executable governed policy bundles and runtime guard evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from types import MappingProxyType
from typing import Mapping

from trellis.platform.context import ExecutionContext, RunMode, default_policy_bundle_id
from trellis.platform.models import ModelLifecycleStatus


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable shallow copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _normalize_token(value: str | None, *, fallback: str = "") -> str:
    """Normalize one identifier token."""
    return str(value or "").strip().lower().replace(" ", "_") or fallback


def _string_tuple(values) -> tuple[str, ...]:
    """Return a stable tuple of unique normalized strings."""
    if not values:
        return ()
    items: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in items:
            items.append(text)
    return tuple(items)


def _run_mode_tuple(values) -> tuple[RunMode, ...]:
    """Normalize one iterable of run modes."""
    return tuple(RunMode.normalize(value) for value in values or ())


def _model_status_tuple(values) -> tuple[ModelLifecycleStatus, ...]:
    """Normalize one iterable of model lifecycle statuses."""
    return tuple(ModelLifecycleStatus.normalize(value) for value in values or ())


def _is_mock_provider_id(provider_id: str | None) -> bool:
    """Return whether one provider id represents mock data."""
    token = _normalize_token(provider_id)
    if not token:
        return False
    return any(part.startswith("mock") for part in token.split("."))


def _selected_model_status(selected_model) -> ModelLifecycleStatus | None:
    """Resolve the selected model lifecycle status when available."""
    if selected_model is None:
        return None
    if isinstance(selected_model, Mapping):
        for key in ("status", "model_status", "selected_version_status"):
            value = selected_model.get(key)
            if value:
                return ModelLifecycleStatus.normalize(value)
        return None
    for attr in ("status", "model_status", "selected_version_status"):
        value = getattr(selected_model, attr, None)
        if value:
            return ModelLifecycleStatus.normalize(value)
    return None


def _selected_model_id(selected_model) -> str:
    """Resolve the selected model id when available."""
    if selected_model is None:
        return ""
    if isinstance(selected_model, Mapping):
        return str(
            selected_model.get("model_id")
            or selected_model.get("id")
            or ""
        ).strip()
    return str(
        getattr(selected_model, "model_id", None)
        or getattr(selected_model, "id", None)
        or ""
    ).strip()


@dataclass(frozen=True)
class PolicyBundle:
    """Executable governed runtime policy bundle."""

    policy_id: str
    name: str = ""
    allowed_run_modes: tuple[RunMode, ...] = ()
    allow_mock_data: bool = False
    require_provider_disclosure: bool = True
    required_provider_families: tuple[str, ...] = ("market_data",)
    allowed_model_statuses: tuple[ModelLifecycleStatus, ...] = ()
    required_provenance_fields: tuple[str, ...] = ()
    metadata: Mapping[str, object] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        """Normalize bundle fields into immutable primitives."""
        policy_id = str(self.policy_id or "").strip()
        if not policy_id:
            raise ValueError("policy_id is required")
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "allowed_run_modes", _run_mode_tuple(self.allowed_run_modes))
        object.__setattr__(self, "required_provider_families", _string_tuple(self.required_provider_families))
        object.__setattr__(self, "allowed_model_statuses", _model_status_tuple(self.allowed_model_statuses))
        object.__setattr__(self, "required_provenance_fields", _string_tuple(self.required_provenance_fields))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "allowed_run_modes": [mode.value for mode in self.allowed_run_modes],
            "allow_mock_data": self.allow_mock_data,
            "require_provider_disclosure": self.require_provider_disclosure,
            "required_provider_families": list(self.required_provider_families),
            "allowed_model_statuses": [status.value for status in self.allowed_model_statuses],
            "required_provenance_fields": list(self.required_provenance_fields),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PolicyBundle:
        """Rehydrate one policy bundle."""
        return cls(
            policy_id=str(payload.get("policy_id", "")).strip(),
            name=str(payload.get("name", "")).strip(),
            allowed_run_modes=payload.get("allowed_run_modes") or (),
            allow_mock_data=bool(payload.get("allow_mock_data", False)),
            require_provider_disclosure=bool(
                payload.get("require_provider_disclosure", True)
            ),
            required_provider_families=payload.get("required_provider_families") or (),
            allowed_model_statuses=payload.get("allowed_model_statuses") or (),
            required_provenance_fields=payload.get("required_provenance_fields") or (),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class PolicyBlocker:
    """Structured policy failure reason."""

    code: str
    message: str
    requirement: str = ""
    field: str = ""
    details: Mapping[str, object] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        """Normalize blocker metadata into immutable primitives."""
        object.__setattr__(self, "code", _normalize_token(self.code))
        object.__setattr__(self, "message", str(self.message or "").strip())
        object.__setattr__(self, "requirement", _normalize_token(self.requirement))
        object.__setattr__(self, "field", str(self.field or "").strip())
        object.__setattr__(self, "details", _freeze_mapping(self.details))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "code": self.code,
            "message": self.message,
            "requirement": self.requirement,
            "field": self.field,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PolicyEvaluation:
    """Deterministic evaluation result for one runtime context."""

    policy_id: str
    run_mode: RunMode
    allowed: bool
    blockers: tuple[PolicyBlocker, ...] = ()
    satisfied_requirements: tuple[str, ...] = ()
    provenance: Mapping[str, object] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        """Normalize evaluation metadata into immutable primitives."""
        object.__setattr__(self, "policy_id", str(self.policy_id or "").strip())
        object.__setattr__(self, "run_mode", RunMode.normalize(self.run_mode))
        object.__setattr__(
            self,
            "blockers",
            tuple(
                blocker if isinstance(blocker, PolicyBlocker) else PolicyBlocker(**blocker)
                for blocker in self.blockers
            ),
        )
        object.__setattr__(self, "satisfied_requirements", _string_tuple(self.satisfied_requirements))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    @property
    def blocker_codes(self) -> tuple[str, ...]:
        """Return the ordered blocker codes."""
        return tuple(blocker.code for blocker in self.blockers)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "policy_id": self.policy_id,
            "run_mode": self.run_mode.value,
            "allowed": self.allowed,
            "blocker_codes": list(self.blocker_codes),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "satisfied_requirements": list(self.satisfied_requirements),
            "provenance": dict(self.provenance),
        }


class PolicyViolationError(RuntimeError):
    """Raised when governed policy blocks execution."""

    def __init__(self, evaluation: PolicyEvaluation):
        self.evaluation = evaluation
        summary = ", ".join(evaluation.blocker_codes) or "unknown_policy_violation"
        super().__init__(f"Governed execution blocked by policy: {summary}")


def _default_policy_bundles() -> dict[str, PolicyBundle]:
    """Return the built-in governed policy bundles."""
    return {
        "policy_bundle.sandbox.default": PolicyBundle(
            policy_id="policy_bundle.sandbox.default",
            name="Sandbox Default",
            allowed_run_modes=(RunMode.SANDBOX,),
            allow_mock_data=True,
            require_provider_disclosure=False,
            required_provider_families=("market_data",),
            allowed_model_statuses=(
                ModelLifecycleStatus.DRAFT,
                ModelLifecycleStatus.VALIDATED,
                ModelLifecycleStatus.APPROVED,
            ),
            required_provenance_fields=(),
        ),
        "policy_bundle.research.default": PolicyBundle(
            policy_id="policy_bundle.research.default",
            name="Research Default",
            allowed_run_modes=(RunMode.RESEARCH,),
            allow_mock_data=False,
            require_provider_disclosure=True,
            required_provider_families=("market_data",),
            allowed_model_statuses=(
                ModelLifecycleStatus.VALIDATED,
                ModelLifecycleStatus.APPROVED,
            ),
            required_provenance_fields=("provider_id", "market_snapshot_id"),
        ),
        "policy_bundle.production.default": PolicyBundle(
            policy_id="policy_bundle.production.default",
            name="Production Default",
            allowed_run_modes=(RunMode.PRODUCTION,),
            allow_mock_data=False,
            require_provider_disclosure=True,
            required_provider_families=("market_data",),
            allowed_model_statuses=(ModelLifecycleStatus.APPROVED,),
            required_provenance_fields=("provider_id", "market_snapshot_id"),
        ),
    }


def get_policy_bundle(
    policy_bundle_id: str | None = None,
    *,
    run_mode: RunMode | str | None = None,
) -> PolicyBundle:
    """Return one built-in policy bundle by id or run mode."""
    bundle_id = str(policy_bundle_id or "").strip() or default_policy_bundle_id(run_mode)
    try:
        return _default_policy_bundles()[bundle_id]
    except KeyError as exc:
        raise ValueError(f"Unknown policy bundle: {bundle_id!r}") from exc


def _resolved_provenance(
    *,
    execution_context: ExecutionContext,
    market_snapshot=None,
    selected_model=None,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the normalized provenance map used by policy evaluation."""
    resolved: dict[str, object] = {
        "run_mode": execution_context.run_mode.value,
        "policy_bundle_id": execution_context.policy_bundle_id,
        "session_id": execution_context.session_id,
    }
    market_binding = execution_context.provider_bindings.market_data.primary
    if market_binding is not None and market_binding.provider_id:
        resolved["provider_id"] = market_binding.provider_id
    snapshot_provider_id = str(getattr(market_snapshot, "provider_id", "") or "").strip()
    if snapshot_provider_id:
        resolved["provider_id"] = snapshot_provider_id
    snapshot_id = str(getattr(market_snapshot, "market_snapshot_id", "") or "").strip()
    if snapshot_id:
        resolved["market_snapshot_id"] = snapshot_id
    model_id = _selected_model_id(selected_model)
    if model_id:
        resolved["model_id"] = model_id
    model_status = _selected_model_status(selected_model)
    if model_status is not None:
        resolved["model_status"] = model_status.value
    resolved.update(dict(provenance or {}))
    return resolved


def evaluate_execution_policy(
    *,
    execution_context: ExecutionContext,
    policy_bundle: PolicyBundle | None = None,
    market_snapshot=None,
    selected_model=None,
    provenance: Mapping[str, object] | None = None,
) -> PolicyEvaluation:
    """Evaluate one governed execution context against an executable policy bundle."""
    bundle = policy_bundle or get_policy_bundle(
        execution_context.policy_bundle_id,
        run_mode=execution_context.run_mode,
    )
    blockers: list[PolicyBlocker] = []
    satisfied: list[str] = []

    if execution_context.run_mode not in bundle.allowed_run_modes:
        blockers.append(
            PolicyBlocker(
                code="run_mode_not_allowed",
                message=(
                    f"Run mode {execution_context.run_mode.value!r} is not allowed by {bundle.policy_id}"
                ),
                requirement="allowed_run_modes",
                field="run_mode",
                details={
                    "allowed_run_modes": [mode.value for mode in bundle.allowed_run_modes],
                },
            )
        )
    else:
        satisfied.append("allowed_run_modes")

    market_data_binding = execution_context.provider_bindings.market_data
    provider_disclosure_required = (
        bundle.require_provider_disclosure
        or execution_context.require_provider_disclosure
    )
    if "market_data" in bundle.required_provider_families:
        if market_data_binding.primary is None:
            blockers.append(
                PolicyBlocker(
                    code="provider_binding_required",
                    message="Governed execution requires an explicit market-data provider binding",
                    requirement="provider_disclosure",
                    field="provider_bindings.market_data.primary",
                )
            )
        elif provider_disclosure_required:
            satisfied.append("provider_disclosure")

    bound_provider_ids = [
        binding.provider_id
        for binding in (market_data_binding.primary, market_data_binding.fallback)
        if binding is not None and binding.provider_id
    ]
    snapshot_provider_id = str(getattr(market_snapshot, "provider_id", "") or "").strip()
    if snapshot_provider_id:
        bound_provider_ids.append(snapshot_provider_id)
    mock_provider_ids = tuple(
        dict.fromkeys(
            provider_id
            for provider_id in bound_provider_ids
            if _is_mock_provider_id(provider_id)
        )
    )
    mock_requested = execution_context.allow_mock_data or bool(mock_provider_ids)
    if mock_requested and not (bundle.allow_mock_data and execution_context.allow_mock_data):
        blockers.append(
            PolicyBlocker(
                code="mock_data_not_allowed",
                message="Governed policy does not allow mock data for this execution context",
                requirement="mock_data_policy",
                field="allow_mock_data",
                details={"provider_ids": list(mock_provider_ids)},
            )
        )
    elif mock_requested:
        satisfied.append("mock_data_policy")

    allowed_model_statuses = bundle.allowed_model_statuses
    selected_model_status = _selected_model_status(selected_model)
    if allowed_model_statuses and selected_model is not None:
        if selected_model_status is None:
            blockers.append(
                PolicyBlocker(
                    code="model_lifecycle_status_required",
                    message="Governed execution requires a selected model lifecycle status",
                    requirement="model_lifecycle",
                    field="selected_model.status",
                    details={
                        "allowed_statuses": [status.value for status in allowed_model_statuses],
                    },
                )
            )
        elif selected_model_status not in allowed_model_statuses:
            blockers.append(
                PolicyBlocker(
                    code="model_lifecycle_not_allowed",
                    message=(
                        f"Model lifecycle status {selected_model_status.value!r} is not allowed by {bundle.policy_id}"
                    ),
                    requirement="model_lifecycle",
                    field="selected_model.status",
                    details={
                        "allowed_statuses": [status.value for status in allowed_model_statuses],
                    },
                )
            )
        else:
            satisfied.append("model_lifecycle")

    resolved_provenance = _resolved_provenance(
        execution_context=execution_context,
        market_snapshot=market_snapshot,
        selected_model=selected_model,
        provenance=provenance,
    )
    for field_name in bundle.required_provenance_fields:
        if not resolved_provenance.get(field_name):
            blockers.append(
                PolicyBlocker(
                    code="missing_provenance_field",
                    message=f"Required provenance field {field_name!r} is missing",
                    requirement="required_provenance_fields",
                    field=field_name,
                )
            )
    if not any(blocker.code == "missing_provenance_field" for blocker in blockers):
        if bundle.required_provenance_fields:
            satisfied.append("required_provenance_fields")

    return PolicyEvaluation(
        policy_id=bundle.policy_id,
        run_mode=execution_context.run_mode,
        allowed=not blockers,
        blockers=tuple(blockers),
        satisfied_requirements=tuple(satisfied),
        provenance=resolved_provenance,
    )


def enforce_execution_policy(
    *,
    execution_context: ExecutionContext,
    policy_bundle: PolicyBundle | None = None,
    market_snapshot=None,
    selected_model=None,
    provenance: Mapping[str, object] | None = None,
) -> PolicyEvaluation:
    """Evaluate policy and raise a structured error when execution is blocked."""
    evaluation = evaluate_execution_policy(
        execution_context=execution_context,
        policy_bundle=policy_bundle,
        market_snapshot=market_snapshot,
        selected_model=selected_model,
        provenance=provenance,
    )
    if not evaluation.allowed:
        raise PolicyViolationError(evaluation)
    return evaluation
