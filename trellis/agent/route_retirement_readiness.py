"""Readiness ledger and masked-authority harness for later route retirement.

This module turns the post-Phase-4 closure requirements into executable
artifacts:

- one seeded readiness ledger for the landed dynamic cohorts
- one reusable masked-authority harness that later-family cutovers can extend

The harness intentionally separates selector-forbidden metadata from the
authoritative snapshot that should remain invariant under those variations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from trellis.agent.dynamic_contract_ir import DynamicContractIR
from trellis.agent.dynamic_lane_admission import compile_dynamic_lane_admission


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _freeze_tuple(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values or ())


@dataclass(frozen=True)
class RouteRetirementGate:
    """One gate in the route-retirement readiness ledger."""

    ready: bool
    evidence_refs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", _freeze_tuple(self.evidence_refs))
        object.__setattr__(self, "notes", _freeze_tuple(self.notes))


@dataclass(frozen=True)
class RouteRetirementReadinessRecord:
    """Machine-checkable readiness row for one later-family migration cohort."""

    cohort_id: str
    semantic_track: str
    proving_families: tuple[str, ...]
    honest_block_relatives: tuple[str, ...]
    representation_closure: RouteRetirementGate
    decomposition_closure: RouteRetirementGate
    lowering_admission: RouteRetirementGate
    parity_or_benchmark: RouteRetirementGate
    provenance_readiness: RouteRetirementGate
    masked_authority_readiness: RouteRetirementGate
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "proving_families", _freeze_tuple(self.proving_families))
        object.__setattr__(
            self,
            "honest_block_relatives",
            _freeze_tuple(self.honest_block_relatives),
        )
        object.__setattr__(self, "notes", _freeze_tuple(self.notes))


_READINESS_GATE_NAMES = (
    "representation_closure",
    "decomposition_closure",
    "lowering_admission",
    "parity_or_benchmark",
    "provenance_readiness",
    "masked_authority_readiness",
)


def missing_route_retirement_gates(record: RouteRetirementReadinessRecord) -> tuple[str, ...]:
    """Return the not-yet-ready route-retirement gates for one cohort."""

    missing: list[str] = []
    for name in _READINESS_GATE_NAMES:
        gate = getattr(record, name)
        if not gate.ready:
            missing.append(name)
    return tuple(missing)


def is_route_retirement_ready(record: RouteRetirementReadinessRecord) -> bool:
    """Return ``True`` when the cohort can enter a real route-retirement ticket."""

    return not missing_route_retirement_gates(record)


@dataclass(frozen=True)
class MaskedAuthorityVariant:
    """Selector-forbidden metadata that a route-retirement harness may perturb."""

    label: str
    route_id: str | None = None
    route_family: str | None = None
    product_instrument: str | None = None
    wrapper_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "route_id", str(self.route_id or "").strip() or None)
        object.__setattr__(self, "route_family", str(self.route_family or "").strip() or None)
        object.__setattr__(
            self,
            "product_instrument",
            str(self.product_instrument or "").strip() or None,
        )
        object.__setattr__(self, "wrapper_metadata", _freeze_mapping(self.wrapper_metadata))


@dataclass(frozen=True)
class AuthoritySelectionSnapshot:
    """Canonical authority snapshot used by masked-authority invariance tests."""

    selection_surface: str
    authoritative_ref: str
    lane_family: str
    semantic_family: str | None = None
    binding_id: str | None = None
    validation_bundle_id: str | None = None
    candidate_lanes: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_surface", str(self.selection_surface or "").strip())
        object.__setattr__(self, "authoritative_ref", str(self.authoritative_ref or "").strip())
        object.__setattr__(self, "lane_family", str(self.lane_family or "").strip())
        object.__setattr__(
            self,
            "semantic_family",
            str(self.semantic_family or "").strip() or None,
        )
        object.__setattr__(self, "binding_id", str(self.binding_id or "").strip() or None)
        object.__setattr__(
            self,
            "validation_bundle_id",
            str(self.validation_bundle_id or "").strip() or None,
        )
        object.__setattr__(self, "candidate_lanes", _freeze_tuple(self.candidate_lanes))
        object.__setattr__(self, "requested_outputs", _freeze_tuple(self.requested_outputs))


class MaskedAuthorityInvariantError(AssertionError):
    """Raised when selector-forbidden metadata changes authoritative selection."""


def default_masked_authority_variants() -> tuple[MaskedAuthorityVariant, ...]:
    """Return the default selector-forbidden metadata perturbations.

    These variants intentionally cover every field called out by the Phase 4
    authority contract: route ids, route families, product-instrument labels,
    and non-semantic trade-envelope metadata.
    """

    return (
        MaskedAuthorityVariant(
            label="baseline",
            route_id="legacy_route_alpha",
            route_family="legacy_family_alpha",
            product_instrument="legacy_product_alpha",
            wrapper_metadata={
                "external_id": "TRADE-ALPHA",
                "booking_tags": ["desk-alpha", "ops-rebook-a"],
                "package_wrapper": {
                    "package_id": "PKG-ALPHA",
                    "position_id": "POS-ALPHA",
                },
            },
        ),
        MaskedAuthorityVariant(
            label="rebooked_wrapper",
            route_id="legacy_route_bravo",
            route_family="legacy_family_bravo",
            product_instrument="legacy_product_bravo",
            wrapper_metadata={
                "external_id": "TRADE-BRAVO",
                "booking_tags": ["desk-bravo", "manual-rebook"],
                "package_wrapper": {
                    "package_id": "PKG-BRAVO",
                    "position_id": "POS-BRAVO",
                },
            },
        ),
        MaskedAuthorityVariant(
            label="audit_clone",
            route_id="legacy_route_charlie",
            route_family="legacy_family_charlie",
            product_instrument="legacy_product_charlie",
            wrapper_metadata={
                "external_id": "TRADE-CHARLIE",
                "booking_tags": ["desk-charlie", "audit-clone"],
                "package_wrapper": {
                    "package_id": "PKG-CHARLIE",
                    "position_id": "POS-CHARLIE",
                },
            },
        ),
    )


def require_masked_authority_invariant(
    variants: Sequence[MaskedAuthorityVariant],
    snapshot_builder: Callable[[MaskedAuthorityVariant], AuthoritySelectionSnapshot],
) -> AuthoritySelectionSnapshot:
    """Require the authoritative selection snapshot to stay invariant.

    The caller decides how each variant is threaded through its compiler or
    probe surface. The resulting snapshots must be identical if the varied
    fields are truly selector-forbidden.
    """

    if not variants:
        raise ValueError("variants must be non-empty")

    baseline_variant = variants[0]
    baseline = snapshot_builder(baseline_variant)
    if not isinstance(baseline, AuthoritySelectionSnapshot):
        raise TypeError("snapshot_builder must return AuthoritySelectionSnapshot")

    for variant in variants[1:]:
        candidate = snapshot_builder(variant)
        if not isinstance(candidate, AuthoritySelectionSnapshot):
            raise TypeError("snapshot_builder must return AuthoritySelectionSnapshot")
        if candidate != baseline:
            raise MaskedAuthorityInvariantError(
                "selector-forbidden metadata changed authoritative selection: "
                f"baseline={baseline_variant.label!r} -> {baseline!r}, "
                f"candidate={variant.label!r} -> {candidate!r}"
            )
    return baseline


def capture_compiled_request_authority_snapshot(compiled_request) -> AuthoritySelectionSnapshot:
    """Capture the authoritative snapshot from the current Phase 4 request path."""

    request = getattr(compiled_request, "request", None)
    metadata = dict(getattr(request, "metadata", {}) or {})
    semantic_blueprint = dict(metadata.get("semantic_blueprint") or {})
    selection = dict(semantic_blueprint.get("contract_ir_solver_selection") or {})
    if not selection:
        selection = dict(semantic_blueprint.get("static_leg_lowering_selection") or {})
    authority = dict(metadata.get("route_binding_authority") or {})
    backend_binding = dict(authority.get("backend_binding") or {})
    authoritative_ref = str(selection.get("declaration_id") or "").strip()
    binding_id = str(backend_binding.get("binding_id") or "").strip() or None
    if not authoritative_ref and binding_id is None:
        raise ValueError("compiled request does not carry route-free authority metadata")

    lane_family = (
        str(backend_binding.get("engine_family") or "").strip()
        or str(selection.get("requested_method") or "").strip()
    )
    if not lane_family:
        raise ValueError("compiled request does not carry a lane family")

    return AuthoritySelectionSnapshot(
        selection_surface="platform_request",
        authoritative_ref=authoritative_ref or str(binding_id or ""),
        lane_family=lane_family,
        binding_id=binding_id,
        validation_bundle_id=str(authority.get("validation_bundle_id") or "").strip() or None,
        requested_outputs=tuple(getattr(request, "requested_outputs", ()) or ()),
    )


def capture_dynamic_lane_probe_authority_snapshot(
    contract: DynamicContractIR,
    *,
    variant: MaskedAuthorityVariant | None = None,
) -> AuthoritySelectionSnapshot:
    """Capture a bounded later-family authority snapshot from dynamic admission.

    ``variant`` exists only to make the selector-forbidden surface explicit.
    Dynamic lane admission remains authoritative on the semantic contract and
    ignores legacy route labels and wrapper metadata entirely.
    """

    if not isinstance(contract, DynamicContractIR):
        raise TypeError("dynamic lane probe requires a DynamicContractIR")
    if variant is not None and not isinstance(variant, MaskedAuthorityVariant):
        raise TypeError("variant must be a MaskedAuthorityVariant")

    admission = compile_dynamic_lane_admission(contract)
    benchmark_plan = admission.benchmark_plan
    return AuthoritySelectionSnapshot(
        selection_surface="dynamic_lane_probe",
        authoritative_ref=benchmark_plan.cohort_id,
        lane_family=admission.lane,
        semantic_family=admission.semantic_family,
        candidate_lanes=tuple(admission.candidate_numerical_lanes),
    )


_DYNAMIC_ROUTE_RETIREMENT_READINESS_LEDGER = (
    RouteRetirementReadinessRecord(
        cohort_id="automatic_event_state",
        semantic_track="dynamic_wrapper",
        proving_families=("autocallable", "tarn"),
        honest_block_relatives=("callable_cms_range_accrual", "prdc_hybrid", "swing_option"),
        representation_closure=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.dynamic_contract_ir",
                "docs/quant/dynamic_contract_ir.rst",
            ),
        ),
        decomposition_closure=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.knowledge.decompose.decompose_to_dynamic_contract_ir",
                "tests/test_agent/test_decompose_static_and_dynamic_contracts.py",
            ),
        ),
        lowering_admission=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.dynamic_lane_admission.compile_dynamic_lane_admission",
                "doc/plan/draft__automatic-event-state-lowering-plan.md",
            ),
        ),
        parity_or_benchmark=RouteRetirementGate(
            ready=False,
            evidence_refs=("doc/plan/draft__automatic-event-state-lowering-plan.md",),
            notes=(
                "benchmark plans exist via DynamicBenchmarkPlan, but no parity-proven executable fresh-build lane has landed yet",
            ),
        ),
        provenance_readiness=RouteRetirementGate(
            ready=False,
            evidence_refs=(
                "doc/plan/draft__contract-ir-phase-4-route-retirement.md",
                "doc/plan/draft__valuation-result-identity-and-provenance.md",
            ),
            notes=(
                "dynamic-lane provenance has not yet been promoted onto the Phase 4 valuation identity surface",
            ),
        ),
        masked_authority_readiness=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.route_retirement_readiness.require_masked_authority_invariant",
                "tests/test_agent/test_route_retirement_readiness.py",
            ),
            notes=(
                "synthetic later-family probe now masks route ids, route families, ProductIR.instrument labels, and wrapper metadata",
            ),
        ),
        notes=(
            "automatic event/state semantics are closed for bounded proving cohorts but not yet cutover-ready",
        ),
    ),
    RouteRetirementReadinessRecord(
        cohort_id="discrete_control",
        semantic_track="dynamic_wrapper",
        proving_families=("callable_bond", "swing_option"),
        honest_block_relatives=("autocallable", "gmwb_financial_control", "insurance_overlay"),
        representation_closure=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.dynamic_contract_ir",
                "docs/quant/dynamic_contract_ir.rst",
            ),
        ),
        decomposition_closure=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.knowledge.decompose.decompose_to_dynamic_contract_ir",
                "tests/test_agent/test_decompose_static_and_dynamic_contracts.py",
            ),
        ),
        lowering_admission=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.dynamic_lane_admission.compile_dynamic_lane_admission",
                "doc/plan/draft__discrete-control-lowering-plan.md",
            ),
        ),
        parity_or_benchmark=RouteRetirementGate(
            ready=False,
            evidence_refs=("doc/plan/draft__discrete-control-lowering-plan.md",),
            notes=(
                "benchmark and parity plans exist, but the route-free executable lane has not been validated for cutover",
            ),
        ),
        provenance_readiness=RouteRetirementGate(
            ready=False,
            evidence_refs=(
                "doc/plan/draft__contract-ir-phase-4-route-retirement.md",
                "doc/plan/draft__valuation-result-identity-and-provenance.md",
            ),
            notes=(
                "controller-role and decision-timing provenance still needs the later-family result packet contract",
            ),
        ),
        masked_authority_readiness=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.route_retirement_readiness.require_masked_authority_invariant",
                "tests/test_agent/test_route_retirement_readiness.py",
            ),
            notes=(
                "bounded callable-bond and swing-style probes can now reuse the shared masked-authority harness",
            ),
        ),
        notes=(
            "discrete-control route retirement remains blocked on real parity and provenance evidence, not on representation",
        ),
    ),
    RouteRetirementReadinessRecord(
        cohort_id="continuous_singular_control",
        semantic_track="dynamic_wrapper",
        proving_families=("gmwb_financial_control",),
        honest_block_relatives=("insurance_overlay", "mortality_overlay", "lapse_overlay"),
        representation_closure=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.dynamic_contract_ir",
                "docs/quant/dynamic_contract_ir.rst",
            ),
        ),
        decomposition_closure=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.knowledge.decompose.decompose_to_dynamic_contract_ir",
                "tests/test_agent/test_decompose_static_and_dynamic_contracts.py",
            ),
        ),
        lowering_admission=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.dynamic_lane_admission.compile_dynamic_lane_admission",
                "doc/plan/draft__continuous-singular-control-lowering-plan.md",
            ),
        ),
        parity_or_benchmark=RouteRetirementGate(
            ready=False,
            evidence_refs=("doc/plan/draft__continuous-singular-control-lowering-plan.md",),
            notes=(
                "approximation-policy and literature-benchmark plans exist, but no cutover-grade parity evidence has landed",
            ),
        ),
        provenance_readiness=RouteRetirementGate(
            ready=False,
            evidence_refs=(
                "doc/plan/draft__contract-ir-phase-4-route-retirement.md",
                "doc/plan/draft__valuation-result-identity-and-provenance.md",
            ),
            notes=(
                "control-magnitude and approximation-policy provenance is still planned work",
            ),
        ),
        masked_authority_readiness=RouteRetirementGate(
            ready=True,
            evidence_refs=(
                "trellis.agent.route_retirement_readiness.require_masked_authority_invariant",
                "tests/test_agent/test_route_retirement_readiness.py",
            ),
            notes=(
                "financial-control-only GMWB probes can now inherit the shared masked-authority contract without claiming insurance-overlay support",
            ),
        ),
        notes=(
            "continuous-control remains bounded to overlay-free financial control until CLX.7 expands the state space",
        ),
    ),
)


def dynamic_route_retirement_readiness_ledger() -> tuple[RouteRetirementReadinessRecord, ...]:
    """Return the seeded later-family readiness ledger for dynamic cohorts."""

    return _DYNAMIC_ROUTE_RETIREMENT_READINESS_LEDGER


def get_dynamic_route_retirement_readiness(cohort_id: str) -> RouteRetirementReadinessRecord:
    """Return one dynamic route-retirement readiness record by cohort id."""

    normalized = str(cohort_id or "").strip().lower()
    for record in _DYNAMIC_ROUTE_RETIREMENT_READINESS_LEDGER:
        if record.cohort_id == normalized:
            return record
    raise KeyError(f"unknown dynamic route-retirement cohort {cohort_id!r}")


__all__ = [
    "AuthoritySelectionSnapshot",
    "MaskedAuthorityInvariantError",
    "MaskedAuthorityVariant",
    "RouteRetirementGate",
    "RouteRetirementReadinessRecord",
    "capture_compiled_request_authority_snapshot",
    "capture_dynamic_lane_probe_authority_snapshot",
    "default_masked_authority_variants",
    "dynamic_route_retirement_readiness_ledger",
    "get_dynamic_route_retirement_readiness",
    "is_route_retirement_ready",
    "missing_route_retirement_gates",
    "require_masked_authority_invariant",
]
