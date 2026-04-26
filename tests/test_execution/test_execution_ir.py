from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, is_dataclass

import pytest

from trellis.execution import (
    ContractExecutionIR,
    ExecutionMetadata,
    KnownCashflowObligation,
    RequirementHints,
    SourceTrack,
    compile_contract_execution_ir,
    contract_execution_summary,
)
from trellis.execution.compiler import UnsupportedExecutionSemantics


def test_execution_ir_is_importable_frozen_and_summary_serializable():
    ir = ContractExecutionIR(
        source_track=SourceTrack(
            source_kind="semantic_contract",
            semantic_id="SEM-1",
            source_ref="semantic:SEM-1",
        ),
        obligations=(
            KnownCashflowObligation(
                obligation_id="cf-1",
                payment_date="2027-01-05",
                currency="USD",
                amount=100.0,
                payer="issuer",
                receiver="holder",
            ),
        ),
        requirement_hints=RequirementHints(
            market_inputs=frozenset({"discount_curve:USD"}),
            timeline_roles=frozenset({"payment"}),
        ),
        execution_metadata=ExecutionMetadata(tags=("unit-test",)),
    )

    assert is_dataclass(ir)
    with pytest.raises(FrozenInstanceError):
        ir.obligations = ()

    payload = asdict(ir)
    assert payload["source_track"]["semantic_id"] == "SEM-1"

    summary = contract_execution_summary(ir)
    assert summary == contract_execution_summary(ir)
    assert summary == {
        "schema_version": "xir.0",
        "source_kind": "semantic_contract",
        "semantic_id": "SEM-1",
        "source_ref": "semantic:SEM-1",
        "obligation_count": 1,
        "obligation_kinds": ("known_cashflow",),
        "observable_count": 0,
        "observable_kinds": (),
        "event_count": 0,
        "state_field_count": 0,
        "decision_action_count": 0,
        "settlement_step_count": 0,
        "requirement_markets": ("discount_curve:USD",),
        "requirement_states": (),
        "timeline_roles": ("payment",),
        "unsupported_reasons": (),
        "route_ids": (),
        "model_families": (),
        "tags": ("unit-test",),
    }


def test_empty_seam_ir_is_route_free_model_free_and_neutral():
    ir = ContractExecutionIR.empty(
        source_track=SourceTrack(
            source_kind="contract_ir",
            semantic_id="SEM-2",
            instrument_class="vanilla_option",
        ),
        unsupported_reasons=("execution lowering not implemented for contract_ir",),
    )

    assert ir.obligations == ()
    assert ir.observables == ()
    assert ir.event_plan.events == ()
    assert ir.state_schema.fields == ()
    assert ir.decision_program.actions == ()
    assert ir.settlement_program.steps == ()

    summary = contract_execution_summary(ir)
    assert summary["route_ids"] == ()
    assert summary["model_families"] == ()
    assert summary["unsupported_reasons"] == (
        "execution lowering not implemented for contract_ir",
    )


def test_string_hints_are_normalized_as_single_values():
    hints = RequirementHints(market_inputs="discount_curve:USD")

    assert hints.market_inputs == frozenset({"discount_curve:USD"})


def test_compiler_entrypoint_accepts_upstream_objects_without_fake_lowering():
    class UpstreamSemanticObject:
        semantic_id = "SEM-3"
        source_kind = "semantic_contract"

        class product:
            instrument_class = "swap"
            payoff_family = "static_leg"

    ir = compile_contract_execution_ir(UpstreamSemanticObject())

    assert isinstance(ir, ContractExecutionIR)
    assert ir.source_track.source_kind == "semantic_contract"
    assert ir.source_track.semantic_id == "SEM-3"
    assert ir.source_track.instrument_class == "swap"
    assert ir.obligations == ()
    assert "not implemented" in ir.requirement_hints.unsupported_reasons[0]
    assert contract_execution_summary(ir)["obligation_count"] == 0


def test_compiler_can_fail_closed_when_requested():
    with pytest.raises(UnsupportedExecutionSemantics):
        compile_contract_execution_ir(object(), fail_on_unsupported=True)


def test_execution_symbols_are_visible_to_import_registry():
    from trellis.agent.knowledge.import_registry import (
        list_module_exports,
        module_exists,
        reset_registry_cache,
    )

    reset_registry_cache()

    assert module_exists("trellis.execution")
    assert "ContractExecutionIR" in list_module_exports("trellis.execution")
    assert "compile_contract_execution_ir" in list_module_exports("trellis.execution")
