from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, is_dataclass

import pytest

from trellis.execution import (
    BermudanBestOfBasketLatticeControls,
    BermudanBestOfBasketMCControls,
    BermudanBestOfBasketMCInputs,
    ContractExecutionIR,
    ExecutionMetadata,
    KnownCashflowObligation,
    RequirementHints,
    SourceTrack,
    admit_execution_capabilities,
    compile_bermudan_best_of_basket_execution_ir,
    compile_contract_execution_ir,
    contract_execution_summary,
    price_bermudan_best_of_basket_lattice,
    price_bermudan_best_of_basket_monte_carlo,
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
        "event_kinds": (),
        "state_field_count": 0,
        "state_fields": (),
        "decision_action_count": 0,
        "decision_action_types": (),
        "settlement_step_count": 0,
        "settlement_kinds": (),
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


def test_bermudan_best_of_basket_execution_ir_records_operator_shape():
    from datetime import date

    ir = compile_bermudan_best_of_basket_execution_ir(
        semantic_id="P001",
        underliers=("AAPL", "MSFT"),
        strike=100.0,
        expiry_date=date(2025, 12, 15),
        observation_dates=(
            date(2025, 3, 15),
            date(2025, 6, 15),
            date(2025, 9, 15),
            date(2025, 12, 15),
        ),
        exercise_dates=(
            date(2025, 6, 15),
            date(2025, 9, 15),
            date(2025, 12, 15),
        ),
        requested_outputs=("price", "greeks", "bounds"),
    )

    assert ir.source_track.semantic_id == "P001"
    assert ir.source_track.product_family == "bermudan_best_of_basket"
    assert ir.source_track.instrument_class == "basket_option"
    assert tuple(
        observable.source_ref
        for observable in ir.observables
        if observable.observable_kind == "spot"
    ) == ("market.spot:AAPL", "market.spot:MSFT")
    assert tuple(event.event_kind for event in ir.event_plan.events).count("observation") == 4
    assert tuple(event.event_kind for event in ir.event_plan.events).count("decision") == 3
    assert tuple(action.action_type for action in ir.decision_program.actions) == (
        "holder_max",
        "holder_max",
        "holder_max",
    )
    assert ir.settlement_program.steps[0].expression == (
        "notional * max(max(spot[AAPL], spot[MSFT]) - 100.0, 0.0)"
    )

    summary = contract_execution_summary(ir)
    assert summary["route_ids"] == ()
    assert summary["model_families"] == ()
    assert summary["observable_kinds"] == (
        "correlation_matrix",
        "curve_quote",
        "spot",
        "surface_quote",
    )
    assert summary["decision_action_types"] == ("holder_max",)
    assert summary["settlement_kinds"] == ("best_of_call_payoff",)
    assert summary["requirement_markets"] == (
        "black_vol_surface:AAPL",
        "black_vol_surface:MSFT",
        "correlation_matrix:AAPL,MSFT",
        "discount_curve:USD",
        "spot:AAPL",
        "spot:MSFT",
    )


def test_p001_benchmark_task_compiles_to_route_free_execution_ir():
    from pathlib import Path
    from dataclasses import asdict

    from trellis.agent.benchmark_contracts import benchmark_contract_execution_ir
    from trellis.agent.task_manifests import load_task_manifest

    root = Path(__file__).resolve().parents[2]
    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=root)
    }

    ir = benchmark_contract_execution_ir(tasks["P001"], root=root)
    payload_text = repr(asdict(ir))

    assert ir.source_track.semantic_id == "P001"
    assert ir.execution_metadata.metadata == (
        ("requested_outputs", ("price", "greeks", "bounds", "comparison")),
        ("validation_policy", "invariants_and_cross_method"),
    )
    assert "market.spot:AAPL" in payload_text
    assert "market.spot:MSFT" in payload_text
    assert "Asset1" not in payload_text
    assert "Asset2" not in payload_text
    assert contract_execution_summary(ir)["route_ids"] == ()


def test_p001_execution_capability_admission_admits_mc_and_blocks_lattice():
    from datetime import date

    ir = compile_bermudan_best_of_basket_execution_ir(
        semantic_id="P001",
        underliers=("AAPL", "MSFT"),
        strike=100.0,
        expiry_date=date(2025, 12, 15),
        observation_dates=(
            date(2025, 3, 15),
            date(2025, 6, 15),
            date(2025, 9, 15),
            date(2025, 12, 15),
        ),
        exercise_dates=(
            date(2025, 6, 15),
            date(2025, 9, 15),
            date(2025, 12, 15),
        ),
    )

    monte_carlo = admit_execution_capabilities(ir, method="monte_carlo")
    assert monte_carlo.admitted is True
    assert monte_carlo.blockers == ()
    assert monte_carlo.required_capabilities == (
        "multi_asset_correlated_diffusion",
        "correlation_matrix",
        "path_simulation",
        "bermudan_holder_exercise",
        "best_of_basket_payoff",
    )

    lattice = admit_execution_capabilities(ir, method="lattice")
    assert lattice.admitted is False
    assert lattice.required_capabilities == (
        "multi_asset_bermudan_state_grid",
        "bermudan_holder_exercise",
        "best_of_basket_payoff",
    )
    assert lattice.blockers[0].blocker_id == "missing_multi_asset_bermudan_state_grid"
    assert lattice.blockers[0].missing_primitive == "multi_asset_bermudan_state_grid"
    assert "short-rate lattice is not compatible" in lattice.blockers[0].message
    assert "build_rate_lattice" not in repr(lattice)

    product_grid = admit_execution_capabilities(
        ir,
        method="lattice",
        available_primitives=("multi_asset_bermudan_state_grid",),
    )
    assert product_grid.admitted is True
    assert product_grid.engine_family == "lattice"
    assert "multi_asset_bermudan_state_grid" in product_grid.matched_capabilities


def test_p001_benchmark_capability_admission_is_method_specific():
    from pathlib import Path

    from trellis.agent.benchmark_contracts import (
        benchmark_contract_execution_admission,
    )
    from trellis.agent.task_manifests import load_task_manifest

    root = Path(__file__).resolve().parents[2]
    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=root)
    }

    monte_carlo = benchmark_contract_execution_admission(
        tasks["P001"],
        method="monte_carlo",
        root=root,
    )
    lattice = benchmark_contract_execution_admission(
        tasks["P001"],
        method="lattice",
        root=root,
    )

    assert monte_carlo.admitted is True
    assert lattice.admitted is False
    assert lattice.blockers[0].blocker_id == "missing_multi_asset_bermudan_state_grid"


def test_p001_monte_carlo_visitor_prices_route_free_execution_ir():
    from datetime import date

    ir = compile_bermudan_best_of_basket_execution_ir(
        semantic_id="P001",
        underliers=("AAPL", "MSFT"),
        strike=100.0,
        notional=1.0,
        expiry_date=date(2025, 12, 15),
        observation_dates=(
            date(2025, 3, 15),
            date(2025, 6, 15),
            date(2025, 9, 15),
            date(2025, 12, 15),
        ),
        exercise_dates=(
            date(2025, 6, 15),
            date(2025, 9, 15),
            date(2025, 12, 15),
        ),
    )

    result = price_bermudan_best_of_basket_monte_carlo(
        ir,
        BermudanBestOfBasketMCInputs(
            valuation_date=date(2024, 11, 15),
            spot_values={"AAPL": 100.0, "MSFT": 95.0},
            volatilities={"AAPL": 0.20, "MSFT": 0.25},
            carry_rates={"AAPL": 0.0, "MSFT": 0.0},
            correlation_matrix=((1.0, 0.35), (0.35, 1.0)),
            risk_free_rate=0.05,
        ),
        controls=BermudanBestOfBasketMCControls(
            n_paths=6000,
            n_steps=128,
            seed=7,
        ),
    )

    assert 5.0 < result.price < 30.0
    assert result.currency == "USD"
    assert result.provenance["source_semantic_id"] == "P001"
    assert result.provenance["method"] == "monte_carlo"
    assert result.provenance["underliers"] == ("AAPL", "MSFT")
    assert result.provenance["route_ids"] == ()
    assert result.provenance["pricing_authority"] == "execution_ir_visitor"
    assert result.diagnostics["policy_class"] == "longstaff_schwartz_multistate"
    assert result.diagnostics["exercise_dates_count"] == 3
    assert "state_space" not in repr(result.provenance)
    assert "_agent/rainbow_option.py" not in repr(result.provenance)


def test_p001_benchmark_monte_carlo_execution_uses_scenario_inputs():
    from pathlib import Path

    from trellis.agent.benchmark_contracts import (
        benchmark_contract_monte_carlo_execution,
    )
    from trellis.agent.task_manifests import load_task_manifest

    root = Path(__file__).resolve().parents[2]
    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=root)
    }

    result = benchmark_contract_monte_carlo_execution(
        tasks["P001"],
        root=root,
        controls=BermudanBestOfBasketMCControls(
            n_paths=4096,
            n_steps=96,
            seed=11,
        ),
    )

    assert result.price > 0.0
    assert result.provenance["source_semantic_id"] == "P001"
    assert result.provenance["scenario_id"] == "equity_rainbow_two_asset"
    assert result.provenance["underliers"] == ("AAPL", "MSFT")


def test_p001_lattice_visitor_prices_route_free_execution_ir():
    from datetime import date

    ir = compile_bermudan_best_of_basket_execution_ir(
        semantic_id="P001",
        underliers=("AAPL", "MSFT"),
        strike=100.0,
        notional=1.0,
        expiry_date=date(2025, 12, 15),
        observation_dates=(
            date(2025, 3, 15),
            date(2025, 6, 15),
            date(2025, 9, 15),
            date(2025, 12, 15),
        ),
        exercise_dates=(
            date(2025, 6, 15),
            date(2025, 9, 15),
            date(2025, 12, 15),
        ),
    )

    result = price_bermudan_best_of_basket_lattice(
        ir,
        BermudanBestOfBasketMCInputs(
            valuation_date=date(2024, 11, 15),
            spot_values={"AAPL": 100.0, "MSFT": 95.0},
            volatilities={"AAPL": 0.20, "MSFT": 0.25},
            carry_rates={"AAPL": 0.0, "MSFT": 0.0},
            correlation_matrix=((1.0, 0.35), (0.35, 1.0)),
            risk_free_rate=0.05,
        ),
        controls=BermudanBestOfBasketLatticeControls(n_steps=48),
    )

    assert 5.0 < result.price < 30.0
    assert result.provenance["source_semantic_id"] == "P001"
    assert result.provenance["method"] == "lattice"
    assert result.provenance["pricing_authority"] == "execution_ir_visitor"
    assert result.provenance["primitive"] == "multi_asset_bermudan_state_grid"
    assert result.provenance["route_ids"] == ()
    assert result.diagnostics["rollback_policy"] == "bermudan_max"
    assert "build_rate_lattice" not in repr(result.provenance)
    assert "state_space" not in repr(result.provenance)


def test_p001_benchmark_lattice_execution_uses_product_state_grid():
    from pathlib import Path

    from trellis.agent.benchmark_contracts import (
        benchmark_contract_lattice_execution,
    )
    from trellis.agent.task_manifests import load_task_manifest

    root = Path(__file__).resolve().parents[2]
    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=root)
    }

    result = benchmark_contract_lattice_execution(
        tasks["P001"],
        root=root,
        controls=BermudanBestOfBasketLatticeControls(n_steps=48),
    )

    assert result.price > 0.0
    assert result.provenance["source_semantic_id"] == "P001"
    assert result.provenance["scenario_id"] == "equity_rainbow_two_asset"
    assert result.provenance["primitive"] == "multi_asset_bermudan_state_grid"
