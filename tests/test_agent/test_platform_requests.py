"""Tests for family-contract aware platform request compilation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trellis.agent.ask import TermSheet
from trellis.session import Session


SETTLE = date(2024, 11, 15)


def _semantic_regression_snapshot(compiled):
    from trellis.agent.semantic_contracts import semantic_contract_summary

    semantic_contract = compiled.semantic_contract
    semantic_blueprint = compiled.semantic_blueprint
    product_ir = compiled.product_ir
    generation_plan = compiled.generation_plan
    return {
        "semantic_contract": semantic_contract_summary(semantic_contract)
        if semantic_contract is not None
        else None,
        "semantic_blueprint": None
        if semantic_blueprint is None
        else {
            "semantic_id": semantic_blueprint.semantic_id,
            "preferred_method": semantic_blueprint.preferred_method,
            "selection_reason": semantic_blueprint.selection_reason,
            "assumption_summary": semantic_blueprint.assumption_summary,
            "primitive_routes": semantic_blueprint.primitive_routes,
            "route_modules": semantic_blueprint.route_modules,
            "target_modules": semantic_blueprint.target_modules,
            "dsl_helper_refs": semantic_blueprint.dsl_lowering.helper_refs,
            "required_market_data": semantic_blueprint.required_market_data,
        },
        "product_ir": None
        if product_ir is None
        else {
            "instrument": product_ir.instrument,
            "payoff_family": product_ir.payoff_family,
            "required_market_data": tuple(sorted(product_ir.required_market_data)),
            "candidate_engine_families": product_ir.candidate_engine_families,
            "supported": product_ir.supported,
        },
        "execution_plan": {
            "action": compiled.execution_plan.action,
            "reason": compiled.execution_plan.reason,
            "route_method": compiled.execution_plan.route_method,
            "requires_build": compiled.execution_plan.requires_build,
        },
        "approved_modules": tuple(generation_plan.approved_modules) if generation_plan else (),
    }


def test_compile_build_request_uses_quanto_semantic_contract_blueprint():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
    )

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.family_blueprint is None
    assert compiled.request.metadata["semantic_contract"]["semantic_id"] == "quanto_option"
    assert compiled.request.metadata["semantic_contract"]["semantic_concept"]["semantic_id"] == "quanto_option"
    assert compiled.request.metadata["semantic_blueprint"]["dsl_route"] == "quanto_adjustment_analytical"
    assert "trellis.models.analytical.quanto.price_quanto_option_analytical" in (
        compiled.request.metadata["semantic_blueprint"]["dsl_helper_refs"]
    )
    assert compiled.request.metadata["semantic_role_ownership"]["selected_stage"] == "route_assembly"
    assert compiled.request.metadata["semantic_role_ownership"]["selected_role"] == "quant"
    assert compiled.request.metadata["semantic_role_ownership"]["artifact_kind"] == "GenerationPlan"
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "quanto_option"
    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == "analytical"
    assert compiled.generation_plan is not None
    assert compiled.execution_plan.reason == "semantic_contract_request"
    assert "trellis.models.resolution.quanto" in compiled.semantic_blueprint.target_modules
    assert "trellis.models.resolution.quanto" in compiled.generation_plan.approved_modules
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "quanto_adjustment_analytical"
    assert compiled.semantic_blueprint.selection_reason == compiled.pricing_plan.selection_reason
    assert compiled.semantic_blueprint.route_modules == tuple(
        dict.fromkeys(
            (
                *compiled.pricing_plan.method_modules,
                *compiled.semantic_blueprint.target_modules,
                *compiled.semantic_blueprint.dsl_lowering.helper_modules,
            )
        )
    )


def test_compile_build_request_preserves_quanto_semantic_binding_hints():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
    )

    assert compiled.semantic_blueprint is not None
    assert "model_parameters" in compiled.semantic_blueprint.connector_binding_hints
    assert compiled.semantic_blueprint.connector_binding_hints["model_parameters"]["capability"] == "model_parameters"
    assert "quanto_correlation" in compiled.semantic_blueprint.connector_binding_hints["model_parameters"]["aliases"]
    assert compiled.semantic_blueprint.valuation_context is not None
    assert compiled.semantic_blueprint.required_data_spec.required_input_ids == compiled.semantic_blueprint.required_market_data
    assert (
        compiled.semantic_blueprint.market_binding_spec.reporting_currency
        == compiled.semantic_blueprint.valuation_context.reporting_policy.reporting_currency
    )
    assert compiled.request.metadata["semantic_blueprint"]["valuation_context"]["market_source"] == "unbound_market_snapshot"
    assert compiled.semantic_blueprint.required_market_data == (
        "discount_curve",
        "forward_curve",
        "underlier_spot",
        "black_vol_surface",
        "fx_rates",
        "model_parameters",
    )
    assert compiled.semantic_blueprint.derivable_market_data == ()


def test_compile_build_request_respects_quanto_preferred_monte_carlo_route():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
        preferred_method="monte_carlo",
    )

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.family_blueprint is None
    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == "monte_carlo"
    assert compiled.pricing_plan.method_modules == ["trellis.models.monte_carlo.engine"]
    assert compiled.generation_plan is not None
    assert "trellis.models.resolution.quanto" in compiled.generation_plan.approved_modules
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "correlated_gbm_monte_carlo"
    assert compiled.execution_plan.route_method == "monte_carlo"
    assert compiled.semantic_blueprint.selection_reason == compiled.pricing_plan.selection_reason


def test_compile_term_sheet_request_uses_quanto_semantic_contract():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_term_sheet_request,
    )

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    term_sheet = TermSheet(
        instrument_type="quanto_option",
        notional=1_000_000,
        currency="USD",
        parameters={
            "underlier": "SAP",
            "underlier_currency": "EUR",
            "domestic_currency": "USD",
            "strike": 120.0,
            "expiry_date": "2025-11-15",
        },
    )

    request = make_term_sheet_request(
        description="Quanto option on SAP settled in USD",
        term_sheet=term_sheet,
        session=session,
        measures=["price", "vega"],
    )
    compiled = compile_platform_request(request)

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.family_blueprint is None
    assert compiled.request.requested_outputs == ("price", "vega")
    assert compiled.semantic_blueprint.requested_outputs == ("price", "vega")
    assert compiled.request.metadata["semantic_contract"]["semantic_id"] == "quanto_option"
    assert compiled.request.metadata["semantic_blueprint"]["requested_outputs"] == ["price", "vega"]
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "quanto_option"
    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == "analytical"
    assert compiled.execution_plan.route_method == "analytical"
    assert compiled.execution_plan.reason == "semantic_contract_request"


@pytest.mark.parametrize(
    "description,instrument_type,term_sheet_kwargs,expected_semantic_id,expected_generation_module,expected_route",
    [
        (
            "European call on AAPL settled in USD",
            "european_option",
            {
                "instrument_type": "european_option",
                "notional": 1_000_000,
                "currency": "USD",
                "parameters": {
                    "underlier": "AAPL",
                    "strike": 120.0,
                    "expiry_date": "2025-11-15",
                },
            },
            "vanilla_option",
            "trellis.models.black",
            "analytical_black76",
        ),
        (
            "Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
            "callable_bond",
            {
                "instrument_type": "callable_bond",
                "notional": 1_000_000,
                "currency": "USD",
                "parameters": {
                    "coupon": 0.05,
                    "start_date": "2025-01-15",
                    "end_date": "2030-01-15",
                    "call_dates": "2026-01-15, 2027-01-15",
                },
            },
            "callable_bond",
            "trellis.models.trees.lattice",
            "exercise_lattice",
        ),
        (
            "European payer swaption with expiry 2026-01-15",
            "swaption",
            {
                "instrument_type": "swaption",
                "notional": 1_000_000,
                "currency": "USD",
                "parameters": {
                    "expiry_date": "2026-01-15",
                },
            },
            "rate_style_swaption",
            "trellis.models.black",
            "analytical_black76",
        ),
    ],
)
def test_representative_term_sheet_requests_use_the_semantic_path(
    description,
    instrument_type,
    term_sheet_kwargs,
    expected_semantic_id,
    expected_generation_module,
    expected_route,
):
    from trellis.agent.platform_requests import compile_platform_request, make_term_sheet_request

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    request = make_term_sheet_request(
        description=description,
        term_sheet=TermSheet(**term_sheet_kwargs),
        session=session,
    )
    compiled = compile_platform_request(request)

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.family_blueprint is None
    assert compiled.request.metadata["semantic_contract"]["semantic_id"] == expected_semantic_id
    assert compiled.execution_plan.reason == "semantic_contract_request"
    assert compiled.execution_plan.route_method in {"analytical", "rate_tree"}
    assert expected_generation_module in compiled.semantic_blueprint.target_modules
    assert expected_generation_module in compiled.generation_plan.approved_modules
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == expected_route


def test_mountain_range_request_drafts_ranked_observation_contract():
    from trellis.agent.platform_requests import compile_build_request

    description = (
        "Himalaya-style ranked observation basket on AAPL, MSFT, NVDA with observation dates "
        "2025-01-15, 2025-02-15, 2025-03-15. At each observation choose the best performer "
        "among the remaining constituents, remove it, lock the simple return, and settle the "
        "average locked returns at maturity."
    )
    compiled = compile_build_request(description, instrument_type="basket_option")
    compiled_again = compile_build_request(description, instrument_type="basket_option")
    snapshot = _semantic_regression_snapshot(compiled)
    snapshot_again = _semantic_regression_snapshot(compiled_again)

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.family_blueprint is None
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "basket_path_payoff"
    assert compiled.product_ir.payoff_family == "basket_path_payoff"
    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == "monte_carlo"
    assert compiled.execution_plan.reason == "semantic_contract_request"
    assert compiled.semantic_contract.product.selection_operator == "best_of_remaining"
    assert compiled.semantic_contract.product.selection_scope == "remaining_constituents"
    assert compiled.request.metadata["semantic_contract"]["semantic_concept"]["semantic_id"] == "ranked_observation_basket"
    assert "basket_option" in compiled.request.metadata["semantic_contract"]["semantic_concept"]["compatibility_wrappers"]
    assert compiled.semantic_blueprint is not None
    assert compiled.semantic_blueprint.primitive_routes == ("correlated_basket_monte_carlo",)
    assert compiled.semantic_blueprint.selection_reason == compiled.pricing_plan.selection_reason
    assert compiled.semantic_blueprint.assumption_summary == compiled.pricing_plan.assumption_summary
    assert compiled.semantic_blueprint.target_modules == (
        "trellis.models.resolution.basket_semantics",
        "trellis.models.monte_carlo.semantic_basket",
    )
    assert compiled.semantic_blueprint.route_modules == tuple(
        dict.fromkeys(
            (
                *compiled.pricing_plan.method_modules,
                *compiled.semantic_blueprint.target_modules,
                *compiled.semantic_blueprint.dsl_lowering.helper_modules,
            )
        )
    )
    assert snapshot == snapshot_again
    assert snapshot["semantic_contract"] == compiled.request.metadata["semantic_contract"]
    assert "trellis.models.resolution.basket_semantics" in snapshot["approved_modules"]
    assert "trellis.models.monte_carlo.semantic_basket" in snapshot["approved_modules"]
    assert "trellis.models.monte_carlo.engine" in snapshot["semantic_blueprint"]["route_modules"]
    assert "trellis.models.processes.correlated_gbm" not in snapshot["approved_modules"]
    assert "himalaya_option" not in repr(snapshot).lower()
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "correlated_basket_monte_carlo"


def test_novel_request_falls_back_with_semantic_gap_metadata():
    from trellis.agent.platform_requests import compile_build_request

    description = (
        "Price a resettable memory note with a holiday-adjusted schedule and monthly coupons."
    )
    compiled = compile_build_request(
        description,
        instrument_type="structured_note",
    )
    compiled_again = compile_build_request(
        description,
        instrument_type="structured_note",
    )

    assert compiled.semantic_contract is None
    assert compiled.semantic_blueprint is None
    assert compiled.family_blueprint is None
    assert compiled.request.metadata["semantic_gap"] == compiled_again.request.metadata["semantic_gap"]
    assert compiled.request.metadata["semantic_gap"]["instrument_type"] == "structured_note"
    assert compiled.request.metadata["semantic_gap"]["requires_clarification"] is False
    assert compiled.request.metadata["semantic_gap"]["can_use_mock_inputs"] is True
    assert "missing_semantic_contract_field" in compiled.request.metadata["semantic_gap"]["gap_types"]
    assert "generate_schedule" in compiled.request.metadata["semantic_gap"]["missing_runtime_primitives"]
    assert "observation_schedule" in compiled.request.metadata["semantic_gap"]["missing_contract_fields"]
    assert compiled.execution_plan.reason == "free_form_build_request"
    assert compiled.product_ir is not None
    assert compiled.pricing_plan is not None
    assert "semantic_gap" in compiled.request.metadata


def test_novel_request_persists_semantic_extension_trace(monkeypatch, tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    import trellis.agent.knowledge.promotion as promotion_mod
    import yaml

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    monkeypatch.setattr(promotion_mod, "_KNOWLEDGE_DIR", knowledge_root)
    monkeypatch.setattr(promotion_mod, "_LESSONS_DIR", knowledge_root / "lessons")
    monkeypatch.setattr(promotion_mod, "_TRACES_DIR", knowledge_root / "traces")
    monkeypatch.setattr(
        promotion_mod,
        "_SEMANTIC_EXTENSION_TRACES_DIR",
        knowledge_root / "traces" / "semantic_extensions",
    )
    monkeypatch.setattr(promotion_mod, "_INDEX_PATH", knowledge_root / "lessons" / "index.yaml")
    monkeypatch.setattr(promotion_mod, "_REPO_ROOT", tmp_path)

    description = (
        "Price a resettable memory note with a holiday-adjusted schedule and monthly coupons."
    )
    compiled = compile_build_request(
        description,
        instrument_type="structured_note",
    )

    assert compiled.request.metadata["semantic_role_ownership"]["selected_stage"] == "primitive_proposal"
    assert compiled.request.metadata["semantic_role_ownership"]["selected_role"] == "quant"
    assert compiled.request.metadata["semantic_role_ownership"]["artifact_kind"] == "SemanticExtensionProposal"
    assert compiled.request.metadata["semantic_extension"]["decision"] == "new_primitive"
    assert compiled.request.metadata["semantic_extension"]["recommended_next_step"]
    assert compiled.request.metadata["semantic_extension_trace"].endswith(".yaml")
    assert Path(compiled.request.metadata["semantic_extension_trace"]).exists()
    assert compiled.request.metadata["semantic_extension"]["trace_key"]
    trace_data = yaml.safe_load(Path(compiled.request.metadata["semantic_extension_trace"]).read_text())
    assert trace_data["semantic_role_ownership"]["selected_stage"] == "trace_handoff"
    assert trace_data["semantic_role_ownership"]["selected_role"] == "knowledge_agent"
    assert trace_data["semantic_role_ownership"]["artifact_kind"] == "semantic_extension_trace"


def test_callable_bond_request_replays_as_supported_after_schedule_primitive_is_added(
    monkeypatch,
    tmp_path,
):
    from trellis.agent.platform_requests import compile_build_request
    import trellis.agent.knowledge.promotion as promotion_mod
    import trellis.agent.semantic_contracts as semantic_contracts_mod

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    monkeypatch.setattr(promotion_mod, "_KNOWLEDGE_DIR", knowledge_root)
    monkeypatch.setattr(promotion_mod, "_LESSONS_DIR", knowledge_root / "lessons")
    monkeypatch.setattr(promotion_mod, "_TRACES_DIR", knowledge_root / "traces")
    monkeypatch.setattr(
        promotion_mod,
        "_SEMANTIC_EXTENSION_TRACES_DIR",
        knowledge_root / "traces" / "semantic_extensions",
    )
    monkeypatch.setattr(promotion_mod, "_INDEX_PATH", knowledge_root / "lessons" / "index.yaml")
    monkeypatch.setattr(promotion_mod, "_REPO_ROOT", tmp_path)

    description = "Callable bond with annual coupons"
    compiled = compile_build_request(
        description,
        instrument_type="callable_bond",
    )

    assert compiled.semantic_contract is None
    assert compiled.request.metadata["semantic_extension"]["decision"] == "new_primitive"
    assert compiled.request.metadata["semantic_extension"]["semantic_concept"]["semantic_id"] == "callable_bond"

    monkeypatch.setattr(
        semantic_contracts_mod,
        "_split_supported_dates",
        lambda *args, **kwargs: ("2026-01-15",),
    )

    replayed = compile_build_request(
        description,
        instrument_type="callable_bond",
    )

    assert replayed.semantic_contract is not None
    assert replayed.request.metadata["semantic_contract"]["semantic_id"] == "callable_bond"
    assert replayed.request.metadata["semantic_role_ownership"]["selected_stage"] == "route_assembly"
    assert replayed.execution_plan.reason == "semantic_contract_request"
    assert replayed.generation_plan is not None
    assert replayed.generation_plan.new_primitive_workflow is None
    assert "semantic_gap" not in replayed.request.metadata


@pytest.mark.parametrize(
    "description,instrument_type,expected_reason,expected_route_method,expected_family_id,expected_semantic_id,expected_instrument,expected_payoff_family,expected_pricing_module,expected_generation_module,expected_primitive_route,expect_semantic_contract",
    [
        (
            "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
            "quanto_option",
            "semantic_contract_request",
            "analytical",
            None,
            "quanto_option",
            "quanto_option",
            "vanilla_option",
            "trellis.models.black",
            "trellis.models.resolution.quanto",
            "quanto_adjustment_analytical",
            True,
        ),
        (
            "Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
            "callable_bond",
            "semantic_contract_request",
            "rate_tree",
            None,
            "callable_bond",
            "callable_bond",
            "callable_fixed_income",
            "trellis.models.trees.lattice",
            "trellis.models.trees.lattice",
            "exercise_lattice",
            True,
        ),
        (
            "European equity call on AAPL with strike 120 and expiry 2025-11-15",
            "european_option",
            "semantic_contract_request",
            "analytical",
            None,
            "vanilla_option",
            "european_option",
            "vanilla_option",
            "trellis.models.black",
            "trellis.models.black",
            "analytical_black76",
            True,
        ),
        (
            "European swaption on a fixed-for-floating swap with expiry 2026-01-15",
            "swaption",
            "semantic_contract_request",
            "analytical",
            None,
            "rate_style_swaption",
            "swaption",
            "swaption",
            "trellis.models.black",
            "trellis.models.black",
            "analytical_black76",
            True,
        ),
    ],
)
def test_representative_derivatives_use_generic_semantic_contracts(
    description,
    instrument_type,
    expected_reason,
    expected_route_method,
    expected_family_id,
    expected_semantic_id,
    expected_instrument,
    expected_payoff_family,
    expected_pricing_module,
    expected_generation_module,
    expected_primitive_route,
    expect_semantic_contract,
):
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        description,
        instrument_type=instrument_type,
    )
    compiled_again = compile_build_request(
        description,
        instrument_type=instrument_type,
    )
    snapshot = _semantic_regression_snapshot(compiled)
    snapshot_again = _semantic_regression_snapshot(compiled_again)

    if expect_semantic_contract:
        assert compiled.semantic_contract is not None
        assert compiled.semantic_blueprint is not None
        assert compiled.family_blueprint is None
        assert compiled.request.metadata["semantic_contract"]["semantic_id"] == expected_semantic_id
    else:
        assert compiled.semantic_contract is None
        assert compiled.semantic_blueprint is None
        assert compiled.request.metadata.get("semantic_contract") is None
        assert compiled.family_blueprint is not None
        assert compiled.family_blueprint.family_id == expected_family_id
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == expected_instrument
    assert compiled.product_ir.payoff_family == expected_payoff_family
    assert compiled.product_ir.payoff_family != "basket_path_payoff"
    assert compiled.execution_plan.reason == expected_reason
    assert compiled.execution_plan.route_method == expected_route_method
    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == expected_route_method
    assert expected_pricing_module in compiled.pricing_plan.method_modules
    if expect_semantic_contract:
        assert compiled.semantic_blueprint is not None
        assert compiled.semantic_contract.product.semantic_id == expected_semantic_id
        assert compiled.semantic_contract.product.instrument_class == expected_instrument
        assert compiled.semantic_contract.product.payoff_family == expected_payoff_family
        assert compiled.semantic_contract.product.underlier_structure
        assert compiled.semantic_blueprint.selection_reason == compiled.pricing_plan.selection_reason
        assert compiled.semantic_blueprint.assumption_summary == compiled.pricing_plan.assumption_summary
        assert compiled.semantic_blueprint.route_modules == tuple(
            dict.fromkeys(
                (
                    *compiled.pricing_plan.method_modules,
                    *compiled.semantic_blueprint.target_modules,
                    *compiled.semantic_blueprint.dsl_lowering.helper_modules,
                )
            )
        )
        assert compiled.semantic_blueprint.primitive_routes == (expected_primitive_route,)
        assert compiled.request.metadata["semantic_blueprint"]["dsl_route"] == expected_primitive_route
        assert expected_generation_module in compiled.semantic_blueprint.target_modules
        assert snapshot == snapshot_again
        assert snapshot["semantic_contract"] == compiled.request.metadata["semantic_contract"]
        assert snapshot["semantic_blueprint"]["selection_reason"] == compiled.pricing_plan.selection_reason
        assert snapshot["semantic_blueprint"]["assumption_summary"] == compiled.pricing_plan.assumption_summary
        assert "trellis.models.resolution.basket_semantics" not in snapshot["approved_modules"]
        assert "trellis.models.monte_carlo.semantic_basket" not in snapshot["approved_modules"]
        assert "himalaya_option" not in repr(snapshot).lower()
    else:
        assert compiled.semantic_blueprint is None
    if expected_generation_module is not None:
        assert expected_generation_module in compiled.generation_plan.approved_modules
    assert "trellis.models.resolution.basket_semantics" not in compiled.generation_plan.approved_modules
    assert "trellis.models.monte_carlo.semantic_basket" not in compiled.generation_plan.approved_modules
    assert "himalaya_option" not in repr(compiled).lower()


def test_request_missing_schedule_returns_semantic_error():
    from trellis.agent.platform_requests import compile_build_request

    with pytest.raises(ValueError, match="observation schedule"):
        compile_build_request(
            "Himalaya-style ranked observation basket on AAPL, MSFT, NVDA with best-performer removal and maturity settlement, but no observation dates were provided.",
            instrument_type="basket_option",
        )
