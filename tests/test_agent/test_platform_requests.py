"""Tests for family-contract aware platform request compilation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trellis.agent.ask import TermSheet
from trellis.session import Session


SETTLE = date(2024, 11, 15)


def _expected_route_modules(compiled) -> tuple[str, ...]:
    from trellis.agent.calibration_contract import _KNOWN_PRIMITIVES

    blueprint = compiled.semantic_blueprint
    calibration_modules: tuple[str, ...] = ()
    calibration_step = getattr(blueprint, "calibration_step", None)
    primitive = getattr(calibration_step, "proven_primitive", "") if calibration_step is not None else ""
    if primitive:
        module = _KNOWN_PRIMITIVES.get(primitive, "")
        if module:
            calibration_modules = (module,)
    return tuple(
        dict.fromkeys(
            (
                *compiled.pricing_plan.method_modules,
                *blueprint.target_modules,
                *blueprint.dsl_lowering.helper_modules,
                *calibration_modules,
            )
        )
    )


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
    assert compiled.request.metadata["semantic_contract"]["semantic_id"] == "quanto_option"
    assert compiled.request.metadata["semantic_contract"]["semantic_concept"]["semantic_id"] == "quanto_option"
    assert compiled.request.metadata["semantic_blueprint"]["dsl_route"] == "quanto_adjustment_analytical"
    assert "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state" in (
        compiled.request.metadata["semantic_blueprint"]["dsl_helper_refs"]
    )
    assert compiled.request.metadata["semantic_blueprint"]["lane_plan"]["lane_family"] == "analytical"
    assert compiled.request.metadata["semantic_blueprint"]["lane_plan"]["plan_kind"] == "exact_target_binding"
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
    assert "trellis.models.analytical.quanto" in compiled.semantic_blueprint.target_modules
    assert "trellis.models.quanto_option" in compiled.generation_plan.approved_modules
    assert "trellis.models.resolution.quanto" in compiled.generation_plan.approved_modules
    assert compiled.semantic_blueprint.lane_plan is not None
    assert compiled.semantic_blueprint.lane_plan.lane_family == "analytical"
    assert compiled.generation_plan.lane_family == "analytical"
    assert compiled.generation_plan.lane_plan_kind == "exact_target_binding"
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "quanto_adjustment_analytical"
    assert compiled.semantic_blueprint.selection_reason == compiled.pricing_plan.selection_reason
    assert compiled.semantic_blueprint.route_modules == _expected_route_modules(compiled)


def test_compile_build_request_attaches_route_binding_authority_packet():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
    )

    authority = compiled.request.metadata["route_binding_authority"]
    backend_binding = authority["backend_binding"]

    assert authority["route_id"] == "quanto_adjustment_analytical"
    assert authority["route_family"] == "analytical"
    assert authority["authority_kind"] == "exact_backend_fit"
    assert authority["compatibility_alias_policy"] == "internal_only"
    assert backend_binding["binding_id"] == "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state"
    assert backend_binding["engine_family"] == "analytical"
    assert backend_binding["exact_backend_fit"] is True
    assert backend_binding["primitive_refs"] == [
        "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state"
    ]
    assert "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state" in backend_binding["helper_refs"]
    assert authority["validation_bundle_id"] == "analytical:quanto_option"
    assert "check_non_negativity" in authority["validation_check_ids"]
    assert backend_binding["admissibility"]["multicurrency_support"] == "native_payout_with_fx"
    assert backend_binding["admissibility_failures"] == []
    assert authority["canary_task_ids"] == ["T105"]
    assert authority["provenance"]["semantic_contract_id"] == "quanto_option"
    assert authority["provenance"]["lane_plan_kind"] == "exact_target_binding"


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


def test_compile_build_request_records_generated_skill_artifacts_in_shared_bundle():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
    )

    assert "## Generated Skills" in compiled.knowledge_text
    assert "## Routing Skills" in compiled.routing_knowledge_text
    assert "route_hint:quanto_adjustment_analytical:route-helper" in (
        compiled.knowledge_summary["selected_artifact_ids"]
    )
    assert "builder" in compiled.knowledge_summary["selected_artifacts_by_audience"]
    assert "routing" in compiled.knowledge_summary["selected_artifacts_by_audience"]


def test_compile_build_request_supports_knowledge_light_profile():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
        knowledge_profile="knowledge_light",
    )

    assert compiled.knowledge_summary["knowledge_profile"] == "knowledge_light"
    assert compiled.knowledge_text.startswith("## Knowledge-Light Mode")
    assert compiled.review_knowledge_text.startswith("## Knowledge-Light Review Mode")
    assert compiled.routing_knowledge_text.startswith("## Knowledge-Light Routing Mode")


def test_platform_request_method_specialization_uses_shared_semantic_authority():
    from trellis.agent.platform_requests import _semantic_contract_with_preferred_method
    from trellis.agent.semantic_contracts import (
        make_rate_style_swaption_contract,
        specialize_semantic_contract_for_method,
    )

    contract = make_rate_style_swaption_contract(
        description="European payer swaption",
        observation_schedule=("2026-01-15",),
        preferred_method="analytical",
    )

    specialized = _semantic_contract_with_preferred_method(
        contract,
        preferred_method="monte_carlo",
    )
    expected = specialize_semantic_contract_for_method(
        contract,
        preferred_method="monte_carlo",
    )

    assert specialized == expected


def test_compile_build_request_emits_fallback_lane_plan_for_american_tree_route():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "American put: tree vs PDE vs LSM at 3 vol levels",
        instrument_type="american_option",
        preferred_method="rate_tree",
        knowledge_profile="knowledge_light",
    )

    assert compiled.semantic_blueprint is None
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "exercise_lattice"
    assert compiled.generation_plan.lane_family == "lattice"
    assert "price_vanilla_equity_option_tree" in " ".join(compiled.generation_plan.lane_construction_steps)


def test_compile_build_request_emits_fallback_lane_plan_for_fx_monte_carlo_route():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "FX vanilla option: Garman-Kohlhagen vs MC",
        instrument_type="european_option",
        preferred_method="monte_carlo",
        knowledge_profile="knowledge_light",
    )

    assert compiled.semantic_blueprint is None
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "monte_carlo_fx_vanilla"
    assert compiled.generation_plan.lane_family == "monte_carlo"
    assert any("FXRate" in step for step in compiled.generation_plan.lane_construction_steps)


def test_compile_build_request_respects_quanto_preferred_monte_carlo_route():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
        preferred_method="monte_carlo",
    )

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == "monte_carlo"
    assert compiled.pricing_plan.method_modules == ["trellis.models.monte_carlo.engine"]
    assert compiled.generation_plan is not None
    assert "trellis.models.resolution.quanto" in compiled.generation_plan.approved_modules
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "correlated_gbm_monte_carlo"
    assert compiled.execution_plan.route_method == "monte_carlo"
    assert compiled.semantic_blueprint.selection_reason == compiled.pricing_plan.selection_reason


def test_compile_build_request_preserves_swaption_conventions_and_hw_bindings():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        (
            "European payer swaption. Expiry: 2025-11-15. Underlying: 5Y fixed-for-float interest "
            "rate swap. Strike (fixed rate): 3%. Fixed leg: semi-annual, 30/360. "
            "Float leg: quarterly 3M SOFR, Act/360. Use the USD OIS curve for discounting "
            "and the SOFR-3M forecast curve for forward rate projection."
        ),
        instrument_type="swaption",
        preferred_method="rate_tree",
    )

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.semantic_contract.product.term_fields["fixed_leg_day_count"] == "THIRTY_360"
    assert compiled.semantic_contract.product.term_fields["float_leg_day_count"] == "ACT_360"
    assert compiled.semantic_contract.product.term_fields["rate_index"] == "USD-SOFR-3M"
    assert compiled.semantic_blueprint.valuation_context is not None
    assert compiled.semantic_blueprint.valuation_context.engine_model_spec is not None
    assert compiled.semantic_blueprint.valuation_context.engine_model_spec.model_name == "hull_white_1f"
    assert (
        compiled.semantic_blueprint.valuation_context.engine_model_spec.rates_curve_roles.rate_index
        == "usd-sofr-3m"
    )
    assert compiled.semantic_blueprint.calibration_step is not None
    assert compiled.semantic_blueprint.market_binding_spec is not None
    assert "model_parameters" in compiled.semantic_blueprint.market_binding_spec.derivable_inputs


def test_compile_comparison_request_keeps_semantic_swaption_method_plans():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_comparison_request,
    )

    request = make_comparison_request(
        description=(
            "European payer swaption. Expiry: 2025-11-15. Underlying: 5Y fixed-for-float "
            "interest rate swap. Strike (fixed rate): 3%. Fixed leg: semi-annual, "
            "30/360. Float leg: quarterly 3M SOFR, Act/360. Use the USD OIS curve "
            "for discounting and the SOFR-3M forecast curve for forward rate projection."
        ),
        instrument_type="swaption",
        methods=["analytical", "rate_tree", "monte_carlo"],
        reference_method="analytical",
    )

    compiled = compile_platform_request(request)

    assert compiled.semantic_contract is not None
    assert compiled.semantic_contract.semantic_id == "rate_style_swaption"
    assert len(compiled.comparison_method_plans) == 3
    assert all(plan.semantic_blueprint is not None for plan in compiled.comparison_method_plans)
    tree_plan = next(plan for plan in compiled.comparison_method_plans if plan.preferred_method == "rate_tree")
    mc_plan = next(plan for plan in compiled.comparison_method_plans if plan.preferred_method == "monte_carlo")
    assert tree_plan.semantic_blueprint.valuation_context.engine_model_spec.model_name == "hull_white_1f"
    assert mc_plan.semantic_blueprint.valuation_context.engine_model_spec.model_name == "hull_white_1f"
    assert tree_plan.semantic_blueprint.calibration_step is not None
    assert mc_plan.semantic_blueprint.calibration_step is not None


def test_compile_comparison_request_preserves_explicit_swaption_comparison_regime_for_all_methods():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_comparison_request,
    )

    request = make_comparison_request(
        description=(
            "European payer swaption. Expiry: 2025-11-15. Underlying: 5Y fixed-for-float "
            "interest rate swap. Strike (fixed rate): 3%. Fixed leg: semi-annual, "
            "30/360. Float leg: quarterly 3M SOFR, Act/360. Use the USD OIS curve "
            "for discounting and the SOFR-3M forecast curve for forward rate projection. "
            "Hull-White model: mean reversion a=0.05, vol sigma=0.01."
        ),
        instrument_type="swaption",
        methods=["analytical", "rate_tree", "monte_carlo"],
        reference_method="analytical",
    )

    compiled = compile_platform_request(request)
    plans = {
        plan.preferred_method: plan
        for plan in compiled.comparison_method_plans
    }

    for method in ("analytical", "rate_tree", "monte_carlo"):
        blueprint = plans[method].semantic_blueprint
        assert blueprint is not None
        engine_model_spec = blueprint.valuation_context.engine_model_spec
        assert engine_model_spec is not None
        assert engine_model_spec.model_name == "hull_white_1f"
        assert engine_model_spec.parameter_overrides["mean_reversion"] == pytest.approx(0.05)
        assert engine_model_spec.parameter_overrides["sigma"] == pytest.approx(0.01)
        assert engine_model_spec.parameter_overrides["quote_family"] == "implied_vol"
        assert engine_model_spec.parameter_overrides["quote_convention"] == "black"
        assert engine_model_spec.parameter_overrides["quote_subject"] == "swaption"


def test_compile_comparison_request_does_not_treat_llm_model_name_as_pricing_model():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_comparison_request,
    )

    request = make_comparison_request(
        description=(
            "European payer swaption. Expiry: 2025-11-15. Underlying: 5Y fixed-for-float "
            "interest rate swap. Strike (fixed rate): 3%. Fixed leg: semi-annual, "
            "30/360. Float leg: quarterly 3M SOFR, Act/360. Use the USD OIS curve "
            "for discounting and the SOFR-3M forecast curve for forward rate projection. "
            "Hull-White model: mean reversion a=0.05, vol sigma=0.01."
        ),
        instrument_type="swaption",
        methods=["analytical", "rate_tree", "monte_carlo"],
        reference_method="analytical",
        model="gpt-5.4-mini",
    )

    compiled = compile_platform_request(request)
    plans = {
        plan.preferred_method: plan
        for plan in compiled.comparison_method_plans
    }

    for method in ("analytical", "rate_tree", "monte_carlo"):
        blueprint = plans[method].semantic_blueprint
        assert blueprint is not None
        assert blueprint.valuation_context.model_spec != "gpt-5.4-mini"
        assert blueprint.valuation_context.engine_model_spec is not None
        assert blueprint.valuation_context.engine_model_spec.model_name == "hull_white_1f"


def test_compile_build_request_preserves_missing_route_state_for_range_accrual_semantics():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        (
            "Range accrual note on SOFR paying 5.25% when SOFR stays between 1.50% "
            "and 3.25% on 2026-01-15, 2026-04-15, 2026-07-15, and 2026-10-15."
        ),
        instrument_type="range_accrual",
    )

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.semantic_blueprint.primitive_routes == ()
    assert compiled.semantic_blueprint.dsl_lowering.route_id is None
    assert compiled.generation_plan is not None
    assert compiled.generation_plan.primitive_plan is None
    assert "primitive_plan_not_available" in compiled.generation_plan.uncertainty_flags
    assert compiled.request.metadata["semantic_blueprint"]["dsl_lowering_errors"][0]["code"] == (
        "missing_primitive_routes"
    )
    assert "route_binding_authority" not in compiled.request.metadata


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


def test_known_family_request_requires_semantic_bridge(monkeypatch):
    from trellis.agent.platform_requests import PlatformRequest, _compile_known_family_request

    monkeypatch.setattr(
        "trellis.agent.platform_requests.family_template_as_semantic_contract",
        lambda family_id: None,
    )

    with pytest.raises(ValueError, match="has no semantic bridge"):
        _compile_known_family_request(
            family_id="quanto_option",
            request=PlatformRequest(
                request_id="test_build_001",
                request_type="build",
                entry_point="executor",
                description="Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
                instrument_type="quanto_option",
            ),
            reason="known_family_build_request",
            description="Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        )


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
            "trellis.models.callable_bond_tree",
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
            "trellis.models.rate_style_swaption",
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
    assert compiled.semantic_blueprint.route_modules == _expected_route_modules(compiled)
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


def test_compile_build_request_routes_rate_cap_family_through_semantic_contract():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Build a pricer for: Cap/floor: Black caplet stack vs MC rate simulation",
        instrument_type="cap",
    )

    assert compiled.execution_plan.reason == "semantic_contract_request"
    assert compiled.execution_plan.route_method == "analytical"
    assert compiled.semantic_contract is not None
    assert compiled.semantic_contract.semantic_id == "rate_cap_floor_strip"
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "cap"
    assert compiled.request.metadata["semantic_contract"]["semantic_id"] == "rate_cap_floor_strip"
    assert "semantic_gap" not in compiled.request.metadata


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
    "description,instrument_type,expected_reason,expected_route_method,expected_semantic_id,expected_instrument,expected_payoff_family,expected_pricing_module,expected_generation_module,expected_primitive_route",
    [
        (
            "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
            "quanto_option",
            "semantic_contract_request",
            "analytical",
            "quanto_option",
            "quanto_option",
            "vanilla_option",
            "trellis.models.black",
            "trellis.models.resolution.quanto",
            "quanto_adjustment_analytical",
        ),
        (
            "Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
            "callable_bond",
            "semantic_contract_request",
            "rate_tree",
            "callable_bond",
            "callable_bond",
            "callable_fixed_income",
            "trellis.models.trees.lattice",
            "trellis.models.callable_bond_tree",
            "exercise_lattice",
        ),
        (
            "European equity call on AAPL with strike 120 and expiry 2025-11-15",
            "european_option",
            "semantic_contract_request",
            "analytical",
            "vanilla_option",
            "european_option",
            "vanilla_option",
            "trellis.models.black",
            "trellis.models.black",
            "analytical_black76",
        ),
        (
            "European swaption on a fixed-for-floating swap with expiry 2026-01-15",
            "swaption",
            "semantic_contract_request",
            "analytical",
            "rate_style_swaption",
            "swaption",
            "swaption",
            "trellis.models.black",
            "trellis.models.rate_style_swaption",
            "analytical_black76",
        ),
    ],
)
def test_representative_derivatives_use_generic_semantic_contracts(
    description,
    instrument_type,
    expected_reason,
    expected_route_method,
    expected_semantic_id,
    expected_instrument,
    expected_payoff_family,
    expected_pricing_module,
    expected_generation_module,
    expected_primitive_route,
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

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.request.metadata["semantic_contract"]["semantic_id"] == expected_semantic_id
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == expected_instrument
    assert compiled.product_ir.payoff_family == expected_payoff_family
    assert compiled.product_ir.payoff_family != "basket_path_payoff"
    assert compiled.execution_plan.reason == expected_reason
    assert compiled.execution_plan.route_method == expected_route_method
    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == expected_route_method
    assert expected_pricing_module in compiled.pricing_plan.method_modules
    assert compiled.semantic_contract.product.semantic_id == expected_semantic_id
    assert compiled.semantic_contract.product.instrument_class == expected_instrument
    assert compiled.semantic_contract.product.payoff_family == expected_payoff_family
    assert compiled.semantic_contract.product.underlier_structure
    assert compiled.semantic_blueprint.selection_reason == compiled.pricing_plan.selection_reason
    assert compiled.semantic_blueprint.assumption_summary == compiled.pricing_plan.assumption_summary
    assert compiled.semantic_blueprint.route_modules == _expected_route_modules(compiled)
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


@pytest.mark.parametrize(
    "preferred_method,expected_route,expected_expr_kind",
    [
        ("analytical", "credit_default_swap_analytical", "ThenExpr"),
        ("monte_carlo", "credit_default_swap_monte_carlo", "ThenExpr"),
    ],
)
def test_compile_build_request_uses_credit_default_swap_semantic_contract_blueprint(
    preferred_method,
    expected_route,
    expected_expr_kind,
):
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        (
            "Single-name CDS on ACME with premium dates "
            "2026-06-20, 2026-09-20, 2026-12-20, 2027-03-20, 2027-06-20"
        ),
        instrument_type="credit_default_swap",
        preferred_method=preferred_method,
    )

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.request.metadata["semantic_contract"]["semantic_id"] == "credit_default_swap"
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "cds"
    assert compiled.product_ir.payoff_family == "credit_default_swap"
    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == preferred_method
    assert compiled.semantic_blueprint.route_modules == _expected_route_modules(compiled)
    assert compiled.semantic_blueprint.primitive_routes == (expected_route,)
    assert compiled.request.metadata["semantic_blueprint"]["dsl_route"] == expected_route
    assert compiled.request.metadata["semantic_blueprint"]["dsl_family_ir_type"] == "CreditDefaultSwapIR"
    assert compiled.request.metadata["semantic_blueprint"]["dsl_expr_kind"] == expected_expr_kind
    assert (
        compiled.request.metadata["semantic_blueprint"]["dsl_family_ir"]["schedule_builder_symbol"]
        == "build_cds_schedule"
    )
    assert "trellis.models.credit_default_swap" in compiled.semantic_blueprint.target_modules


def test_compile_build_request_uses_nth_to_default_semantic_contract_blueprint():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "First-to-default basket on ACME, BRAVO, CHARLIE, DELTA, ECHO maturing 2029-11-15",
        instrument_type="nth_to_default",
    )

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert compiled.request.metadata["semantic_contract"]["semantic_id"] == "nth_to_default"
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "nth_to_default"
    assert compiled.product_ir.payoff_family == "nth_to_default"
    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == "copula"
    assert compiled.semantic_blueprint.route_modules == _expected_route_modules(compiled)
    assert compiled.semantic_blueprint.primitive_routes == ("nth_to_default_monte_carlo",)
    assert compiled.request.metadata["semantic_blueprint"]["dsl_route"] == "nth_to_default_monte_carlo"
    assert compiled.request.metadata["semantic_blueprint"]["dsl_family_ir_type"] == "NthToDefaultIR"
    assert compiled.request.metadata["semantic_blueprint"]["dsl_expr_kind"] == "ContractAtom"
    assert (
        compiled.request.metadata["semantic_blueprint"]["dsl_family_ir"]["helper_symbol"]
        == "price_nth_to_default_basket"
    )
    assert "trellis.instruments.nth_to_default" in compiled.semantic_blueprint.target_modules
