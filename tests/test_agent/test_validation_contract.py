"""Tests for compiled validation-contract metadata."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


_RANKED_BASKET_DESCRIPTION = (
    "Himalaya-style ranked observation basket on AAPL, MSFT, NVDA with observation dates "
    "2025-01-15, 2025-02-15, 2025-03-15. At each observation choose the best performer "
    "among the remaining constituents, remove it, lock the simple return, and settle the "
    "average locked returns at maturity."
)


def test_compile_build_request_attaches_validation_contract_summary():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
    )

    contract = compiled.validation_contract

    assert contract is not None
    assert contract.bundle_id == "analytical:quanto_option"
    assert (
        contract.backend_binding_id
        == "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state"
    )
    assert (
        contract.exact_bundle_id
        == "analytical:quanto_option@trellis.models.quanto_option.price_quanto_option_analytical_from_market_state"
    )
    assert contract.route_id == "quanto_adjustment_analytical"
    assert contract.route_family == "analytical"
    assert {check.check_id for check in contract.deterministic_checks} >= {
        "check_non_negativity",
        "check_price_sanity",
        "quanto_adjustment_applied",
        "fx_conversion_applied_before_settlement",
    }
    assert "black_vol_surface" in contract.required_market_data
    assert compiled.request.metadata["validation_contract"]["bundle_id"] == contract.bundle_id
    assert (
        compiled.request.metadata["validation_contract"]["backend_binding_id"]
        == contract.backend_binding_id
    )
    assert (
        compiled.request.metadata["validation_contract"]["exact_bundle_id"]
        == contract.exact_bundle_id
    )
    assert compiled.request.metadata["validation_contract"]["route_id"] == contract.route_id
    assert (
        compiled.request.metadata["validation_contract"]["deterministic_checks"][0]["check_id"]
        == contract.deterministic_checks[0].check_id
    )


def test_callable_bond_validation_contract_carries_reference_bound_relation():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        instrument_type="callable_bond",
    )

    contract = compiled.validation_contract

    assert contract is not None
    assert contract.bundle_id == "rate_tree:callable_bond"
    assert "check_bounded_by_reference" in {
        check.check_id for check in contract.deterministic_checks
    }
    reference_relations = {
        relation.target_id: relation.relation
        for relation in contract.comparison_relations
    }
    assert reference_relations["bounded_by_reference"] == "<="
    bounded_check = next(
        check
        for check in contract.deterministic_checks
        if check.check_id == "check_bounded_by_reference"
    )
    assert bounded_check.harness_requirements == (
        "payoff_factory",
        "reference_factory",
        "market_state_factory",
    )


def test_puttable_bond_validation_contract_uses_lower_bound_relation():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Puttable bond with annual coupons and holder put dates 2026-01-15, 2027-01-15",
        instrument_type="puttable_bond",
    )

    contract = compiled.validation_contract

    assert contract is not None
    bounded_check = next(
        check
        for check in contract.deterministic_checks
        if check.check_id == "check_bounded_by_reference"
    )
    assert bounded_check.relation == ">="
    reference_relations = {
        relation.target_id: relation.relation
        for relation in contract.comparison_relations
    }
    assert reference_relations["bounded_by_reference"] == ">="


def test_swaption_validation_contract_carries_helper_consistency_check():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "European swaption on a fixed-for-floating swap with expiry 2026-01-15",
        instrument_type="swaption",
    )

    contract = compiled.validation_contract

    assert contract is not None
    helper_check = next(
        check
        for check in contract.deterministic_checks
        if check.check_id == "check_rate_style_swaption_helper_consistency"
    )
    assert helper_check.category == "route_specific"
    assert helper_check.relation == "within_tolerance"
    assert helper_check.harness_requirements == (
        "payoff_factory",
        "market_state_factory",
    )


def test_compile_validation_contract_prefers_more_specific_product_family_over_generic_request_family():
    from trellis.agent.validation_contract import compile_validation_contract

    contract = compile_validation_contract(
        pricing_plan=type("PricingPlan", (), {"method": "analytical", "required_market_data": {"discount_curve", "black_vol_surface"}})(),
        instrument_type="european_option",
        product_ir=type("ProductIR", (), {"instrument": "zcb_option", "required_market_data": {"discount_curve", "black_vol_surface"}})(),
    )

    assert contract is not None
    assert contract.bundle_id == "analytical:zcb_option"
    assert contract.backend_binding_id is None
    assert contract.exact_bundle_id is None
    assert contract.instrument_type == "zcb_option"


def test_route_less_semantic_request_keeps_validation_contract_truthful():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        (
            "Range accrual note on SOFR paying 5.25% when SOFR stays between 1.50% "
            "and 3.25% on 2026-01-15, 2026-04-15, 2026-07-15, and 2026-10-15."
        ),
        instrument_type="range_accrual",
    )

    contract = compiled.validation_contract

    assert contract is not None
    assert contract.backend_binding_id is None
    assert contract.exact_bundle_id is None
    assert contract.route_id is None
    assert contract.route_family is None
    check_ids = {check.check_id for check in contract.deterministic_checks}
    assert "fixing_history_bound_to_past_schedule_points" in check_ids
    assert "principal_redeems_at_maturity" in check_ids
    assert "check_vol_sensitivity" not in check_ids
    assert "check_vol_monotonicity" not in check_ids
    assert contract.lowering_errors == (
        "route_selection:missing_primitive_routes:No primitive routes declared for DSL lowering.",
    )
    assert contract.review_hints["has_lowering_errors"] is True


@pytest.mark.parametrize(
    "description,instrument_type,expected_semantic_id,expected_route,expected_bundle,required_inputs,expected_admissibility",
    [
        (
            "European call on AAPL with strike 120 and expiry 2025-11-15",
            "european_option",
            "vanilla_option",
            "analytical_black76",
            "analytical:european_option",
            {"discount_curve", "underlier_spot", "black_vol_surface"},
            (),
        ),
        (
            "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
            "quanto_option",
            "quanto_option",
            "quanto_adjustment_analytical",
            "analytical:quanto_option",
            {
                "discount_curve",
                "forward_curve",
                "underlier_spot",
                "black_vol_surface",
                "fx_rates",
                "model_parameters",
            },
            (),
        ),
        (
            "Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
            "callable_bond",
            "callable_bond",
            "exercise_lattice",
            "rate_tree:callable_bond",
            {"discount_curve", "black_vol_surface"},
            (),
        ),
        (
            "European swaption on a fixed-for-floating swap with expiry 2026-01-15",
            "swaption",
            "rate_style_swaption",
            "analytical_black76",
            "analytical:swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            (),
        ),
        (
            (
                "Single-name CDS on ACME with premium dates "
                "2026-06-20, 2026-09-20, 2026-12-20, 2027-03-20, 2027-06-20"
            ),
            "credit_default_swap",
            "credit_default_swap",
            "credit_default_swap_analytical",
            "analytical:credit_default_swap",
            {"discount_curve", "credit_curve"},
            (),
        ),
        (
            "First-to-default basket on ACME, BRAVO, CHARLIE, DELTA, ECHO maturing 2029-11-15",
            "nth_to_default",
            "nth_to_default",
            "nth_to_default_monte_carlo",
            "copula:nth_to_default",
            {"discount_curve", "credit_curve"},
            (),
        ),
        (
            _RANKED_BASKET_DESCRIPTION,
            "basket_option",
            "ranked_observation_basket",
            "correlated_basket_monte_carlo",
            "monte_carlo:basket_option",
            {
                "discount_curve",
                "underlier_spots",
                "black_vol_surface",
                "correlation_matrix",
            },
            (),
        ),
    ],
)
def test_representative_semantic_requests_compile_stable_validation_contracts(
    description,
    instrument_type,
    expected_semantic_id,
    expected_route,
    expected_bundle,
    required_inputs,
    expected_admissibility,
):
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(description, instrument_type=instrument_type)
    compiled_again = compile_build_request(description, instrument_type=instrument_type)

    contract = compiled.validation_contract
    contract_again = compiled_again.validation_contract

    assert compiled.semantic_contract is not None
    assert compiled.semantic_blueprint is not None
    assert contract is not None
    assert contract_again is not None

    assert compiled.semantic_contract.semantic_id == expected_semantic_id
    assert compiled.request.metadata["semantic_contract"]["semantic_id"] == expected_semantic_id
    assert compiled.request.metadata["semantic_blueprint"]["dsl_route"] == expected_route
    assert contract.route_id == expected_route
    assert contract.bundle_id == expected_bundle
    assert contract.backend_binding_id
    assert contract.exact_bundle_id == f"{expected_bundle}@{contract.backend_binding_id}"
    assert set(contract.required_market_data) == required_inputs
    assert contract.lowering_errors == ()
    assert contract.admissibility_failures == expected_admissibility
    assert contract.review_hints["has_lowering_errors"] is False
    assert contract.review_hints["has_admissibility_failures"] == bool(expected_admissibility)
    assert contract.review_hints["has_exact_validation_identity"] is True

    assert contract == contract_again
    assert compiled.request.metadata["validation_contract"]["bundle_id"] == expected_bundle
    assert (
        compiled.request.metadata["validation_contract"]["exact_bundle_id"]
        == contract.exact_bundle_id
    )
    assert compiled.request.metadata["validation_contract"]["route_id"] == expected_route


def test_platform_trace_persists_validation_contract_summary(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import load_platform_traces, record_platform_trace

    compiled = compile_build_request(
        (
            "Single-name CDS on ACME with premium dates "
            "2026-06-20, 2026-09-20, 2026-12-20, 2027-03-20, 2027-06-20"
        ),
        instrument_type="credit_default_swap",
        preferred_method="analytical",
    )

    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )
    traces = load_platform_traces(root=tmp_path)
    raw = Path(trace_path).read_text()

    assert "validation_contract:" in raw
    assert len(traces) == 1
    assert traces[0].validation_contract["route_id"] == "credit_default_swap_analytical"
    assert traces[0].validation_contract["bundle_id"] == "analytical:credit_default_swap"
    assert (
        traces[0].validation_contract["backend_binding_id"]
        == "trellis.models.credit_default_swap.price_cds_analytical"
    )
    assert (
        traces[0].validation_contract["exact_bundle_id"]
        == "analytical:credit_default_swap@trellis.models.credit_default_swap.price_cds_analytical"
    )


def test_validate_build_emits_validation_contract_summary_in_bundle_events(monkeypatch):
    from trellis.agent.executor import _validate_build
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.planner import SPECIALIZED_SPECS

    class DummyPayoff:
        requirements = {"discount_curve", "black_vol_surface", "fx_rates", "model_parameters"}

        def __init__(self, spec):
            self.spec = spec

        def evaluate(self, market_state):
            return 1.0

    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        "trellis.agent.executor._record_platform_event",
        lambda compiled_request, event, **kwargs: events.append((event, kwargs.get("details") or {})),
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_quanto_required_inputs",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_quanto_cross_currency_semantics",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_sensitivity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_monotonicity",
        lambda *args, **kwargs: [],
    )

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
    )

    failures = _validate_build(
        DummyPayoff,
        code="def evaluate(self, market_state):\n    return 1.0\n",
        description="Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        spec_schema=SPECIALIZED_SPECS["quanto_option_analytical"],
        validation="fast",
        compiled_request=compiled,
        pricing_plan=compiled.pricing_plan,
        product_ir=compiled.product_ir,
        attempt_number=1,
    )

    assert failures == []
    selected = next(details for event, details in events if event == "validation_bundle_selected")
    executed = next(details for event, details in events if event == "validation_bundle_executed")
    assert selected["validation_contract"]["route_id"] == "quanto_adjustment_analytical"
    assert selected["validation_contract"]["bundle_id"] == "analytical:quanto_option"
    assert (
        selected["validation_contract"]["exact_bundle_id"]
        == "analytical:quanto_option@trellis.models.quanto_option.price_quanto_option_analytical_from_market_state"
    )
    assert selected["route_binding_authority"]["route_id"] == "quanto_adjustment_analytical"
    assert isinstance(selected["route_binding_authority"]["canary_task_ids"], list)
    assert executed["validation_contract"]["bundle_id"] == "analytical:quanto_option"
    assert executed["route_binding_authority"]["validation_bundle_id"] == "analytical:quanto_option"
    assert (
        executed["route_binding_authority"]["exact_validation_bundle_id"]
        == "analytical:quanto_option@trellis.models.quanto_option.price_quanto_option_analytical_from_market_state"
    )


def test_validate_build_passes_validation_contract_to_review_policy(monkeypatch):
    from types import SimpleNamespace

    from trellis.agent.executor import _validate_build
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.planner import SPECIALIZED_SPECS

    class DummyPayoff:
        requirements = {"discount_curve", "black_vol_surface", "fx_rates", "model_parameters"}

        def __init__(self, spec):
            self.spec = spec

        def evaluate(self, market_state):
            return 1.0

    captured: dict[str, object] = {}

    def _determine_review_policy(**kwargs):
        captured["validation_contract"] = kwargs.get("validation_contract")
        return SimpleNamespace(
            run_critic=False,
            run_model_validator_llm=False,
            risk_level="low",
            critic_reason="low_risk_supported_vanilla_analytical",
            model_validator_reason="low_risk_supported_vanilla_analytical",
            critic_mode="skip",
            critic_json_max_retries=0,
            critic_allow_text_fallback=False,
            critic_text_max_retries=0,
        )

    monkeypatch.setattr(
        "trellis.agent.review_policy.determine_review_policy",
        _determine_review_policy,
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_quanto_required_inputs",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_quanto_cross_currency_semantics",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_sensitivity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_monotonicity",
        lambda *args, **kwargs: [],
    )

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
    )

    failures = _validate_build(
        DummyPayoff,
        code="def evaluate(self, market_state):\n    return 1.0\n",
        description="Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        spec_schema=SPECIALIZED_SPECS["quanto_option_analytical"],
        validation="standard",
        compiled_request=compiled,
        pricing_plan=compiled.pricing_plan,
        product_ir=compiled.product_ir,
        attempt_number=1,
    )

    assert failures == []
    assert captured["validation_contract"] is compiled.validation_contract


def test_validate_build_emits_reference_oracle_event_for_single_method_zcb_option(monkeypatch):
    from trellis.agent.executor import _validate_build
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.planner import STATIC_SPECS
    from trellis.models.zcb_option import price_zcb_option_jamshidian
    from trellis.core.types import DayCountConvention

    class ZCBSpec:
        notional = 100.0
        strike = 63.0
        expiry_date = date(2027, 11, 15)
        bond_maturity_date = date(2032, 11, 15)
        day_count = DayCountConvention.ACT_365
        option_type = "call"

    class HelperBackedPayoff:
        requirements = {"discount_curve", "black_vol_surface"}

        def __init__(self, spec):
            self.spec = spec

        def evaluate(self, market_state):
            return float(price_zcb_option_jamshidian(market_state, self.spec, mean_reversion=0.1))

    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        "trellis.agent.executor._record_platform_event",
        lambda compiled_request, event, **kwargs: events.append((event, kwargs.get("details") or {})),
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_sensitivity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_monotonicity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.executor._make_test_payoff",
        lambda payoff_cls, spec_schema, settle: payoff_cls(ZCBSpec()),
    )

    compiled = compile_build_request(
        "European zero-coupon bond option expiring 2027-11-15 on a bond maturing 2032-11-15",
        instrument_type="zcb_option",
    )

    failures = _validate_build(
        HelperBackedPayoff,
        code="def evaluate(self, market_state):\n    return 1.0\n",
        description="European zero-coupon bond option expiring 2027-11-15 on a bond maturing 2032-11-15",
        spec_schema=STATIC_SPECS["zcb_option"],
        validation="fast",
        compiled_request=compiled,
        pricing_plan=compiled.pricing_plan,
        product_ir=compiled.product_ir,
        attempt_number=1,
    )

    assert failures == []
    oracle_details = next(details for event, details in events if event == "reference_oracle_executed")
    assert oracle_details["oracle"]["oracle_id"] == "zcb_option_jamshidian_exact"
    assert oracle_details["oracle"]["passed"] is True


def test_validate_build_reference_oracle_catches_zcb_magnitude_error(monkeypatch):
    from trellis.agent.executor import _validate_build
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.planner import STATIC_SPECS
    from trellis.models.zcb_option import price_zcb_option_jamshidian
    from trellis.core.types import DayCountConvention

    class ZCBSpec:
        notional = 100.0
        strike = 63.0
        expiry_date = date(2027, 11, 15)
        bond_maturity_date = date(2032, 11, 15)
        day_count = DayCountConvention.ACT_365
        option_type = "call"

    class BrokenPayoff:
        requirements = {"discount_curve", "black_vol_surface"}

        def __init__(self, spec):
            self.spec = spec

        def evaluate(self, market_state):
            return float(
                price_zcb_option_jamshidian(market_state, self.spec, mean_reversion=0.1)
                / self.spec.notional
            )

    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        "trellis.agent.executor._record_platform_event",
        lambda compiled_request, event, **kwargs: events.append((event, kwargs.get("details") or {})),
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_sensitivity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_monotonicity",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.executor._make_test_payoff",
        lambda payoff_cls, spec_schema, settle: payoff_cls(ZCBSpec()),
    )

    compiled = compile_build_request(
        "European zero-coupon bond option expiring 2027-11-15 on a bond maturing 2032-11-15",
        instrument_type="zcb_option",
    )

    failures = _validate_build(
        BrokenPayoff,
        code="def evaluate(self, market_state):\n    return 1.0\n",
        description="European zero-coupon bond option expiring 2027-11-15 on a bond maturing 2032-11-15",
        spec_schema=STATIC_SPECS["zcb_option"],
        validation="fast",
        compiled_request=compiled,
        pricing_plan=compiled.pricing_plan,
        product_ir=compiled.product_ir,
        attempt_number=1,
    )

    assert failures
    assert any("zcb_option_jamshidian_exact" in failure for failure in failures)
    oracle_details = next(details for event, details in events if event == "reference_oracle_executed")
    assert oracle_details["oracle"]["passed"] is False


def test_reference_oracle_runs_only_for_single_method_builds():
    from types import SimpleNamespace

    from trellis.agent.executor import _should_run_reference_oracle

    assert _should_run_reference_oracle(None) is True
    assert _should_run_reference_oracle(
        SimpleNamespace(comparison_spec=None, comparison_method_plans=())
    ) is True
    assert _should_run_reference_oracle(
        SimpleNamespace(comparison_spec=object(), comparison_method_plans=())
    ) is False
    assert _should_run_reference_oracle(
        SimpleNamespace(comparison_spec=None, comparison_method_plans=(object(),))
    ) is False
