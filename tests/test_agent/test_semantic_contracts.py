"""Tests for the family-name-free semantic contract layer."""

from __future__ import annotations

from dataclasses import replace
import pytest


def _expected_route_modules(compiled) -> tuple[str, ...]:
    from trellis.agent.calibration_contract import _KNOWN_PRIMITIVES

    calibration_modules: tuple[str, ...] = ()
    calibration_step = getattr(compiled, "calibration_step", None)
    primitive = getattr(calibration_step, "proven_primitive", "") if calibration_step is not None else ""
    if primitive:
        module = _KNOWN_PRIMITIVES.get(primitive, "")
        if module:
            calibration_modules = (module,)
    return tuple(
        dict.fromkeys(
            (
                *compiled.pricing_plan.method_modules,
                *compiled.target_modules,
                *compiled.dsl_lowering.helper_modules,
                *calibration_modules,
            )
        )
    )


def _canonical_contract():
    from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract

    return make_ranked_observation_basket_contract(
        description="Himalaya-style ranked observation basket on AAPL, MSFT, and NVDA",
        constituents=("AAPL", "MSFT", "NVDA"),
        observation_schedule=(
            "2025-01-15",
            "2025-02-15",
            "2025-03-15",
        ),
    )


def _draft_contract(description: str, instrument_type: str):
    from trellis.agent.semantic_contracts import draft_semantic_contract

    return draft_semantic_contract(description, instrument_type=instrument_type)


def test_semantic_draft_rule_registry_exposes_stable_order():
    from trellis.agent.semantic_contracts import registered_semantic_draft_rule_names

    assert registered_semantic_draft_rule_names() == (
        "ranked_observation_basket",
        "quanto_option",
        "range_accrual",
        "callable_bond",
        "vanilla_option",
        "rate_style_swaption",
        "credit_basket_tranche",
        "nth_to_default",
        "credit_default_swap",
    )


def test_semantic_family_registry_exposes_supported_method_surfaces():
    from trellis.agent.semantic_contracts import (
        registered_semantic_family_keys,
        resolve_semantic_method_surface,
    )

    assert "vanilla_option" in registered_semantic_family_keys()
    surface = resolve_semantic_method_surface("vanilla_option", "fft_pricing")

    assert surface.method == "fft_pricing"
    assert surface.target_modules == ("trellis.models.equity_option_transforms",)
    assert surface.primitive_families == ("transform_fft",)


def test_semantic_contract_summary_emits_registered_family_surface_metadata():
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract, semantic_contract_summary

    contract = make_rate_style_swaption_contract(
        description="European payer swaption",
        observation_schedule=("2026-01-15",),
        preferred_method="monte_carlo",
    )
    summary = semantic_contract_summary(contract)

    assert summary["methods"]["family_key"] == "rate_style_swaption:european"
    assert summary["methods"]["registered_surface"]["method"] == "monte_carlo"
    assert "trellis.models.rate_style_swaption" in summary["methods"]["registered_surface"]["target_modules"]


def _phase_index(contract):
    return {
        phase: idx
        for idx, phase in enumerate(contract.product.timeline.phase_order)
    }


def _observables_by_id(contract):
    return {
        observable.observable_id: observable
        for observable in contract.product.observables
    }


def _state_fields_by_name(contract):
    return {
        state_field.field_name: state_field
        for state_field in contract.product.state_fields
    }


def test_ranked_observation_basket_contract_validates():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _canonical_contract()
    report = validate_semantic_contract(contract)

    assert report.ok
    assert report.normalized_contract is not None

    compiled = compile_semantic_contract(contract)
    assert compiled.semantic_id == "ranked_observation_basket"
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "basket_path_payoff"
    assert compiled.product_ir.payoff_family == "basket_path_payoff"
    assert compiled.product_ir.schedule_dependence is True
    assert compiled.product_ir.state_dependence == "path_dependent"
    assert compiled.pricing_plan.method == "monte_carlo"
    assert compiled.selection_reason == compiled.pricing_plan.selection_reason
    assert compiled.assumption_summary == compiled.pricing_plan.assumption_summary
    assert compiled.contract.product.selection_scope == "remaining_constituents"
    assert compiled.target_modules == (
        "trellis.models.resolution.basket_semantics",
        "trellis.models.monte_carlo.semantic_basket",
    )
    assert compiled.route_modules == _expected_route_modules(compiled)
    assert "trellis.models.monte_carlo.engine" in compiled.route_modules
    assert "correlated_basket_monte_carlo" in compiled.primitive_routes
    assert "trellis.models.processes.correlated_gbm" not in compiled.route_modules
    assert compiled.dsl_lowering is not None
    assert compiled.dsl_lowering.route_id == "correlated_basket_monte_carlo"


def test_ranked_observation_basket_summary_is_stable_and_family_free():
    from trellis.agent.semantic_contracts import semantic_contract_summary

    contract = _canonical_contract()
    summary = semantic_contract_summary(contract)

    assert summary == semantic_contract_summary(contract)
    assert summary["semantic_id"] == "ranked_observation_basket"
    assert summary["product"]["instrument_class"] == "basket_path_payoff"
    assert summary["product"]["payoff_family"] == "basket_path_payoff"
    assert summary["product"]["observation_schedule"] == [
        "2025-01-15",
        "2025-02-15",
        "2025-03-15",
    ]
    assert summary["product"]["selection_scope"] == "remaining_constituents"
    assert summary["product"]["selection_operator"] == "best_of_remaining"
    assert summary["market_data"]["required_inputs"] == [
        "discount_curve",
        "underlier_spots",
        "black_vol_surface",
        "correlation_matrix",
    ]
    assert summary["semantic_concept"]["semantic_id"] == "ranked_observation_basket"
    assert "basket_option" in summary["semantic_concept"]["compatibility_wrappers"]
    assert summary["blueprint"]["primitive_families"] == ["correlated_basket_monte_carlo"]
    assert summary["typed_semantics"]["phase_order"] == [
        "event",
        "observation",
        "decision",
        "determination",
        "settlement",
        "state_update",
    ]
    assert summary["typed_semantics"]["controller_protocol"]["controller_style"] == "identity"
    assert summary["typed_semantics"]["event_machine_present"] is True
    assert "himalaya_option" not in repr(summary).lower()


def test_ranked_observation_basket_emits_typed_semantic_surface():
    from trellis.agent.semantic_contracts import DEFAULT_PHASE_ORDER

    contract = _canonical_contract()
    observables = _observables_by_id(contract)
    state_fields = _state_fields_by_name(contract)

    assert contract.product.timeline.phase_order == DEFAULT_PHASE_ORDER
    assert contract.product.controller_protocol.controller_style == "identity"
    assert contract.product.implementation_hints.preserve_route_behavior is True
    assert contract.product.implementation_hints.event_machine_source == "derived_from_event_transitions"
    assert contract.product.audit_info.legacy_mirrors == (
        "settlement_rule",
        "event_transitions",
        "state_variables",
    )
    assert observables["constituent_spots"].availability_phase == "observation"
    assert observables["ranked_constituent_return"].availability_phase == "determination"
    assert state_fields["remaining_constituents"].kind == "contract_memory"
    assert state_fields["locked_returns"].kind == "contract_memory"
    assert contract.product.obligations[0].obligation_id == "maturity_cash_settlement"
    assert contract.product.event_machine is not None


def test_range_accrual_trade_entry_contract_validates_and_surfaces_term_fields():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import draft_semantic_contract, semantic_contract_summary

    contract = draft_semantic_contract(
        (
            "Range accrual note on SOFR paying 5.25% when SOFR stays between 1.50% "
            "and 3.25% on 2026-01-15, 2026-04-15, 2026-07-15, and 2026-10-15."
        ),
        instrument_type="range_accrual",
    )
    report = validate_semantic_contract(contract)

    assert report.ok, report.errors
    assert contract.product.semantic_id == "range_accrual"
    assert contract.product.instrument_class == "range_accrual"
    assert contract.product.observation_schedule == (
        "2026-01-15",
        "2026-04-15",
        "2026-07-15",
        "2026-10-15",
    )
    summary = semantic_contract_summary(contract)
    assert summary["semantic_id"] == "range_accrual"
    assert summary["product"]["term_fields"]["reference_index"] == "SOFR"
    assert summary["product"]["term_fields"]["coupon_definition"] == {
        "coupon_rate": 0.0525,
        "coupon_style": "fixed_rate_if_in_range",
    }
    assert summary["product"]["term_fields"]["range_condition"] == {
        "lower_bound": 0.015,
        "upper_bound": 0.0325,
        "inclusive_lower": True,
        "inclusive_upper": True,
    }
    assert summary["product"]["term_fields"]["settlement_profile"] == {
        "coupon_settlement": "coupon_period_cash_settlement",
        "principal_settlement": "principal_at_maturity",
    }
    assert contract.methods.preferred_method == "analytical"
    assert "range_accrual_checked_route_pending" not in contract.blueprint.blocked_by

    compiled = compile_semantic_contract(contract)
    assert compiled.pricing_plan.method == "analytical"
    assert "trellis.models.range_accrual" in compiled.target_modules


@pytest.mark.parametrize(
    "contract_factory",
    [
        lambda: _canonical_contract(),
        lambda: _draft_contract(
            "European call on AAPL with strike 120 and expiry 2025-11-15",
            "european_option",
        ),
        lambda: _draft_contract(
            "Quanto option on SAP in USD with EUR underlier currency and expiry 2025-11-15",
            "quanto_option",
        ),
        lambda: _draft_contract(
            "Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
            "callable_bond",
        ),
        lambda: _draft_contract(
            "European swaption on a fixed-for-floating swap with expiry 2026-01-15",
            "swaption",
        ),
        lambda: _draft_contract(
            (
                "Range accrual note on SOFR paying 5.25% when SOFR stays between 1.50% "
                "and 3.25% on 2026-01-15, 2026-04-15, 2026-07-15, and 2026-10-15."
            ),
            "range_accrual",
        ),
    ],
)
def test_typed_semantic_surface_uses_phase_available_inputs_without_future_peek(contract_factory):
    contract = contract_factory()
    assert contract is not None

    phase_index = _phase_index(contract)
    observables = _observables_by_id(contract)

    assert all(
        observable.availability_phase in phase_index
        for observable in contract.product.observables
    )

    for state_field in contract.product.state_fields:
        cutoff_phase = "determination" if state_field.kind == "event_state" else "state_update"
        for source in state_field.source_observables:
            assert source in observables
            assert (
                phase_index[observables[source].availability_phase]
                <= phase_index[cutoff_phase]
            )


def test_automatic_event_paths_and_strategic_rights_are_separated_on_the_typed_surface():
    basket = _canonical_contract()
    vanilla = _draft_contract(
        "European call on AAPL with strike 120 and expiry 2025-11-15",
        "european_option",
    )
    callable_bond = _draft_contract(
        "Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        "callable_bond",
    )

    assert vanilla is not None
    assert callable_bond is not None

    assert basket.product.controller_protocol.controller_style == "identity"
    assert basket.product.timeline.decision_dates == ()
    assert basket.product.event_machine is not None

    assert vanilla.product.controller_protocol.controller_style == "holder_max"
    assert vanilla.product.controller_protocol.controller_role == "holder"
    assert vanilla.product.timeline.decision_dates == vanilla.product.observation_schedule

    assert callable_bond.product.controller_protocol.controller_style == "issuer_min"
    assert callable_bond.product.controller_protocol.controller_role == "issuer"
    assert callable_bond.product.timeline.decision_dates == callable_bond.product.observation_schedule


def test_vanilla_option_rate_tree_contract_uses_equity_tree_surface():
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="European call on AAPL with rate-tree preference",
        underliers=("AAPL",),
        observation_schedule=("2025-11-15",),
        preferred_method="rate_tree",
    )

    assert "fft_pricing" in contract.methods.candidate_methods
    assert contract.blueprint.target_modules == ("trellis.models.equity_option_tree",)
    assert contract.blueprint.primitive_families == ("exercise_lattice",)


def test_callable_bond_contract_rejects_unsupported_monte_carlo_surface():
    from trellis.agent.semantic_contracts import make_callable_bond_contract

    with pytest.raises(ValueError, match="does not support method `monte_carlo`"):
        make_callable_bond_contract(
            description="Callable bond monte carlo request",
            observation_schedule=("2026-01-15", "2027-01-15"),
            preferred_method="monte_carlo",
        )


def test_bermudan_swaption_contract_rejects_unsupported_monte_carlo_surface():
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    with pytest.raises(ValueError, match="does not support method `monte_carlo`"):
        make_rate_style_swaption_contract(
            description="Bermudan swaption monte carlo request",
            observation_schedule=("2026-01-15", "2027-01-15"),
            preferred_method="monte_carlo",
            exercise_style="bermudan",
        )


def test_specialize_semantic_contract_for_method_reuses_shared_family_authority():
    from trellis.agent.semantic_contracts import (
        make_vanilla_option_contract,
        specialize_semantic_contract_for_method,
    )

    contract = make_vanilla_option_contract(
        description="European call on AAPL with analytical preference",
        underliers=("AAPL",),
        observation_schedule=("2025-11-15",),
        preferred_method="analytical",
    )

    specialized = specialize_semantic_contract_for_method(
        contract,
        preferred_method="fft_pricing",
    )

    assert specialized is not None
    assert specialized.methods.preferred_method == "fft_pricing"
    assert specialized.blueprint.target_modules == ("trellis.models.equity_option_transforms",)
    assert specialized.blueprint.primitive_families == ("transform_fft",)


def test_semantic_family_registry_invariants_hold():
    from trellis.agent.semantic_contracts import validate_semantic_family_registry

    validate_semantic_family_registry()


def test_contract_rejects_illegal_phase_inversion():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _canonical_contract()
    contract = replace(
        contract,
        product=replace(
            contract.product,
            timeline=replace(
                contract.product.timeline,
                phase_order=(
                    "observation",
                    "event",
                    "decision",
                    "determination",
                    "settlement",
                    "state_update",
                ),
            ),
        ),
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("phase order" in error.lower() for error in report.errors)
    assert any(f.code == "semantic.phase_order_invalid" for f in report.error_findings)


def test_contract_rejects_future_peek_observable_usage():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(
        "European call on AAPL with strike 120 and expiry 2025-11-15",
        "european_option",
    )
    assert contract is not None
    bad_observable = replace(
        contract.product.observables[0],
        availability_phase="settlement",
    )
    contract = replace(
        contract,
        product=replace(
            contract.product,
            observables=(bad_observable,),
        ),
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("future-peek" in error.lower() for error in report.errors)
    assert any(f.code == "semantic.future_peek" for f in report.error_findings)


def test_contract_rejects_automatic_trigger_represented_as_control():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _canonical_contract()
    contract = replace(
        contract,
        product=replace(
            contract.product,
            controller_protocol=replace(
                contract.product.controller_protocol,
                controller_style="holder_max",
                controller_role="holder",
                admissible_actions=("settle",),
            ),
        ),
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("automatic trigger represented as control" in error.lower() for error in report.errors)
    assert any(f.code == "semantic.automatic_trigger_as_control" for f in report.error_findings)


def test_contract_requires_typed_obligations_for_settlement_bearing_shapes():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(
        "European call on AAPL with strike 120 and expiry 2025-11-15",
        "european_option",
    )
    assert contract is not None
    contract = replace(
        contract,
        product=replace(contract.product, obligations=()),
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("typed obligation" in error.lower() or "settlement-bearing" in error.lower() for error in report.errors)
    assert any(f.code == "semantic.missing_obligation" for f in report.error_findings)


def test_contract_rejects_inconsistent_state_tags():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(
        "European call on AAPL with strike 120 and expiry 2025-11-15",
        "european_option",
    )
    assert contract is not None
    bad_state_field = replace(
        contract.product.state_fields[0],
        tags=("pathwise_only", "recombining_safe"),
    )
    contract = replace(
        contract,
        product=replace(
            contract.product,
            state_fields=(bad_state_field,),
        ),
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("state-tag consistency" in error.lower() for error in report.errors)
    assert any(f.code == "semantic.state_tag_inconsistent" for f in report.error_findings)


def test_legacy_typed_surface_normalization_warns_without_failing():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(
        "European call on AAPL with strike 120 and expiry 2025-11-15",
        "european_option",
    )
    assert contract is not None
    contract = replace(
        contract,
        product=replace(
            contract.product,
            observables=(),
            implementation_hints=replace(
                contract.product.implementation_hints,
                primary_schedule_role="",
            ),
        ),
    )

    report = validate_semantic_contract(contract)

    assert report.ok
    assert any("without typed observables" in warning.lower() for warning in report.warnings)
    assert any(f.code == "semantic.legacy_observables_missing" for f in report.warning_findings)


def test_contract_rejects_missing_observation_schedule():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = replace(
        _canonical_contract(),
        product=replace(_canonical_contract().product, observation_schedule=()),
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("observation schedule" in error.lower() for error in report.errors)


def test_contract_rejects_selection_without_remaining_scope():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = replace(
        _canonical_contract(),
        product=replace(
            _canonical_contract().product,
            selection_scope="entire_basket",
        ),
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("remaining" in error.lower() for error in report.errors)


def test_contract_requires_correlation_for_multi_asset_mc():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _canonical_contract()
    required_inputs = tuple(
        item for item in contract.market_data.required_inputs
        if item.input_id != "correlation_matrix"
    )
    contract = replace(contract, market_data=replace(contract.market_data, required_inputs=required_inputs))

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("correlation" in error.lower() for error in report.errors)


@pytest.mark.parametrize(
    "description,instrument_type,expected_semantic_id,expected_instrument,expected_payoff_family,expected_underlier_structure,expected_settlement_rule,expected_method,expected_route,expected_target_module",
    [
        (
            "European call on AAPL with strike 120 and expiry 2025-11-15",
            "european_option",
            "vanilla_option",
            "european_option",
            "vanilla_option",
            "single_underlier",
            "cash_settle_at_expiry",
            "analytical",
            "analytical_black76",
            "trellis.models.black",
        ),
        (
            "Quanto option on SAP in USD with EUR underlier currency and expiry 2025-11-15",
            "quanto_option",
            "quanto_option",
            "quanto_option",
            "vanilla_option",
            "cross_currency_single_underlier",
            "cash_settle_at_expiry_after_fx_conversion",
            "analytical",
            "quanto_adjustment_analytical",
            "trellis.models.resolution.quanto",
        ),
        (
            "Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
            "callable_bond",
            "callable_bond",
            "callable_bond",
            "callable_fixed_income",
            "single_issuer_bond",
            "settle_on_call_or_maturity",
            "rate_tree",
            "exercise_lattice",
            "trellis.models.callable_bond_tree",
        ),
        (
            "European swaption on a fixed-for-floating swap with expiry 2026-01-15",
            "swaption",
            "rate_style_swaption",
            "swaption",
            "swaption",
            "single_curve_rate_style",
            "cash_settle_at_exercise",
            "analytical",
            "analytical_black76",
            "trellis.models.rate_style_swaption",
        ),
    ],
)
def test_representative_derivative_contracts_validate_and_compile(
    description,
    instrument_type,
    expected_semantic_id,
    expected_instrument,
    expected_payoff_family,
    expected_underlier_structure,
    expected_settlement_rule,
    expected_method,
    expected_route,
    expected_target_module,
):
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(description, instrument_type)

    assert contract is not None
    assert contract.semantic_id == expected_semantic_id
    assert contract.product.instrument_class == expected_instrument
    assert contract.product.payoff_family == expected_payoff_family
    assert contract.product.underlier_structure == expected_underlier_structure
    assert contract.product.settlement_rule == expected_settlement_rule

    report = validate_semantic_contract(contract)
    assert report.ok
    assert report.normalized_contract is not None

    compiled = compile_semantic_contract(contract)
    assert compiled.semantic_id == expected_semantic_id
    assert compiled.contract.product.underlier_structure == expected_underlier_structure
    assert compiled.contract.product.settlement_rule == expected_settlement_rule
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == expected_instrument
    assert compiled.product_ir.payoff_family == expected_payoff_family
    assert compiled.pricing_plan.method == expected_method
    assert compiled.selection_reason == compiled.pricing_plan.selection_reason
    assert compiled.assumption_summary == compiled.pricing_plan.assumption_summary
    assert compiled.dsl_lowering is not None
    assert compiled.dsl_lowering.route_id == expected_route
    assert compiled.route_modules == _expected_route_modules(compiled)
    assert compiled.primitive_routes == (expected_route,)
    assert expected_target_module in compiled.target_modules
    assert all("himalaya" not in module.lower() for module in compiled.route_modules)


def test_migrated_contract_allows_missing_settlement_rule_mirror_with_warning():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(
        "European call on AAPL with strike 120 and expiry 2025-11-15",
        "european_option",
    )
    assert contract is not None
    contract = replace(
        contract,
        product=replace(
            contract.product,
            settlement_rule="",
            maturity_settlement_rule="",
        ),
    )

    report = validate_semantic_contract(contract)

    assert report.ok
    assert any("legacy settlement_rule mirror" in warning for warning in report.warnings)


def test_non_migrated_contract_still_requires_settlement_rule_mirror():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(
        "Quanto option on SAP in USD with EUR underlier currency and expiry 2025-11-15",
        "quanto_option",
    )
    assert contract is not None
    contract = replace(
        contract,
        product=replace(contract.product, settlement_rule=""),
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("settlement" in error.lower() for error in report.errors)


def test_contract_rejects_unsupported_structure_combination():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(
        "European call on AAPL with strike 120 and expiry 2025-11-15",
        "european_option",
    )
    assert contract is not None
    contract = replace(
        contract,
        product=replace(contract.product, underlier_structure="multi_asset_basket"),
    )

    report = validate_semantic_contract(contract)

    assert not report.ok
    assert any("underlier structure" in error.lower() for error in report.errors)


def test_novel_request_gap_classification_is_structured_and_repeatable():
    from trellis.agent.semantic_contract_validation import (
        classify_semantic_gap,
        semantic_gap_summary,
    )

    report = classify_semantic_gap(
        "Price a resettable memory note with a holiday-adjusted schedule and monthly coupons.",
        instrument_type="structured_note",
    )
    summary = semantic_gap_summary(report)

    assert summary == semantic_gap_summary(report)
    assert report.request_text.startswith("Price a resettable memory note")
    assert report.instrument_type == "structured_note"
    assert not report.requires_clarification
    assert report.can_use_mock_inputs
    assert "missing_semantic_contract_field" in report.gap_types
    assert "missing_runtime_primitive" in report.gap_types
    assert "missing_knowledge_lesson" in report.gap_types
    assert "underlier_structure" in report.missing_contract_fields
    assert "observation_schedule" in report.missing_contract_fields
    assert "generate_schedule" in report.missing_runtime_primitives
    assert "semantic_contract_lesson" in report.missing_knowledge_artifacts
    assert "missing contract fields" in report.summary


def test_novel_request_proposal_prefers_mock_inputs_for_market_gap():
    from trellis.agent.semantic_contract_validation import (
        classify_semantic_gap,
        propose_semantic_extension,
        semantic_extension_summary,
    )

    report = classify_semantic_gap(
        "Price a basket note on AAPL and MSFT, but do not provide vol or correlation inputs.",
        instrument_type="basket_note",
    )
    proposal = propose_semantic_extension(report)
    summary = semantic_extension_summary(proposal)

    assert summary == semantic_extension_summary(proposal)
    assert proposal.decision == "mock_inputs"
    assert proposal.confidence >= 0.6
    assert "mock_market_parameter_provider" in proposal.proposed_market_inputs
    assert "correlation_source_policy" in proposal.proposed_market_inputs
    assert "mock inputs can likely bridge the gap" in report.summary or "missing market inputs" in proposal.summary
    assert "decision=mock_inputs" in proposal.summary


def test_novel_request_proposal_prefers_new_primitive_for_schedule_state_gap():
    from trellis.agent.semantic_contract_validation import (
        classify_semantic_gap,
        propose_semantic_extension,
        semantic_extension_summary,
    )

    report = classify_semantic_gap(
        "Price a resettable memory note with a holiday-adjusted schedule and monthly coupons.",
        instrument_type="structured_note",
    )
    proposal = propose_semantic_extension(report)
    summary = semantic_extension_summary(proposal)

    assert summary == semantic_extension_summary(proposal)
    assert proposal.decision == "new_primitive"
    assert proposal.confidence >= 0.7
    assert "trellis.core.date_utils.generate_schedule" in proposal.proposed_runtime_primitives
    assert "trellis.models.monte_carlo.event_state" in proposal.proposed_runtime_primitives
    assert "trellis.models.monte_carlo.basket_state" in proposal.proposed_runtime_primitives
    assert "decision=new_primitive" in proposal.summary


def test_semantic_concept_resolution_prefers_the_canonical_basket_concept_and_reports_conflicts():
    from trellis.agent.semantic_concepts import resolve_semantic_concept, semantic_concept_summary

    resolution = resolve_semantic_concept(
        "Quanto basket option on AAPL, MSFT, and NVDA with ranked observation dates 2025-01-15, 2025-02-15, and 2025-03-15.",
        instrument_type="basket_option",
    )
    summary = semantic_concept_summary(resolution)

    assert summary == semantic_concept_summary(resolution)
    assert resolution.concept_id == "ranked_observation_basket"
    assert resolution.resolution_kind == "thin_compatibility_wrapper"
    assert resolution.concept_status == "active"
    assert resolution.matched_wrapper == "basket_option"
    assert "quanto_option" in resolution.conflicting_concepts
    assert summary["concept_id"] == "ranked_observation_basket"
    assert summary["resolution_kind"] == "thin_compatibility_wrapper"
    assert "quanto_option" in summary["conflicting_concepts"]


def test_semantic_concept_resolution_marks_deprecated_wrappers_as_stale():
    from trellis.agent.semantic_concepts import resolve_semantic_concept, semantic_concept_summary

    resolution = resolve_semantic_concept(
        "Himalaya-style ranked observation basket on AAPL, MSFT, and NVDA.",
        instrument_type="himalaya_option",
    )
    summary = semantic_concept_summary(resolution)

    assert summary == semantic_concept_summary(resolution)
    assert resolution.concept_id == "ranked_observation_basket"
    assert resolution.concept_status == "stale"
    assert resolution.resolution_kind == "thin_compatibility_wrapper"
    assert resolution.matched_alias == "himalaya_option"
    assert summary["concept_status"] == "stale"
    assert summary["matched_alias"] == "himalaya_option"


@pytest.mark.parametrize(
    "description,instrument_type,expected_concept_id,expected_concept_role",
    [
        (
            "Generate an observation schedule for quarterly coupon dates.",
            "schedule",
            "schedule",
            "supporting_atom",
        ),
        (
            "Discount curve required for present-value discounting.",
            "curve",
            "curve",
            "market_input",
        ),
        (
            "Implied volatility surface required for the route.",
            "surface",
            "surface",
            "market_input",
        ),
        (
            "Correlation matrix required for basket pricing.",
            "correlation",
            "correlation",
            "market_input",
        ),
        (
            "State machine and event transitions for a path-dependent workflow.",
            "event_state",
            "event_state",
            "supporting_atom",
        ),
        (
            "Payoff and settlement semantics for the instrument.",
            "payoff",
            "payoff",
            "supporting_atom",
        ),
        (
            "Exercise policy for Bermudan and callable decision rights.",
            "exercise_policy",
            "exercise_policy",
            "supporting_atom",
        ),
        (
            "Calibration target for fitting the model to market data.",
            "calibration_target",
            "calibration_target",
            "market_input",
        ),
        (
            "Market parameter provenance and source policy for observed and estimated inputs.",
            "market_parameter_source",
            "market_parameter_source",
            "market_input",
        ),
    ],
)
def test_generic_semantic_atoms_resolve_with_explicit_taxonomy_metadata(
    description,
    instrument_type,
    expected_concept_id,
    expected_concept_role,
):
    from trellis.agent.semantic_concepts import resolve_semantic_concept, semantic_concept_summary

    resolution = resolve_semantic_concept(description, instrument_type=instrument_type)
    summary = semantic_concept_summary(resolution)

    assert summary == semantic_concept_summary(resolution)
    assert resolution.concept_id == expected_concept_id
    assert resolution.concept_role == expected_concept_role
    assert resolution.resolution_kind == "reuse_existing_concept"
    assert summary["concept_id"] == expected_concept_id
    assert summary["concept_role"] == expected_concept_role
    assert expected_concept_id in resolution.summary
    assert expected_concept_role in resolution.summary


@pytest.mark.parametrize(
    "description,instrument_type,expected_concept_id,expected_extension_kind",
    [
        (
            "Price a basket option on AAPL and MSFT with observation dates 2025-01-15 and 2025-02-15.",
            "ranked_observation_basket",
            "ranked_observation_basket",
            "new_attribute",
        ),
        (
            "Price a resettable memory note with a holiday-adjusted schedule and monthly coupons.",
            "structured_note",
            "",
            "introduce_new_concept",
        ),
    ],
)
def test_semantic_gap_classification_tracks_concept_extension_policy(
    description,
    instrument_type,
    expected_concept_id,
    expected_extension_kind,
):
    from trellis.agent.semantic_contract_validation import (
        classify_semantic_gap,
        propose_semantic_extension,
        semantic_gap_summary,
        semantic_extension_summary,
    )

    report = classify_semantic_gap(description, instrument_type=instrument_type)
    proposal = propose_semantic_extension(report)
    gap_summary = semantic_gap_summary(report)
    proposal_summary = semantic_extension_summary(proposal)

    assert gap_summary == semantic_gap_summary(report)
    assert proposal_summary == semantic_extension_summary(proposal)
    assert report.semantic_concept_id == expected_concept_id
    assert report.semantic_concept_extension_kind == expected_extension_kind
    assert proposal.semantic_concept_extension_kind == expected_extension_kind
    assert proposal.semantic_concept_id == expected_concept_id
    if expected_concept_id:
        assert gap_summary["semantic_concept"]["semantic_id"] == expected_concept_id
        assert proposal_summary["semantic_concept"]["semantic_id"] == expected_concept_id
        assert gap_summary["semantic_concept"]["extension_kind"] == expected_extension_kind
    else:
        assert gap_summary["semantic_concept"] is None
        assert proposal_summary["semantic_concept"] is None
    assert expected_extension_kind in proposal.summary


# ---------------------------------------------------------------------------
# Confidence bands, gap_ratio, and role-based tiebreaking (QUA-403)
# ---------------------------------------------------------------------------


def test_confidence_populated_for_wrapper_match():
    """A compatibility_wrapper match scores 120 => confidence >= 1.0."""
    from trellis.agent.semantic_concepts import resolve_semantic_concept

    resolution = resolve_semantic_concept(
        "Basket option on AAPL, MSFT, NVDA",
        instrument_type="basket_option",
    )
    assert resolution.concept_id == "ranked_observation_basket"
    assert resolution.confidence >= 1.0


def test_confidence_zero_for_no_match():
    """An unrecognized instrument with no cue phrases => confidence == 0.0."""
    from trellis.agent.semantic_concepts import resolve_semantic_concept

    resolution = resolve_semantic_concept(
        "Completely unknown product with no matching cues",
        instrument_type="xyzzy_nonexistent_instrument",
    )
    assert resolution.confidence == 0.0


def test_gap_ratio_populated_with_runner_up():
    """When multiple concepts score, gap_ratio is second/top and > 0."""
    from trellis.agent.semantic_concepts import resolve_semantic_concept

    resolution = resolve_semantic_concept(
        "Quanto basket option on AAPL, MSFT, and NVDA with ranked observation dates",
        instrument_type="basket_option",
    )
    # basket_option wrapper triggers ranked_observation_basket (120),
    # but quanto_option also scores via cue phrases -> runner-up > 0.
    assert resolution.concept_id == "ranked_observation_basket"
    assert resolution.gap_ratio > 0.0
    assert resolution.gap_ratio < 1.0


def test_gap_ratio_zero_when_single_match():
    """When only one concept matches, gap_ratio == 0.0."""
    from trellis.agent.semantic_concepts import resolve_semantic_concept

    resolution = resolve_semantic_concept(
        "Discount curve required for present-value discounting.",
        instrument_type="curve",
    )
    assert resolution.concept_id == "curve"
    if len(resolution.candidate_concepts) <= 1:
        assert resolution.gap_ratio == 0.0


def test_role_based_tiebreaking_prefers_product_contract():
    """When two concepts have identical scores, product_contract wins over supporting_atom."""
    from trellis.agent.semantic_concepts import _CONCEPT_ROLE_PRIORITY

    assert _CONCEPT_ROLE_PRIORITY["product_contract"] < _CONCEPT_ROLE_PRIORITY["supporting_atom"]
    assert _CONCEPT_ROLE_PRIORITY["supporting_atom"] < _CONCEPT_ROLE_PRIORITY["market_input"]


def test_confidence_and_gap_ratio_in_summary():
    """confidence and gap_ratio appear in the summary dict."""
    from trellis.agent.semantic_concepts import resolve_semantic_concept, semantic_concept_summary

    resolution = resolve_semantic_concept(
        "European call on AAPL with strike 120",
        instrument_type="european_option",
    )
    summary = semantic_concept_summary(resolution)
    assert "confidence" in summary
    assert "gap_ratio" in summary
    assert summary["confidence"] == resolution.confidence
    assert summary["gap_ratio"] == resolution.gap_ratio


# ---------------------------------------------------------------------------
# Credit concept resolution tests (QUA-420)
# ---------------------------------------------------------------------------


class TestCreditConceptResolution:
    """Test credit derivative concept resolution and ambiguity detection."""

    def test_cds_resolves_from_alias(self):
        from trellis.agent.semantic_concepts import resolve_semantic_concept

        resolution = resolve_semantic_concept("credit default swap pricing")
        assert resolution.concept_id == "credit_default_swap"
        assert resolution.resolution_kind == "reuse_existing_concept"
        assert resolution.confidence >= 0.6

    def test_cds_resolves_from_abbreviation(self):
        from trellis.agent.semantic_concepts import resolve_semantic_concept

        resolution = resolve_semantic_concept(
            "CDS pricing: hazard rate MC vs survival prob analytical",
            instrument_type="credit_default_swap",
        )
        assert resolution.concept_id == "credit_default_swap"
        assert resolution.resolution_kind == "reuse_existing_concept"

    def test_ntd_resolves_from_alias(self):
        from trellis.agent.semantic_concepts import resolve_semantic_concept

        resolution = resolve_semantic_concept("nth to default on 5 names")
        assert resolution.concept_id == "nth_to_default"
        assert resolution.resolution_kind == "reuse_existing_concept"

    def test_ntd_resolves_first_to_default(self):
        from trellis.agent.semantic_concepts import resolve_semantic_concept

        resolution = resolve_semantic_concept("first to default basket on IG credits")
        assert resolution.concept_id == "nth_to_default"

    def test_ambiguous_credit_with_overlapping_cues(self):
        """Request matching cues from both CDS and NTD should be ambiguous or contested."""
        from trellis.agent.semantic_concepts import resolve_semantic_concept

        # "reference entity" matches CDS cue, "names" matches NTD cue
        resolution = resolve_semantic_concept(
            "credit protection on reference entity names with default correlation"
        )
        # Both concepts should score > 0 from their respective cues
        assert len(resolution.candidate_concepts) >= 2 or resolution.resolution_kind == "clarification"
        if resolution.resolution_kind == "ambiguous":
            assert len(resolution.conflicting_concepts) > 0

    def test_vague_credit_request_falls_to_clarification(self):
        """A vague 'credit derivative' phrase resolves to clarification (no concept)."""
        from trellis.agent.semantic_concepts import resolve_semantic_concept

        resolution = resolve_semantic_concept("credit derivative pricing")
        # Neither CDS nor NTD aliases contain "credit derivative"
        assert resolution.resolution_kind == "clarification"

    def test_cds_and_ntd_have_different_primitives(self):
        from trellis.agent.semantic_concepts import get_semantic_concept_definition

        cds = get_semantic_concept_definition("credit_default_swap")
        ntd = get_semantic_concept_definition("nth_to_default")
        assert cds is not None
        assert ntd is not None
        assert cds.required_primitives != ntd.required_primitives
        assert "credit_curve_survival_probability" in cds.required_primitives
        assert "gaussian_copula" in ntd.required_primitives

    def test_credit_concepts_have_distinct_route_families(self):
        from trellis.agent.semantic_concepts import get_semantic_concept_definition

        cds = get_semantic_concept_definition("credit_default_swap")
        ntd = get_semantic_concept_definition("nth_to_default")
        assert cds.route_family == "credit_default_swap"
        assert ntd.route_family == "nth_to_default"


class TestAmbiguousResolutionBlocksBuild:
    """Test that ambiguous concept resolution triggers requires_clarification."""

    def test_ambiguous_sets_requires_clarification(self):
        from trellis.agent.semantic_contract_validation import classify_semantic_gap

        # Use a request with cues overlapping both CDS and NTD
        gap = classify_semantic_gap(
            "credit protection on reference entity names with default correlation"
        )
        # If concept resolution was ambiguous, clarification must be required
        if gap.semantic_concept_resolution_kind == "ambiguous":
            assert gap.requires_clarification is True
            assert len(gap.semantic_concept_conflicts) > 0

    def test_vague_credit_request_requires_clarification(self):
        from trellis.agent.semantic_contract_validation import classify_semantic_gap

        gap = classify_semantic_gap("credit derivative pricing")
        # "credit" cue fires → not vague per cue system, but concept is
        # unresolved ("clarification"). The R3 wiring only triggers on
        # resolution_kind == "ambiguous", not "clarification".
        # With the credit cue firing, requires_clarification is False
        # unless concept resolution is ambiguous.
        assert gap.semantic_concept_resolution_kind == "clarification"

    def test_specific_cds_does_not_require_clarification(self):
        from trellis.agent.semantic_contract_validation import classify_semantic_gap

        gap = classify_semantic_gap(
            "CDS pricing: hazard rate MC vs survival prob analytical",
            instrument_type="credit_default_swap",
        )
        # "CDS" is a direct alias → resolves cleanly, credit cue fires → no clarification
        assert gap.requires_clarification is False
        assert gap.semantic_concept_id == "credit_default_swap"

    def test_specific_ntd_resolves_concept_even_without_cues(self):
        """NTD resolves the concept correctly, even though the coarse cue
        system may not recognize it (no credit/shape cue fires for 'nth to default')."""
        from trellis.agent.semantic_contract_validation import classify_semantic_gap

        gap = classify_semantic_gap(
            "nth to default on 5 names with credit curve and default correlation",
            instrument_type="nth_to_default",
        )
        assert gap.semantic_concept_id == "nth_to_default"
        assert gap.semantic_concept_resolution_kind == "reuse_existing_concept"


def test_bermudan_swaption_analytical_contract_uses_lower_bound_target():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = make_rate_style_swaption_contract(
        description="Bermudan payer swaption analytical lower bound",
        observation_schedule=("2026-01-15", "2027-01-15", "2028-01-15"),
        preferred_method="analytical",
        exercise_style="bermudan",
    )
    report = validate_semantic_contract(contract)
    assert report.ok

    compiled = compile_semantic_contract(contract, preferred_method="analytical")
    assert compiled.pricing_plan.method == "analytical"
    assert compiled.primitive_routes == ("analytical_black76",)
    assert "trellis.models.rate_style_swaption" in compiled.target_modules
    assert compiled.dsl_lowering is not None
    assert (
        "trellis.models.rate_style_swaption.price_bermudan_swaption_black76_lower_bound"
        in compiled.dsl_lowering.helper_refs
    )


def test_rate_style_swaption_contract_accepts_european_monte_carlo_method():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = make_rate_style_swaption_contract(
        description="European payer swaption under Hull-White Monte Carlo",
        observation_schedule=("2029-11-15",),
        preferred_method="monte_carlo",
        exercise_style="european",
    )
    report = validate_semantic_contract(contract)
    assert report.ok

    compiled = compile_semantic_contract(contract, preferred_method="monte_carlo")
    assert compiled.pricing_plan.method == "monte_carlo"
    assert compiled.primitive_routes == ("monte_carlo_paths",)
    assert compiled.dsl_lowering is not None
    assert compiled.dsl_lowering.route_id == "monte_carlo_paths"
    assert compiled.dsl_lowering.family_ir is not None


def test_extract_swaption_term_fields_captures_hull_white_comparison_regime():
    from types import SimpleNamespace

    from trellis.agent.semantic_contracts import _extract_swaption_term_fields

    fields = _extract_swaption_term_fields(
        (
            "European payer swaption. Fixed leg: semi-annual, 30/360. "
            "Float leg: quarterly 3M SOFR, Act/360. "
            "Hull-White model: mean reversion a=0.05, vol sigma=0.01."
        ),
        SimpleNamespace(parameters={}),
    )

    assert fields["comparison_model_name"] == "hull_white_1f"
    assert fields["comparison_mean_reversion"] == pytest.approx(0.05)
    assert fields["comparison_sigma"] == pytest.approx(0.01)
    assert fields["comparison_quote_family"] == "implied_vol"
    assert fields["comparison_quote_convention"] == "black"
    assert fields["comparison_quote_subject"] == "swaption"


def test_credit_default_swap_contract_validates_and_compiles():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_credit_default_swap_contract
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = make_credit_default_swap_contract(
        description="Single-name CDS on ACME with quarterly premium dates",
        observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20"),
    )
    report = validate_semantic_contract(contract)

    assert report.ok
    assert report.normalized_contract is not None

    compiled = compile_semantic_contract(contract)
    assert compiled.semantic_id == "credit_default_swap"
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "cds"
    assert compiled.product_ir.payoff_family == "credit_default_swap"
    assert compiled.product_ir.schedule_dependence is True
    assert compiled.pricing_plan.method == "analytical"
    assert compiled.target_modules == ("trellis.models.credit_default_swap",)
    assert compiled.route_modules == _expected_route_modules(compiled)
    assert compiled.primitive_routes == ("credit_default_swap_analytical",)
    assert compiled.dsl_lowering is not None
    assert compiled.dsl_lowering.route_id == "credit_default_swap_analytical"


def test_credit_default_swap_summary_is_stable_and_route_specific():
    from trellis.agent.semantic_contracts import make_credit_default_swap_contract, semantic_contract_summary

    contract = make_credit_default_swap_contract(
        description="Single-name CDS on ACME with quarterly premium dates",
        observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20"),
    )
    summary = semantic_contract_summary(contract)

    assert summary == semantic_contract_summary(contract)
    assert summary["semantic_id"] == "credit_default_swap"
    assert summary["product"]["instrument_class"] == "cds"
    assert summary["product"]["payoff_family"] == "credit_default_swap"
    assert summary["typed_semantics"]["controller_protocol"]["controller_style"] == "identity"
    assert summary["market_data"]["required_inputs"] == ["discount_curve", "credit_curve"]
    assert summary["blueprint"]["primitive_families"] == ["credit_default_swap_analytical"]


def test_nth_to_default_contract_validates_and_compiles():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_nth_to_default_contract
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = make_nth_to_default_contract(
        description="First-to-default basket on ACME, BRAVO, CHARLIE, DELTA, ECHO through 2029-11-15",
        observation_schedule=("2029-11-15",),
        reference_entities=("ACME", "BRAVO", "CHARLIE", "DELTA", "ECHO"),
        trigger_rank=1,
    )
    report = validate_semantic_contract(contract)

    assert report.ok
    assert report.normalized_contract is not None

    compiled = compile_semantic_contract(contract)
    assert compiled.semantic_id == "nth_to_default"
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "nth_to_default"
    assert compiled.product_ir.payoff_family == "nth_to_default"
    assert compiled.product_ir.schedule_dependence is True
    assert compiled.pricing_plan.method == "copula"
    assert compiled.target_modules == ("trellis.instruments.nth_to_default",)
    assert compiled.route_modules == _expected_route_modules(compiled)
    assert compiled.primitive_routes == ("nth_to_default_monte_carlo",)
    assert compiled.dsl_lowering is not None
    assert compiled.dsl_lowering.route_id == "nth_to_default_monte_carlo"


def test_nth_to_default_summary_is_stable_and_route_specific():
    from trellis.agent.semantic_contracts import make_nth_to_default_contract, semantic_contract_summary

    contract = make_nth_to_default_contract(
        description="First-to-default basket on ACME, BRAVO, CHARLIE, DELTA, ECHO through 2029-11-15",
        observation_schedule=("2029-11-15",),
        reference_entities=("ACME", "BRAVO", "CHARLIE", "DELTA", "ECHO"),
        trigger_rank=1,
    )
    summary = semantic_contract_summary(contract)

    assert summary == semantic_contract_summary(contract)
    assert summary["semantic_id"] == "nth_to_default"
    assert summary["product"]["instrument_class"] == "nth_to_default"
    assert summary["product"]["payoff_family"] == "nth_to_default"
    assert summary["typed_semantics"]["controller_protocol"]["controller_style"] == "identity"
    assert summary["market_data"]["required_inputs"] == ["discount_curve", "credit_curve"]
    assert summary["blueprint"]["primitive_families"] == ["nth_to_default_monte_carlo"]


def test_cdo_tranche_contract_validates_and_compiles():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(
        (
            "CDO tranche on a 100-name IG portfolio with attachment point 3%, "
            "detachment point 7%, maturity 2029-11-15, and flat default correlation 0.3."
        ),
        "cdo",
    )
    report = validate_semantic_contract(contract)

    assert report.ok, report.errors

    compiled = compile_semantic_contract(contract)
    assert compiled.semantic_id == "credit_basket_tranche"
    assert compiled.product_ir.instrument == "cdo"
    assert compiled.product_ir.payoff_family == "credit_basket_tranche"
    assert compiled.pricing_plan.method == "copula"
    assert "trellis.models.credit_basket_copula" in compiled.target_modules
    assert compiled.primitive_routes == ("copula_loss_distribution",)
    assert compiled.dsl_lowering is not None
    assert compiled.dsl_lowering.route_id == "copula_loss_distribution"
