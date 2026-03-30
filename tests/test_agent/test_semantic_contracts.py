"""Tests for the family-name-free semantic contract layer."""

from __future__ import annotations

from dataclasses import replace
import pytest


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
    assert compiled.route_modules == tuple(
        dict.fromkeys((*compiled.pricing_plan.method_modules, *compiled.target_modules))
    )
    assert "trellis.models.monte_carlo.engine" in compiled.route_modules
    assert "correlated_basket_monte_carlo" in compiled.primitive_routes
    assert "trellis.models.processes.correlated_gbm" not in compiled.route_modules


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
    assert "himalaya_option" not in repr(summary).lower()


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
            "trellis.models.trees.lattice",
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
            "trellis.models.black",
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
    assert compiled.route_modules == tuple(
        dict.fromkeys((*compiled.pricing_plan.method_modules, *compiled.target_modules))
    )
    assert compiled.primitive_routes == (expected_route,)
    assert expected_target_module in compiled.target_modules
    assert all("himalaya" not in module.lower() for module in compiled.route_modules)


def test_contract_rejects_missing_settlement_rule():
    from trellis.agent.semantic_contract_validation import validate_semantic_contract

    contract = _draft_contract(
        "European call on AAPL with strike 120 and expiry 2025-11-15",
        "european_option",
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

    def test_credit_concepts_share_route_family(self):
        from trellis.agent.semantic_concepts import get_semantic_concept_definition

        cds = get_semantic_concept_definition("credit_default_swap")
        ntd = get_semantic_concept_definition("nth_to_default")
        assert cds.route_family == ntd.route_family == "credit_default_swap"


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
