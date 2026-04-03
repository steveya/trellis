"""Tests for deterministic validation-bundle selection and execution."""

from __future__ import annotations


def test_select_validation_bundle_for_analytical_european_option():
    from trellis.agent.validation_bundles import select_validation_bundle

    bundle = select_validation_bundle(
        instrument_type="european_option",
        method="analytical",
    )

    assert bundle.bundle_id == "analytical:european_option"
    assert "check_non_negativity" in bundle.checks
    assert "check_price_sanity" in bundle.checks
    assert "check_vol_sensitivity" in bundle.checks
    assert "check_vol_monotonicity" in bundle.checks
    assert "universal" in bundle.categories
    assert "no_arbitrage" in bundle.categories


def test_select_validation_bundle_for_analytical_swaption_includes_helper_consistency():
    from trellis.agent.validation_bundles import select_validation_bundle

    bundle = select_validation_bundle(
        instrument_type="swaption",
        method="analytical",
    )

    assert bundle.bundle_id == "analytical:swaption"
    assert "check_rate_style_swaption_helper_consistency" in bundle.checks
    assert "route_specific" in bundle.categories


def test_execute_validation_bundle_respects_validation_level(monkeypatch):
    from trellis.agent.validation_bundles import ValidationBundle, execute_validation_bundle

    calls: list[str] = []

    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda payoff, market_state: calls.append("check_non_negativity") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda payoff, market_state: calls.append("check_price_sanity") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_sensitivity",
        lambda payoff_factory, market_state_factory: calls.append("check_vol_sensitivity") or [],
    )

    bundle = ValidationBundle(
        bundle_id="demo",
        instrument_type="european_option",
        method="analytical",
        checks=(
            "check_non_negativity",
            "check_price_sanity",
            "check_vol_sensitivity",
        ),
        categories={"universal": ("check_non_negativity", "check_price_sanity")},
    )

    execution = execute_validation_bundle(
        bundle,
        validation_level="fast",
        test_payoff=object(),
        market_state=object(),
        payoff_factory=lambda: object(),
        market_state_factory=lambda **kwargs: object(),
    )

    assert calls == ["check_non_negativity", "check_price_sanity"]
    assert execution.failures == ()
    assert execution.executed_checks == ("check_non_negativity", "check_price_sanity")
    assert execution.skipped_checks == ("check_vol_sensitivity",)


def test_select_validation_bundle_for_quanto_family_includes_family_checks():
    from trellis.agent.family_contract_compiler import compile_family_contract
    from trellis.agent.family_contract_templates import get_family_contract_template
    from trellis.agent.validation_bundles import select_validation_bundle

    blueprint = compile_family_contract(get_family_contract_template("quanto_option"))

    bundle = select_validation_bundle(
        instrument_type="unknown",
        method="analytical",
        product_ir=blueprint.product_ir,
        family_blueprint=blueprint,
    )

    assert bundle.bundle_id == "analytical:quanto_option"
    assert "check_quanto_required_inputs" in bundle.checks
    assert "check_quanto_cross_currency_semantics" in bundle.checks
    assert "product_family" in bundle.categories


def test_select_validation_bundle_for_cds_includes_credit_family_checks_first():
    from trellis.agent.validation_bundles import select_validation_bundle

    bundle = select_validation_bundle(
        instrument_type="credit_default_swap",
        method="monte_carlo",
    )

    assert bundle.bundle_id == "monte_carlo:credit_default_swap"
    assert bundle.checks[:2] == (
        "check_cds_spread_quote_normalization",
        "check_cds_credit_curve_sensitivity",
    )
    assert "product_family" in bundle.categories


def test_execute_validation_bundle_uses_decreasing_vol_direction_for_callable_bond(monkeypatch):
    from trellis.agent.validation_bundles import ValidationBundle, execute_validation_bundle

    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda payoff, market_state: [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda payoff, market_state: [],
    )

    def _check_vol_monotonicity(payoff_factory, market_state_factory, **kwargs):
        seen["expected_direction"] = kwargs.get("expected_direction")
        return []

    monkeypatch.setattr(
        "trellis.agent.invariants.check_vol_monotonicity",
        _check_vol_monotonicity,
    )

    bundle = ValidationBundle(
        bundle_id="rate_tree:callable_bond",
        instrument_type="callable_bond",
        method="rate_tree",
        checks=(
            "check_non_negativity",
            "check_price_sanity",
            "check_vol_monotonicity",
        ),
        categories={"universal": ("check_non_negativity", "check_price_sanity")},
    )

    execution = execute_validation_bundle(
        bundle,
        validation_level="standard",
        test_payoff=object(),
        market_state=object(),
        payoff_factory=lambda: object(),
        market_state_factory=lambda **kwargs: object(),
    )

    assert execution.failures == ()
    assert seen["expected_direction"] == "decreasing"


def test_execute_validation_bundle_uses_configured_bound_relation(monkeypatch):
    from trellis.agent.validation_bundles import ValidationBundle, execute_validation_bundle

    seen: dict[str, object] = {}

    def _check_bounded_by_reference(*args, **kwargs):
        seen["relation"] = kwargs.get("relation")
        return []

    monkeypatch.setattr(
        "trellis.agent.invariants.check_bounded_by_reference",
        _check_bounded_by_reference,
    )

    bundle = ValidationBundle(
        bundle_id="rate_tree:puttable_bond",
        instrument_type="puttable_bond",
        method="rate_tree",
        checks=("check_bounded_by_reference",),
        categories={"product_family": ("check_bounded_by_reference",)},
    )

    execution = execute_validation_bundle(
        bundle,
        validation_level="standard",
        test_payoff=object(),
        market_state=object(),
        payoff_factory=lambda: object(),
        market_state_factory=lambda **kwargs: object(),
        reference_factory=lambda: object(),
        check_relations={"check_bounded_by_reference": ">="},
    )

    assert execution.failures == ()
    assert seen["relation"] == ">="


def test_execute_validation_bundle_runs_quanto_family_checks(monkeypatch):
    from trellis.agent.validation_bundles import ValidationBundle, execute_validation_bundle

    calls: list[str] = []

    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda payoff, market_state: calls.append("check_non_negativity") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda payoff, market_state: calls.append("check_price_sanity") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_quanto_required_inputs",
        lambda payoff, market_state: calls.append("check_quanto_required_inputs") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_quanto_cross_currency_semantics",
        lambda payoff_factory, market_state_factory: calls.append("check_quanto_cross_currency_semantics") or [],
    )

    bundle = ValidationBundle(
        bundle_id="monte_carlo:quanto_option",
        instrument_type="quanto_option",
        method="monte_carlo",
        checks=(
            "check_non_negativity",
            "check_price_sanity",
            "check_quanto_required_inputs",
            "check_quanto_cross_currency_semantics",
        ),
        categories={
            "universal": ("check_non_negativity", "check_price_sanity"),
            "product_family": (
                "check_quanto_required_inputs",
                "check_quanto_cross_currency_semantics",
            ),
        },
    )

    execution = execute_validation_bundle(
        bundle,
        validation_level="fast",
        test_payoff=object(),
        market_state=object(),
        payoff_factory=lambda: object(),
        market_state_factory=lambda **kwargs: object(),
    )

    assert execution.failures == ()
    assert execution.executed_checks == (
        "check_non_negativity",
        "check_price_sanity",
        "check_quanto_required_inputs",
        "check_quanto_cross_currency_semantics",
    )
    assert execution.skipped_checks == ()
    assert calls == list(execution.executed_checks)


def test_execute_validation_bundle_returns_structured_failure_details(monkeypatch):
    from trellis.agent.invariants import InvariantFailure
    from trellis.agent.validation_bundles import ValidationBundle, execute_validation_bundle

    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda *args, **kwargs: [
            InvariantFailure(
                check="check_non_negativity",
                message="Price is negative: -2.000000",
                actual=-2.0,
                context={"spot": 100.0, "fx_pairs": ("EURUSD",)},
            )
        ],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda *args, **kwargs: [],
    )

    bundle = ValidationBundle(
        bundle_id="demo",
        instrument_type="european_option",
        method="analytical",
        checks=("check_non_negativity", "check_price_sanity"),
        categories={"universal": ("check_non_negativity", "check_price_sanity")},
    )

    execution = execute_validation_bundle(
        bundle,
        validation_level="fast",
        test_payoff=object(),
        market_state=object(),
    )

    assert execution.failures == ("Price is negative: -2.000000",)
    assert len(execution.failure_details) == 1
    detail = execution.failure_details[0]
    assert detail.check == "check_non_negativity"
    assert detail.actual == -2.0
    assert detail.context["spot"] == 100.0


def test_execute_validation_bundle_runs_cds_checks_before_universal(monkeypatch):
    from trellis.agent.validation_bundles import ValidationBundle, execute_validation_bundle

    calls: list[str] = []

    monkeypatch.setattr(
        "trellis.agent.invariants.check_cds_spread_quote_normalization",
        lambda *args, **kwargs: calls.append("check_cds_spread_quote_normalization") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_cds_credit_curve_sensitivity",
        lambda *args, **kwargs: calls.append("check_cds_credit_curve_sensitivity") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda *args, **kwargs: calls.append("check_non_negativity") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda *args, **kwargs: calls.append("check_price_sanity") or [],
    )

    bundle = ValidationBundle(
        bundle_id="monte_carlo:credit_default_swap",
        instrument_type="credit_default_swap",
        method="monte_carlo",
        checks=(
            "check_cds_spread_quote_normalization",
            "check_cds_credit_curve_sensitivity",
            "check_non_negativity",
            "check_price_sanity",
        ),
        categories={
            "product_family": (
                "check_cds_spread_quote_normalization",
                "check_cds_credit_curve_sensitivity",
            ),
            "universal": ("check_non_negativity", "check_price_sanity"),
        },
    )

    execution = execute_validation_bundle(
        bundle,
        validation_level="standard",
        test_payoff=object(),
        market_state=object(),
        payoff_factory=lambda: object(),
        market_state_factory=lambda **kwargs: object(),
    )

    assert execution.failures == ()
    assert calls == [
        "check_cds_spread_quote_normalization",
        "check_cds_credit_curve_sensitivity",
        "check_non_negativity",
        "check_price_sanity",
    ]


def test_execute_validation_bundle_runs_swaption_helper_consistency(monkeypatch):
    from trellis.agent.validation_bundles import ValidationBundle, execute_validation_bundle

    calls: list[str] = []

    monkeypatch.setattr(
        "trellis.agent.invariants.check_non_negativity",
        lambda *args, **kwargs: calls.append("check_non_negativity") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_price_sanity",
        lambda *args, **kwargs: calls.append("check_price_sanity") or [],
    )
    monkeypatch.setattr(
        "trellis.agent.invariants.check_rate_style_swaption_helper_consistency",
        lambda *args, **kwargs: calls.append("check_rate_style_swaption_helper_consistency") or [],
    )

    bundle = ValidationBundle(
        bundle_id="analytical:swaption",
        instrument_type="swaption",
        method="analytical",
        checks=(
            "check_non_negativity",
            "check_price_sanity",
            "check_rate_style_swaption_helper_consistency",
        ),
        categories={
            "universal": ("check_non_negativity", "check_price_sanity"),
            "route_specific": ("check_rate_style_swaption_helper_consistency",),
        },
    )

    execution = execute_validation_bundle(
        bundle,
        validation_level="fast",
        test_payoff=object(),
        market_state=object(),
        payoff_factory=lambda: object(),
        market_state_factory=lambda **kwargs: object(),
    )

    assert execution.failures == ()
    assert execution.executed_checks == (
        "check_non_negativity",
        "check_price_sanity",
        "check_rate_style_swaption_helper_consistency",
    )
    assert calls == list(execution.executed_checks)
