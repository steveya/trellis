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
