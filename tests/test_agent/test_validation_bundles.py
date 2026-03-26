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
