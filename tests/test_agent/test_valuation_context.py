"""Tests for tranche-1 valuation-context normalization."""

from __future__ import annotations

from types import SimpleNamespace


def test_build_valuation_context_normalizes_requested_outputs_and_snapshot_metadata():
    from trellis.agent.valuation_context import build_valuation_context

    ctx = build_valuation_context(
        market_snapshot=SimpleNamespace(source="mock"),
        model_spec="black_scholes",
        reporting_currency="USD",
        requested_outputs=["PV", "vega", "scenario"],
    )

    assert ctx.market_source == "mock"
    assert ctx.market_snapshot_handle == "mock"
    assert ctx.model_spec == "black_scholes"
    assert ctx.reporting_policy.reporting_currency == "USD"
    assert ctx.requested_outputs == ("price", "vega", "scenario_pnl")


def test_normalize_valuation_context_merges_legacy_requested_measures():
    from trellis.agent.valuation_context import (
        ReportingPolicy,
        ValuationContext,
        normalize_valuation_context,
    )

    ctx = ValuationContext(
        market_source="manual",
        requested_outputs=("price",),
        reporting_policy=ReportingPolicy(reporting_currency="USD"),
    )

    normalized = normalize_valuation_context(
        ctx,
        requested_measures=["price", "delta"],
    )

    assert normalized.market_source == "manual"
    assert normalized.reporting_policy.reporting_currency == "USD"
    assert normalized.requested_outputs == ("price", "delta")
