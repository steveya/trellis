"""Tests for valuation-context normalization."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot


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


def test_build_valuation_context_uses_snapshot_id_when_available():
    from trellis.agent.valuation_context import build_valuation_context

    snapshot = MarketSnapshot(
        as_of=date(2024, 11, 15),
        source="mock",
        discount_curves={"discount": YieldCurve.flat(0.045)},
        default_discount_curve="discount",
        provenance={
            "source": "mock",
            "source_kind": "synthetic_snapshot",
            "provider_id": "market_data.mock",
            "snapshot_id": "snapshot_mock_20241115",
        },
    )

    context = build_valuation_context(market_snapshot=snapshot)

    assert context.market_source == "mock"
    assert context.market_snapshot_handle == "snapshot_mock_20241115"
