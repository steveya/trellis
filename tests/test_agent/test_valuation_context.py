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


def test_engine_model_spec_validation_and_summary_are_stable():
    from trellis.agent.valuation_context import (
        EngineModelSpec,
        PotentialSpec,
        RatesCurveRoleSpec,
        SourceSpec,
        build_valuation_context,
        validate_engine_model_spec,
        valuation_context_summary,
    )

    engine_model_spec = EngineModelSpec(
        model_family="rates",
        model_name="hull_white_1f",
        state_semantics=("short_rate",),
        potential=PotentialSpec(discount_term="risk_free_rate"),
        sources=(SourceSpec(source_kind="coupon_stream"),),
        calibration_requirements=("bootstrap_curve", "fit_hw_strip"),
        backend_hints=("lattice",),
        parameter_overrides={"mean_reversion": 0.05, "sigma": 0.01},
        rates_curve_roles=RatesCurveRoleSpec(
            discount_curve_role="discount_curve",
            forecast_curve_role="forward_curve",
        ),
    )

    assert validate_engine_model_spec(engine_model_spec) == ()

    context = build_valuation_context(
        engine_model_spec=engine_model_spec,
        reporting_currency="USD",
    )

    assert context.engine_model_spec == engine_model_spec
    summary = valuation_context_summary(context)
    assert summary["engine_model_spec"]["model_family"] == "rates"
    assert summary["engine_model_spec"]["model_name"] == "hull_white_1f"
    assert summary["engine_model_spec"]["rates_curve_roles"] == {
        "discount_curve_role": "discount_curve",
        "forecast_curve_role": "forward_curve",
        "rate_index": "",
    }
    assert summary["engine_model_spec"]["parameter_overrides"] == {
        "mean_reversion": 0.05,
        "sigma": 0.01,
    }


def test_legacy_model_spec_shim_builds_engine_model_spec_for_supported_rates_model():
    from trellis.agent.valuation_context import build_valuation_context

    context = build_valuation_context(model_spec="hull_white")

    assert context.model_spec == "hull_white"
    assert context.engine_model_spec is not None
    assert context.engine_model_spec.model_family == "rates"
    assert context.engine_model_spec.model_name == "hull_white_1f"
    assert context.engine_model_spec.rates_curve_roles is not None
