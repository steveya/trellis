"""Tests for governed platform runtime records."""

from __future__ import annotations

from datetime import date

from trellis.book import Book
from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot
from trellis.instruments.bond import Bond
from trellis.pipeline import Pipeline
from trellis.session import Session


SETTLE = date(2024, 11, 15)


def _snapshot(source: str = "treasury_gov") -> MarketSnapshot:
    return MarketSnapshot(
        as_of=SETTLE,
        source=source,
        discount_curves={"discount": YieldCurve.flat(0.045)},
        default_discount_curve="discount",
        provenance={"source": source, "source_kind": "direct_quote"},
    )


def _book() -> Book:
    return Book(
        {
            "10Y": Bond(
                face=100,
                coupon=0.05,
                maturity_date=date(2034, 11, 15),
                maturity=10,
                frequency=2,
            )
        }
    )


def test_platform_requests_namespace_reexports_the_request_compiler():
    from trellis.agent.platform_requests import PlatformRequest as AgentPlatformRequest
    from trellis.platform.requests import PlatformRequest as PlatformPlatformRequest

    assert PlatformPlatformRequest is AgentPlatformRequest


def test_execution_context_round_trips_through_serialized_payload():
    from trellis.platform.context import (
        ExecutionContext,
        ProviderBinding,
        ProviderBindingSet,
        ProviderBindings,
        RunMode,
    )

    context = ExecutionContext(
        session_id="sess_ctx_001",
        run_mode="research",
        provider_bindings=ProviderBindings(
            market_data=ProviderBindingSet(
                primary=ProviderBinding("market_data.treasury_gov"),
            ),
            pricing_engine=ProviderBindingSet(
                primary=ProviderBinding("pricing_engine.local"),
            ),
            model_store=ProviderBindingSet(
                primary=ProviderBinding("model_store.local"),
            ),
            validation_engine=ProviderBindingSet(
                primary=ProviderBinding("validation_engine.local"),
            ),
        ),
        policy_bundle_id="policy_bundle.research.default",
        allow_mock_data=False,
        metadata={"caller": "unit"},
    )

    restored = ExecutionContext.from_dict(context.to_dict())

    assert restored == context
    assert restored.run_mode is RunMode.RESEARCH
    assert restored.provider_bindings.market_data.primary.provider_id == "market_data.treasury_gov"
    assert restored.metadata["caller"] == "unit"


def test_session_to_execution_context_maps_mock_sessions_to_explicit_sandbox_runtime():
    from trellis.platform.context import RunMode

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)

    context = session.to_execution_context()

    assert context.run_mode is RunMode.SANDBOX
    assert context.allow_mock_data is True
    assert context.provider_bindings.market_data.primary.provider_id == "market_data.mock"
    assert context.policy_bundle_id == "policy_bundle.sandbox.default"


def test_session_to_execution_context_uses_snapshot_provider_identity_for_live_data():
    from trellis.platform.context import RunMode

    session = Session(market_snapshot=_snapshot("treasury_gov"), settlement=SETTLE)

    context = session.to_execution_context()

    assert context.run_mode is RunMode.RESEARCH
    assert context.allow_mock_data is False
    assert context.provider_bindings.market_data.primary.provider_id == "market_data.treasury_gov"
    assert context.policy_bundle_id == "policy_bundle.research.default"


def test_session_to_execution_context_prefers_snapshot_provenance_provider_id():
    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="treasury_gov",
        discount_curves={"discount": YieldCurve.flat(0.045)},
        default_discount_curve="discount",
        provenance={
            "source": "treasury_gov",
            "source_kind": "provider_snapshot",
            "provider_id": "market_data.treasury_gov.bound",
            "snapshot_id": "snapshot_bound_001",
        },
    )
    session = Session(market_snapshot=snapshot, settlement=SETTLE)

    context = session.to_execution_context()

    assert context.provider_bindings.market_data.primary.provider_id == "market_data.treasury_gov.bound"


def test_pipeline_to_execution_context_uses_configured_runtime_surface():
    from trellis.platform.context import RunMode

    pipeline = (
        Pipeline()
        .instruments(_book())
        .market_data(source="mock", as_of="2024-11-15")
    )

    context = pipeline.to_execution_context()

    assert context.run_mode is RunMode.SANDBOX
    assert context.provider_bindings.market_data.primary.provider_id == "market_data.mock"
