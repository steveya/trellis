"""Tests for governed provider bindings and snapshot resolution."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.data.base import BaseDataProvider
from trellis.data.mock import MockDataProvider
from trellis.platform.context import (
    ExecutionContext,
    ProviderBinding,
    ProviderBindingSet,
    ProviderBindings,
    RunMode,
)


SETTLE = date(2024, 11, 15)
SAMPLE_YIELDS = {
    0.25: 0.045,
    0.5: 0.046,
    1.0: 0.047,
    2.0: 0.048,
    5.0: 0.045,
    10.0: 0.044,
    30.0: 0.046,
}


class YieldOnlyProvider(BaseDataProvider):
    """Simple test provider that exposes only Treasury-like yields."""

    def fetch_yields(self, as_of: date | None = None) -> dict[float, float]:
        return dict(SAMPLE_YIELDS)


class BrokenProvider(BaseDataProvider):
    """Provider stub that fails during governed resolution."""

    def fetch_yields(self, as_of: date | None = None) -> dict[float, float]:
        raise RuntimeError("provider offline")


def _governed_context(
    *,
    primary: str | None,
    fallback: str | None = None,
    run_mode: RunMode = RunMode.RESEARCH,
    allow_mock_data: bool = False,
) -> ExecutionContext:
    return ExecutionContext(
        session_id="sess_provider_test",
        run_mode=run_mode,
        provider_bindings=ProviderBindings(
            market_data=ProviderBindingSet(
                primary=None if primary is None else ProviderBinding(primary),
                fallback=None if fallback is None else ProviderBinding(fallback),
            )
        ),
        allow_mock_data=allow_mock_data,
        require_provider_disclosure=True,
        policy_bundle_id=f"policy_bundle.{run_mode.value}.default",
    )


def test_provider_registry_exposes_stable_market_data_records():
    from trellis.platform.providers import ProviderRegistry

    registry = ProviderRegistry()

    record = registry.get_provider("market_data.mock")

    assert record.provider_id == "market_data.mock"
    assert record.kind == "market_data"
    assert record.is_mock is True
    assert record.supports_snapshots is True
    assert "market_snapshot" in record.capabilities
    assert "fixing_history" in record.capabilities


def test_governed_resolution_assigns_provider_and_snapshot_identity():
    from trellis.platform.providers import ProviderRecord, ProviderRegistry, resolve_governed_market_snapshot

    registry = ProviderRegistry(
        records=(
            ProviderRecord(
                provider_id="market_data.treasury_gov.bound",
                kind="market_data",
                display_name="Treasury.gov Bound",
                capabilities=("discount_curve", "market_snapshot"),
                connection_mode="http_pull",
                source="treasury_gov",
            ),
        ),
        provider_factories={
            "market_data.treasury_gov.bound": YieldOnlyProvider,
        },
    )

    snapshot = resolve_governed_market_snapshot(
        execution_context=_governed_context(primary="market_data.treasury_gov.bound"),
        as_of=SETTLE,
        registry=registry,
    )

    assert snapshot.provider_id == "market_data.treasury_gov.bound"
    assert snapshot.market_snapshot_id == snapshot.snapshot_id
    assert snapshot.snapshot_id.startswith("snapshot_")
    assert snapshot.provenance["provider_id"] == "market_data.treasury_gov.bound"
    assert snapshot.provenance["source"] == "treasury_gov"


def test_governed_resolution_requires_explicit_provider_binding():
    from trellis.platform.providers import ProviderBindingRequiredError, resolve_governed_market_snapshot

    with pytest.raises(ProviderBindingRequiredError, match="market-data provider binding"):
        resolve_governed_market_snapshot(
            execution_context=_governed_context(primary=None),
            as_of=SETTLE,
        )


def test_governed_resolution_does_not_silently_fall_back_to_mock():
    from trellis.platform.providers import ProviderRecord, ProviderRegistry, ProviderResolutionError, resolve_governed_market_snapshot

    registry = ProviderRegistry(
        records=(
            ProviderRecord(
                provider_id="market_data.live_unit",
                kind="market_data",
                display_name="Live Unit Provider",
                capabilities=("discount_curve", "market_snapshot"),
                connection_mode="http_pull",
                source="treasury_gov",
            ),
        ),
        provider_factories={
            "market_data.live_unit": BrokenProvider,
        },
    )

    with pytest.raises(ProviderResolutionError, match="market_data.live_unit"):
        resolve_governed_market_snapshot(
            execution_context=_governed_context(primary="market_data.live_unit"),
            as_of=SETTLE,
            registry=registry,
        )


def test_governed_resolution_allows_explicit_mock_fallback_only_when_policy_allows():
    from trellis.platform.providers import MockDataNotAllowedError, ProviderRecord, ProviderRegistry, resolve_governed_market_snapshot

    registry = ProviderRegistry(
        records=(
            ProviderRecord(
                provider_id="market_data.live_unit",
                kind="market_data",
                display_name="Live Unit Provider",
                capabilities=("discount_curve", "market_snapshot"),
                connection_mode="http_pull",
                source="treasury_gov",
            ),
            ProviderRecord(
                provider_id="market_data.mock_unit",
                kind="market_data",
                display_name="Mock Unit Provider",
                capabilities=("discount_curve", "market_snapshot"),
                connection_mode="embedded",
                is_mock=True,
                source="mock",
            ),
        ),
        provider_factories={
            "market_data.live_unit": BrokenProvider,
            "market_data.mock_unit": MockDataProvider,
        },
    )

    with pytest.raises(MockDataNotAllowedError, match="market_data.mock_unit"):
        resolve_governed_market_snapshot(
            execution_context=_governed_context(
                primary="market_data.live_unit",
                fallback="market_data.mock_unit",
                run_mode=RunMode.RESEARCH,
                allow_mock_data=False,
            ),
            as_of=SETTLE,
            registry=registry,
        )

    snapshot = resolve_governed_market_snapshot(
        execution_context=_governed_context(
            primary="market_data.live_unit",
            fallback="market_data.mock_unit",
            run_mode=RunMode.SANDBOX,
            allow_mock_data=True,
        ),
        as_of=SETTLE,
        registry=registry,
    )

    assert snapshot.provider_id == "market_data.mock_unit"
    assert snapshot.source == "mock"
    assert snapshot.provenance["resolution_kind"] == "fallback"
