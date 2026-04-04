"""Opt-in local demo bootstrap helpers for the Trellis MCP transport."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from trellis.data.mock import MockDataProvider
from trellis.models.vol_surface import FlatVol
from trellis.platform.context import ProviderBinding, ProviderBindingSet, ProviderBindings, RunMode
from trellis.platform.models import ModelLifecycleStatus, ModelRecord, ModelVersionRecord
from trellis.platform.providers import ProviderRecord, ProviderRegistry


_DEMO_MODEL_ID = "vanilla_option_demo"
_DEMO_MODEL_VERSION = "v1"


class DemoMockDataProvider(MockDataProvider):
    """Embedded mock provider variant tuned for prompt-flow option demos."""

    def fetch_market_snapshot(self, as_of: date | None = None):
        snapshot = super().fetch_market_snapshot(as_of=as_of)
        demo_vol_name = "aapl_demo_flat"
        demo_metadata = {
            **dict(snapshot.metadata),
            "demo_mode": True,
            "demo_profile": "local_prompt_flow",
        }
        demo_provenance = {
            **dict(snapshot.provenance),
            "demo_mode": True,
            "source_ref": "local_prompt_flow_demo",
        }
        return replace(
            snapshot,
            vol_surfaces={**dict(snapshot.vol_surfaces), demo_vol_name: FlatVol(0.20)},
            metadata=demo_metadata,
            provenance=demo_provenance,
            default_vol_surface=demo_vol_name,
            default_underlier_spot="AAPL",
        )


def build_demo_provider_registry() -> ProviderRegistry:
    """Return a provider registry with the mock provider tuned for local demos."""
    return ProviderRegistry(
        records=(
            ProviderRecord(
                provider_id="market_data.mock",
                kind="market_data",
                display_name="Embedded Mock Market Data (Local Demo)",
                capabilities=(
                    "discount_curve",
                    "market_snapshot",
                    "underlier_spot",
                    "black_vol_surface",
                    "forecast_curve",
                    "credit_curve",
                    "fx_rates",
                ),
                connection_mode="embedded",
                is_mock=True,
                supports_snapshots=True,
                source="mock",
                config_summary={"profile": "local_prompt_flow"},
            ),
        ),
        provider_factories={"market_data.mock": DemoMockDataProvider},
    )


def enable_local_demo_mode(services, *, session_id: str = "demo") -> dict[str, object]:
    """Seed a sandbox/mock session plus one approved vanilla-option model for demos."""
    _ensure_demo_model(services.model_registry)
    return _ensure_demo_session(services, session_id=session_id)


def _ensure_demo_model(model_registry) -> None:
    model = model_registry.get_model(_DEMO_MODEL_ID)
    if model is None:
        model_registry.create_model(
            ModelRecord(
                model_id=_DEMO_MODEL_ID,
                semantic_id="vanilla_option",
                semantic_version="c2.1",
                product_family="equity_option",
                instrument_class="european_option",
                payoff_family="vanilla_option",
                exercise_style="european",
                underlier_structure="single_underlier",
                payout_currency="USD",
                reporting_currency="USD",
                required_market_data=("discount_curve", "underlier_spot", "black_vol_surface"),
                supported_method_families=("analytical",),
                tags=("builtin", "demo", "mcp"),
                metadata={"demo_mode": True, "seed_source": "trellis_mcp_local_demo"},
            )
        )

    version = model_registry.get_version(_DEMO_MODEL_ID, _DEMO_MODEL_VERSION)
    if version is None:
        version = model_registry.create_version(
            ModelVersionRecord(
                model_id=_DEMO_MODEL_ID,
                version=_DEMO_MODEL_VERSION,
                contract_summary={
                    "semantic_id": "vanilla_option",
                    "exercise_style": "european",
                    "underlier_structure": "single_underlier",
                },
                methodology_summary={
                    "method_family": "analytical",
                    "candidate_source": "builtin_demo",
                },
                engine_binding={
                    "engine_id": "pricing_engine.local",
                    "version": "1",
                    "adapter_id": "european_option_analytical",
                },
                metadata={"demo_mode": True},
            ),
            actor="mcp_demo",
            reason="seed_demo_model",
            notes="Local MCP prompt-flow demo seed",
            metadata={"demo_mode": True},
        )
        artifact_uris = {
            "contract_uri": model_registry.write_version_artifact(
                _DEMO_MODEL_ID,
                _DEMO_MODEL_VERSION,
                "contract",
                dict(version.contract_summary),
            ),
            "methodology_uri": model_registry.write_version_artifact(
                _DEMO_MODEL_ID,
                _DEMO_MODEL_VERSION,
                "methodology",
                dict(version.methodology_summary),
            ),
            "code_uri": model_registry.write_version_artifact(
                _DEMO_MODEL_ID,
                _DEMO_MODEL_VERSION,
                "code",
                (
                    "from trellis.instruments.option import EuropeanOption\n"
                    "\n"
                    "# Demo-only governed MCP seed model using the checked analytical route.\n"
                ),
            ),
            "validation-plan_uri": model_registry.write_version_artifact(
                _DEMO_MODEL_ID,
                _DEMO_MODEL_VERSION,
                "validation-plan",
                {"profile": "local_prompt_flow_demo"},
            ),
        }
        version = model_registry.save_version(
            replace(
                version,
                artifacts={**dict(version.artifacts), **artifact_uris},
            )
        )

    if version.status is ModelLifecycleStatus.DRAFT:
        version = model_registry.transition_version(
            _DEMO_MODEL_ID,
            _DEMO_MODEL_VERSION,
            ModelLifecycleStatus.VALIDATED,
            actor="mcp_demo",
            reason="seed_demo_validation",
            notes="Local MCP prompt-flow demo seed",
        )
    if version.status is ModelLifecycleStatus.VALIDATED:
        model_registry.transition_version(
            _DEMO_MODEL_ID,
            _DEMO_MODEL_VERSION,
            ModelLifecycleStatus.APPROVED,
            actor="mcp_demo",
            reason="seed_demo_approval",
            notes="Local MCP prompt-flow demo seed",
        )


def _ensure_demo_session(services, *, session_id: str) -> dict[str, object]:
    session_service = services.session_service
    record = session_service.ensure_record(session_id)
    persisted = session_service.save(
        replace(
            record,
            run_mode=RunMode.SANDBOX,
            provider_bindings=ProviderBindings(
                market_data=ProviderBindingSet(
                    primary=ProviderBinding("market_data.mock"),
                    fallback=None,
                ),
                pricing_engine=record.provider_bindings.pricing_engine,
                model_store=record.provider_bindings.model_store,
                validation_engine=record.provider_bindings.validation_engine,
            ),
            active_policy="policy_bundle.sandbox.default",
            allow_mock_data=True,
            require_provider_disclosure=False,
            metadata={**dict(record.metadata), "demo_mode": True},
            connected_providers=tuple(
                provider.provider_id for provider in services.provider_registry.list_providers()
            ),
        )
    )
    return session_service.get_context(persisted.session_id)


__all__ = [
    "build_demo_provider_registry",
    "enable_local_demo_mode",
]
