"""End-to-end tests for the governed MCP ``trellis.price.trade`` MVP."""

from __future__ import annotations

from datetime import date


def _provider_registry():
    from trellis.curves.yield_curve import YieldCurve
    from trellis.data.schema import MarketSnapshot
    from trellis.models.vol_surface import FlatVol
    from trellis.platform.providers import ProviderRecord, ProviderRegistry

    class StaticLiveProvider:
        def fetch_market_snapshot(self, as_of):
            return MarketSnapshot(
                as_of=as_of,
                source="test_live",
                discount_curves={"discount": YieldCurve.flat(0.05)},
                vol_surfaces={"default": FlatVol(0.20)},
                underlier_spots={"AAPL": 123.0},
                default_discount_curve="discount",
                default_vol_surface="default",
                default_underlier_spot="AAPL",
                provenance={
                    "source": "test_live",
                    "source_kind": "provider_snapshot",
                    "source_ref": "fetch_market_snapshot",
                },
            )

    return ProviderRegistry(
        records=(
            ProviderRecord(
                provider_id="market_data.test_live",
                kind="market_data",
                display_name="Static Test Live Provider",
                capabilities=("discount_curve", "market_snapshot", "underlier_spot", "black_vol_surface"),
                source="test_live",
            ),
        ),
        provider_factories={"market_data.test_live": StaticLiveProvider},
    )


def _seed_model(registry, *, approved: bool) -> None:
    from trellis.platform.models import (
        ModelLifecycleStatus,
        ModelRecord,
        ModelVersionRecord,
    )

    registry.create_model(
        ModelRecord(
            model_id="vanilla_option_approved",
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
        )
    )
    registry.create_version(
        ModelVersionRecord(
            model_id="vanilla_option_approved",
            version="v1",
            contract_summary={"semantic_id": "vanilla_option"},
            methodology_summary={"method_family": "analytical"},
            engine_binding={
                "engine_id": "pricing_engine.local",
                "version": "1",
                "adapter_id": "european_option_analytical",
            },
        ),
        actor="builder",
        reason="seed",
    )
    registry.transition_version(
        "vanilla_option_approved",
        "v1",
        ModelLifecycleStatus.VALIDATED,
        actor="validator",
        reason="seed_validation",
    )
    if approved:
        registry.transition_version(
            "vanilla_option_approved",
            "v1",
            ModelLifecycleStatus.APPROVED,
            actor="reviewer",
            reason="seed_approval",
        )


def _trade_payload() -> dict[str, object]:
    return {
        "instrument_type": "european_option",
        "description": "European call on AAPL with strike 120 expiring 2026-12-31",
        "underliers": ("AAPL",),
        "observation_schedule": ("2026-12-31",),
        "payout_currency": "USD",
        "reporting_currency": "USD",
        "preferred_method": "analytical",
        "strike": 120.0,
        "option_type": "call",
        "notional": 1.0,
    }


def test_price_trade_executes_approved_model_and_persists_run_and_audit(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_model(server.services.model_registry, approved=True)

    server.call_tool(
        "trellis.providers.configure",
        {
            "session_id": "sess_price_success",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.test_live"},
                }
            },
        },
    )
    server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_price_success", "run_mode": "production"},
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_success",
            "structured_trade": _trade_payload(),
            "output_mode": "structured",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "succeeded"
    assert payload["result"]["price"] > 0.0
    assert payload["provenance"]["model_id"] == "vanilla_option_approved"
    assert payload["provenance"]["model_version"] == "v1"
    assert payload["provenance"]["engine_id"] == "pricing_engine.local"
    assert payload["provenance"]["provider_id"] == "market_data.test_live"
    assert payload["provenance"]["market_snapshot_id"].startswith("snapshot_")
    assert payload["audit_uri"] == f"trellis://runs/{payload['run_id']}/audit"

    run_payload = server.call_tool("trellis.run.get", {"run_id": payload["run_id"]})
    audit_payload = server.call_tool("trellis.run.get_audit", {"run_id": payload["run_id"]})

    assert run_payload["run"]["status"] == "succeeded"
    assert run_payload["run"]["selected_model"]["model_id"] == "vanilla_option_approved"
    assert audit_payload["audit"]["execution"]["selected_model"]["version"] == "v1"
    assert audit_payload["audit"]["outputs"]["price"] == payload["result"]["price"]


def test_price_trade_blocks_when_provider_binding_is_missing(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_model(server.services.model_registry, approved=True)
    server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_price_blocked_provider", "run_mode": "production"},
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_blocked_provider",
            "structured_trade": _trade_payload(),
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "blocked"
    assert "provider_binding_required" in payload["result"]["blocker_codes"]

    run_payload = server.call_tool("trellis.run.get", {"run_id": payload["run_id"]})
    assert run_payload["run"]["status"] == "blocked"


def test_price_trade_rejects_validated_only_match_in_mvp(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_model(server.services.model_registry, approved=False)
    server.call_tool(
        "trellis.providers.configure",
        {
            "session_id": "sess_price_blocked_model",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.test_live"},
                }
            },
        },
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_blocked_model",
            "structured_trade": _trade_payload(),
            "valuation_date": date(2026, 4, 4).isoformat(),
        },
    )

    assert payload["status"] == "blocked"
    assert payload["result"]["reason"] == "no_approved_model_match"
    assert payload["result"]["selected_candidate"]["status"] == "validated"
