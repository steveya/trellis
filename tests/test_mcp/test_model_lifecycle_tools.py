"""Tool-contract tests for governed MCP model lifecycle operations."""

from __future__ import annotations

import pytest


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


def _candidate_payload() -> dict[str, object]:
    return {
        "structured_trade": _trade_payload(),
        "model_id": "vanilla_option_candidate",
        "version": "v1",
        "method_family": "analytical",
        "engine_binding": {
            "engine_id": "pricing_engine.local",
            "version": "1",
            "adapter_id": "european_option_analytical",
        },
        "implementation_source": "class VanillaOptionCandidate:\n    pass\n",
        "module_path": "trellis/instruments/_agent/vanilla_option_candidate.py",
        "validation_plan": {
            "bundle": "deterministic_manifest_v1",
            "checks": ["contract_summary", "methodology_summary", "engine_binding"],
        },
        "assumptions": ["black76_inputs"],
        "actor": "mcp_test",
        "reason": "seed_candidate",
    }


def test_generate_candidate_creates_draft_version_and_persists_artifacts(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )

    payload = server.call_tool("trellis.model.generate_candidate", _candidate_payload())

    assert payload["model"]["model_id"] == "vanilla_option_candidate"
    assert payload["model"]["status"] == "draft"
    assert payload["version"]["version"] == "v1"
    assert payload["version"]["status"] == "draft"
    assert payload["artifact_uris"]["contract"] == (
        "trellis://models/vanilla_option_candidate/versions/v1/contract"
    )
    assert payload["artifact_uris"]["code"] == (
        "trellis://models/vanilla_option_candidate/versions/v1/code"
    )

    stored_version = server.services.model_registry.get_version("vanilla_option_candidate", "v1")

    assert stored_version is not None
    assert stored_version.status.value == "draft"
    assert stored_version.artifacts["contract_uri"] == payload["artifact_uris"]["contract"]
    assert stored_version.artifacts["code_uri"] == payload["artifact_uris"]["code"]


def test_validate_candidate_persists_report_without_auto_approving(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    server.call_tool("trellis.model.generate_candidate", _candidate_payload())

    payload = server.call_tool(
        "trellis.model.validate",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v1",
            "actor": "validator",
            "reason": "deterministic_validation",
        },
    )

    assert payload["validation"]["status"] == "passed"
    assert payload["validation"]["summary"]["all_checks_passed"] is True
    assert payload["version"]["status"] == "draft"
    assert payload["model"]["latest_validated_version"] == ""

    stored_validation = server.services.validation_store.get_validation(
        payload["validation"]["validation_id"]
    )
    stored_version = server.services.model_registry.get_version("vanilla_option_candidate", "v1")

    assert stored_validation is not None
    assert stored_version is not None
    assert stored_version.status.value == "draft"
    assert payload["validation"]["validation_id"] in stored_version.validation_refs


def test_promote_controls_execution_eligibility_and_deprecation(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    server.call_tool("trellis.model.generate_candidate", _candidate_payload())
    server.call_tool(
        "trellis.model.validate",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v1",
            "actor": "validator",
            "reason": "deterministic_validation",
        },
    )
    server.call_tool(
        "trellis.providers.configure",
        {
            "session_id": "sess_lifecycle",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.test_live"},
                }
            },
        },
    )
    server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_lifecycle", "run_mode": "production"},
    )

    blocked_draft = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_lifecycle",
            "structured_trade": _trade_payload(),
            "valuation_date": "2026-04-04",
        },
    )
    assert blocked_draft["status"] == "blocked"
    assert blocked_draft["result"]["reason"] == "no_approved_model_match"

    validated = server.call_tool(
        "trellis.model.promote",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v1",
            "to_status": "validated",
            "actor": "reviewer",
            "reason": "validation_review_complete",
        },
    )
    assert validated["version"]["status"] == "validated"

    blocked_validated = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_lifecycle",
            "structured_trade": _trade_payload(),
            "valuation_date": "2026-04-04",
        },
    )
    assert blocked_validated["status"] == "blocked"
    assert blocked_validated["result"]["selected_candidate"]["status"] == "validated"

    approved = server.call_tool(
        "trellis.model.promote",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v1",
            "to_status": "approved",
            "actor": "reviewer",
            "reason": "manual_approval",
        },
    )
    assert approved["version"]["status"] == "approved"

    succeeded = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_lifecycle",
            "structured_trade": _trade_payload(),
            "valuation_date": "2026-04-04",
        },
    )
    assert succeeded["status"] == "succeeded"
    assert succeeded["provenance"]["model_id"] == "vanilla_option_candidate"

    deprecated = server.call_tool(
        "trellis.model.promote",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v1",
            "to_status": "deprecated",
            "actor": "reviewer",
            "reason": "withdrawn",
        },
    )
    assert deprecated["version"]["status"] == "deprecated"

    blocked_deprecated = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_lifecycle",
            "structured_trade": _trade_payload(),
            "valuation_date": "2026-04-04",
        },
    )
    assert blocked_deprecated["status"] == "blocked"
    assert blocked_deprecated["result"]["selected_candidate"]["status"] == "deprecated"


def test_promote_requires_fresh_validation_for_persisted_version(tmp_path):
    from trellis.mcp.errors import TrellisMcpError
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    server.call_tool("trellis.model.generate_candidate", _candidate_payload())
    server.call_tool(
        "trellis.model.validate",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v1",
            "actor": "validator",
            "reason": "deterministic_validation",
        },
    )
    server.call_tool(
        "trellis.model.persist",
        {
            "model_id": "vanilla_option_candidate",
            "base_version": "v1",
            "new_version": "v2",
            "actor": "model_store",
            "reason": "metadata_revision_only",
        },
    )

    with pytest.raises(TrellisMcpError) as excinfo:
        server.call_tool(
            "trellis.model.promote",
            {
                "model_id": "vanilla_option_candidate",
                "version": "v2",
                "to_status": "validated",
                "actor": "reviewer",
                "reason": "attempt_without_fresh_validation",
            },
        )
    assert excinfo.value.code == "validation_required"

    server.call_tool(
        "trellis.model.validate",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v2",
            "actor": "validator",
            "reason": "deterministic_validation",
        },
    )
    validated = server.call_tool(
        "trellis.model.promote",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v2",
            "to_status": "validated",
            "actor": "reviewer",
            "reason": "validated_after_fresh_report",
        },
    )
    assert validated["version"]["status"] == "validated"
