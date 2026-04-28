"""Tests for the governed Trellis MCP resource surface."""

from __future__ import annotations


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
        "validation_plan": {"bundle": "deterministic_manifest_v1"},
        "actor": "mcp_test",
        "reason": "seed_candidate",
    }


def _cycle_report() -> dict[str, object]:
    return {
        "request_id": "executor_resource_candidate",
        "status": "succeeded",
        "outcome": "build_completed",
        "success": True,
        "pricing_method": "analytical",
        "validation_contract_id": "validation:vanilla_option:analytical",
        "stage_statuses": {
            "quant": "passed",
            "validation_bundle": "passed",
            "critic": "passed",
            "arbiter": "passed",
            "model_validator": "skipped",
        },
        "failure_count": 0,
        "deterministic_blockers": [],
        "conceptual_blockers": [],
        "calibration_blockers": [],
        "residual_limitations": [],
        "residual_risks": [],
    }


def test_resource_templates_and_model_policy_resources(tmp_path):
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
            "reason": "manifest_validation",
        },
    )

    templates = server.list_resources()
    assert "trellis://models/{model_id}" in templates
    assert "trellis://runs/{run_id}/audit" in templates
    assert "trellis://providers/{provider_id}" in templates
    assert "trellis://policies/{policy_id}" in templates

    model = server.read_resource("trellis://models/vanilla_option_candidate")
    versions = server.read_resource("trellis://models/vanilla_option_candidate/versions")
    contract = server.read_resource("trellis://models/vanilla_option_candidate/versions/v1/contract")
    code = server.read_resource("trellis://models/vanilla_option_candidate/versions/v1/code")
    validation = server.read_resource(
        "trellis://models/vanilla_option_candidate/versions/v1/validation-report"
    )
    provider = server.read_resource("trellis://providers/market_data.test_live")
    policy = server.read_resource("trellis://policies/policy_bundle.research.default")

    assert model["model"]["model_id"] == "vanilla_option_candidate"
    assert [item["version"] for item in versions["versions"]] == ["v1"]
    assert contract["contract"]["semantic_id"] == "vanilla_option"
    assert "VanillaOptionCandidate" in code["source_code"]
    assert validation["validation_report"]["all_checks_passed"] is True
    assert provider["provider"]["provider_id"] == "market_data.test_live"
    assert policy["policy"]["policy_id"] == "policy_bundle.research.default"


def test_run_and_snapshot_resources_resolve_from_persisted_state(tmp_path):
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
            "reason": "manifest_validation",
        },
    )
    server.call_tool(
        "trellis.model.promote",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v1",
            "to_status": "validated",
            "actor": "reviewer",
            "reason": "validation_review_complete",
        },
    )
    server.call_tool(
        "trellis.model.promote",
        {
            "model_id": "vanilla_option_candidate",
            "version": "v1",
            "to_status": "approved",
            "actor": "reviewer",
            "reason": "manual_approval",
            "metadata": {"cycle_report": _cycle_report()},
        },
    )
    server.call_tool(
        "trellis.providers.configure",
        {
            "session_id": "sess_resources",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.test_live"},
                }
            },
        },
    )
    server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_resources", "run_mode": "production"},
    )
    priced = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_resources",
            "structured_trade": _trade_payload(),
            "valuation_date": "2026-04-04",
        },
    )
    bundle = server.call_tool(
        "trellis.snapshot.persist_run",
        {
            "run_id": priced["run_id"],
            "tolerances": {"price_abs": 1e-6},
            "random_seed": 5,
            "calendars": ["NYSE"],
        },
    )

    run = server.read_resource(f"trellis://runs/{priced['run_id']}")
    audit = server.read_resource(f"trellis://runs/{priced['run_id']}/audit")
    inputs = server.read_resource(f"trellis://runs/{priced['run_id']}/inputs")
    outputs = server.read_resource(f"trellis://runs/{priced['run_id']}/outputs")
    logs = server.read_resource(f"trellis://runs/{priced['run_id']}/logs")
    snapshot = server.read_resource(f"trellis://market-snapshots/{bundle['snapshot']['snapshot_id']}")

    assert run["run"]["run_id"] == priced["run_id"]
    assert audit["audit"]["outputs"]["price"] == priced["result"]["price"]
    assert inputs["inputs"]["trade_identity"]["semantic_id"] == "vanilla_option"
    assert outputs["outputs"]["price"] == priced["result"]["price"]
    assert "events" in logs
    assert snapshot["snapshot"]["payload"]["run_id"] == priced["run_id"]
    assert snapshot["snapshot"]["payload"]["random_seed"] == 5
