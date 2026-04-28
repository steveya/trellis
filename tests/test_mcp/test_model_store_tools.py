"""Tool-contract tests for governed MCP model-store and reproducibility surfaces."""

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
        "request_id": "executor_model_store_candidate",
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


def test_persist_versions_list_history_and_diff(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    server.call_tool("trellis.model.generate_candidate", _candidate_payload())

    persisted = server.call_tool(
        "trellis.model.persist",
        {
            "model_id": "vanilla_option_candidate",
            "base_version": "v1",
            "new_version": "v2",
            "actor": "model_store",
            "reason": "persist_revision",
            "methodology_summary": {
                "method_family": "analytical",
                "revision": "refined_formula",
            },
            "implementation_source": (
                "class VanillaOptionCandidate:\n"
                "    def price(self):\n"
                "        return 1.0\n"
            ),
            "module_path": "trellis/instruments/_agent/vanilla_option_candidate_v2.py",
        },
    )

    assert persisted["version"]["version"] == "v2"
    assert persisted["version"]["lineage"]["parent_version"] == "v1"

    versions = server.call_tool(
        "trellis.model.versions.list",
        {"model_id": "vanilla_option_candidate"},
    )
    assert [item["version"] for item in versions["versions"]] == ["v1", "v2"]

    diff = server.call_tool(
        "trellis.model.diff",
        {
            "model_id": "vanilla_option_candidate",
            "left_version": "v1",
            "right_version": "v2",
        },
    )["diff"]

    assert diff["model_id"] == "vanilla_option_candidate"
    assert diff["methodology_summary_changed"] is True
    assert diff["code_changed"] is True
    assert diff["lineage"]["right"]["parent_version"] == "v1"


def test_persisted_version_starts_unvalidated_and_keeps_code_resource(tmp_path):
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

    persisted = server.call_tool(
        "trellis.model.persist",
        {
            "model_id": "vanilla_option_candidate",
            "base_version": "v1",
            "new_version": "v2",
            "actor": "model_store",
            "reason": "metadata_revision_only",
        },
    )

    assert persisted["version"]["version"] == "v2"
    assert persisted["version"]["validation_summary"] == {}
    assert persisted["version"]["validation_refs"] == []

    code_resource = server.read_resource(
        "trellis://models/vanilla_option_candidate/versions/v2/code"
    )
    assert "VanillaOptionCandidate" in code_resource["source_code"]
    assert code_resource["module_path"] == (
        "trellis/instruments/_agent/vanilla_option_candidate.py"
    )


def test_persist_run_creates_reproducibility_bundle_and_attaches_it_to_run(tmp_path):
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
            "session_id": "sess_bundle",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.test_live"},
                }
            },
        },
    )
    server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_bundle", "run_mode": "production"},
    )
    priced = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_bundle",
            "structured_trade": _trade_payload(),
            "valuation_date": "2026-04-04",
        },
    )

    bundle = server.call_tool(
        "trellis.snapshot.persist_run",
        {
            "run_id": priced["run_id"],
            "tolerances": {"price_abs": 1e-6},
            "random_seed": 7,
            "calendars": ["NYSE"],
        },
    )

    assert bundle["bundle_uri"] == f"trellis://market-snapshots/{bundle['snapshot']['snapshot_id']}"
    assert bundle["snapshot"]["source"] == "reproducibility_bundle"
    assert bundle["snapshot"]["payload"]["run_id"] == priced["run_id"]
    assert bundle["snapshot"]["payload"]["random_seed"] == 7
    assert bundle["snapshot"]["payload"]["tolerances"]["price_abs"] == 1e-6

    run = server.call_tool("trellis.run.get", {"run_id": priced["run_id"]})["run"]
    assert any(
        artifact["artifact_kind"] == "reproducibility_bundle"
        for artifact in run["artifacts"]
    )


def test_persist_run_embeds_concrete_market_snapshot_for_live_provider_run(tmp_path):
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
            "session_id": "sess_bundle_snapshot_contract",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.test_live"},
                }
            },
        },
    )
    server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_bundle_snapshot_contract", "run_mode": "production"},
    )
    priced = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_bundle_snapshot_contract",
            "structured_trade": _trade_payload(),
            "valuation_date": "2026-04-04",
        },
    )

    bundle = server.call_tool(
        "trellis.snapshot.persist_run",
        {
            "run_id": priced["run_id"],
            "tolerances": {"price_abs": 1e-6},
        },
    )

    serialized_snapshot = bundle["snapshot"]["payload"]["market_snapshot"]
    snapshot_contract = serialized_snapshot["payload"]["snapshot_contract"]

    assert snapshot_contract["as_of"] == "2026-04-04"
    assert snapshot_contract["discount_curves"]["discount"]["kind"] == "zero_rates"
    assert snapshot_contract["vol_surfaces"]["default"]["kind"] == "flat"
    assert snapshot_contract["underlier_spots"] == {"AAPL": 123.0}

    rehydrated = server.services.snapshot_service.load_market_snapshot(
        bundle["snapshot"]["snapshot_id"]
    )
    assert rehydrated.as_of.isoformat() == "2026-04-04"
    assert rehydrated.discount_curve().discount(1.0) > 0.0
    assert rehydrated.vol_surface().black_vol(1.0, 123.0) == 0.20
