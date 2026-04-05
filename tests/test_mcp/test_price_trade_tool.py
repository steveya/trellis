"""End-to-end tests for the governed MCP ``trellis.price.trade`` MVP."""

from __future__ import annotations

from datetime import date
import json


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


def _trade_payload_with_expiry_date_alias() -> dict[str, object]:
    payload = _trade_payload()
    payload.pop("observation_schedule")
    payload["expiry_date"] = "2026-12-31"
    return payload


def _seed_range_accrual_model(registry) -> None:
    from trellis.platform.models import (
        ModelLifecycleStatus,
        ModelRecord,
        ModelVersionRecord,
    )

    registry.create_model(
        ModelRecord(
            model_id="range_accrual_checked",
            semantic_id="range_accrual",
            semantic_version="c2.1",
            product_family="rates_exotic",
            instrument_class="range_accrual",
            payoff_family="range_accrual_coupon",
            exercise_style="none",
            underlier_structure="single_curve_rate_style",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("discount_curve", "fixing_history", "forward_curve"),
            supported_method_families=("analytical",),
        )
    )
    registry.create_version(
        ModelVersionRecord(
            model_id="range_accrual_checked",
            version="v1",
            contract_summary={"semantic_id": "range_accrual"},
            methodology_summary={"method_family": "analytical"},
            engine_binding={
                "engine_id": "pricing_engine.local",
                "version": "1",
                "adapter_id": "range_accrual_discounted",
            },
        ),
        actor="builder",
        reason="seed",
    )
    registry.transition_version(
        "range_accrual_checked",
        "v1",
        ModelLifecycleStatus.VALIDATED,
        actor="validator",
        reason="seed_validation",
    )
    registry.transition_version(
        "range_accrual_checked",
        "v1",
        ModelLifecycleStatus.APPROVED,
        actor="reviewer",
        reason="seed_approval",
    )


def _seed_callable_bond_model(registry) -> None:
    from trellis.platform.models import (
        ModelLifecycleStatus,
        ModelRecord,
        ModelVersionRecord,
    )

    registry.create_model(
        ModelRecord(
            model_id="callable_bond_checked",
            semantic_id="callable_bond",
            semantic_version="c2.1",
            product_family="callable_bond",
            instrument_class="callable_bond",
            payoff_family="callable_fixed_income",
            exercise_style="issuer_call",
            underlier_structure="single_issuer_bond",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("black_vol_surface", "discount_curve"),
            supported_method_families=("rate_tree",),
        )
    )
    registry.create_version(
        ModelVersionRecord(
            model_id="callable_bond_checked",
            version="v1",
            contract_summary={"semantic_id": "callable_bond"},
            methodology_summary={"method_family": "rate_tree"},
            engine_binding={
                "engine_id": "pricing_engine.local",
                "version": "1",
                "adapter_id": "callable_bond_tree",
            },
        ),
        actor="builder",
        reason="seed",
    )
    registry.transition_version(
        "callable_bond_checked",
        "v1",
        ModelLifecycleStatus.VALIDATED,
        actor="validator",
        reason="seed_validation",
    )
    registry.transition_version(
        "callable_bond_checked",
        "v1",
        ModelLifecycleStatus.APPROVED,
        actor="reviewer",
        reason="seed_approval",
    )


def _seed_bermudan_swaption_model(registry) -> None:
    from trellis.platform.models import (
        ModelLifecycleStatus,
        ModelRecord,
        ModelVersionRecord,
    )

    registry.create_model(
        ModelRecord(
            model_id="bermudan_swaption_checked",
            semantic_id="rate_style_swaption",
            semantic_version="c2.1",
            product_family="rates_option",
            instrument_class="swaption",
            payoff_family="swaption",
            exercise_style="bermudan",
            underlier_structure="single_curve_rate_style",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("black_vol_surface", "discount_curve", "forward_curve"),
            supported_method_families=("rate_tree",),
        )
    )
    registry.create_version(
        ModelVersionRecord(
            model_id="bermudan_swaption_checked",
            version="v1",
            contract_summary={"semantic_id": "rate_style_swaption"},
            methodology_summary={"method_family": "rate_tree"},
            engine_binding={
                "engine_id": "pricing_engine.local",
                "version": "1",
                "adapter_id": "bermudan_swaption_tree",
            },
        ),
        actor="builder",
        reason="seed",
    )
    registry.transition_version(
        "bermudan_swaption_checked",
        "v1",
        ModelLifecycleStatus.VALIDATED,
        actor="validator",
        reason="seed_validation",
    )
    registry.transition_version(
        "bermudan_swaption_checked",
        "v1",
        ModelLifecycleStatus.APPROVED,
        actor="reviewer",
        reason="seed_approval",
    )


def _file_import_manifest(tmp_path) -> str:
    (tmp_path / "discount.yaml").write_text(
        "kind: flat\nrate: 0.05\n",
        encoding="utf-8",
    )
    (tmp_path / "vol.yaml").write_text(
        "kind: flat\nvol: 0.20\n",
        encoding="utf-8",
    )
    (tmp_path / "spots.json").write_text(
        json.dumps({"AAPL": 123.0}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "market_snapshot.yaml"
    manifest_path.write_text(
        "\n".join(
            (
                "as_of: 2026-04-04",
                "source: desk_upload",
                "defaults:",
                "  discount_curve: discount",
                "  vol_surface: default",
                "  underlier_spot: AAPL",
                "discount_curves:",
                "  discount: discount.yaml",
                "vol_surfaces:",
                "  default: vol.yaml",
                "underlier_spots:",
                "  file: spots.json",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return str(manifest_path)


def _range_accrual_manifest(tmp_path) -> str:
    (tmp_path / "discount.yaml").write_text(
        "kind: flat\nrate: 0.04\n",
        encoding="utf-8",
    )
    (tmp_path / "forecast.yaml").write_text(
        "kind: flat\nrate: 0.025\n",
        encoding="utf-8",
    )
    (tmp_path / "fixings.csv").write_text(
        "date,value\n2026-01-15,0.02\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "range_accrual_snapshot.yaml"
    manifest_path.write_text(
        "\n".join(
            (
                "as_of: 2026-04-04",
                "source: range_accrual_desk",
                "defaults:",
                "  discount_curve: usd_ois",
                "  forecast_curve: SOFR",
                "  fixing_history: SOFR",
                "discount_curves:",
                "  usd_ois: discount.yaml",
                "forecast_curves:",
                "  SOFR: forecast.yaml",
                "fixing_histories:",
                "  SOFR: fixings.csv",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return str(manifest_path)


def _callable_rates_manifest(tmp_path) -> str:
    (tmp_path / "discount.yaml").write_text(
        "kind: flat\nrate: 0.05\n",
        encoding="utf-8",
    )
    (tmp_path / "vol.yaml").write_text(
        "kind: flat\nvol: 0.20\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "callable_rates_snapshot.yaml"
    manifest_path.write_text(
        "\n".join(
            (
                "as_of: 2026-04-04",
                "source: callable_desk",
                "defaults:",
                "  discount_curve: usd_ois",
                "  vol_surface: usd_rates_vol",
                "discount_curves:",
                "  usd_ois: discount.yaml",
                "vol_surfaces:",
                "  usd_rates_vol: vol.yaml",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return str(manifest_path)


def _range_accrual_manifest_without_forecast(tmp_path) -> str:
    (tmp_path / "discount.yaml").write_text(
        "kind: flat\nrate: 0.04\n",
        encoding="utf-8",
    )
    (tmp_path / "fixings.csv").write_text(
        "date,value\n2026-01-15,0.02\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "range_accrual_snapshot_no_forecast.yaml"
    manifest_path.write_text(
        "\n".join(
            (
                "as_of: 2026-04-04",
                "source: range_accrual_desk",
                "defaults:",
                "  discount_curve: usd_ois",
                "  fixing_history: SOFR",
                "discount_curves:",
                "  usd_ois: discount.yaml",
                "fixing_histories:",
                "  SOFR: fixings.csv",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return str(manifest_path)


def _range_accrual_trade_payload() -> dict[str, object]:
    return {
        "instrument_type": "range_accrual",
        "description": (
            "Range accrual note on SOFR paying 5.25% when SOFR stays between 1.50% "
            "and 3.25% on 2026-01-15, 2026-04-15, 2026-07-15, and 2026-10-15."
        ),
        "reference_index": "SOFR",
        "coupon_rate": 0.0525,
        "lower_bound": 0.015,
        "upper_bound": 0.0325,
        "observation_schedule": (
            "2026-01-15",
            "2026-04-15",
            "2026-07-15",
            "2026-10-15",
        ),
        "accrual_start_dates": (
            "2025-10-15",
            "2026-01-15",
            "2026-04-15",
            "2026-07-15",
        ),
        "payout_currency": "USD",
        "reporting_currency": "USD",
        "preferred_method": "analytical",
        "notional": 1_000_000.0,
    }


def _callable_bond_trade_payload() -> dict[str, object]:
    return {
        "instrument_type": "callable_bond",
        "description": (
            "Callable USD bond paying 5% with issuer call dates on 2028-01-15, "
            "2030-01-15, and 2032-01-15."
        ),
        "notional": 1_000_000.0,
        "coupon": 0.05,
        "start_date": "2025-01-15",
        "end_date": "2035-01-15",
        "call_dates": ("2028-01-15", "2030-01-15", "2032-01-15"),
        "payout_currency": "USD",
        "reporting_currency": "USD",
        "preferred_method": "rate_tree",
    }


def _bermudan_swaption_trade_payload() -> dict[str, object]:
    return {
        "instrument_type": "bermudan_swaption",
        "description": (
            "Bermudan payer swaption with exercise dates on 2027-11-15, 2028-11-15, "
            "and 2029-11-15 into a swap ending 2032-11-15."
        ),
        "notional": 5_000_000.0,
        "strike": 0.04,
        "exercise_schedule": ("2027-11-15", "2028-11-15", "2029-11-15"),
        "swap_end": "2032-11-15",
        "payout_currency": "USD",
        "reporting_currency": "USD",
        "preferred_method": "rate_tree",
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


def test_price_trade_executes_approved_model_with_expiry_date_alias(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_model(server.services.model_registry, approved=True)

    server.call_tool(
        "trellis.providers.configure",
        {
            "session_id": "sess_price_alias_success",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.test_live"},
                }
            },
        },
    )
    server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_price_alias_success", "run_mode": "production"},
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_alias_success",
            "structured_trade": _trade_payload_with_expiry_date_alias(),
            "output_mode": "structured",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "succeeded"
    assert payload["result"]["price"] > 0.0
    assert payload["provenance"]["model_id"] == "vanilla_option_approved"


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


def test_price_trade_blocked_parse_reports_truthful_desk_review(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_model(server.services.model_registry, approved=True)
    server.call_tool(
        "trellis.providers.configure",
        {
            "session_id": "sess_price_blocked_parse_review",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.test_live"},
                }
            },
        },
    )
    server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_price_blocked_parse_review", "run_mode": "production"},
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_blocked_parse_review",
            "structured_trade": {
                "instrument_type": "european_option",
                "underliers": ("AAPL",),
                "strike": 120.0,
                "option_type": "call",
            },
            "output_mode": "audit",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "blocked"
    driver = payload["desk_review"]["driver_narrative"]
    assert "blocked" in driver["headline"].lower()
    assert any("before route selection" in bullet.lower() for bullet in driver["bullets"])
    assert all("approved route" not in bullet.lower() for bullet in driver["bullets"])


def test_price_trade_blocked_provider_reports_truthful_desk_review(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_model(server.services.model_registry, approved=True)
    server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_price_blocked_provider_review", "run_mode": "production"},
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_blocked_provider_review",
            "structured_trade": _trade_payload(),
            "output_mode": "audit",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "blocked"
    driver = payload["desk_review"]["driver_narrative"]
    assert "blocked" in driver["headline"].lower()
    assert any("before pricing execution" in bullet.lower() for bullet in driver["bullets"])
    assert any("provider_binding_required" in bullet for bullet in driver["bullets"])
    assert all("approved route" not in bullet.lower() for bullet in driver["bullets"])


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


def test_price_trade_uses_activated_imported_snapshot(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_model(server.services.model_registry, approved=True)

    imported = server.call_tool(
        "trellis.snapshot.import_files",
        {
            "session_id": "sess_price_file_import",
            "manifest_path": _file_import_manifest(tmp_path),
            "activate_session": True,
            "reference_date": "2026-04-04",
        },
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_file_import",
            "structured_trade": _trade_payload(),
            "output_mode": "structured",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "succeeded"
    assert payload["result"]["price"] > 0.0
    assert payload["provenance"]["provider_id"] == "market_data.file_import"
    assert payload["provenance"]["market_snapshot_id"] == imported["snapshot"]["snapshot_id"]

    run_payload = server.call_tool("trellis.run.get", {"run_id": payload["run_id"]})
    snapshot_payload = server.read_resource(
        f"trellis://market-snapshots/{imported['snapshot']['snapshot_id']}"
    )

    assert run_payload["run"]["market_snapshot_id"] == imported["snapshot"]["snapshot_id"]
    assert snapshot_payload["snapshot"]["payload"]["manifest"]["as_of"] == "2026-04-04"


def test_price_trade_prices_range_accrual_from_imported_snapshot(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_range_accrual_model(server.services.model_registry)

    imported = server.call_tool(
        "trellis.snapshot.import_files",
        {
            "session_id": "sess_price_range_accrual",
            "manifest_path": _range_accrual_manifest(tmp_path),
            "activate_session": True,
            "reference_date": "2026-04-04",
        },
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_range_accrual",
            "structured_trade": _range_accrual_trade_payload(),
            "output_mode": "structured",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "succeeded"
    assert payload["result"]["price"] > payload["result"]["principal_leg_pv"] > 0.0
    assert payload["result"]["risk"]["parallel_curve_pv01"] > 0.0
    assert len(payload["result"]["scenarios"]) == 4
    assert payload["result"]["validation_bundle"]["route_id"] == "range_accrual_discounted_cashflow_v1"
    assert payload["provenance"]["provider_id"] == "market_data.file_import"
    assert payload["provenance"]["market_snapshot_id"] == imported["snapshot"]["snapshot_id"]

    scenario_prices = {
        item["name"]: item["price"]
        for item in payload["result"]["scenarios"]
    }
    assert scenario_prices["rates_up_100bp"] < payload["result"]["price"]

    check_ids = {
        item["check_id"]: item["status"]
        for item in payload["result"]["validation_bundle"]["checks"]
    }
    assert check_ids["historical_fixing_coverage"] == "passed"
    assert check_ids["pv_reconciliation"] == "passed"


def test_price_trade_projects_trader_review_bundle_for_range_accrual(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_range_accrual_model(server.services.model_registry)

    imported = server.call_tool(
        "trellis.snapshot.import_files",
        {
            "session_id": "sess_price_range_accrual_review",
            "manifest_path": _range_accrual_manifest_without_forecast(tmp_path),
            "activate_session": True,
            "reference_date": "2026-04-04",
        },
    )
    trade_payload = _range_accrual_trade_payload()
    trade_payload.pop("accrual_start_dates")
    trade_payload["call_dates"] = ("2026-07-15",)

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_range_accrual_review",
            "structured_trade": trade_payload,
            "output_mode": "audit",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "succeeded"
    assert payload["desk_review"]["route_summary"]["adapter_id"] == "range_accrual_discounted"
    assert payload["desk_review"]["route_summary"]["method_family"] == "analytical"
    assert payload["desk_review"]["schedule_summary"]["observation_count"] == 4
    assert payload["desk_review"]["schedule_summary"]["call_event_count"] == 1
    assert payload["desk_review"]["scenario_summary"]["scenario_count"] == 4
    assert payload["desk_review"]["audit_refs"]["audit_uri"] == payload["audit_uri"]
    assert (
        payload["desk_review"]["audit_refs"]["snapshot_uri"]
        == f"trellis://market-snapshots/{imported['snapshot']['snapshot_id']}"
    )
    assert any(
        "explicit imported market snapshot" in item.lower()
        for item in payload["desk_review"]["assumptions"]["explicit_inputs"]
    )
    assert any(
        "defaulted payment_dates" in item.lower()
        for item in payload["desk_review"]["assumptions"]["defaulted_inputs"]
    )
    assert any(
        "projection proxy" in item.lower()
        for item in payload["desk_review"]["assumptions"]["synthetic_inputs"]
    )
    assert any(
        "inferred accrual_start_dates" in item.lower()
        for item in payload["desk_review"]["assumptions"]["synthetic_inputs"]
    )
    driver = payload["desk_review"]["driver_narrative"]
    assert "range accrual" in driver["headline"].lower()
    assert any("coupon leg" in bullet.lower() for bullet in driver["bullets"])
    assert driver["linked_route"]["adapter_id"] == "range_accrual_discounted"
    assert driver["linked_assumptions"]["synthetic_inputs"]
    scenario_commentary = payload["desk_review"]["scenario_commentary"]
    assert scenario_commentary["availability"] == "available"
    assert scenario_commentary["scenario_count"] == 4
    assert scenario_commentary["linked_route"]["adapter_id"] == "range_accrual_discounted"
    assert scenario_commentary["dominant_scenario"]["shift_bps"] in {-100.0, -50.0, 50.0, 100.0}
    assert payload["audit"]["outputs"]["price"] == payload["result"]["price"]


def test_price_trade_prices_callable_bond_with_schedule_projection(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_callable_bond_model(server.services.model_registry)

    imported = server.call_tool(
        "trellis.snapshot.import_files",
        {
            "session_id": "sess_price_callable_bond",
            "manifest_path": _callable_rates_manifest(tmp_path),
            "activate_session": True,
            "reference_date": "2026-04-04",
        },
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_callable_bond",
            "structured_trade": _callable_bond_trade_payload(),
            "output_mode": "audit",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "succeeded"
    assert payload["result"]["price"] > 0.0
    assert payload["result"]["oas_duration"] > 0.0
    assert payload["result"]["validation_bundle"]["route_id"] == "callable_bond_tree_v1"
    explain = payload["result"]["callable_scenario_explain"]
    assert explain["metadata"]["controller_role"] == "issuer"
    assert {"-100.0", "-50.0", "50.0", "100.0"} <= {str(key) for key in explain["values"].keys()}
    assert payload["desk_review"]["trade_summary"]["semantic_id"] == "callable_bond"
    assert payload["desk_review"]["route_summary"]["adapter_id"] == "callable_bond_tree"
    assert payload["desk_review"]["route_summary"]["method_family"] == "rate_tree"
    assert payload["desk_review"]["schedule_summary"]["exercise_event_count"] == 3
    assert payload["desk_review"]["schedule_summary"]["schedule_role"] == "decision_dates"
    assert len(payload["desk_review"]["schedule_summary"]["projected_events"]) == 4
    assert payload["desk_review"]["scenario_summary"]["scenario_count"] == 4
    assert payload["desk_review"]["audit_refs"]["snapshot_uri"] == (
        f"trellis://market-snapshots/{imported['snapshot']['snapshot_id']}"
    )
    assert any(
        "defaulted call_price" in item.lower()
        for item in payload["desk_review"]["assumptions"]["defaulted_inputs"]
    )
    assert any(
        "defaulted frequency" in item.lower()
        for item in payload["desk_review"]["assumptions"]["defaulted_inputs"]
    )
    assert any(
        "defaulted day_count" in item.lower()
        for item in payload["desk_review"]["assumptions"]["defaulted_inputs"]
    )
    driver = payload["desk_review"]["driver_narrative"]
    assert "callable-bond" in driver["headline"].lower()
    assert any("call option value" in bullet.lower() for bullet in driver["bullets"])
    assert driver["linked_route"]["adapter_id"] == "callable_bond_tree"
    scenario_commentary = payload["desk_review"]["scenario_commentary"]
    assert scenario_commentary["availability"] == "available"
    assert "call optionality" in scenario_commentary["headline"].lower()
    assert scenario_commentary["dominant_scenario"]["scenario"]
    assert scenario_commentary["linked_assumptions"]["defaulted_inputs"]


def test_price_trade_prices_bermudan_swaption_with_schedule_projection(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(
        state_root=tmp_path / "mcp_state",
        provider_registry=_provider_registry(),
    )
    _seed_bermudan_swaption_model(server.services.model_registry)

    imported = server.call_tool(
        "trellis.snapshot.import_files",
        {
            "session_id": "sess_price_bermudan_swaption",
            "manifest_path": _callable_rates_manifest(tmp_path),
            "activate_session": True,
            "reference_date": "2026-04-04",
        },
    )

    payload = server.call_tool(
        "trellis.price.trade",
        {
            "session_id": "sess_price_bermudan_swaption",
            "structured_trade": _bermudan_swaption_trade_payload(),
            "output_mode": "audit",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "succeeded"
    assert payload["result"]["price"] >= 0.0
    assert payload["result"]["validation_bundle"]["route_id"] == "bermudan_swaption_tree_v1"
    assert payload["desk_review"]["trade_summary"]["instrument_class"] == "bermudan_swaption"
    assert payload["desk_review"]["route_summary"]["adapter_id"] == "bermudan_swaption_tree"
    assert payload["desk_review"]["route_summary"]["method_family"] == "rate_tree"
    assert payload["desk_review"]["schedule_summary"]["exercise_event_count"] == 3
    assert payload["desk_review"]["schedule_summary"]["schedule_role"] == "exercise_dates"
    assert len(payload["desk_review"]["schedule_summary"]["projected_events"]) == 4
    assert payload["desk_review"]["audit_refs"]["snapshot_uri"] == (
        f"trellis://market-snapshots/{imported['snapshot']['snapshot_id']}"
    )
    assert any(
        "defaulted swap_frequency" in item.lower()
        for item in payload["desk_review"]["assumptions"]["defaulted_inputs"]
    )
    assert any(
        "defaulted day_count" in item.lower()
        for item in payload["desk_review"]["assumptions"]["defaulted_inputs"]
    )
    assert any(
        "defaulted is_payer" in item.lower()
        for item in payload["desk_review"]["assumptions"]["defaulted_inputs"]
    )
    driver = payload["desk_review"]["driver_narrative"]
    assert "bermudan swaption" in driver["headline"].lower()
    assert driver["linked_route"]["adapter_id"] == "bermudan_swaption_tree"
    scenario_commentary = payload["desk_review"]["scenario_commentary"]
    assert scenario_commentary["availability"] == "unavailable"
    assert "no route-specific scenario ladder" in scenario_commentary["headline"].lower()
