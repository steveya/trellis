"""Tests for governed MCP run-summary and audit retrieval tools."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def test_run_get_returns_canonical_run_summary(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server
    from trellis.platform.context import build_execution_context
    from trellis.platform.runs import build_run_record

    server = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")
    context = build_execution_context(
        session_id="sess_run_tool_success",
        market_source="treasury_gov",
        run_mode="research",
    )
    server.services.run_ledger.create_run(
        build_run_record(
            run_id="run_tool_success",
            request_id="request_tool_success",
            status="succeeded",
            action="price_trade",
            execution_context=context,
            trade_identity={"instrument_type": "european_option", "trade_id": "trade_001"},
            selected_model={
                "model_id": "vanilla_option_analytical",
                "version": "v2",
                "status": "approved",
            },
            selected_engine={"engine_id": "pricing_engine.local", "version": "1"},
            market_snapshot_id="snapshot_123",
            valuation_timestamp="2026-04-04T05:20:00+00:00",
            warnings=("using_cached_snapshot",),
            result_summary={"price": 12.34},
            provenance={"route_family": "analytical"},
            policy_outcome={
                "policy_id": "policy_bundle.research.default",
                "allowed": True,
                "blocker_codes": [],
                "blockers": [],
            },
        )
    )

    payload = server.call_tool("trellis.run.get", {"run_id": "run_tool_success"})

    assert payload["run"]["run_id"] == "run_tool_success"
    assert payload["run"]["status"] == "succeeded"
    assert payload["run"]["selected_model"]["model_id"] == "vanilla_option_analytical"
    assert payload["run"]["result_summary"]["price"] == 12.34


def test_run_get_audit_returns_canonical_bundle_for_blocked_run(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server
    from trellis.platform.context import build_execution_context
    from trellis.platform.runs import ArtifactReference, build_run_record

    server = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")
    trace_path = _write_yaml(
        tmp_path / "platform" / "request_blocked.yaml",
        {
            "request_id": "request_blocked",
            "request_type": "trade",
            "entry_point": "mcp",
            "action": "price_trade",
            "status": "blocked",
            "outcome": "request_blocked",
            "instrument_type": "credit_default_swap",
            "blocker_codes": ["provider_binding_missing", "policy_denied"],
            "details": {"reason": "market-data provider binding is required"},
            "events": [
                {
                    "event": "request_blocked",
                    "status": "error",
                    "timestamp": "2026-04-04T05:25:00+00:00",
                    "details": {"reason": "market-data provider binding is required"},
                }
            ],
        },
    )
    context = build_execution_context(
        session_id="sess_run_tool_blocked",
        market_source="treasury_gov",
        run_mode="production",
    )
    server.services.run_ledger.create_run(
        build_run_record(
            run_id="run_tool_blocked",
            request_id="request_blocked",
            status="blocked",
            action="price_trade",
            execution_context=context,
            trade_identity={"instrument_type": "credit_default_swap", "trade_id": "trade_blocked"},
            result_summary={"error": "policy blocked execution"},
            warnings=("provider_binding_missing",),
            policy_outcome={
                "policy_id": "policy_bundle.production.default",
                "allowed": False,
                "blocker_codes": ["provider_binding_missing", "policy_denied"],
                "blockers": [
                    {
                        "code": "provider_binding_missing",
                        "message": "Explicit market-data binding is required.",
                        "requirement": "provider_disclosure",
                        "field": "provider_bindings.market_data.primary",
                    }
                ],
            },
            artifacts=(
                ArtifactReference(
                    artifact_id="platform_trace",
                    artifact_kind="platform_trace",
                    uri=str(trace_path),
                ),
            ),
        )
    )

    payload = server.call_tool("trellis.run.get_audit", {"run_id": "run_tool_blocked"})

    assert payload["run_id"] == "run_tool_blocked"
    assert payload["audit"]["run"]["status"] == "blocked"
    assert payload["audit"]["execution"]["policy_outcome"]["allowed"] is False
    assert payload["audit"]["diagnostics"]["blocked"] is True
    assert payload["audit"]["diagnostics"]["failure_context"]["trace_reason"] == (
        "market-data provider binding is required"
    )


def test_run_tools_raise_structured_error_for_unknown_run_ids(tmp_path):
    import pytest

    from trellis.mcp.errors import TrellisMcpError
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")

    with pytest.raises(TrellisMcpError) as run_excinfo:
        server.call_tool("trellis.run.get", {"run_id": "run_missing"})

    assert run_excinfo.value.code == "unknown_run"

    with pytest.raises(TrellisMcpError) as audit_excinfo:
        server.call_tool("trellis.run.get_audit", {"run_id": "run_missing"})

    assert audit_excinfo.value.code == "unknown_run"
