"""Tests for the governed MCP session and provider tool surface."""

from __future__ import annotations

import pytest


def test_session_context_and_provider_configuration_persist_across_tool_calls(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")

    context = server.call_tool(
        "trellis.session.get_context",
        {"session_id": "sess_tools_001"},
    )

    assert context["session"]["run_mode"] == "research"
    assert context["session"]["provider_bindings"]["market_data"]["primary"] is None

    configured = server.call_tool(
        "trellis.providers.configure",
        {
            "session_id": "sess_tools_001",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.treasury_gov"},
                }
            },
        },
    )

    assert configured["session"]["provider_bindings"]["market_data"]["primary"]["provider_id"] == "market_data.treasury_gov"

    listed = server.call_tool(
        "trellis.providers.list",
        {"session_id": "sess_tools_001"},
    )
    by_id = {item["provider_id"]: item for item in listed["providers"]}

    assert "market_data.treasury_gov" in by_id
    assert "market_data.primary" in by_id["market_data.treasury_gov"]["bound_as"]

    persisted = server.call_tool(
        "trellis.session.get_context",
        {"session_id": "sess_tools_001"},
    )

    assert persisted["session"]["provider_bindings"]["market_data"]["primary"]["provider_id"] == "market_data.treasury_gov"


def test_run_mode_set_updates_policy_and_rejects_disallowed_mock_binding(tmp_path):
    from trellis.mcp.errors import TrellisMcpError
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")

    server.call_tool(
        "trellis.providers.configure",
        {
            "session_id": "sess_tools_002",
            "provider_bindings": {
                "market_data": {
                    "primary": {"provider_id": "market_data.mock"},
                }
            },
        },
    )

    sandbox = server.call_tool(
        "trellis.run_mode.set",
        {"session_id": "sess_tools_002", "run_mode": "sandbox"},
    )
    assert sandbox["session"]["run_mode"] == "sandbox"
    assert sandbox["session"]["active_policy"] == "policy_bundle.sandbox.default"

    with pytest.raises(TrellisMcpError) as excinfo:
        server.call_tool(
            "trellis.run_mode.set",
            {"session_id": "sess_tools_002", "run_mode": "production"},
        )

    assert excinfo.value.code == "mock_data_not_allowed"


def test_provider_configuration_rejects_unknown_provider_ids(tmp_path):
    from trellis.mcp.errors import TrellisMcpError
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")

    with pytest.raises(TrellisMcpError) as excinfo:
        server.call_tool(
            "trellis.providers.configure",
            {
                "session_id": "sess_tools_003",
                "provider_bindings": {
                    "market_data": {
                        "primary": {"provider_id": "market_data.not_real"},
                    }
                },
            },
        )

    assert excinfo.value.code == "unknown_provider"
