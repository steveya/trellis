"""Tests for the local streamable HTTP MCP transport wrapper."""

from __future__ import annotations

import asyncio
import json

import pytest


mcp = pytest.importorskip("mcp")


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


def test_http_transport_bootstrap_exposes_same_surface_as_shell(tmp_path):
    from trellis.mcp.http_transport import bootstrap_http_mcp_server
    from trellis.mcp.server import bootstrap_mcp_server

    shell = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")
    server = bootstrap_http_mcp_server(state_root=tmp_path / "mcp_state")

    assert server.endpoint_url == "http://127.0.0.1:8000/mcp"
    assert set(server.shell.list_tools()) == set(shell.list_tools())
    assert set(server.shell.list_resources()) == set(shell.list_resources())
    assert set(server.shell.list_prompts()) == set(shell.list_prompts())
    assert {tool.name for tool in server.transport._tool_manager.list_tools()} == set(shell.list_tools())
    assert {
        template.uri_template for template in server.transport._resource_manager.list_templates()
    } == set(shell.list_resources())
    assert {prompt.name for prompt in server.transport._prompt_manager.list_prompts()} == set(
        shell.list_prompts()
    )
    assert server.streamable_http_app() is not None


def test_http_transport_dispatches_tools_resources_and_prompts(tmp_path):
    from trellis.mcp.http_transport import bootstrap_http_mcp_server

    server = bootstrap_http_mcp_server(state_root=tmp_path / "mcp_state")

    tool = server.transport._tool_manager.get_tool("trellis.session.get_context")
    assert tool is not None
    tool_result = asyncio.run(tool.run({"arguments": {"session_id": "sess_http"}}))

    resource = asyncio.run(
        server.transport._resource_manager.get_resource(
            "trellis://policies/policy_bundle.research.default"
        )
    )
    assert resource is not None
    resource_payload = json.loads(asyncio.run(resource.read()))

    prompt = server.transport._prompt_manager.get_prompt("price_trade")
    assert prompt is not None
    messages = asyncio.run(prompt.render({"session_id": "sess_http"}))
    message_text = "\n".join(
        getattr(getattr(message, "content", None), "text", "")
        for message in messages
    )

    assert tool_result["session"]["session_id"] == "sess_http"
    assert resource_payload["policy"]["policy_id"] == "policy_bundle.research.default"
    assert "trellis.price.trade" in message_text
    assert "trellis://runs/{run_id}/audit" in message_text


def test_http_transport_demo_mode_seeds_mock_sandbox_session_and_model(tmp_path):
    from trellis.mcp.http_transport import bootstrap_http_mcp_server

    server = bootstrap_http_mcp_server(
        state_root=tmp_path / "mcp_state",
        demo_mode=True,
        demo_session_id="demo",
    )

    context = server.shell.call_tool(
        "trellis.session.get_context",
        {"session_id": "demo"},
    )
    match = server.shell.call_tool(
        "trellis.model.match",
        {"structured_trade": _trade_payload()},
    )
    providers = server.shell.call_tool(
        "trellis.providers.list",
        {"session_id": "demo", "kind": "market_data"},
    )

    assert context["session"]["run_mode"] == "sandbox"
    assert context["session"]["allow_mock_data"] is True
    assert context["session"]["active_policy"] == "policy_bundle.sandbox.default"
    assert (
        context["session"]["provider_bindings"]["market_data"]["primary"]["provider_id"]
        == "market_data.mock"
    )
    assert match["match_type"] == "exact_approved_match"
    assert match["selected_candidate"]["model_id"] == "vanilla_option_demo"
    provider_ids = {provider["provider_id"] for provider in providers["providers"]}
    assert "market_data.mock" in provider_ids


def test_http_transport_demo_mode_prices_standard_aapl_option_flow(tmp_path):
    from trellis.mcp.http_transport import bootstrap_http_mcp_server

    server = bootstrap_http_mcp_server(
        state_root=tmp_path / "mcp_state",
        demo_mode=True,
        demo_session_id="demo",
    )

    payload = server.shell.call_tool(
        "trellis.price.trade",
        {
            "session_id": "demo",
            "structured_trade": _trade_payload(),
            "output_mode": "structured",
            "valuation_date": "2026-04-04",
        },
    )

    assert payload["status"] == "succeeded"
    assert payload["result"]["price"] > 0.0
    assert payload["provenance"]["provider_id"] == "market_data.mock"
    assert payload["provenance"]["model_id"] == "vanilla_option_demo"
