"""Tests for the transport-neutral MCP bootstrap shell."""

from __future__ import annotations


def test_bootstrap_server_exposes_shared_service_container(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")

    assert server.services.config.state_root == tmp_path / "mcp_state"
    assert "trellis.session.get_context" in server.list_tools()
    assert "trellis.providers.list" in server.list_tools()
    assert "trellis.model.generate_candidate" in server.list_tools()
    assert "trellis.model.validate" in server.list_tools()
    assert "trellis.model.promote" in server.list_tools()
    assert "trellis.model.persist" in server.list_tools()
    assert "trellis.model.versions.list" in server.list_tools()
    assert "trellis.model.diff" in server.list_tools()
    assert "trellis.snapshot.import_files" in server.list_tools()
    assert "trellis.snapshot.persist_run" in server.list_tools()
    assert "trellis.price.trade" in server.list_tools()
    assert "trellis.run.get" in server.list_tools()
    assert "trellis.run.get_audit" in server.list_tools()
    assert "trellis://models/{model_id}" in server.list_resources()
    assert "trellis://runs/{run_id}/audit" in server.list_resources()
    assert "trellis://market-snapshots/{snapshot_id}" in server.list_resources()
    assert "exotic_desk_one_trade" in server.list_prompts()
    assert "price_trade" in server.list_prompts()
    assert "validate_candidate_model" in server.list_prompts()
