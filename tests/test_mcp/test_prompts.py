"""Tests for the thin Trellis MCP prompt and host-packaging surface."""

from __future__ import annotations


def test_prompt_registry_lists_thin_workflows_and_reuses_tools_resources(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")

    prompts = server.list_prompts()

    assert "price_trade" in prompts
    assert "compare_model_versions" in prompts
    assert "validate_candidate_model" in prompts

    price_prompt = server.get_prompt("price_trade", {"session_id": "sess_prompt"})
    validate_prompt = server.get_prompt(
        "validate_candidate_model",
        {"model_id": "vanilla_option_candidate", "version": "v1"},
    )

    assert "trellis.price.trade" in price_prompt["tools"]
    assert "trellis://runs/{run_id}/audit" in price_prompt["resources"]
    assert "trellis.model.validate" in validate_prompt["tools"]
    assert (
        "trellis://models/vanilla_option_candidate/versions/v1/validation-report"
        in validate_prompt["resources"]
    )


def test_host_packaging_manifest_targets_one_common_contract(tmp_path):
    from trellis.mcp.server import bootstrap_mcp_server

    server = bootstrap_mcp_server(state_root=tmp_path / "mcp_state")
    manifest = server.describe_host_packaging()

    assert manifest["common_contract"]["bootstrap_entrypoint"] == "trellis.mcp.server.bootstrap_mcp_server"
    assert set(manifest["common_contract"]["tools"]) == set(server.list_tools())
    assert set(manifest["common_contract"]["resources"]) == set(server.list_resources())
    assert set(manifest["common_contract"]["prompts"]) == set(server.list_prompts())
    assert manifest["hosts"]["claude"]["uses_common_contract"] is True
    assert manifest["hosts"]["codex"]["uses_common_contract"] is True
    assert manifest["hosts"]["chatgpt"]["uses_common_contract"] is True
