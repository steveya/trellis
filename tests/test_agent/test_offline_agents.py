"""Tests for local-agent offline execution guards."""

from __future__ import annotations

import os

import pytest


def test_offline_local_agent_guard_blocks_text_and_json_llm_calls():
    from trellis.agent.config import llm_generate, llm_generate_json
    from trellis.agent.offline_agents import offline_local_agent_llm_guard

    with offline_local_agent_llm_guard():
        with pytest.raises(RuntimeError, match="forbids live LLM text calls"):
            llm_generate("hello")
        with pytest.raises(RuntimeError, match="forbids live LLM JSON calls"):
            llm_generate_json("hello")


def test_offline_local_agent_run_scope_sets_execution_policy_without_learning_skip_flags(
    monkeypatch,
):
    from trellis.agent.offline_agents import offline_local_agent_run_scope

    monkeypatch.delenv("TRELLIS_OFFLINE_LOCAL_AGENTS", raising=False)
    monkeypatch.delenv("TRELLIS_SKIP_POST_BUILD_REFLECTION", raising=False)
    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_CONSOLIDATION", "preserve")

    with offline_local_agent_run_scope():
        assert os.environ["TRELLIS_OFFLINE_LOCAL_AGENTS"] == "1"
        assert "TRELLIS_SKIP_POST_BUILD_REFLECTION" not in os.environ
        assert os.environ["TRELLIS_SKIP_POST_BUILD_CONSOLIDATION"] == "preserve"

    assert "TRELLIS_OFFLINE_LOCAL_AGENTS" not in os.environ
    assert "TRELLIS_SKIP_POST_BUILD_REFLECTION" not in os.environ
    assert os.environ["TRELLIS_SKIP_POST_BUILD_CONSOLIDATION"] == "preserve"


def test_offline_local_agent_scope_skips_llm_review_policy():
    from trellis.agent.offline_agents import offline_local_agent_run_scope
    from trellis.agent.review_policy import determine_review_policy

    with offline_local_agent_run_scope():
        policy = determine_review_policy(
            validation="thorough",
            method="pde_solver",
            instrument_type="heston_option",
        )

    assert policy.run_critic is False
    assert policy.run_model_validator_llm is False
    assert policy.critic_reason == "offline_local_agents"
    assert policy.model_validator_reason == "offline_local_agents"
