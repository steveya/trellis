from __future__ import annotations

import pytest


def test_generation_policy_and_artifact_origin_payload_are_typed():
    from trellis.agent.generation_policy import generation_evidence_payload

    assert generation_evidence_payload(
        policy="builder_synthesis_required",
        artifact_origin="model_generated_source",
        agent_synthesis_attempted=True,
        agent_synthesis_observed=True,
    ) == {
        "policy": "builder_synthesis_required",
        "artifact_origin": "model_generated_source",
        "agent_synthesis_attempted": True,
        "agent_synthesis_observed": True,
    }


@pytest.mark.parametrize(
    ("fresh_build", "recovery_mode", "execution_mode", "reason"),
    [
        (False, "assisted", "live", "fresh_build_required"),
        (True, "strict", "live", "recovery_mode_strict"),
        (
            True,
            "assisted",
            "deterministic_replay",
            "execution_mode_deterministic_replay",
        ),
    ],
)
def test_builder_synthesis_policy_fails_closed_when_controls_cannot_prove_it(
    fresh_build,
    recovery_mode,
    execution_mode,
    reason,
):
    from trellis.agent.generation_policy import (
        GenerationPolicyError,
        validate_generation_policy_request,
    )

    with pytest.raises(GenerationPolicyError) as exc_info:
        validate_generation_policy_request(
            policy="builder_synthesis_required",
            fresh_build=fresh_build,
            recovery_mode=recovery_mode,
            execution_mode=execution_mode,
        )

    assert exc_info.value.reason == reason


def test_builder_synthesis_policy_allows_assisted_live_execution():
    from trellis.agent.generation_policy import validate_generation_policy_request

    validate_generation_policy_request(
        policy="builder_synthesis_required",
        fresh_build=True,
        recovery_mode="assisted",
        execution_mode="live",
    )
