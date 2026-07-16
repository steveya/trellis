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


def test_builder_synthesis_context_rejects_untrusted_llm_override():
    from trellis.agent.config import llm_override_scope
    from trellis.agent.generation_policy import (
        GenerationPolicyError,
        validate_builder_synthesis_context,
    )

    with llm_override_scope(
        generate=lambda *_args, **_kwargs: "replayed source",
        generate_json=lambda *_args, **_kwargs: {},
    ):
        with pytest.raises(GenerationPolicyError) as exc_info:
            validate_builder_synthesis_context(
                policy="builder_synthesis_required",
            )

    assert exc_info.value.reason == "llm_override_untrusted"


def test_build_payoff_rejects_untrusted_llm_override_before_planning():
    from trellis.agent.config import llm_override_scope
    from trellis.agent.executor import build_payoff
    from trellis.agent.generation_policy import GenerationPolicyError

    with llm_override_scope(
        generate=lambda *_args, **_kwargs: "replayed source",
        generate_json=lambda *_args, **_kwargs: {},
    ):
        with pytest.raises(GenerationPolicyError) as exc_info:
            build_payoff(
                "European option",
                fresh_build=True,
                generation_policy="builder_synthesis_required",
            )

    assert exc_info.value.reason == "llm_override_untrusted"


@pytest.mark.parametrize(
    ("cassette_mode", "expected_reason"),
    [
        ("replay", "execution_mode_cassette_replay"),
        ("record", None),
    ],
)
def test_builder_synthesis_context_distinguishes_cassette_record_from_replay(
    monkeypatch,
    cassette_mode,
    expected_reason,
):
    from trellis.agent.generation_policy import (
        GenerationPolicyError,
        validate_builder_synthesis_context,
    )

    monkeypatch.setattr(
        "trellis.agent.cassette.current_llm_cassette_context",
        lambda: {"mode": cassette_mode},
    )
    monkeypatch.setattr(
        "trellis.agent.config.current_llm_override_context",
        lambda: {
            "source": f"cassette_{cassette_mode}",
            "model_source_observable": cassette_mode == "record",
        },
    )

    if expected_reason is None:
        validate_builder_synthesis_context(policy="builder_synthesis_required")
    else:
        with pytest.raises(GenerationPolicyError) as exc_info:
            validate_builder_synthesis_context(policy="builder_synthesis_required")
        assert exc_info.value.reason == expected_reason
