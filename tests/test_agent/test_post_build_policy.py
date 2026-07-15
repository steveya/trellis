"""Tests for deterministic post-build model-stage eligibility."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    (
        "execution_mode",
        "artifact_policy",
        "recovery_mode",
        "expected_allowed",
        "expected_primary_reason",
    ),
    [
        (
            "live",
            "fresh_generated",
            "strict",
            False,
            "recovery_mode_strict",
        ),
        (
            "live",
            "forced_rebuild",
            "assisted",
            True,
            "assisted_model_backed_stage",
        ),
        (
            "cassette_record",
            "fresh_generated",
            "remediation",
            True,
            "remediation_model_backed_stage",
        ),
        (
            "deterministic_replay",
            "fresh_generated",
            "assisted",
            False,
            "execution_mode_deterministic_replay",
        ),
        (
            "cassette_replay",
            "forced_rebuild",
            "remediation",
            False,
            "execution_mode_cassette_replay",
        ),
        (
            "live",
            "cached_existing",
            "assisted",
            True,
            "assisted_model_backed_stage",
        ),
    ],
)
def test_post_build_policy_matrix(
    execution_mode,
    artifact_policy,
    recovery_mode,
    expected_allowed,
    expected_primary_reason,
):
    from trellis.agent.post_build_policy import determine_post_build_learning_policy

    policy = determine_post_build_learning_policy(
        execution_mode=execution_mode,
        artifact_policy=artifact_policy,
        recovery_mode=recovery_mode,
    )

    assert policy.run_reflection is expected_allowed
    assert policy.run_consolidation is expected_allowed
    assert policy.reflection_reason == expected_primary_reason
    assert policy.consolidation_reason == expected_primary_reason
    assert policy.to_payload()["policy_source"] == "task_execution_policy"


def test_cached_strict_replay_records_every_fail_closed_reason():
    from trellis.agent.post_build_policy import determine_post_build_learning_policy

    policy = determine_post_build_learning_policy(
        execution_mode="deterministic_replay",
        artifact_policy="cached_existing",
        recovery_mode="strict",
    )

    assert policy.run_reflection is False
    assert policy.skip_reasons == (
        "execution_mode_deterministic_replay",
        "recovery_mode_strict",
    )


def test_unknown_execution_or_artifact_policy_fails_closed():
    from trellis.agent.post_build_policy import determine_post_build_learning_policy

    policy = determine_post_build_learning_policy(
        execution_mode="mystery_runner",
        artifact_policy="mystery_artifact",
        recovery_mode="assisted",
    )

    assert policy.run_reflection is False
    assert policy.run_consolidation is False
    assert policy.skip_reasons == (
        "execution_mode_mystery_runner_not_model_eligible",
        "artifact_policy_mystery_artifact_not_model_eligible",
    )


def test_policy_payload_round_trips_from_request_metadata():
    from trellis.agent.post_build_policy import (
        determine_post_build_learning_policy,
        post_build_policy_from_request_metadata,
    )

    expected = determine_post_build_learning_policy(
        execution_mode="live",
        artifact_policy="forced_rebuild",
        recovery_mode="assisted",
    )

    actual = post_build_policy_from_request_metadata(
        {"post_build_learning_policy": expected.to_payload()}
    )

    assert actual == expected


def test_missing_request_policy_defaults_offline_scope_to_deterministic_skip(monkeypatch):
    from trellis.agent.post_build_policy import post_build_policy_from_request_metadata

    monkeypatch.setenv("TRELLIS_OFFLINE_LOCAL_AGENTS", "1")

    policy = post_build_policy_from_request_metadata(None)

    assert policy.execution_mode == "deterministic_replay"
    assert policy.run_reflection is False
    assert policy.run_consolidation is False
    assert policy.policy_source == "offline_execution_policy"
