"""Tests for approved early-exercise Monte Carlo policy classes."""

from __future__ import annotations


def test_policy_registry_marks_implemented_and_planned_policy_classes():
    from trellis.agent.early_exercise_policy import approved_early_exercise_policy_classes

    policies = {policy.name: policy.status for policy in approved_early_exercise_policy_classes()}

    assert policies == {
        "longstaff_schwartz": "implemented",
        "tsitsiklis_van_roy": "implemented",
        "primal_dual_mc": "implemented",
        "stochastic_mesh": "implemented",
    }


def test_policy_registry_canonicalizes_aliases():
    from trellis.agent.early_exercise_policy import canonicalize_early_exercise_policy

    assert canonicalize_early_exercise_policy("longstaff_schwartz") == "longstaff_schwartz"
    assert canonicalize_early_exercise_policy("tv_regression") == "tsitsiklis_van_roy"
    assert canonicalize_early_exercise_policy("dual_mc") == "primal_dual_mc"
    assert canonicalize_early_exercise_policy("foo") is None
