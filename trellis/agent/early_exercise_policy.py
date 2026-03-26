"""Approved early-exercise Monte Carlo policy classes.

This module captures the policy-family contract separately from currently
implemented imports. The goal is to keep semantic and planning logic aligned
around a broader set of valid early-exercise Monte Carlo constructs without
pretending that every approved class is already implemented in Trellis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyExercisePolicyClass:
    """Single approved early-exercise Monte Carlo policy class."""

    name: str
    status: str
    summary: str
    aliases: tuple[str, ...]


_APPROVED_POLICY_CLASSES = (
    EarlyExercisePolicyClass(
        name="longstaff_schwartz",
        status="implemented",
        summary="Least-squares Monte Carlo with backward continuation regression.",
        aliases=("longstaff_schwartz",),
    ),
    EarlyExercisePolicyClass(
        name="tsitsiklis_van_roy",
        status="implemented",
        summary="Continuation-value regression / approximate dynamic programming.",
        aliases=(
            "tsitsiklis_van_roy",
            "tv_regression",
            "continuation_regression",
        ),
    ),
    EarlyExercisePolicyClass(
        name="primal_dual_mc",
        status="implemented",
        summary="Primal lower bound plus optimistic upper-bound diagnostic for early exercise.",
        aliases=(
            "primal_dual_mc",
            "dual_mc",
        ),
    ),
    EarlyExercisePolicyClass(
        name="stochastic_mesh",
        status="implemented",
        summary="Stochastic-mesh continuation weighting for optimal stopping.",
        aliases=("stochastic_mesh",),
    ),
)


def approved_early_exercise_policy_classes() -> tuple[EarlyExercisePolicyClass, ...]:
    """Return the approved early-exercise policy classes."""
    return _APPROVED_POLICY_CLASSES


def canonicalize_early_exercise_policy(name: str) -> str | None:
    """Map a symbol or alias onto a canonical policy-class name."""
    base = name.rsplit(".", 1)[-1]
    for policy in _APPROVED_POLICY_CLASSES:
        if base == policy.name or base in policy.aliases:
            return policy.name
    return None


def detect_early_exercise_policy_calls(call_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return canonical approved policy names detected in a sequence of calls."""
    detected: list[str] = []
    for call_name in call_names:
        canonical = canonicalize_early_exercise_policy(call_name)
        if canonical is None or canonical in detected:
            continue
        detected.append(canonical)
    return tuple(detected)


def render_early_exercise_policy_summary(*, include_status: bool = True) -> str:
    """Render a compact summary of approved policy classes."""
    parts: list[str] = []
    for policy in _APPROVED_POLICY_CLASSES:
        if include_status:
            parts.append(f"`{policy.name}` [{policy.status}]")
        else:
            parts.append(f"`{policy.name}`")
    return ", ".join(parts)


def implemented_early_exercise_policy_classes() -> tuple[EarlyExercisePolicyClass, ...]:
    """Return the subset of approved policy classes already implemented."""
    return tuple(
        policy
        for policy in _APPROVED_POLICY_CLASSES
        if policy.status == "implemented"
    )


def render_implemented_early_exercise_policy_summary() -> str:
    """Render a compact summary of implemented policy classes only."""
    return ", ".join(
        f"`{policy.name}`"
        for policy in implemented_early_exercise_policy_classes()
    )
