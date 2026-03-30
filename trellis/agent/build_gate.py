"""Build gate checkpoint — prevent wasted LLM calls on doomed builds.

Two gate functions:

* ``evaluate_pre_flight_gate`` — runs in ``autonomous.py`` immediately after
  ``gap_check()``, using only the ``GapReport``.  Blocks very-low-confidence
  builds before the executor is entered.

* ``evaluate_pre_generation_gate`` — runs in ``executor.py`` after the
  ``GenerationPlan`` is assembled but before ``_generate_skeleton()`` (the
  first LLM call).  Uses the richer context of ``ResolvedInstructionSet``
  and ``PrimitivePlan`` to catch instruction conflicts and hard blockers.

Both return a frozen ``BuildGateDecision``.
"""

from __future__ import annotations

from trellis.agent.knowledge.schema import BuildGateDecision, BuildGateThresholds

_DEFAULT_THRESHOLDS = BuildGateThresholds()


# ---------------------------------------------------------------------------
# Pre-flight gate (autonomous.py, after gap_check)
# ---------------------------------------------------------------------------

def evaluate_pre_flight_gate(
    gap_report,
    *,
    thresholds: BuildGateThresholds | None = None,
) -> BuildGateDecision:
    """Evaluate whether a build should proceed based on knowledge readiness.

    Parameters
    ----------
    gap_report:
        ``GapReport`` from ``gap_check()``.
    thresholds:
        Optional overrides; falls back to conservative defaults.

    Returns
    -------
    BuildGateDecision
        decision is one of "proceed", "narrow_route", "block".
    """
    t = thresholds or _DEFAULT_THRESHOLDS
    confidence = gap_report.confidence

    if confidence < t.block_below:
        return BuildGateDecision(
            decision="block",
            reason=(
                f"Knowledge confidence {confidence:.0%} is below the "
                f"block threshold ({t.block_below:.0%}). "
                f"Gaps: {'; '.join(gap_report.missing) or 'none listed'}"
            ),
            gap_confidence=confidence,
            gate_source="pre_flight",
        )

    if t.require_promoted_route and not gap_report.has_promoted_route:
        route_msg = ""
        if gap_report.route_gap is not None:
            route_msg = f" ({gap_report.route_gap.message})"
        return BuildGateDecision(
            decision="block",
            reason=f"No promoted route available{route_msg}",
            gap_confidence=confidence,
            gate_source="pre_flight",
        )

    if confidence < t.narrow_below:
        return BuildGateDecision(
            decision="narrow_route",
            reason=(
                f"Knowledge confidence {confidence:.0%} is below the "
                f"narrow threshold ({t.narrow_below:.0%}); "
                f"restricting route candidates"
            ),
            gap_confidence=confidence,
            gate_source="pre_flight",
        )

    return BuildGateDecision(
        decision="proceed",
        reason="Knowledge readiness sufficient",
        gap_confidence=confidence,
        gate_source="pre_flight",
    )


# ---------------------------------------------------------------------------
# Pre-generation gate (executor.py, after build_generation_plan)
# ---------------------------------------------------------------------------

def evaluate_pre_generation_gate(
    gap_report,
    generation_plan,
    *,
    thresholds: BuildGateThresholds | None = None,
) -> BuildGateDecision:
    """Evaluate whether code generation should proceed.

    Uses the richer context available after ``build_generation_plan()``
    has assembled the ``GenerationPlan`` with resolved instructions and
    primitive plans.

    Parameters
    ----------
    gap_report:
        ``GapReport`` from ``gap_check()`` (may be None if not threaded).
    generation_plan:
        ``GenerationPlan`` from ``build_generation_plan()``.
    thresholds:
        Optional overrides; falls back to conservative defaults.
    """
    t = thresholds or _DEFAULT_THRESHOLDS

    # 1. Check hard blockers from the primitive plan
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    blocker_report = getattr(generation_plan, "blocker_report", None)
    if blocker_report is not None and getattr(blocker_report, "should_block", False):
        blocker_ids = tuple(
            getattr(b, "id", str(b))
            for b in getattr(blocker_report, "blockers", ())
        )
        return BuildGateDecision(
            decision="block",
            reason=(
                f"Hard blockers detected: {getattr(blocker_report, 'summary', 'unknown')}. "
                f"Blocker IDs: {', '.join(blocker_ids) or 'none'}"
            ),
            gap_confidence=gap_report.confidence if gap_report is not None else 0.0,
            gate_source="pre_generation",
        )

    # 2. Check instruction conflicts
    resolved = getattr(generation_plan, "resolved_instructions", None)
    if resolved is not None:
        conflicts = getattr(resolved, "conflicts", ())
        if len(conflicts) > t.max_unresolved_conflicts:
            conflict_summaries = tuple(
                f"{getattr(c, 'scope_key', '?')}: {getattr(c, 'winner_id', '?')} vs "
                f"{getattr(c, 'loser_id', '?')}"
                for c in conflicts
            )
            return BuildGateDecision(
                decision="clarify",
                reason=(
                    f"{len(conflicts)} unresolved instruction conflict(s) "
                    f"exceed threshold ({t.max_unresolved_conflicts})"
                ),
                gap_confidence=gap_report.confidence if gap_report is not None else 0.0,
                unresolved_conflicts=conflict_summaries,
                gate_source="pre_generation",
            )

    # 3. Confidence check (if gap_report threaded)
    if gap_report is not None:
        confidence = gap_report.confidence
        if confidence < t.block_below:
            return BuildGateDecision(
                decision="block",
                reason=(
                    f"Knowledge confidence {confidence:.0%} is below the "
                    f"block threshold ({t.block_below:.0%})"
                ),
                gap_confidence=confidence,
                gate_source="pre_generation",
            )
        if confidence < t.narrow_below and primitive_plan is None:
            return BuildGateDecision(
                decision="narrow_route",
                reason=(
                    f"Knowledge confidence {confidence:.0%} is thin and "
                    f"no primitive plan resolved; restricting route candidates"
                ),
                gap_confidence=confidence,
                gate_source="pre_generation",
            )

    return BuildGateDecision(
        decision="proceed",
        reason="Pre-generation checks passed",
        gap_confidence=gap_report.confidence if gap_report is not None else 0.0,
        gate_source="pre_generation",
    )
