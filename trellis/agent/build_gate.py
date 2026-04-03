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


def _semantic_validation_gate_decision(report, *, gate_source: str) -> BuildGateDecision | None:
    """Map a semantic validation report onto the existing build-gate surface."""
    if report is None:
        return None
    if not hasattr(report, "normalized_contract") or not hasattr(report, "errors") or not hasattr(report, "warnings"):
        return None

    errors = tuple(getattr(report, "errors", ()) or ())
    warnings = tuple(getattr(report, "warnings", ()) or ())
    if errors:
        preview = "; ".join(errors[:2])
        if len(errors) > 2:
            preview += f"; +{len(errors) - 2} more"
        return BuildGateDecision(
            decision="block",
            reason=f"Semantic contract validation failed: {preview}",
            gap_confidence=0.0,
            gate_source=gate_source,
        )
    if warnings:
        preview = "; ".join(warnings[:2])
        if len(warnings) > 2:
            preview += f"; +{len(warnings) - 2} more"
        return BuildGateDecision(
            decision="proceed",
            reason=f"Semantic contract validation passed with warnings: {preview}",
            gap_confidence=1.0,
            gate_source=gate_source,
        )
    return BuildGateDecision(
        decision="proceed",
        reason="Semantic contract validation passed",
        gap_confidence=1.0,
        gate_source=gate_source,
    )


def _route_admissibility_gate_decision(
    generation_plan,
    *,
    semantic_blueprint=None,
    gap_confidence: float,
    gate_source: str,
) -> BuildGateDecision | None:
    """Map typed route-admissibility failures onto the build-gate surface."""
    if generation_plan is None or semantic_blueprint is None:
        return None
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    route_id = getattr(primitive_plan, "route", "") if primitive_plan is not None else ""
    if not route_id:
        return None

    from trellis.agent.route_registry import (
        evaluate_route_admissibility,
        find_route_by_id,
    )

    route_spec = find_route_by_id(route_id)
    if route_spec is None:
        return None
    decision = evaluate_route_admissibility(
        route_spec,
        semantic_blueprint=semantic_blueprint,
        product_ir=getattr(semantic_blueprint, "product_ir", None),
    )
    if decision.ok:
        return None

    preview = "; ".join(decision.failures[:2])
    if len(decision.failures) > 2:
        preview += f"; +{len(decision.failures) - 2} more"
    return BuildGateDecision(
        decision="block",
        reason=f"Route admissibility failed for `{route_id}`: {preview}",
        gap_confidence=gap_confidence,
        route_admissibility_failures=decision.failures,
        gate_source=gate_source,
    )


def _lane_plan_summary(
    generation_plan,
    *,
    semantic_blueprint=None,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """Return the lane family, kind, exact bindings, and steps available."""
    lane_family = str(getattr(generation_plan, "lane_family", "") or "")
    lane_kind = str(getattr(generation_plan, "lane_plan_kind", "") or "")
    exact_refs = tuple(getattr(generation_plan, "lane_exact_binding_refs", ()) or ())
    steps = tuple(getattr(generation_plan, "lane_construction_steps", ()) or ())
    if lane_family or lane_kind or exact_refs or steps:
        return lane_family, lane_kind, exact_refs, steps

    lane_plan = getattr(semantic_blueprint, "lane_plan", None)
    if lane_plan is None:
        return "", "", (), ()
    return (
        str(getattr(lane_plan, "lane_family", "") or ""),
        str(getattr(lane_plan, "plan_kind", "") or ""),
        tuple(getattr(lane_plan, "exact_target_refs", ()) or ()),
        tuple(getattr(lane_plan, "construction_steps", ()) or ()),
    )


def _lane_obligation_gate_decision(
    generation_plan,
    *,
    semantic_blueprint=None,
    gap_confidence: float,
    gate_source: str,
) -> BuildGateDecision | None:
    """Require either an exact backend binding or a constructive lane plan."""
    if generation_plan is None:
        return None
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    lane_family, lane_kind, exact_refs, steps = _lane_plan_summary(
        generation_plan,
        semantic_blueprint=semantic_blueprint,
    )
    if primitive_plan is not None:
        return None
    if exact_refs or steps:
        return None

    if not lane_family and not lane_kind:
        return BuildGateDecision(
            decision="narrow_route",
            reason=(
                "No primitive plan resolved and the compiler did not emit a constructive "
                "lane plan; narrowing generation instead of guessing a backend."
            ),
            gap_confidence=gap_confidence,
            gate_source=gate_source,
        )

    return BuildGateDecision(
        decision="block",
        reason=(
            f"Lane `{lane_family or 'unknown'}` has no exact backend binding and no "
            "constructive steps; generation would be route guessing."
        ),
        gap_confidence=gap_confidence,
        gate_source=gate_source,
    )


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
    semantic_decision = _semantic_validation_gate_decision(
        gap_report,
        gate_source="pre_flight",
    )
    if semantic_decision is not None:
        return semantic_decision

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
    semantic_blueprint=None,
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
    semantic_decision = _semantic_validation_gate_decision(
        gap_report,
        gate_source="pre_generation",
    )
    if semantic_decision is not None:
        return semantic_decision

    route_admissibility_decision = _route_admissibility_gate_decision(
        generation_plan,
        semantic_blueprint=semantic_blueprint,
        gap_confidence=gap_report.confidence if gap_report is not None else 0.0,
        gate_source="pre_generation",
    )
    if route_admissibility_decision is not None:
        return route_admissibility_decision

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
        lane_family, lane_kind, exact_refs, steps = _lane_plan_summary(
            generation_plan,
            semantic_blueprint=semantic_blueprint,
        )
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
        if primitive_plan is None and not (lane_family or lane_kind or exact_refs or steps):
            return BuildGateDecision(
                decision="narrow_route",
                reason=(
                    "No primitive plan resolved and no compiler-backed lane plan "
                    "was attached; restricting generation instead of guessing."
                ),
                gap_confidence=confidence,
                gate_source="pre_generation",
            )

    lane_obligation_decision = _lane_obligation_gate_decision(
        generation_plan,
        semantic_blueprint=semantic_blueprint,
        gap_confidence=gap_report.confidence if gap_report is not None else 0.0,
        gate_source="pre_generation",
    )
    if lane_obligation_decision is not None:
        return lane_obligation_decision

    return BuildGateDecision(
        decision="proceed",
        reason="Pre-generation checks passed",
        gap_confidence=gap_report.confidence if gap_report is not None else 0.0,
        gate_source="pre_generation",
    )
