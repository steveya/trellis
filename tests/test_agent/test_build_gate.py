"""Tests for the build gate checkpoint (QUA-410).

Covers:
- Pre-flight gate: proceed, narrow_route, block decisions
- Pre-generation gate: hard blockers, instruction conflicts, proceed
- Custom thresholds
- Spec-schema hint wiring into planner
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from trellis.agent.build_gate import (
    evaluate_pre_flight_gate,
    evaluate_pre_generation_gate,
)
from trellis.agent.knowledge.schema import (
    BuildGateDecision,
    BuildGateThresholds,
)


# ---------------------------------------------------------------------------
# Minimal stubs for gap_report and generation_plan
# ---------------------------------------------------------------------------

@dataclass
class _StubGapReport:
    confidence: float = 0.8
    has_promoted_route: bool = True
    route_gap: object = None
    missing: list = field(default_factory=list)


@dataclass
class _StubBlockerReport:
    should_block: bool = False
    summary: str = ""
    blockers: tuple = ()


@dataclass
class _StubConflict:
    scope_key: str = "method:monte_carlo"
    winner_id: str = "route:a"
    loser_id: str = "route:b"


@dataclass
class _StubResolvedInstructions:
    effective_instructions: tuple = ()
    conflicts: tuple = ()


@dataclass
class _StubPrimitivePlan:
    route_family: str = "analytical"
    engine_family: str = "analytical"
    blockers: tuple = ()


@dataclass
class _StubGenerationPlan:
    method: str = "analytical"
    instrument_type: str | None = None
    inspected_modules: tuple = ()
    approved_modules: tuple = ()
    symbols_to_reuse: tuple = ()
    proposed_tests: tuple = ()
    primitive_plan: _StubPrimitivePlan | None = None
    blocker_report: _StubBlockerReport | None = None
    resolved_instructions: _StubResolvedInstructions | None = None


# ---------------------------------------------------------------------------
# Pre-flight gate tests
# ---------------------------------------------------------------------------

class TestPreFlightGate:
    def test_proceed_on_high_confidence(self):
        gap = _StubGapReport(confidence=0.8)
        decision = evaluate_pre_flight_gate(gap)
        assert decision.decision == "proceed"
        assert decision.gate_source == "pre_flight"

    def test_block_on_very_low_confidence(self):
        gap = _StubGapReport(confidence=0.2)
        decision = evaluate_pre_flight_gate(gap)
        assert decision.decision == "block"
        assert "20%" in decision.reason
        assert decision.gap_confidence == 0.2

    def test_narrow_on_medium_confidence(self):
        gap = _StubGapReport(confidence=0.45)
        decision = evaluate_pre_flight_gate(gap)
        assert decision.decision == "narrow_route"
        assert decision.gap_confidence == 0.45

    def test_block_when_no_promoted_route_required(self):
        gap = _StubGapReport(confidence=0.9, has_promoted_route=False)
        thresholds = BuildGateThresholds(require_promoted_route=True)
        decision = evaluate_pre_flight_gate(gap, thresholds=thresholds)
        assert decision.decision == "block"
        assert "promoted route" in decision.reason.lower()

    def test_proceed_when_no_promoted_route_not_required(self):
        gap = _StubGapReport(confidence=0.9, has_promoted_route=False)
        # Default: require_promoted_route=False
        decision = evaluate_pre_flight_gate(gap)
        assert decision.decision == "proceed"

    def test_custom_thresholds(self):
        gap = _StubGapReport(confidence=0.5)
        # With default thresholds (block<0.3, narrow<0.55): narrow_route
        decision_default = evaluate_pre_flight_gate(gap)
        assert decision_default.decision == "narrow_route"

        # With custom thresholds: proceed because 0.5 >= 0.4
        thresholds = BuildGateThresholds(block_below=0.2, narrow_below=0.4)
        decision_custom = evaluate_pre_flight_gate(gap, thresholds=thresholds)
        assert decision_custom.decision == "proceed"

    def test_exact_boundary_block(self):
        """Confidence exactly at block_below should still block (< not <=)."""
        gap = _StubGapReport(confidence=0.3)
        # 0.3 is NOT < 0.3, so should NOT block
        decision = evaluate_pre_flight_gate(gap)
        assert decision.decision != "block"

    def test_exact_boundary_narrow(self):
        gap = _StubGapReport(confidence=0.55)
        # 0.55 is NOT < 0.55, so should proceed
        decision = evaluate_pre_flight_gate(gap)
        assert decision.decision == "proceed"


# ---------------------------------------------------------------------------
# Pre-generation gate tests
# ---------------------------------------------------------------------------

class TestPreGenerationGate:
    def test_proceed_clean_plan(self):
        gap = _StubGapReport(confidence=0.8)
        plan = _StubGenerationPlan(
            primitive_plan=_StubPrimitivePlan(),
            resolved_instructions=_StubResolvedInstructions(),
        )
        decision = evaluate_pre_generation_gate(gap, plan)
        assert decision.decision == "proceed"
        assert decision.gate_source == "pre_generation"

    def test_block_on_hard_blocker(self):
        gap = _StubGapReport(confidence=0.8)
        plan = _StubGenerationPlan(
            blocker_report=_StubBlockerReport(
                should_block=True,
                summary="Missing required primitive: generate_schedule",
                blockers=(_StubConflict(scope_key="blocker_1"),),
            ),
        )
        decision = evaluate_pre_generation_gate(gap, plan)
        assert decision.decision == "block"
        assert "generate_schedule" in decision.reason

    def test_clarify_on_instruction_conflicts(self):
        gap = _StubGapReport(confidence=0.8)
        plan = _StubGenerationPlan(
            resolved_instructions=_StubResolvedInstructions(
                conflicts=(
                    _StubConflict(scope_key="method:mc", winner_id="r1", loser_id="r2"),
                ),
            ),
        )
        decision = evaluate_pre_generation_gate(gap, plan)
        assert decision.decision == "clarify"
        assert len(decision.unresolved_conflicts) == 1
        assert "r1" in decision.unresolved_conflicts[0]

    def test_allow_conflicts_with_raised_threshold(self):
        gap = _StubGapReport(confidence=0.8)
        plan = _StubGenerationPlan(
            resolved_instructions=_StubResolvedInstructions(
                conflicts=(
                    _StubConflict(),
                ),
            ),
        )
        thresholds = BuildGateThresholds(max_unresolved_conflicts=1)
        decision = evaluate_pre_generation_gate(gap, plan, thresholds=thresholds)
        assert decision.decision == "proceed"

    def test_block_on_low_confidence_in_pre_gen(self):
        gap = _StubGapReport(confidence=0.2)
        plan = _StubGenerationPlan()
        decision = evaluate_pre_generation_gate(gap, plan)
        assert decision.decision == "block"

    def test_narrow_when_no_primitive_plan_and_thin_knowledge(self):
        gap = _StubGapReport(confidence=0.45)
        plan = _StubGenerationPlan(primitive_plan=None)
        decision = evaluate_pre_generation_gate(gap, plan)
        assert decision.decision == "narrow_route"

    def test_proceed_when_no_primitive_plan_but_high_confidence(self):
        gap = _StubGapReport(confidence=0.8)
        plan = _StubGenerationPlan(primitive_plan=None)
        decision = evaluate_pre_generation_gate(gap, plan)
        assert decision.decision == "proceed"

    def test_none_gap_report_proceeds_when_clean(self):
        plan = _StubGenerationPlan(
            primitive_plan=_StubPrimitivePlan(),
            resolved_instructions=_StubResolvedInstructions(),
        )
        decision = evaluate_pre_generation_gate(None, plan)
        assert decision.decision == "proceed"
        assert decision.gap_confidence == 0.0

    def test_blocker_takes_priority_over_conflicts(self):
        gap = _StubGapReport(confidence=0.8)
        plan = _StubGenerationPlan(
            blocker_report=_StubBlockerReport(
                should_block=True,
                summary="Critical blocker",
            ),
            resolved_instructions=_StubResolvedInstructions(
                conflicts=(_StubConflict(),),
            ),
        )
        decision = evaluate_pre_generation_gate(gap, plan)
        # Blocker check runs first
        assert decision.decision == "block"


# ---------------------------------------------------------------------------
# BuildGateDecision dataclass tests
# ---------------------------------------------------------------------------

class TestBuildGateDecision:
    def test_frozen(self):
        d = BuildGateDecision(decision="proceed", reason="ok", gap_confidence=0.8)
        with pytest.raises(AttributeError):
            d.decision = "block"  # type: ignore[misc]

    def test_defaults(self):
        d = BuildGateDecision(decision="proceed", reason="ok", gap_confidence=0.8)
        assert d.unresolved_conflicts == ()
        assert d.missing_required_inputs == ()
        assert d.suggested_fallback_route is None
        assert d.gate_source == ""


class TestBuildGateThresholds:
    def test_defaults(self):
        t = BuildGateThresholds()
        assert t.block_below == 0.3
        assert t.narrow_below == 0.55
        assert t.max_unresolved_conflicts == 0
        assert t.require_promoted_route is False

    def test_custom(self):
        t = BuildGateThresholds(block_below=0.1, narrow_below=0.3)
        assert t.block_below == 0.1
        assert t.narrow_below == 0.3


# ---------------------------------------------------------------------------
# Spec-schema hint wiring
# ---------------------------------------------------------------------------

class TestSpecSchemaHintWiring:
    def test_hint_used_when_available(self):
        """Verify plan_build uses spec_schema_hint when it matches a known spec."""
        from trellis.agent.planner import plan_build, SPECIALIZED_SPECS

        # Find a known spec key to use as hint
        if not SPECIALIZED_SPECS:
            pytest.skip("No SPECIALIZED_SPECS defined")
        hint_key = next(iter(SPECIALIZED_SPECS))
        expected_spec = SPECIALIZED_SPECS[hint_key]

        plan = plan_build(
            "some generic description",
            requirements=set(),
            instrument_type=None,
            preferred_method=None,
            spec_schema_hint=hint_key,
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == expected_spec.class_name

    def test_fallback_to_regex_when_no_hint(self):
        """Verify plan_build falls back to regex matching when no hint."""
        from trellis.agent.planner import plan_build

        plan = plan_build(
            "European option on equity",
            requirements=set(),
            instrument_type="european_option",
            preferred_method="analytical",
            spec_schema_hint=None,
        )
        # Should still resolve via regex/STATIC_SPECS fallback
        assert plan.spec_schema is not None

    def test_fallback_when_hint_unknown(self):
        """Verify plan_build falls back when hint doesn't match any spec."""
        from trellis.agent.planner import plan_build

        plan = plan_build(
            "European option on equity",
            requirements=set(),
            instrument_type="european_option",
            preferred_method="analytical",
            spec_schema_hint="nonexistent_spec_xyz",
        )
        # Should still resolve via regex fallback
        assert plan.spec_schema is not None
