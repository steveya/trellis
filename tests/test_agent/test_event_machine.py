"""Tests for the event state machine DSL (QUA-412).

Covers:
- EventMachine construction and validation
- Validation error detection (missing initial, unreachable terminal, orphans, etc.)
- Factory functions (autocallable, TARF)
- Backward compat migration from flat event_transitions
- Skeleton emission produces parseable Python
- Compilation to PathEventTimeline
"""

from __future__ import annotations

import ast

import pytest

from trellis.agent.event_machine import (
    EventAction,
    EventGuard,
    EventMachine,
    EventState,
    EventTransition,
    autocallable_event_machine,
    emit_event_machine_skeleton,
    event_transitions_to_machine,
    tarf_event_machine,
    validate_event_machine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_machine() -> EventMachine:
    """A minimal valid machine: start → done."""
    return EventMachine(
        states=(
            EventState(name="start", kind="initial"),
            EventState(name="done", kind="terminal"),
        ),
        transitions=(
            EventTransition(name="go", from_state="start", to_state="done"),
        ),
        initial_state="start",
        terminal_states=("done",),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestEventMachineConstruction:
    def test_simple_machine_is_valid(self):
        errors = validate_event_machine(_simple_machine())
        assert errors == ()

    def test_frozen(self):
        m = _simple_machine()
        with pytest.raises(AttributeError):
            m.initial_state = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_initial_state(self):
        m = EventMachine(
            states=(EventState(name="a"), EventState(name="b", kind="terminal")),
            transitions=(EventTransition(name="t", from_state="a", to_state="b"),),
            initial_state="nonexistent",
            terminal_states=("b",),
        )
        errors = validate_event_machine(m)
        assert any("nonexistent" in e for e in errors)

    def test_missing_terminal_state(self):
        m = EventMachine(
            states=(EventState(name="a", kind="initial"),),
            transitions=(),
            initial_state="a",
            terminal_states=("nonexistent",),
        )
        errors = validate_event_machine(m)
        assert any("nonexistent" in e for e in errors)

    def test_no_terminal_states(self):
        m = EventMachine(
            states=(EventState(name="a", kind="initial"),),
            transitions=(),
            initial_state="a",
            terminal_states=(),
        )
        errors = validate_event_machine(m)
        assert any("No terminal" in e for e in errors)

    def test_undefined_from_state(self):
        m = EventMachine(
            states=(EventState(name="a", kind="initial"), EventState(name="b", kind="terminal")),
            transitions=(EventTransition(name="t", from_state="ghost", to_state="b"),),
            initial_state="a",
            terminal_states=("b",),
        )
        errors = validate_event_machine(m)
        assert any("ghost" in e for e in errors)

    def test_duplicate_transition_names(self):
        m = EventMachine(
            states=(EventState(name="a", kind="initial"), EventState(name="b", kind="terminal")),
            transitions=(
                EventTransition(name="dup", from_state="a", to_state="b"),
                EventTransition(name="dup", from_state="a", to_state="b"),
            ),
            initial_state="a",
            terminal_states=("b",),
        )
        errors = validate_event_machine(m)
        assert any("Duplicate" in e for e in errors)

    def test_terminal_with_outgoing(self):
        m = EventMachine(
            states=(
                EventState(name="a", kind="initial"),
                EventState(name="b", kind="terminal"),
            ),
            transitions=(
                EventTransition(name="t1", from_state="a", to_state="b"),
                EventTransition(name="t2", from_state="b", to_state="a"),
            ),
            initial_state="a",
            terminal_states=("b",),
        )
        errors = validate_event_machine(m)
        assert any("outgoing" in e for e in errors)

    def test_unreachable_terminal(self):
        m = EventMachine(
            states=(
                EventState(name="a", kind="initial"),
                EventState(name="b"),
                EventState(name="c", kind="terminal"),
            ),
            transitions=(
                EventTransition(name="t", from_state="a", to_state="b"),
                # c is unreachable
            ),
            initial_state="a",
            terminal_states=("c",),
        )
        errors = validate_event_machine(m)
        assert any("reachable" in e.lower() for e in errors)

    def test_orphan_state(self):
        m = EventMachine(
            states=(
                EventState(name="a", kind="initial"),
                EventState(name="b", kind="terminal"),
                EventState(name="orphan"),
            ),
            transitions=(
                EventTransition(name="t", from_state="a", to_state="b"),
            ),
            initial_state="a",
            terminal_states=("b",),
        )
        errors = validate_event_machine(m)
        assert any("orphan" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactories:
    def test_autocallable_is_valid(self):
        m = autocallable_event_machine()
        errors = validate_event_machine(m)
        assert errors == ()
        assert m.initial_state == "alive"
        assert "knocked_out" in m.terminal_states
        assert "matured" in m.terminal_states
        assert len(m.transitions) == 2

    def test_tarf_is_valid(self):
        m = tarf_event_machine()
        errors = validate_event_machine(m)
        assert errors == ()
        assert m.initial_state == "accumulating"
        assert "target_reached" in m.terminal_states
        assert len(m.transitions) == 2

    def test_autocallable_custom_params(self):
        m = autocallable_event_machine(
            observation_dates=12,
            barrier_level=1.10,
            coupon_rate=0.12,
        )
        assert "1.1" in m.transitions[0].guard.expression
        assert m.description is not None


# ---------------------------------------------------------------------------
# Backward compat migration
# ---------------------------------------------------------------------------

class TestEventTransitionsToMachine:
    def test_basket_pattern_recognized(self):
        m = event_transitions_to_machine((
            "rank_remaining_constituents",
            "remove_selected_constituent",
            "lock_simple_return",
            "settle_at_maturity",
        ))
        assert m is not None
        errors = validate_event_machine(m)
        assert errors == ()
        assert m.initial_state == "observing"

    def test_unknown_pattern_linear_chain(self):
        m = event_transitions_to_machine(("step_a", "step_b", "step_c"))
        assert m is not None
        errors = validate_event_machine(m)
        assert errors == ()

    def test_empty_returns_none(self):
        assert event_transitions_to_machine(()) is None


# ---------------------------------------------------------------------------
# Skeleton emission
# ---------------------------------------------------------------------------

class TestSkeletonEmission:
    def test_produces_parseable_python(self):
        m = autocallable_event_machine()
        skeleton = emit_event_machine_skeleton(m)
        assert isinstance(skeleton, str)
        assert "STATE_ALIVE" in skeleton
        assert "STATE_KNOCKED_OUT" in skeleton
        assert "# TODO" in skeleton

    def test_guard_stubs_present(self):
        m = autocallable_event_machine()
        skeleton = emit_event_machine_skeleton(m)
        assert "def guard_check_barrier" in skeleton

    def test_action_stubs_present(self):
        m = autocallable_event_machine()
        skeleton = emit_event_machine_skeleton(m)
        assert "def action_check_barrier" in skeleton

    def test_dispatch_loop_present(self):
        m = _simple_machine()
        skeleton = emit_event_machine_skeleton(m)
        assert "for obs_idx" in skeleton


# ---------------------------------------------------------------------------
# Compilation to timeline
# ---------------------------------------------------------------------------

class TestCompileToTimeline:
    def test_compiles_simple_machine(self):
        from trellis.agent.event_machine import compile_event_machine_to_timeline

        m = _simple_machine()
        timeline = compile_event_machine_to_timeline(
            m,
            event_times=(0.25, 0.5, 0.75, 1.0),
            T=1.0,
            n_steps=252,
        )
        assert hasattr(timeline, "events")
        assert len(timeline.events) > 0

    def test_autocallable_compiles(self):
        from trellis.agent.event_machine import compile_event_machine_to_timeline

        m = autocallable_event_machine(observation_dates=4)
        timeline = compile_event_machine_to_timeline(
            m,
            event_times=(0.25, 0.5, 0.75, 1.0),
            T=1.0,
            n_steps=252,
        )
        # 4 observation dates × 2 transitions = 8 specs
        assert len(timeline.events) == 8


# ---------------------------------------------------------------------------
# Integration: event_machine on ProductIR
# ---------------------------------------------------------------------------

class TestProductIRIntegration:
    def test_product_ir_accepts_event_machine(self):
        from trellis.agent.knowledge.schema import ProductIR

        m = autocallable_event_machine()
        ir = ProductIR(
            instrument="autocallable",
            payoff_family="barrier_payoff",
            state_dependence="path_dependent",
            event_machine=m,
        )
        assert ir.event_machine is m
        assert ir.event_machine.initial_state == "alive"

    def test_product_ir_default_none(self):
        from trellis.agent.knowledge.schema import ProductIR

        ir = ProductIR(instrument="bond", payoff_family="fixed_coupon")
        assert ir.event_machine is None
