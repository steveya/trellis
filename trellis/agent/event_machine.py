"""Declarative event state machine for path-dependent derivatives.

Higher-level DSL abstraction over the runtime ``PathEventSpec`` /
``PathEventState`` system in ``models/monte_carlo/event_state.py``.

The ``EventMachine`` describes the event lifecycle declaratively —
states, transitions, guards, terminal conditions — and can be:

* **validated** for structural correctness (reachability, terminals)
* **compiled** down to a ``PathEventTimeline`` for the existing runtime
* **emitted** as a Python code skeleton the LLM fills in
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventState:
    """A named state in the event machine."""

    name: str
    kind: str = "intermediate"  # "initial" | "intermediate" | "terminal"
    description: str = ""
    state_variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventGuard:
    """A condition that must be true for a transition to fire."""

    expression: str  # e.g. "spot >= barrier_level"
    guard_type: str = "value_condition"  # "value_condition" | "schedule_trigger" | "count_threshold"
    parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventAction:
    """An action executed when a transition fires."""

    action_type: str  # "lock_return" | "remove_constituent" | "record_barrier_hit" | "pay_coupon" | "exercise" | "settle"
    description: str = ""
    parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventTransition:
    """A directed edge in the event machine."""

    name: str
    from_state: str
    to_state: str
    guard: EventGuard | None = None
    action: EventAction | None = None
    priority: int = 0
    event_kind: str = "observation"  # maps to PathEventSpec.kind


@dataclass(frozen=True)
class EventMachine:
    """Declarative event state machine for a path-dependent product."""

    states: tuple[EventState, ...]
    transitions: tuple[EventTransition, ...]
    initial_state: str
    terminal_states: tuple[str, ...] = ()
    description: str = ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_event_machine(machine: EventMachine) -> tuple[str, ...]:
    """Validate an ``EventMachine`` for structural correctness.

    Returns a tuple of error strings (empty if valid).
    """
    errors: list[str] = []
    state_names = {s.name for s in machine.states}

    # 1. Initial state must exist
    if machine.initial_state not in state_names:
        errors.append(
            f"Initial state '{machine.initial_state}' is not in defined states: "
            f"{sorted(state_names)}"
        )

    # 2. Terminal states must exist
    for ts in machine.terminal_states:
        if ts not in state_names:
            errors.append(
                f"Terminal state '{ts}' is not in defined states: {sorted(state_names)}"
            )

    # 3. Must have at least one terminal state
    if not machine.terminal_states:
        errors.append("No terminal states defined")

    # 4. Transition endpoints must reference defined states
    transition_names: list[str] = []
    for t in machine.transitions:
        if t.from_state not in state_names:
            errors.append(
                f"Transition '{t.name}' references undefined from_state '{t.from_state}'"
            )
        if t.to_state not in state_names:
            errors.append(
                f"Transition '{t.name}' references undefined to_state '{t.to_state}'"
            )
        transition_names.append(t.name)

    # 5. No duplicate transition names
    seen: set[str] = set()
    for tn in transition_names:
        if tn in seen:
            errors.append(f"Duplicate transition name: '{tn}'")
        seen.add(tn)

    # 6. Terminal states must have no outgoing transitions
    terminal_set = set(machine.terminal_states)
    for t in machine.transitions:
        if t.from_state in terminal_set:
            errors.append(
                f"Terminal state '{t.from_state}' has outgoing transition '{t.name}'"
            )

    # 7. At least one terminal is reachable from initial_state (BFS)
    if machine.initial_state in state_names and machine.terminal_states:
        adj: dict[str, list[str]] = {s: [] for s in state_names}
        for t in machine.transitions:
            if t.from_state in adj:
                adj[t.from_state].append(t.to_state)
        reachable = _reachable_from(machine.initial_state, adj)
        reachable_terminals = reachable & terminal_set
        if not reachable_terminals:
            errors.append(
                f"No terminal state is reachable from initial state "
                f"'{machine.initial_state}'. Reachable: {sorted(reachable)}"
            )

        # 8. No orphan states (every non-initial state should be reachable)
        orphans = state_names - reachable - {machine.initial_state}
        # Filter: only flag as orphan if the state is non-terminal or unreachable
        orphans -= terminal_set - reachable  # already flagged above
        if orphans:
            errors.append(
                f"Orphan states not reachable from '{machine.initial_state}': "
                f"{sorted(orphans)}"
            )

    return tuple(errors)


def _reachable_from(start: str, adj: dict[str, list[str]]) -> set[str]:
    """BFS reachability from *start*."""
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj.get(node, ()):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited


# ---------------------------------------------------------------------------
# Compilation to PathEventTimeline
# ---------------------------------------------------------------------------

def compile_event_machine_to_timeline(
    machine: EventMachine,
    *,
    event_times: tuple[float, ...] | list[float],
    T: float,
    n_steps: int,
) -> object:
    """Compile an ``EventMachine`` into a ``PathEventTimeline``.

    Maps each ``EventTransition`` to one or more ``PathEventSpec`` entries
    using the existing runtime types from ``event_state.py``.

    Parameters
    ----------
    machine : EventMachine
        The declarative state machine.
    event_times : sequence of float
        Observation times as year fractions.
    T : float
        Maturity (final time).
    n_steps : int
        Total number of Monte Carlo time steps.
    """
    from trellis.models.monte_carlo.event_state import (
        PathEventSpec,
        PathEventTimeline,
        event_step_indices,
    )

    step_indices = event_step_indices(event_times, T, n_steps)
    specs: list[PathEventSpec] = []

    for i, t_step in enumerate(step_indices):
        for transition in machine.transitions:
            payload: dict[str, object] = {
                "from_state": transition.from_state,
                "to_state": transition.to_state,
            }
            if transition.guard is not None:
                payload["guard_expression"] = transition.guard.expression
                payload["guard_type"] = transition.guard.guard_type
                payload["guard_parameters"] = transition.guard.parameters
            if transition.action is not None:
                payload["action_type"] = transition.action.action_type
                payload["action_parameters"] = transition.action.parameters

            if transition.guard:
                payload["trigger_condition"] = transition.guard.expression
            specs.append(PathEventSpec(
                name=f"{transition.name}_{i}",
                kind=transition.event_kind,
                step=t_step,
                priority=transition.priority,
                payload=payload,
            ))

    return PathEventTimeline(events=tuple(specs))


# ---------------------------------------------------------------------------
# Skeleton emission for LLM code generation
# ---------------------------------------------------------------------------

def emit_event_machine_skeleton(machine: EventMachine) -> str:
    """Emit a Python code skeleton for the LLM to fill in.

    Generates a state enum, guard/action function stubs, and a dispatch
    loop with ``# TODO`` markers.
    """
    lines: list[str] = [
        "# --- Event machine skeleton (auto-generated) ---",
        "",
        "# States",
    ]
    for s in machine.states:
        marker = ""
        if s.kind == "initial":
            marker = "  # initial"
        elif s.kind == "terminal":
            marker = "  # terminal"
        lines.append(f'STATE_{s.name.upper()} = "{s.name}"{marker}')

    lines.append("")
    lines.append(f'current_state = STATE_{machine.initial_state.upper()}')
    lines.append("")

    # Guard stubs
    guards = [t for t in machine.transitions if t.guard is not None]
    if guards:
        lines.append("# Guards")
        for t in guards:
            params = ", ".join(t.guard.parameters) if t.guard.parameters else "spot, **kwargs"
            lines.append(f"def guard_{t.name}({params}):")
            lines.append(f'    """Guard: {t.guard.expression}"""')
            lines.append(f"    # TODO: implement guard condition")
            lines.append(f"    return {t.guard.expression}")
            lines.append("")

    # Action stubs
    actions = [t for t in machine.transitions if t.action is not None]
    if actions:
        lines.append("# Actions")
        for t in actions:
            params = ", ".join(t.action.parameters) if t.action.parameters else "state, **kwargs"
            lines.append(f"def action_{t.name}({params}):")
            lines.append(f'    """Action: {t.action.action_type}"""')
            lines.append(f"    # TODO: implement {t.action.action_type}")
            lines.append(f"    pass")
            lines.append("")

    # Dispatch loop
    lines.append("# Dispatch loop")
    lines.append("for obs_idx, obs_date in enumerate(observation_dates):")
    lines.append("    cross_section = path_matrix[:, step_indices[obs_idx]]")
    for t in machine.transitions:
        guard_call = f"guard_{t.name}(cross_section)" if t.guard else "True"
        action_call = f"action_{t.name}(cross_section)" if t.action else "pass"
        lines.append(f'    if current_state == STATE_{t.from_state.upper()} and {guard_call}:')
        lines.append(f"        {action_call}")
        lines.append(f'        current_state = STATE_{t.to_state.upper()}')
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory functions for common patterns
# ---------------------------------------------------------------------------

def autocallable_event_machine(
    *,
    observation_dates: int = 4,
    barrier_level: float = 1.05,
    coupon_rate: float = 0.08,
) -> EventMachine:
    """Build the canonical autocallable event machine.

    States: alive → (knocked_out | matured)
    At each observation date, if spot >= barrier, knock out and pay coupon.
    At maturity, if still alive, pay terminal redemption.
    """
    return EventMachine(
        states=(
            EventState(name="alive", kind="initial", description="Note is active"),
            EventState(name="knocked_out", kind="terminal", description="Early redemption triggered"),
            EventState(name="matured", kind="terminal", description="Reached maturity without knock-out"),
        ),
        transitions=(
            EventTransition(
                name="check_barrier",
                from_state="alive",
                to_state="knocked_out",
                guard=EventGuard(
                    expression=f"spot >= {barrier_level}",
                    guard_type="value_condition",
                    parameters=("spot",),
                ),
                action=EventAction(
                    action_type="pay_coupon",
                    description=f"Pay coupon at rate {coupon_rate} and redeem at par",
                    parameters=("notional", "coupon_rate"),
                ),
                event_kind="barrier",
            ),
            EventTransition(
                name="settle_at_maturity",
                from_state="alive",
                to_state="matured",
                guard=EventGuard(
                    expression="is_maturity",
                    guard_type="schedule_trigger",
                    parameters=(),
                ),
                action=EventAction(
                    action_type="settle",
                    description="Terminal redemption based on final spot",
                    parameters=("spot", "notional"),
                ),
                priority=-1,  # lower priority than barrier check
                event_kind="settlement",
            ),
        ),
        initial_state="alive",
        terminal_states=("knocked_out", "matured"),
        description=f"Autocallable note: {observation_dates} observations, "
                    f"barrier={barrier_level}, coupon={coupon_rate}",
    )


def tarf_event_machine(
    *,
    fixing_dates: int = 12,
    target_level: float = 0.10,
) -> EventMachine:
    """Build the canonical TARF (Target Accrual Redemption Forward) event machine.

    States: accumulating → (target_reached | matured)
    At each fixing, accumulate gain. If cumulative gain >= target, terminate.
    """
    return EventMachine(
        states=(
            EventState(
                name="accumulating",
                kind="initial",
                description="Accumulating gains at each fixing",
                state_variables=("cumulative_gain",),
            ),
            EventState(name="target_reached", kind="terminal", description="Target accrual level reached"),
            EventState(name="matured", kind="terminal", description="All fixings exhausted"),
        ),
        transitions=(
            EventTransition(
                name="accumulate_and_check",
                from_state="accumulating",
                to_state="target_reached",
                guard=EventGuard(
                    expression=f"cumulative_gain >= {target_level}",
                    guard_type="count_threshold",
                    parameters=("cumulative_gain",),
                ),
                action=EventAction(
                    action_type="settle",
                    description="Terminate with accumulated gain capped at target",
                    parameters=("cumulative_gain", "target_level"),
                ),
                event_kind="observation",
            ),
            EventTransition(
                name="final_settlement",
                from_state="accumulating",
                to_state="matured",
                guard=EventGuard(
                    expression="is_final_fixing",
                    guard_type="schedule_trigger",
                    parameters=(),
                ),
                action=EventAction(
                    action_type="settle",
                    description="Settle with total accumulated gain",
                    parameters=("cumulative_gain",),
                ),
                priority=-1,
                event_kind="settlement",
            ),
        ),
        initial_state="accumulating",
        terminal_states=("target_reached", "matured"),
        description=f"TARF: {fixing_dates} fixings, target={target_level}",
    )


# ---------------------------------------------------------------------------
# Backward compatibility: flat event_transitions → EventMachine
# ---------------------------------------------------------------------------

_KNOWN_BASKET_TRANSITIONS = frozenset({
    "rank_remaining_constituents",
    "remove_selected_constituent",
    "lock_simple_return",
    "settle_at_maturity",
})


def event_transitions_to_machine(
    transitions: tuple[str, ...],
    *,
    state_dependence: str = "path_dependent",
) -> EventMachine | None:
    """Best-effort conversion from flat ``event_transitions`` to ``EventMachine``.

    Returns ``None`` if the transitions cannot be meaningfully structured.
    Recognizes the ranked-observation basket pattern.
    """
    if not transitions:
        return None

    transition_set = frozenset(transitions)

    # Recognize ranked observation basket
    if transition_set == _KNOWN_BASKET_TRANSITIONS or _KNOWN_BASKET_TRANSITIONS <= transition_set:
        return EventMachine(
            states=(
                EventState(name="observing", kind="initial", description="Processing observation dates"),
                EventState(name="settled", kind="terminal", description="All observations complete"),
            ),
            transitions=(
                EventTransition(
                    name="rank_and_select",
                    from_state="observing",
                    to_state="observing",
                    action=EventAction(
                        action_type="lock_return",
                        description="Rank remaining constituents, select best/worst, lock return",
                    ),
                    event_kind="observation",
                ),
                EventTransition(
                    name="settle",
                    from_state="observing",
                    to_state="settled",
                    guard=EventGuard(
                        expression="all_observations_complete",
                        guard_type="schedule_trigger",
                    ),
                    action=EventAction(action_type="settle"),
                    priority=-1,
                    event_kind="settlement",
                ),
            ),
            initial_state="observing",
            terminal_states=("settled",),
            description="Ranked observation basket (auto-converted from flat transitions)",
        )

    # Unknown pattern — build a linear chain
    chain_states = [
        EventState(name="start", kind="initial"),
        EventState(name="done", kind="terminal"),
    ]
    chain_transitions = []
    for i, label in enumerate(transitions):
        chain_transitions.append(EventTransition(
            name=label,
            from_state="start",
            to_state="done" if i == len(transitions) - 1 else "start",
            action=EventAction(action_type=label),
            event_kind="observation",
        ))
    return EventMachine(
        states=tuple(chain_states),
        transitions=tuple(chain_transitions),
        initial_state="start",
        terminal_states=("done",),
        description=f"Linear chain (auto-converted from {len(transitions)} flat transitions)",
    )
