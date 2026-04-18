"""Contract pattern evaluator against ProductIR (QUA-918 / Phase 1.5.B).

This module makes the :class:`~trellis.agent.contract_pattern.ContractPattern`
AST built in QUA-917 executable: given a pattern and a
:class:`~trellis.agent.knowledge.schema.ProductIR`, :func:`evaluate_pattern`
decides whether the pattern matches and returns any captured bindings for
named wildcards.

Design notes
------------

**Target surface.** The evaluator matches against ``ProductIR`` only.  Phase 2
may extend to ``ContractIR`` (tracked separately); the AST is expressive
enough that the walker does not need to change when a richer target is added,
only the per-slot matchers (:func:`_match_exercise`, :func:`_match_underlying`,
etc.) that bridge pattern fields onto concrete target fields.

**Bare strings vs :class:`AtomPattern`.** ``UnderlyingPattern.kind``,
``ExercisePattern.style``, and ``ObservationPattern.kind`` can carry either a
bare string (ergonomic construction, e.g. ``ExercisePattern(style="european")``)
or a structured :class:`AtomPattern` / composite.  The field-level matcher
:func:`_match_field_multivalued` treats the two forms equivalently by
dispatching on the pattern type and comparing against the target value(s).

**Instrument-level payoff tags.** ``PayoffPattern(kind="swaption_payoff")``
(zero-arity instrument tag) matches against ``ProductIR.payoff_family`` via
the hardcoded mapping in :data:`_INSTRUMENT_TAG_TO_PAYOFF_FAMILIES`.  This
map is deliberately kept colocated with the evaluator so QUA-919's schema
integration and QUA-920's route migration can extend it without touching the
AST module.

**Structural payoffs.** Composite payoff shapes such as
``max(sub(spot, strike), 0)`` are canonical templates for particular
``payoff_family`` values.  :func:`_match_structural_payoff` recognises the
shapes currently exercised by the analytical_black76 canonical patterns
(vanilla call/put) and matches them against ``payoff_family="vanilla_option"``.
Unrecognised structural shapes fall through to an explicit mismatch rather
than accidentally matching the wrong family.

**Schedule patterns.** ``SchedulePattern.frequency`` has no corresponding
``ProductIR`` field yet.  Per QUA-917's forward-compat notes the evaluator
treats wildcards / absent frequency as a trivial match.  A concrete-value
frequency pattern currently produces a principled mismatch with a
``mismatch_reason`` explaining the limitation — the AST parser does not
produce such schedules for any existing canonical pattern.

Follow-ups for downstream slices
--------------------------------

- QUA-919 wires the evaluator into ``conditional_primitives.when`` dispatch.
  Migrated clauses will call :func:`evaluate_pattern` rather than
  ``_matches_condition``.
- QUA-920 migrates the four canonical ``analytical_black76`` when-clauses
  to :class:`ContractPattern` form.  The parity tests in
  ``tests/test_agent/test_contract_pattern_eval.py`` lock the intended
  semantics.
- Phase 3 ``@solves_pattern`` uses :func:`evaluate_pattern` to dispatch
  kernel implementations.
- Phase 2 extends the walker to match against the richer ``ContractIR``;
  the AST does not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from trellis.agent.contract_pattern import (
    AndPattern,
    AtomPattern,
    ConstantPattern,
    ContractPattern,
    ExercisePattern,
    NotPattern,
    ObservationPattern,
    OrPattern,
    PayoffPattern,
    SchedulePattern,
    SpotPattern,
    StrikePattern,
    UnderlyingPattern,
    Wildcard,
)
from trellis.agent.knowledge.schema import ProductIR


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    """Outcome of evaluating a :class:`ContractPattern` against a target.

    ``ok`` is the top-level boolean success flag.  ``bindings`` collects
    captured named wildcards: ``{name: captured_value}``.  ``mismatch_reason``
    is a short human-readable explanation when ``ok`` is ``False``; it is
    ``None`` on success.
    """

    ok: bool
    bindings: dict[str, Any] = field(default_factory=dict)
    mismatch_reason: str | None = None


# ---------------------------------------------------------------------------
# Instrument-level payoff tag mapping
# ---------------------------------------------------------------------------


# Instrument-level payoff tags are zero-arity :class:`PayoffPattern` head
# tags that stand in for a whole product family.  They map to the set of
# ``ProductIR.payoff_family`` strings that should be considered a match.
#
# Extend this table as new product families land.  The union semantics
# mirror the expanded-family semantics used by ``_expanded_payoff_families``
# in :mod:`trellis.agent.route_registry` so QUA-920's migration remains a
# mechanical rename.
_INSTRUMENT_TAG_TO_PAYOFF_FAMILIES: Mapping[str, frozenset[str]] = {
    "vanilla_payoff": frozenset({"vanilla_option"}),
    "swaption_payoff": frozenset({"swaption", "rate_style_swaption"}),
    "basket_payoff": frozenset({"basket_option", "basket_path_payoff"}),
    "variance_payoff": frozenset({"variance_swap", "variance_option"}),
    "digital_payoff": frozenset({"digital_option"}),
    "barrier_payoff": frozenset({"barrier_option"}),
    "rate_payoff": frozenset(
        {
            "rate_cap_floor_strip",
            "range_accrual_coupon",
            "fixed_coupon",
        }
    ),
    "lookback_payoff": frozenset({"lookback_option"}),
    "asian_payoff": frozenset({"asian_option"}),
    "cliquet_payoff": frozenset({"cliquet_option"}),
    "chooser_payoff": frozenset({"chooser_option"}),
    "compound_payoff": frozenset({"compound_option"}),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_pattern(pattern: ContractPattern, target: ProductIR) -> MatchResult:
    """Evaluate ``pattern`` against ``target`` and return a :class:`MatchResult`.

    A :class:`ContractPattern` matches a :class:`ProductIR` when every
    non-``None`` sub-pattern matches the corresponding slice of the target.
    Captured named wildcards from different slices must bind consistently
    (same name → same value); a binding conflict is a hard mismatch.

    Unrecognised payoff shapes, unsupported schedule constraints, or target
    fields outside the evaluator's current vocabulary produce a principled
    mismatch rather than silently succeeding.
    """
    if not isinstance(pattern, ContractPattern):
        raise TypeError(
            f"evaluate_pattern expects a ContractPattern, got {type(pattern).__name__}"
        )
    if not isinstance(target, ProductIR):
        raise TypeError(
            f"evaluate_pattern expects a ProductIR target, got {type(target).__name__}"
        )

    bindings: dict[str, Any] = {}

    # Top-level fields compose with AND semantics: every non-None slot must
    # match; a mismatch on any slot is the final result.
    if pattern.payoff is not None:
        outcome = _match_payoff(pattern.payoff, target, bindings)
        if not outcome.ok:
            return outcome
        bindings = outcome.bindings

    if pattern.exercise is not None:
        outcome = _match_exercise(pattern.exercise, target, bindings)
        if not outcome.ok:
            return outcome
        bindings = outcome.bindings

    if pattern.observation is not None:
        outcome = _match_observation(pattern.observation, target, bindings)
        if not outcome.ok:
            return outcome
        bindings = outcome.bindings

    if pattern.underlying is not None:
        outcome = _match_underlying(pattern.underlying, target, bindings)
        if not outcome.ok:
            return outcome
        bindings = outcome.bindings

    return MatchResult(ok=True, bindings=bindings)


# ---------------------------------------------------------------------------
# Per-slot matchers
# ---------------------------------------------------------------------------


def _match_exercise(
    pattern: ExercisePattern,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    style_outcome = _match_field(
        pattern.style,
        target.exercise_style,
        bindings,
        field_label="exercise_style",
    )
    if not style_outcome.ok:
        return style_outcome

    if pattern.schedule is None:
        return style_outcome

    return _match_schedule(pattern.schedule, target, style_outcome.bindings)


def _match_schedule(
    pattern: SchedulePattern,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    # QUA-917 only exposes ``frequency``; ProductIR has no direct frequency
    # field yet (QUA-918 follow-up).  A wildcard / absent frequency is a
    # trivial match; anything else is recorded as unsupported and currently
    # mismatches rather than silently accepting.
    if pattern.frequency is None:
        return MatchResult(ok=True, bindings=bindings)
    if isinstance(pattern.frequency, Wildcard):
        new_bindings = bindings
        if pattern.frequency.name is not None:
            # Best-effort capture: no frequency is available on ProductIR
            # today, so wildcard capture yields ``None``.  This keeps the
            # capture stable when a frequency field is added later.
            new_bindings = _bind(
                bindings, pattern.frequency.name, None, field_label="schedule.frequency"
            )
            if new_bindings is None:
                return MatchResult(
                    ok=False,
                    bindings=bindings,
                    mismatch_reason=(
                        f"binding conflict on '{pattern.frequency.name}' "
                        "while matching schedule.frequency"
                    ),
                )
        return MatchResult(ok=True, bindings=new_bindings)

    return MatchResult(
        ok=False,
        bindings=bindings,
        mismatch_reason=(
            "schedule.frequency matching against a concrete value is not "
            "yet implemented; ProductIR has no frequency field"
        ),
    )


def _match_observation(
    pattern: ObservationPattern,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    # Map ``ProductIR.state_dependence`` to an observation-kind space that
    # pattern authors can write against ergonomically.  Exact string match
    # is the default; the small alias table below covers the common cases
    # used by routes.yaml and the conditional-primitive dispatcher.
    observed_kind = _normalise_observation_kind(target)
    return _match_field_multivalued(
        pattern.kind,
        candidate_values=observed_kind,
        bindings=bindings,
        field_label="observation.kind",
        primary_value=observed_kind[0] if observed_kind else None,
    )


def _normalise_observation_kind(target: ProductIR) -> tuple[str, ...]:
    """Expand ``state_dependence`` into observation-kind aliases.

    A pattern author writing ``observation: {kind: terminal}`` expects
    that to match any "terminal_*" state label.  This helper keeps the
    alias rules localised here.
    """
    state = (target.state_dependence or "").strip()
    candidates: list[str] = []
    if state:
        candidates.append(state)
        if state.startswith("terminal"):
            candidates.append("terminal")
        if "schedule" in state:
            candidates.append("schedule")
        if state.startswith("path") or state == "pathwise_only":
            candidates.append("path_dependent")
    return tuple(candidates)


def _match_underlying(
    pattern: UnderlyingPattern,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    # ``UnderlyingPattern.kind`` matches either ``model_family`` or any
    # member of ``candidate_engine_families``.  Pattern authors use the
    # same vocabulary as routes.yaml ``model_family`` clauses; engine
    # families let dispatch target analytical-only routes etc.
    candidate_values: tuple[str, ...] = tuple(
        v for v in (target.model_family, *target.candidate_engine_families) if v
    )
    kind_outcome = _match_field_multivalued(
        pattern.kind,
        candidate_values=candidate_values,
        bindings=bindings,
        field_label="underlying.kind",
        primary_value=target.model_family,
    )
    if not kind_outcome.ok:
        return kind_outcome

    if pattern.dynamics is None:
        return kind_outcome

    # ``dynamics`` currently has no dedicated ProductIR field; the closest
    # surrogate is ``model_family`` itself (equity_diffusion, rate_style,
    # etc.).  We reuse it so pattern authors can write an explicit
    # dynamics constraint on the same vocabulary.
    return _match_field(
        pattern.dynamics,
        target.model_family,
        kind_outcome.bindings,
        field_label="underlying.dynamics",
    )


def _match_payoff(
    pattern: Any,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    # Composite wrappers first so they work regardless of the child kind.
    if isinstance(pattern, AndPattern):
        return _match_and_payoff(pattern, target, bindings)
    if isinstance(pattern, OrPattern):
        return _match_or_payoff(pattern, target, bindings)
    if isinstance(pattern, NotPattern):
        return _match_not_payoff(pattern, target, bindings)

    if isinstance(pattern, PayoffPattern):
        return _match_payoff_head(pattern, target, bindings)

    if isinstance(pattern, SpotPattern):
        # A top-level bare ``Spot`` payoff is not a full vanilla shape; the
        # canonical vanilla/digital payoffs always come wrapped in a head
        # tag.  Treat this as an unsupported shape.
        return MatchResult(
            ok=False,
            bindings=bindings,
            mismatch_reason=(
                "top-level SpotPattern has no ProductIR surrogate; wrap it "
                "in a head-tagged PayoffPattern such as max(sub(spot, strike), 0)"
            ),
        )

    if isinstance(pattern, StrikePattern):
        return MatchResult(
            ok=False,
            bindings=bindings,
            mismatch_reason=(
                "top-level StrikePattern has no ProductIR surrogate; "
                "strike leaves must appear inside a head-tagged payoff"
            ),
        )

    if isinstance(pattern, ConstantPattern):
        return MatchResult(
            ok=False,
            bindings=bindings,
            mismatch_reason="top-level ConstantPattern is not matchable against ProductIR",
        )

    return MatchResult(
        ok=False,
        bindings=bindings,
        mismatch_reason=f"unsupported payoff pattern type {type(pattern).__name__}",
    )


def _match_payoff_head(
    pattern: PayoffPattern,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    # Instrument-level zero-arity tags match ``payoff_family`` directly.
    if pattern.kind in _INSTRUMENT_TAG_TO_PAYOFF_FAMILIES and not pattern.args:
        return _match_instrument_tag(pattern.kind, target, bindings)

    # Structural head tags (max/sub/mul/constant/etc.) match recognised
    # canonical templates against the coarse ``payoff_family`` label.
    return _match_structural_payoff(pattern, target, bindings)


def _match_instrument_tag(
    tag: str,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    expected = _INSTRUMENT_TAG_TO_PAYOFF_FAMILIES[tag]
    observed_families: set[str] = set()
    if target.payoff_family:
        observed_families.add(target.payoff_family)
    observed_families.update(t for t in target.payoff_traits if t)
    # Also admit the tag itself appearing verbatim in payoff_traits so
    # authors can tag products structurally without inventing new families.
    observed_families.add(tag)

    if expected & observed_families:
        return MatchResult(ok=True, bindings=bindings)
    return MatchResult(
        ok=False,
        bindings=bindings,
        mismatch_reason=(
            f"payoff tag {tag!r} expected one of "
            f"{sorted(expected)}, got payoff_family={target.payoff_family!r} "
            f"traits={list(target.payoff_traits)}"
        ),
    )


def _match_structural_payoff(
    pattern: PayoffPattern,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    # Canonical vanilla call/put shape: max(sub(spot, strike), constant(0)).
    if _is_vanilla_intrinsic_shape(pattern):
        if target.payoff_family == "vanilla_option":
            return MatchResult(ok=True, bindings=bindings)
        return MatchResult(
            ok=False,
            bindings=bindings,
            mismatch_reason=(
                f"structural vanilla payoff expected payoff_family='vanilla_option', "
                f"got {target.payoff_family!r}"
            ),
        )

    return MatchResult(
        ok=False,
        bindings=bindings,
        mismatch_reason=(
            f"structural payoff kind={pattern.kind!r} is not a recognised canonical "
            "template; wrap instrument-level payoffs in their zero-arity tag instead"
        ),
    )


def _is_vanilla_intrinsic_shape(pattern: PayoffPattern) -> bool:
    """Return True when ``pattern`` is ``max(sub(spot, strike), constant 0)``."""
    if pattern.kind != "max" or len(pattern.args) != 2:
        return False
    sub_node, zero_node = pattern.args
    if not isinstance(sub_node, PayoffPattern) or sub_node.kind != "sub":
        return False
    if len(sub_node.args) != 2:
        return False
    lhs, rhs = sub_node.args
    if not isinstance(lhs, SpotPattern):
        return False
    if not isinstance(rhs, StrikePattern):
        return False
    if not isinstance(zero_node, ConstantPattern):
        return False
    zero_value = zero_node.value
    if isinstance(zero_value, Wildcard):
        return True
    try:
        return float(zero_value) == 0.0
    except (TypeError, ValueError):
        return False


def _match_and_payoff(
    pattern: AndPattern,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    current = bindings
    for child in pattern.patterns:
        outcome = _match_payoff(child, target, current)
        if not outcome.ok:
            return outcome
        current = outcome.bindings
    return MatchResult(ok=True, bindings=current)


def _match_or_payoff(
    pattern: OrPattern,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    last_reason: str | None = None
    for child in pattern.patterns:
        outcome = _match_payoff(child, target, bindings)
        if outcome.ok:
            return outcome
        last_reason = outcome.mismatch_reason
    return MatchResult(
        ok=False,
        bindings=bindings,
        mismatch_reason=last_reason or "no OR branch matched",
    )


def _match_not_payoff(
    pattern: NotPattern,
    target: ProductIR,
    bindings: dict[str, Any],
) -> MatchResult:
    outcome = _match_payoff(pattern.pattern, target, bindings)
    if outcome.ok:
        return MatchResult(
            ok=False,
            bindings=bindings,
            mismatch_reason="NOT sub-pattern unexpectedly matched",
        )
    # NOT discards the sub-pattern's would-be bindings (even if they were
    # partial), matching the intuition that a negated pattern should not
    # leak captures from a branch that was supposed to fail.
    return MatchResult(ok=True, bindings=bindings)


# ---------------------------------------------------------------------------
# Field-level matching (AtomPattern / Wildcard / composites / bare values)
# ---------------------------------------------------------------------------


def _match_field(
    pattern: Any,
    observed: Any,
    bindings: dict[str, Any],
    *,
    field_label: str,
) -> MatchResult:
    return _match_field_multivalued(
        pattern,
        candidate_values=(observed,) if observed is not None else (),
        bindings=bindings,
        field_label=field_label,
        primary_value=observed,
    )


def _match_field_multivalued(
    pattern: Any,
    *,
    candidate_values: tuple[Any, ...],
    bindings: dict[str, Any],
    field_label: str,
    primary_value: Any,
) -> MatchResult:
    """Match a field pattern against a set of candidate values.

    ``primary_value`` is the value used when capturing a named wildcard.
    This lets multi-valued fields (e.g. underlying.kind backed by
    model_family + candidate_engine_families) match against any member
    while still binding the canonical model_family.
    """
    if isinstance(pattern, Wildcard):
        # Wildcards always succeed; a named wildcard captures the primary
        # value if one is available.  Absent observed values still succeed
        # (the wildcard is "don't care").
        if pattern.name is None:
            return MatchResult(ok=True, bindings=bindings)
        new_bindings = _bind(bindings, pattern.name, primary_value, field_label=field_label)
        if new_bindings is None:
            return MatchResult(
                ok=False,
                bindings=bindings,
                mismatch_reason=(
                    f"binding conflict on {pattern.name!r} while matching {field_label}"
                ),
            )
        return MatchResult(ok=True, bindings=new_bindings)

    if isinstance(pattern, AtomPattern):
        return _match_atom_against_candidates(
            pattern.value, candidate_values, bindings, field_label
        )

    if isinstance(pattern, AndPattern):
        current = bindings
        for child in pattern.patterns:
            outcome = _match_field_multivalued(
                child,
                candidate_values=candidate_values,
                bindings=current,
                field_label=field_label,
                primary_value=primary_value,
            )
            if not outcome.ok:
                return outcome
            current = outcome.bindings
        return MatchResult(ok=True, bindings=current)

    if isinstance(pattern, OrPattern):
        last_reason: str | None = None
        for child in pattern.patterns:
            outcome = _match_field_multivalued(
                child,
                candidate_values=candidate_values,
                bindings=bindings,
                field_label=field_label,
                primary_value=primary_value,
            )
            if outcome.ok:
                return outcome
            last_reason = outcome.mismatch_reason
        return MatchResult(
            ok=False,
            bindings=bindings,
            mismatch_reason=last_reason
            or f"no OR branch matched for {field_label}",
        )

    if isinstance(pattern, NotPattern):
        outcome = _match_field_multivalued(
            pattern.pattern,
            candidate_values=candidate_values,
            bindings=bindings,
            field_label=field_label,
            primary_value=primary_value,
        )
        if outcome.ok:
            return MatchResult(
                ok=False,
                bindings=bindings,
                mismatch_reason=f"NOT sub-pattern unexpectedly matched for {field_label}",
            )
        return MatchResult(ok=True, bindings=bindings)

    # Bare literal (str / int / float / bool / None).  Compare directly
    # against any observed candidate.
    return _match_atom_against_candidates(pattern, candidate_values, bindings, field_label)


def _match_atom_against_candidates(
    expected: Any,
    candidate_values: tuple[Any, ...],
    bindings: dict[str, Any],
    field_label: str,
) -> MatchResult:
    if not candidate_values:
        return MatchResult(
            ok=False,
            bindings=bindings,
            mismatch_reason=(
                f"{field_label} expected {expected!r} but target has no value"
            ),
        )
    if any(value == expected for value in candidate_values):
        return MatchResult(ok=True, bindings=bindings)
    return MatchResult(
        ok=False,
        bindings=bindings,
        mismatch_reason=(
            f"{field_label} expected {expected!r}, got {list(candidate_values)}"
        ),
    )


# ---------------------------------------------------------------------------
# Binding helpers
# ---------------------------------------------------------------------------


def _bind(
    bindings: dict[str, Any],
    name: str,
    value: Any,
    *,
    field_label: str,
) -> dict[str, Any] | None:
    """Add ``name -> value`` to ``bindings`` or return ``None`` on conflict.

    A conflict happens when ``name`` is already bound to a different value.
    We return a fresh dict on success so callers can assume bindings are
    treated immutably within a match run.
    """
    if name in bindings:
        if bindings[name] != value:
            return None
        return bindings
    new_bindings = dict(bindings)
    new_bindings[name] = value
    return new_bindings


__all__ = [
    "MatchResult",
    "evaluate_pattern",
]
