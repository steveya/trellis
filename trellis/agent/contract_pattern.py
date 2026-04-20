"""Contract pattern AST and parser (QUA-917 / Phase 1.5.A).

This module introduces a pure pattern-matching substrate that mirrors the
concrete contract vocabulary from :mod:`trellis.agent.dsl_algebra` one
level up: it represents *patterns* over concrete contract terms rather
than the concrete terms themselves.  A ``ContractPattern`` is therefore a
shape that can match many different :class:`ProductIR` / contract
instances, with explicit anonymous or named wildcards where structural
freedom is desired.

The slice is intentionally narrow:

- types + frozen-dataclass AST only, so Phase 1 registry dispatch and
  Phase 3 ``@solves_pattern`` decorators can share the vocabulary,
- structured YAML parser (the form routes.yaml will eventually write),
- round-trippable ``dump_contract_pattern`` so parse/serialize parity can
  be tested today,
- clear :class:`ContractPatternParseError` on malformed inputs rather than
  silently building a wrong AST.

Not in scope for QUA-917 (tracked separately):

- evaluator against ``ProductIR`` — QUA-918,
- conditional_primitives YAML schema integration — QUA-919,
- string-expression parser (e.g. ``Payoff(Max(Sub(Spot(_), Strike(_))...``);
  the structured YAML form is the priority surface because it is the form
  ``routes.yaml`` will actually use.  Adding a string-form parser on top
  of this AST is a follow-up.

The AST reuses naming from :mod:`dsl_algebra` where that vocabulary
already exists (``Spot``, ``Strike``, ``Max``, ``Sub`` etc.) rather than
inventing parallel types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ContractPatternParseError(ValueError):
    """Raised when a pattern payload does not describe a valid pattern.

    The parser always surfaces a structural failure instead of producing a
    silently-wrong AST.  Callers can catch this to report clean user-facing
    diagnostics.
    """


# ---------------------------------------------------------------------------
# Leaf patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Wildcard:
    """Anonymous or named wildcard leaf.

    ``name=None`` matches anything without capturing it.  A non-empty
    ``name`` captures the matched sub-term under that binding identifier
    when an evaluator runs (QUA-918).  At the AST level this slice only
    stores the identifier.
    """

    name: str | None = None


@dataclass(frozen=True)
class AtomPattern:
    """Match one concrete literal value.

    Used for string tags (e.g. an exercise style of ``"european"``) and
    numeric literals (e.g. a constant strike at ``0.0``) when the pattern
    author wants to pin down a specific value rather than wildcard it.
    """

    value: Any


# A field-level pattern is any of: a concrete literal atom, a wildcard, or
# an AST-level composition that eventually resolves to one of those.
FieldPattern = Union["Wildcard", "AtomPattern", "AndPattern", "OrPattern", "NotPattern"]


# ---------------------------------------------------------------------------
# Payoff-level patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayoffPattern:
    """Shape of a payoff expression.

    ``kind`` is the head tag (``max``, ``sub``, ``mul``, ``indicator``,
    ``scaled``, ``basket_payoff``, ``swaption_payoff`` etc.) mirroring the
    vocabulary of :mod:`dsl_algebra` and the pricing stack; ``args`` is the
    tuple of sub-patterns to recurse into.  Leaf payoff shapes use
    :class:`SpotPattern`, :class:`StrikePattern`, or :class:`ConstantPattern`.
    """

    kind: str
    args: tuple["Pattern", ...] = ()


@dataclass(frozen=True)
class SpotPattern:
    """Payoff leaf ``Spot(underlier)``."""

    underlier: FieldPattern = Wildcard()


@dataclass(frozen=True)
class StrikePattern:
    """Payoff leaf ``Strike(value)``."""

    value: FieldPattern = Wildcard()


@dataclass(frozen=True)
class ConstantPattern:
    """Payoff leaf ``Constant(value)``.

    ``value`` may be a concrete numeric literal or a :class:`Wildcard` when
    the pattern author wants to allow any constant.
    """

    value: Any


# ---------------------------------------------------------------------------
# Field-level helper patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulePattern:
    """Shape of a schedule the exercise / observation references."""

    frequency: FieldPattern | None = None


@dataclass(frozen=True)
class ExercisePattern:
    """Match the exercise style (and optionally the exercise schedule)."""

    style: FieldPattern
    schedule: SchedulePattern | None = None


@dataclass(frozen=True)
class UnderlyingPattern:
    """Match the underlying kind (and optionally its dynamics / process)."""

    kind: FieldPattern
    dynamics: FieldPattern | None = None


@dataclass(frozen=True)
class ObservationPattern:
    """Match the observation kind (terminal / schedule / path-dependent)."""

    kind: FieldPattern


# ---------------------------------------------------------------------------
# Composite patterns (AND / OR / NOT)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AndPattern:
    """Conjunction — all sub-patterns must match."""

    patterns: tuple["Pattern", ...]


@dataclass(frozen=True)
class OrPattern:
    """Disjunction — any sub-pattern may match."""

    patterns: tuple["Pattern", ...]


@dataclass(frozen=True)
class NotPattern:
    """Negation — the sub-pattern must not match."""

    pattern: "Pattern"


# ---------------------------------------------------------------------------
# Top-level contract pattern
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractPattern:
    """Full pattern over a contract / ProductIR.

    Each field is optional.  A missing field means "don't care about this
    aspect," mirroring how ``ProductIR`` fields default in the knowledge
    schema.  This lets a pattern say "payoff must be max(sub(spot, strike),
    0) and I don't care about the exercise schedule" without having to
    enumerate the whole space.
    """

    payoff: "PatternPayoffLike | None" = None
    exercise: ExercisePattern | None = None
    observation: ObservationPattern | None = None
    underlying: UnderlyingPattern | None = None


# Alias for the payoff-slot union to keep the top-level dataclass readable.
PatternPayoffLike = Union[
    PayoffPattern,
    SpotPattern,
    StrikePattern,
    ConstantPattern,
    AndPattern,
    OrPattern,
    NotPattern,
]


# Broadest alias used by composite traversal.
Pattern = Union[
    PayoffPattern,
    SpotPattern,
    StrikePattern,
    ConstantPattern,
    ExercisePattern,
    UnderlyingPattern,
    ObservationPattern,
    SchedulePattern,
    AndPattern,
    OrPattern,
    NotPattern,
    Wildcard,
    AtomPattern,
    ContractPattern,
]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


_TOP_LEVEL_FIELDS = ("payoff", "exercise", "observation", "underlying")

_PAYOFF_COMPOSITE_KINDS = {"and", "or", "not"}

# Known payoff head tags.  The parser is strict: an unknown kind is an
# error rather than being quietly preserved, because this slice is
# explicitly about expressiveness parity with routes.yaml when-clauses.
_PAYOFF_KINDS = {
    "max",
    "min",
    "sum",
    "sub",
    "mul",
    "scaled",
    "indicator",
    "forward",
    "swap_rate",
    "annuity",
    "linear_basket",
    "arithmetic_mean",
    "variance_observable",
    "curve_quote",
    "surface_quote",
    "constant",
    "spot",
    "strike",
    "wildcard",
    "literal",
    "and",
    "or",
    "not",
    # Instrument-level payoff head tags used by routes.yaml when-clauses.
    # These are treated as opaque identifiers on a PayoffPattern with no
    # args; the AST stays expressive enough to encode the existing
    # analytical_black76 `payoff_family: swaption / basket_option` clauses.
    "swaption_payoff",
    "basket_payoff",
    "variance_payoff",
    "digital_payoff",
    "barrier_payoff",
    "vanilla_payoff",
    "rate_payoff",
    "lookback_payoff",
    "asian_payoff",
    "cliquet_payoff",
    "chooser_payoff",
    "compound_payoff",
}


def parse_contract_pattern(payload: Any) -> ContractPattern:
    """Parse a structured YAML-style payload into a :class:`ContractPattern`.

    The expected top-level shape is::

        {
          "payoff": <payoff pattern>,
          "exercise": {"style": <field>, "schedule": {...}?},
          "observation": {"kind": <field>},
          "underlying": {"kind": <field>, "dynamics": <field>?},
        }

    For ergonomic embedding inside larger documents, a top-level
    ``{"contract_pattern": {...}}`` wrapper is also accepted.

    Raises :class:`ContractPatternParseError` on any structural problem.
    """
    if not isinstance(payload, Mapping):
        raise ContractPatternParseError(
            f"contract pattern payload must be a mapping, got {type(payload).__name__}"
        )

    # Accept the wrapped form routes.yaml will likely use.
    if set(payload.keys()) == {"contract_pattern"}:
        return parse_contract_pattern(payload["contract_pattern"])

    unknown = [key for key in payload.keys() if key not in _TOP_LEVEL_FIELDS]
    if unknown:
        raise ContractPatternParseError(
            "unknown contract pattern field(s): "
            + ", ".join(repr(key) for key in unknown)
        )

    payoff = _parse_payoff_slot(payload.get("payoff"))
    exercise = _parse_exercise(payload.get("exercise"))
    observation = _parse_observation(payload.get("observation"))
    underlying = _parse_underlying(payload.get("underlying"))

    return ContractPattern(
        payoff=payoff,
        exercise=exercise,
        observation=observation,
        underlying=underlying,
    )


def parse_payoff_pattern(payload: Any) -> Pattern:
    """Shortcut: parse a single payoff expression in isolation."""
    if payload is None:
        raise ContractPatternParseError("payoff payload must not be None")
    return _parse_payoff_node(payload)


# ---------------------------------------------------------------------------
# Per-slot parsers
# ---------------------------------------------------------------------------


def _parse_payoff_slot(node: Any) -> PatternPayoffLike | None:
    if node is None:
        return None
    if not isinstance(node, Mapping):
        raise ContractPatternParseError(
            f"'payoff' must be a mapping, got {type(node).__name__}"
        )
    result = _parse_payoff_node(node)
    if isinstance(
        result,
        (PayoffPattern, SpotPattern, StrikePattern, ConstantPattern, AndPattern, OrPattern, NotPattern),
    ):
        return result
    raise ContractPatternParseError(
        "'payoff' must resolve to a payoff pattern, not a leaf wildcard/literal"
    )


def _parse_exercise(node: Any) -> ExercisePattern | None:
    if node is None:
        return None
    if not isinstance(node, Mapping):
        raise ContractPatternParseError(
            f"'exercise' must be a mapping, got {type(node).__name__}"
        )
    unknown = [key for key in node.keys() if key not in {"style", "schedule"}]
    if unknown:
        raise ContractPatternParseError(
            "unknown 'exercise' field(s): " + ", ".join(repr(k) for k in unknown)
        )
    if "style" not in node:
        raise ContractPatternParseError("'exercise' requires a 'style' field")
    style = _parse_field_pattern(node["style"])
    schedule_payload = node.get("schedule")
    schedule = _parse_schedule(schedule_payload) if schedule_payload is not None else None
    return ExercisePattern(style=style, schedule=schedule)


def _parse_observation(node: Any) -> ObservationPattern | None:
    if node is None:
        return None
    if not isinstance(node, Mapping):
        raise ContractPatternParseError(
            f"'observation' must be a mapping, got {type(node).__name__}"
        )
    unknown = [key for key in node.keys() if key not in {"kind"}]
    if unknown:
        raise ContractPatternParseError(
            "unknown 'observation' field(s): " + ", ".join(repr(k) for k in unknown)
        )
    if "kind" not in node:
        raise ContractPatternParseError("'observation' requires a 'kind' field")
    return ObservationPattern(kind=_parse_field_pattern(node["kind"]))


def _parse_underlying(node: Any) -> UnderlyingPattern | None:
    if node is None:
        return None
    if not isinstance(node, Mapping):
        raise ContractPatternParseError(
            f"'underlying' must be a mapping, got {type(node).__name__}"
        )
    unknown = [key for key in node.keys() if key not in {"kind", "dynamics"}]
    if unknown:
        raise ContractPatternParseError(
            "unknown 'underlying' field(s): " + ", ".join(repr(k) for k in unknown)
        )
    if "kind" not in node:
        raise ContractPatternParseError("'underlying' requires a 'kind' field")
    kind = _parse_field_pattern(node["kind"])
    dynamics_payload = node.get("dynamics")
    dynamics = _parse_field_pattern(dynamics_payload) if dynamics_payload is not None else None
    return UnderlyingPattern(kind=kind, dynamics=dynamics)


def _parse_schedule(node: Any) -> SchedulePattern:
    if not isinstance(node, Mapping):
        raise ContractPatternParseError(
            f"'schedule' must be a mapping, got {type(node).__name__}"
        )
    unknown = [key for key in node.keys() if key not in {"frequency"}]
    if unknown:
        raise ContractPatternParseError(
            "unknown 'schedule' field(s): " + ", ".join(repr(k) for k in unknown)
        )
    frequency = None
    if "frequency" in node:
        frequency = _parse_field_pattern(node["frequency"])
    return SchedulePattern(frequency=frequency)


# ---------------------------------------------------------------------------
# Field-level parsing (atoms, wildcards, composites)
# ---------------------------------------------------------------------------


def _parse_field_pattern(payload: Any) -> FieldPattern:
    """Parse a field-level pattern (atom, wildcard, or composite)."""
    if isinstance(payload, Wildcard):
        return payload
    if isinstance(payload, AtomPattern):
        return payload
    if isinstance(payload, (AndPattern, OrPattern, NotPattern)):
        return payload

    if isinstance(payload, str):
        return _parse_string_shorthand(payload, atom_factory=lambda v: v)
    if isinstance(payload, (int, float, bool)) or payload is None:
        return AtomPattern(value=payload)

    if isinstance(payload, Mapping):
        return _parse_mapping_as_field(payload)

    raise ContractPatternParseError(
        f"field pattern must be str/number/mapping, got {type(payload).__name__}"
    )


def _parse_string_shorthand(raw: str, *, atom_factory) -> FieldPattern:
    """Interpret a bare string: wildcard shorthand vs literal atom."""
    if raw == "_":
        return Wildcard()
    if raw.startswith("_") and len(raw) > 1:
        name = raw[1:]
        return Wildcard(name=name)
    # Plain literal — wrap as atom so evaluator can distinguish from
    # wildcard when comparing.  The atom_factory lets callers decide
    # whether to unwrap on the top-level fast path (e.g. exercise.style
    # stored as a raw string for ergonomic comparisons).
    return atom_factory(raw) if callable(atom_factory) else AtomPattern(value=raw)


def _parse_mapping_as_field(payload: Mapping[str, Any]) -> FieldPattern:
    """Parse a mapping payload into a field-level pattern."""
    if "kind" not in payload:
        raise ContractPatternParseError(
            "field pattern mapping is missing required 'kind' key"
        )
    kind = payload["kind"]
    if not isinstance(kind, str):
        raise ContractPatternParseError(
            f"field pattern 'kind' must be a string, got {type(kind).__name__}"
        )

    if kind == "wildcard":
        return _parse_wildcard_mapping(payload)
    if kind == "literal":
        if "value" not in payload:
            raise ContractPatternParseError("'literal' requires a 'value' field")
        return AtomPattern(value=payload["value"])
    if kind == "and":
        return AndPattern(
            patterns=_parse_composite_children(
                payload, multi_key="patterns", kind_label="and", min_children=1
            )
        )
    if kind == "or":
        return OrPattern(
            patterns=_parse_composite_children(
                payload, multi_key="patterns", kind_label="or", min_children=1
            )
        )
    if kind == "not":
        if "pattern" not in payload:
            raise ContractPatternParseError("'not' requires a 'pattern' field")
        child = _parse_field_pattern(payload["pattern"])
        return NotPattern(pattern=child)

    raise ContractPatternParseError(
        f"unknown field pattern kind {kind!r}"
    )


def _parse_wildcard_mapping(payload: Mapping[str, Any]) -> Wildcard:
    unknown = [key for key in payload.keys() if key not in {"kind", "name"}]
    if unknown:
        raise ContractPatternParseError(
            "unknown 'wildcard' field(s): " + ", ".join(repr(k) for k in unknown)
        )
    name = payload.get("name")
    if name is None:
        return Wildcard()
    if not isinstance(name, str):
        raise ContractPatternParseError(
            f"wildcard 'name' must be a string, got {type(name).__name__}"
        )
    return Wildcard(name=name)


def _parse_composite_children(
    payload: Mapping[str, Any],
    *,
    multi_key: str,
    kind_label: str,
    min_children: int,
) -> tuple[Pattern, ...]:
    if multi_key not in payload:
        raise ContractPatternParseError(
            f"{kind_label!r} requires a {multi_key!r} field"
        )
    raw_children = payload[multi_key]
    if not isinstance(raw_children, Sequence) or isinstance(raw_children, (str, bytes)):
        raise ContractPatternParseError(
            f"{kind_label!r} {multi_key!r} must be a sequence"
        )
    if len(raw_children) < min_children:
        raise ContractPatternParseError(
            f"{kind_label!r} requires at least {min_children} child pattern(s)"
        )
    children = tuple(_parse_payoff_node(child) for child in raw_children)
    return children


# ---------------------------------------------------------------------------
# Payoff-node parsing
# ---------------------------------------------------------------------------


def _parse_payoff_node(payload: Any) -> Pattern:
    """Parse a payoff-context node.

    This is the workhorse used for the ``payoff`` slot, for ``args`` of
    head-tagged payoffs, and for the children of ``and`` / ``or`` / ``not``
    composites.
    """
    if isinstance(payload, (PayoffPattern, SpotPattern, StrikePattern, ConstantPattern)):
        return payload
    if isinstance(payload, (AndPattern, OrPattern, NotPattern)):
        return payload
    if isinstance(payload, (Wildcard, AtomPattern)):
        return payload

    if isinstance(payload, str):
        # In payoff contexts a bare string is either an underscore-wildcard
        # or an opaque literal atom.  Literals show up mainly as observation
        # tags or exercise-style leaves reused inside OR/NOT composites.
        if payload == "_":
            return Wildcard()
        if payload.startswith("_") and len(payload) > 1:
            return Wildcard(name=payload[1:])
        return AtomPattern(value=payload)

    if isinstance(payload, (int, float, bool)) or payload is None:
        return AtomPattern(value=payload)

    if not isinstance(payload, Mapping):
        raise ContractPatternParseError(
            f"payoff node must be str/number/mapping, got {type(payload).__name__}"
        )
    if "kind" not in payload:
        raise ContractPatternParseError(
            "payoff node mapping is missing required 'kind' key"
        )

    kind = payload["kind"]
    if not isinstance(kind, str):
        raise ContractPatternParseError(
            f"payoff node 'kind' must be a string, got {type(kind).__name__}"
        )

    if kind == "wildcard":
        return _parse_wildcard_mapping(payload)
    if kind == "literal":
        if "value" not in payload:
            raise ContractPatternParseError("'literal' requires a 'value' field")
        return AtomPattern(value=payload["value"])
    if kind == "and":
        return AndPattern(
            patterns=_parse_composite_children(
                payload, multi_key="patterns", kind_label="and", min_children=1
            )
        )
    if kind == "or":
        return OrPattern(
            patterns=_parse_composite_children(
                payload, multi_key="patterns", kind_label="or", min_children=1
            )
        )
    if kind == "not":
        if "pattern" not in payload:
            raise ContractPatternParseError("'not' requires a 'pattern' field")
        return NotPattern(pattern=_parse_payoff_node(payload["pattern"]))
    if kind == "spot":
        underlier_payload = payload.get("underlier", "_")
        return SpotPattern(underlier=_parse_field_pattern(underlier_payload))
    if kind == "strike":
        value_payload = payload.get("value", "_")
        return StrikePattern(value=_parse_field_pattern(value_payload))
    if kind == "constant":
        if "value" not in payload:
            raise ContractPatternParseError("'constant' requires a 'value' field")
        value = payload["value"]
        if isinstance(value, Mapping) or (isinstance(value, str) and value.startswith("_")):
            wrapped = _parse_field_pattern(value)
            return ConstantPattern(value=wrapped)
        if isinstance(value, (int, float)):
            return ConstantPattern(value=float(value))
        if isinstance(value, str):
            return ConstantPattern(value=value)
        raise ContractPatternParseError(
            f"'constant' value must be a number, string, or wildcard mapping, "
            f"got {type(value).__name__}"
        )

    if kind in _PAYOFF_KINDS:
        # A head-tagged PayoffPattern.  ``args`` is required for head tags
        # with known arity (``max``, ``min``, ``sum``, ``sub``, ``mul``,
        # ``scaled``, ``indicator``); instrument-level payoff family tags
        # (``basket_payoff`` etc.) may legitimately omit ``args``.
        args_payload = payload.get("args")
        if args_payload is None:
            _require_args_optional(kind)
            return PayoffPattern(kind=kind, args=())
        if not isinstance(args_payload, Sequence) or isinstance(args_payload, (str, bytes)):
            raise ContractPatternParseError(
                f"payoff node 'args' must be a sequence for kind {kind!r}"
            )
        args = tuple(_parse_payoff_node(child) for child in args_payload)
        _validate_arity(kind, args)
        return PayoffPattern(kind=kind, args=args)

    raise ContractPatternParseError(f"unknown payoff node kind {kind!r}")


_ARITY_TABLE: dict[str, int | None] = {
    # Arity-exact head tags.
    "sub": 2,
    "mul": 2,
    "scaled": 2,
    "indicator": 1,
    "forward": 2,
    "swap_rate": 2,
    "annuity": 2,
    "arithmetic_mean": 2,
    "variance_observable": 2,
    "curve_quote": 3,
    "surface_quote": 3,
    # Variadic (>=1) head tags — we accept any non-zero arity.
    "max": None,
    "min": None,
    "sum": None,
    "linear_basket": None,
    # Instrument-level payoff tags have no structural sub-args requirement.
    "swaption_payoff": 0,
    "basket_payoff": 0,
    "variance_payoff": 0,
    "digital_payoff": 0,
    "barrier_payoff": 0,
    "vanilla_payoff": 0,
    "rate_payoff": 0,
    "lookback_payoff": 0,
    "asian_payoff": 0,
    "cliquet_payoff": 0,
    "chooser_payoff": 0,
    "compound_payoff": 0,
}


def _validate_arity(kind: str, args: tuple[Pattern, ...]) -> None:
    expected = _ARITY_TABLE.get(kind)
    if expected is None:
        # Variadic — require at least one child.
        if kind in {"max", "min", "sum", "linear_basket"} and len(args) < 1:
            raise ContractPatternParseError(
                f"payoff node {kind!r} requires at least one 'args' entry"
            )
        return
    if expected == 0:
        # Instrument-level tags accept either zero args or none; nothing to check.
        return
    if len(args) != expected:
        raise ContractPatternParseError(
            f"payoff node {kind!r} requires exactly {expected} 'args' entries, got {len(args)}"
        )


def _require_args_optional(kind: str) -> None:
    """Ensure missing ``args`` is only legal for instrument-level head tags."""
    expected = _ARITY_TABLE.get(kind)
    if expected is None:
        raise ContractPatternParseError(
            f"payoff node {kind!r} requires an 'args' list"
        )
    if expected == 0:
        return
    raise ContractPatternParseError(
        f"payoff node {kind!r} requires an 'args' list"
    )


# ---------------------------------------------------------------------------
# Serializer (dump)
# ---------------------------------------------------------------------------


def dump_contract_pattern(pattern: ContractPattern) -> dict[str, Any]:
    """Serialize a :class:`ContractPattern` back to a YAML-friendly mapping.

    The result round-trips through :func:`parse_contract_pattern` — this is
    how QUA-917's parse/dump parity is tested.
    """
    if not isinstance(pattern, ContractPattern):
        raise TypeError(
            f"dump_contract_pattern expects ContractPattern, got {type(pattern).__name__}"
        )
    out: dict[str, Any] = {}
    if pattern.payoff is not None:
        out["payoff"] = _dump_pattern(pattern.payoff)
    if pattern.exercise is not None:
        out["exercise"] = _dump_exercise(pattern.exercise)
    if pattern.observation is not None:
        out["observation"] = _dump_observation(pattern.observation)
    if pattern.underlying is not None:
        out["underlying"] = _dump_underlying(pattern.underlying)
    return out


def _dump_exercise(node: ExercisePattern) -> dict[str, Any]:
    out: dict[str, Any] = {"style": _dump_field(node.style)}
    if node.schedule is not None:
        out["schedule"] = _dump_schedule(node.schedule)
    return out


def _dump_observation(node: ObservationPattern) -> dict[str, Any]:
    return {"kind": _dump_field(node.kind)}


def _dump_underlying(node: UnderlyingPattern) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": _dump_field(node.kind)}
    if node.dynamics is not None:
        out["dynamics"] = _dump_field(node.dynamics)
    return out


def _dump_schedule(node: SchedulePattern) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if node.frequency is not None:
        out["frequency"] = _dump_field(node.frequency)
    return out


def _dump_field(node: Any) -> Any:
    """Serialize a field-level pattern back to its YAML-friendly form."""
    if isinstance(node, Wildcard):
        return {"kind": "wildcard", "name": node.name} if node.name else "_"
    if isinstance(node, AtomPattern):
        # Always serialize explicit atoms via the {"kind": "literal", ...}
        # form so the parser does not re-interpret a bare underscore-prefixed
        # string as wildcard shorthand on reparse.
        return {"kind": "literal", "value": node.value}
    if isinstance(node, AndPattern):
        return {
            "kind": "and",
            "patterns": [_dump_pattern(child) for child in node.patterns],
        }
    if isinstance(node, OrPattern):
        return {
            "kind": "or",
            "patterns": [_dump_pattern(child) for child in node.patterns],
        }
    if isinstance(node, NotPattern):
        return {"kind": "not", "pattern": _dump_pattern(node.pattern)}
    if isinstance(node, str):
        # A field sometimes carries a bare string (e.g. exercise style)
        # when the caller constructed it directly with a literal rather than
        # an AtomPattern.  Serialize it as-is so downstream ergonomic
        # comparisons (``pattern.exercise.style == "european"``) continue to
        # work, and so that underscore-prefixed strings round-trip back
        # through the wildcard shorthand.
        return node
    if isinstance(node, (int, float, bool)) or node is None:
        return {"kind": "literal", "value": node}
    raise TypeError(
        f"cannot serialize field-level pattern of type {type(node).__name__}"
    )


def _dump_pattern(node: Any) -> Any:
    """Serialize a payoff-context pattern."""
    if isinstance(node, PayoffPattern):
        out: dict[str, Any] = {"kind": node.kind}
        if node.args:
            out["args"] = [_dump_pattern(child) for child in node.args]
        return out
    if isinstance(node, SpotPattern):
        return {"kind": "spot", "underlier": _dump_field(node.underlier)}
    if isinstance(node, StrikePattern):
        return {"kind": "strike", "value": _dump_field(node.value)}
    if isinstance(node, ConstantPattern):
        value = node.value
        if isinstance(value, Wildcard):
            return {"kind": "constant", "value": _dump_field(value)}
        return {"kind": "constant", "value": value}
    if isinstance(node, (AndPattern, OrPattern, NotPattern)):
        return _dump_field(node)
    if isinstance(node, (Wildcard, AtomPattern)):
        return _dump_field(node)
    raise TypeError(
        f"cannot serialize payoff-context pattern of type {type(node).__name__}"
    )


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "AndPattern",
    "AtomPattern",
    "ConstantPattern",
    "ContractPattern",
    "ContractPatternParseError",
    "ExercisePattern",
    "FieldPattern",
    "NotPattern",
    "ObservationPattern",
    "OrPattern",
    "Pattern",
    "PatternPayoffLike",
    "PayoffPattern",
    "SchedulePattern",
    "SpotPattern",
    "StrikePattern",
    "UnderlyingPattern",
    "Wildcard",
    "dump_contract_pattern",
    "parse_contract_pattern",
    "parse_payoff_pattern",
]
