"""Tests for the ContractPattern AST and parser (QUA-917 / Phase 1.5.A).

The pattern substrate is pure types + parsing in this slice.  No evaluator
runs against ProductIR here (QUA-918) and no route schema reads it yet
(QUA-919).  These tests therefore focus on:

- canonical AST construction for the four existing ``analytical_black76``
  when-clauses, confirming the AST is expressive enough to encode them,
- round-trip parse -> serialize -> parse fidelity,
- wildcard semantics (anonymous vs named capture),
- AST-level composition via ``AndPattern`` / ``OrPattern`` / ``NotPattern``,
- clear error behaviour on malformed input (not a silently wrong AST).
"""

from __future__ import annotations

import pytest

from trellis.agent.contract_pattern import (
    AndPattern,
    AtomPattern,
    ConstantPattern,
    ContractPattern,
    ContractPatternParseError,
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
    dump_contract_pattern,
    parse_contract_pattern,
    parse_payoff_pattern,
)


# ---------------------------------------------------------------------------
# Structural smoke tests
# ---------------------------------------------------------------------------


class TestPatternConstruction:
    def test_wildcard_anonymous_carries_no_binding(self):
        node = Wildcard()
        assert node.name is None

    def test_wildcard_named_captures_binding_name(self):
        node = Wildcard(name="K")
        assert node.name == "K"

    def test_contract_pattern_missing_fields_default_to_none(self):
        pattern = ContractPattern()
        assert pattern.payoff is None
        assert pattern.exercise is None
        assert pattern.observation is None
        assert pattern.underlying is None

    def test_payoff_pattern_children_are_immutable_tuples(self):
        payoff = PayoffPattern(
            kind="max",
            args=(
                PayoffPattern(
                    kind="sub",
                    args=(SpotPattern(), StrikePattern()),
                ),
                ConstantPattern(value=0.0),
            ),
        )
        # Frozen dataclasses forbid reassignment.
        with pytest.raises(Exception):
            payoff.kind = "min"  # type: ignore[misc]
        with pytest.raises(Exception):
            payoff.args = ()  # type: ignore[misc]


class TestAtomPatternHelpers:
    def test_spot_pattern_underlier_defaults_to_wildcard(self):
        spot = SpotPattern()
        assert isinstance(spot.underlier, Wildcard)

    def test_strike_pattern_value_defaults_to_wildcard(self):
        strike = StrikePattern()
        assert isinstance(strike.value, Wildcard)

    def test_constant_pattern_carries_literal(self):
        node = ConstantPattern(value=0.0)
        assert node.value == 0.0

    def test_constant_pattern_accepts_wildcard_literal(self):
        node = ConstantPattern(value=Wildcard(name="zero"))
        assert isinstance(node.value, Wildcard)
        assert node.value.name == "zero"


# ---------------------------------------------------------------------------
# analytical_black76 canonical when-clauses (fixture parity)
# ---------------------------------------------------------------------------


class TestBlack76CanonicalPatterns:
    def test_vanilla_option_payoff_encoded_as_max_sub_spot_strike(self):
        payload = {
            "payoff": {
                "kind": "max",
                "args": [
                    {
                        "kind": "sub",
                        "args": [
                            {"kind": "spot", "underlier": "_"},
                            {"kind": "strike", "value": "_"},
                        ],
                    },
                    {"kind": "constant", "value": 0},
                ],
            },
            "exercise": {"style": "european"},
        }
        pattern = parse_contract_pattern(payload)

        assert isinstance(pattern, ContractPattern)
        assert pattern.exercise is not None
        assert pattern.exercise.style == "european"

        assert isinstance(pattern.payoff, PayoffPattern)
        assert pattern.payoff.kind == "max"
        outer_args = pattern.payoff.args
        assert len(outer_args) == 2
        sub_node, zero_node = outer_args
        assert isinstance(sub_node, PayoffPattern) and sub_node.kind == "sub"
        assert isinstance(zero_node, ConstantPattern) and zero_node.value == 0.0

        lhs, rhs = sub_node.args
        assert isinstance(lhs, SpotPattern)
        assert isinstance(lhs.underlier, Wildcard) and lhs.underlier.name is None
        assert isinstance(rhs, StrikePattern)
        assert isinstance(rhs.value, Wildcard) and rhs.value.name is None

    def test_basket_european_equity_diffusion(self):
        payload = {
            "payoff": {"kind": "basket_payoff"},
            "exercise": {"style": "european"},
            "underlying": {"kind": "linear_basket", "dynamics": "equity_diffusion"},
        }
        pattern = parse_contract_pattern(payload)

        assert pattern.payoff is not None and pattern.payoff.kind == "basket_payoff"
        assert pattern.exercise is not None and pattern.exercise.style == "european"
        assert pattern.underlying is not None
        assert pattern.underlying.kind == "linear_basket"
        assert pattern.underlying.dynamics == "equity_diffusion"

    def test_swaption_bermudan(self):
        payload = {
            "payoff": {"kind": "swaption_payoff"},
            "exercise": {"style": "bermudan"},
        }
        pattern = parse_contract_pattern(payload)

        assert pattern.payoff is not None and pattern.payoff.kind == "swaption_payoff"
        assert pattern.exercise is not None and pattern.exercise.style == "bermudan"

    def test_swaption_european(self):
        payload = {
            "payoff": {"kind": "swaption_payoff"},
            "exercise": {"style": "european"},
        }
        pattern = parse_contract_pattern(payload)

        assert pattern.payoff is not None and pattern.payoff.kind == "swaption_payoff"
        assert pattern.exercise is not None and pattern.exercise.style == "european"


# ---------------------------------------------------------------------------
# Wildcard parsing semantics
# ---------------------------------------------------------------------------


class TestWildcardParsing:
    def test_bare_underscore_parses_as_anonymous_wildcard(self):
        payload = {
            "payoff": {"kind": "spot", "underlier": "_"},
        }
        pattern = parse_contract_pattern(payload)
        assert isinstance(pattern.payoff, SpotPattern)
        assert isinstance(pattern.payoff.underlier, Wildcard)
        assert pattern.payoff.underlier.name is None

    def test_underscore_prefixed_identifier_is_named_capture(self):
        payload = {
            "payoff": {"kind": "strike", "value": "_K"},
        }
        pattern = parse_contract_pattern(payload)
        assert isinstance(pattern.payoff, StrikePattern)
        assert isinstance(pattern.payoff.value, Wildcard)
        assert pattern.payoff.value.name == "K"

    def test_named_wildcard_via_dict_form(self):
        payload = {
            "payoff": {
                "kind": "strike",
                "value": {"kind": "wildcard", "name": "strike_binding"},
            },
        }
        pattern = parse_contract_pattern(payload)
        assert isinstance(pattern.payoff, StrikePattern)
        assert isinstance(pattern.payoff.value, Wildcard)
        assert pattern.payoff.value.name == "strike_binding"

    def test_wildcard_with_explicit_null_name_is_anonymous(self):
        payload = {
            "payoff": {
                "kind": "spot",
                "underlier": {"kind": "wildcard"},
            },
        }
        pattern = parse_contract_pattern(payload)
        assert isinstance(pattern.payoff, SpotPattern)
        assert isinstance(pattern.payoff.underlier, Wildcard)
        assert pattern.payoff.underlier.name is None


# ---------------------------------------------------------------------------
# AND / OR / NOT composition (AST-level; no evaluation yet)
# ---------------------------------------------------------------------------


class TestCompositionPatterns:
    def test_and_pattern_parses_with_multiple_children(self):
        payload = {
            "payoff": {
                "kind": "and",
                "patterns": [
                    {"kind": "spot", "underlier": "_"},
                    {"kind": "strike", "value": "_"},
                ],
            },
        }
        pattern = parse_contract_pattern(payload)
        assert isinstance(pattern.payoff, AndPattern)
        assert len(pattern.payoff.patterns) == 2
        assert isinstance(pattern.payoff.patterns[0], SpotPattern)
        assert isinstance(pattern.payoff.patterns[1], StrikePattern)

    def test_or_pattern_parses_with_multiple_children(self):
        payload = {
            "exercise": {
                "style": {
                    "kind": "or",
                    "patterns": [
                        {"kind": "literal", "value": "european"},
                        {"kind": "literal", "value": "bermudan"},
                    ],
                },
            },
        }
        pattern = parse_contract_pattern(payload)
        assert isinstance(pattern.exercise, ExercisePattern)
        assert isinstance(pattern.exercise.style, OrPattern)
        assert len(pattern.exercise.style.patterns) == 2

    def test_not_pattern_parses_with_single_child(self):
        payload = {
            "exercise": {
                "style": {
                    "kind": "not",
                    "pattern": {"kind": "literal", "value": "american"},
                },
            },
        }
        pattern = parse_contract_pattern(payload)
        assert isinstance(pattern.exercise, ExercisePattern)
        assert isinstance(pattern.exercise.style, NotPattern)

    def test_and_pattern_requires_at_least_one_child(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern(
                {"payoff": {"kind": "and", "patterns": []}}
            )

    def test_or_pattern_requires_at_least_one_child(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern(
                {"payoff": {"kind": "or", "patterns": []}}
            )

    def test_not_pattern_requires_a_child(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern({"payoff": {"kind": "not"}})


# ---------------------------------------------------------------------------
# Round-trip (parse -> dump -> parse)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize(
        "payload",
        [
            # 1. simple contract pattern with only an exercise
            {"exercise": {"style": "european"}},
            # 2. canonical vanilla option payoff
            {
                "payoff": {
                    "kind": "max",
                    "args": [
                        {
                            "kind": "sub",
                            "args": [
                                {"kind": "spot", "underlier": "_"},
                                {"kind": "strike", "value": "_K"},
                            ],
                        },
                        {"kind": "constant", "value": 0},
                    ],
                },
                "exercise": {"style": "european"},
            },
            # 3. basket option clause
            {
                "payoff": {"kind": "basket_payoff"},
                "exercise": {"style": "european"},
                "underlying": {
                    "kind": "linear_basket",
                    "dynamics": "equity_diffusion",
                },
            },
            # 4. OR-composed exercise style
            {
                "exercise": {
                    "style": {
                        "kind": "or",
                        "patterns": [
                            {"kind": "literal", "value": "european"},
                            {"kind": "literal", "value": "bermudan"},
                        ],
                    },
                },
            },
            # 5. NOT-composed exercise style
            {
                "exercise": {
                    "style": {
                        "kind": "not",
                        "pattern": {"kind": "literal", "value": "american"},
                    },
                },
            },
            # 6. observation pattern
            {"observation": {"kind": "terminal"}},
            # 7. nested AND on the payoff (expressive stress)
            {
                "payoff": {
                    "kind": "and",
                    "patterns": [
                        {"kind": "spot", "underlier": "_"},
                        {"kind": "strike", "value": "_"},
                    ],
                },
            },
        ],
    )
    def test_parse_dump_parse_idempotent(self, payload):
        first = parse_contract_pattern(payload)
        dumped = dump_contract_pattern(first)
        second = parse_contract_pattern(dumped)
        assert first == second


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_payload_must_be_mapping(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern([])  # type: ignore[arg-type]

    def test_unknown_payoff_kind_raises_with_clear_message(self):
        with pytest.raises(ContractPatternParseError) as excinfo:
            parse_contract_pattern({"payoff": {"kind": "__bogus__"}})
        assert "__bogus__" in str(excinfo.value)

    def test_payoff_must_be_mapping_or_composite(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern({"payoff": 42})  # type: ignore[dict-item]

    def test_exercise_must_be_mapping(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern({"exercise": "european"})  # type: ignore[dict-item]

    def test_unknown_top_level_field_raises(self):
        with pytest.raises(ContractPatternParseError) as excinfo:
            parse_contract_pattern({"payoff": {"kind": "max", "args": []}, "wat": 1})
        assert "wat" in str(excinfo.value)

    def test_max_payoff_requires_args(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern({"payoff": {"kind": "max"}})

    def test_sub_payoff_requires_two_args(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern(
                {
                    "payoff": {
                        "kind": "sub",
                        "args": [{"kind": "spot", "underlier": "_"}],
                    }
                }
            )

    def test_constant_requires_value_key(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern({"payoff": {"kind": "constant"}})

    def test_wildcard_name_must_be_string(self):
        with pytest.raises(ContractPatternParseError):
            parse_contract_pattern(
                {"payoff": {"kind": "spot", "underlier": {"kind": "wildcard", "name": 5}}}
            )


class TestSchedulePattern:
    def test_schedule_pattern_parses_frequency_only(self):
        payload = {
            "exercise": {
                "style": "bermudan",
                "schedule": {"frequency": "annual"},
            },
        }
        pattern = parse_contract_pattern(payload)
        assert pattern.exercise is not None
        assert isinstance(pattern.exercise.schedule, SchedulePattern)
        assert pattern.exercise.schedule.frequency == "annual"

    def test_schedule_pattern_accepts_wildcard_frequency(self):
        payload = {
            "exercise": {
                "style": "bermudan",
                "schedule": {"frequency": "_"},
            },
        }
        pattern = parse_contract_pattern(payload)
        assert pattern.exercise is not None
        assert isinstance(pattern.exercise.schedule, SchedulePattern)
        assert isinstance(pattern.exercise.schedule.frequency, Wildcard)


class TestTopLevelParser:
    def test_parser_accepts_nested_under_contract_pattern_key(self):
        payload = {
            "contract_pattern": {
                "exercise": {"style": "european"},
            }
        }
        pattern = parse_contract_pattern(payload)
        assert pattern.exercise is not None
        assert pattern.exercise.style == "european"

    def test_parse_payoff_pattern_shortcut(self):
        pattern = parse_payoff_pattern(
            {
                "kind": "max",
                "args": [
                    {
                        "kind": "sub",
                        "args": [
                            {"kind": "spot", "underlier": "_"},
                            {"kind": "strike", "value": "_"},
                        ],
                    },
                    {"kind": "constant", "value": 0},
                ],
            }
        )
        assert isinstance(pattern, PayoffPattern)
        assert pattern.kind == "max"
        assert len(pattern.args) == 2

    def test_atom_pattern_literal(self):
        # Underlying kind supplied as a plain literal string.
        payload = {"underlying": {"kind": "equity_spot"}}
        pattern = parse_contract_pattern(payload)
        assert pattern.underlying is not None
        assert pattern.underlying.kind == "equity_spot"

    def test_atom_pattern_wrapped_atompattern(self):
        # Atoms can also be written out via the {"kind": "atom", "value": ...}
        # dict form for uniformity with the composite vocabulary.
        payload = {
            "underlying": {
                "kind": {"kind": "literal", "value": "equity_spot"},
            },
        }
        pattern = parse_contract_pattern(payload)
        assert pattern.underlying is not None
        assert isinstance(pattern.underlying.kind, AtomPattern)
        assert pattern.underlying.kind.value == "equity_spot"
