"""Tests for the ContractPattern evaluator against ProductIR (QUA-918 / Phase 1.5.B).

The evaluator walks a :class:`ContractPattern` against a :class:`ProductIR`
and decides whether the pattern matches, returning any captured bindings for
named wildcards.  These tests cover:

- Per-pattern-kind matching against canonical ``ProductIR`` fixtures for
  vanilla calls, basket options, payer/receiver swaptions, variance swaps,
  digitals etc.
- Wildcard binding semantics including conflict detection.
- AND / OR / NOT composition.
- Parity with the existing string-tag ``when`` dispatch in
  :mod:`trellis.agent.route_registry` for the four
  ``analytical_black76`` canonical when-clauses.
- Bare-string vs :class:`AtomPattern`-wrapped field values.
"""

from __future__ import annotations

import pytest

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
    parse_contract_pattern,
)
from trellis.agent.contract_pattern_eval import (
    MatchResult,
    evaluate_pattern,
)
from trellis.agent.knowledge.schema import ProductIR


# ---------------------------------------------------------------------------
# ProductIR fixtures (one helper each so tests stay readable)
# ---------------------------------------------------------------------------


def _vanilla_european_call_ir() -> ProductIR:
    return ProductIR(
        instrument="vanilla_call",
        payoff_family="vanilla_option",
        payoff_traits=("vanilla_option",),
        exercise_style="european",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="equity_diffusion",
        candidate_engine_families=("analytical", "monte_carlo"),
    )


def _basket_european_equity_ir() -> ProductIR:
    return ProductIR(
        instrument="basket_option",
        payoff_family="basket_option",
        payoff_traits=("basket_payoff",),
        exercise_style="european",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="equity_diffusion",
        candidate_engine_families=("analytical",),
    )


def _swaption_bermudan_ir() -> ProductIR:
    return ProductIR(
        instrument="swaption",
        payoff_family="swaption",
        exercise_style="bermudan",
        state_dependence="schedule_state",
        schedule_dependence=True,
        model_family="rate_style",
        candidate_engine_families=("analytical",),
    )


def _swaption_european_ir() -> ProductIR:
    return ProductIR(
        instrument="swaption",
        payoff_family="swaption",
        exercise_style="european",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="rate_style",
        candidate_engine_families=("analytical",),
    )


def _variance_swap_ir() -> ProductIR:
    return ProductIR(
        instrument="variance_swap",
        payoff_family="variance_swap",
        exercise_style="none",
        state_dependence="path_dependent",
        schedule_dependence=True,
        model_family="equity_diffusion",
        candidate_engine_families=("analytical",),
    )


def _cash_or_nothing_digital_ir() -> ProductIR:
    return ProductIR(
        instrument="digital_option",
        payoff_family="digital_option",
        payoff_traits=("digital_payoff",),
        exercise_style="european",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="equity_diffusion",
        candidate_engine_families=("analytical",),
    )


def _american_vanilla_call_ir() -> ProductIR:
    return ProductIR(
        instrument="vanilla_call_american",
        payoff_family="vanilla_option",
        exercise_style="american",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="equity_diffusion",
        candidate_engine_families=("pde", "monte_carlo"),
    )


# ---------------------------------------------------------------------------
# Leaf pattern semantics
# ---------------------------------------------------------------------------


class TestLeafPatterns:
    def test_empty_contract_pattern_matches_anything(self):
        pattern = ContractPattern()
        result = evaluate_pattern(pattern, _vanilla_european_call_ir())
        assert result.ok is True
        assert result.bindings == {}

    def test_exercise_style_matches_bare_string(self):
        pattern = ContractPattern(exercise=ExercisePattern(style="european"))
        result = evaluate_pattern(pattern, _vanilla_european_call_ir())
        assert result.ok is True

    def test_exercise_style_mismatch_fails(self):
        pattern = ContractPattern(exercise=ExercisePattern(style="bermudan"))
        result = evaluate_pattern(pattern, _vanilla_european_call_ir())
        assert result.ok is False
        assert result.mismatch_reason is not None
        assert "exercise_style" in result.mismatch_reason

    def test_exercise_style_wildcard_matches_anything(self):
        pattern = ContractPattern(exercise=ExercisePattern(style=Wildcard()))
        for ir in [
            _vanilla_european_call_ir(),
            _american_vanilla_call_ir(),
            _swaption_bermudan_ir(),
        ]:
            assert evaluate_pattern(pattern, ir).ok is True

    def test_observation_matches_terminal(self):
        pattern = ContractPattern(observation=ObservationPattern(kind="terminal"))
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True

    def test_observation_mismatch_on_path_dependent(self):
        pattern = ContractPattern(observation=ObservationPattern(kind="terminal"))
        result = evaluate_pattern(pattern, _variance_swap_ir())
        assert result.ok is False

    def test_underlying_kind_matches_model_family(self):
        pattern = ContractPattern(
            underlying=UnderlyingPattern(kind="equity_diffusion"),
        )
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True

    def test_underlying_kind_mismatch_fails(self):
        pattern = ContractPattern(
            underlying=UnderlyingPattern(kind="equity_diffusion"),
        )
        result = evaluate_pattern(pattern, _swaption_bermudan_ir())
        assert result.ok is False


# ---------------------------------------------------------------------------
# Payoff pattern semantics
# ---------------------------------------------------------------------------


class TestPayoffPatterns:
    def test_instrument_tag_swaption_matches_swaption_family(self):
        pattern = ContractPattern(payoff=PayoffPattern(kind="swaption_payoff"))
        assert evaluate_pattern(pattern, _swaption_european_ir()).ok is True
        assert evaluate_pattern(pattern, _swaption_bermudan_ir()).ok is True

    def test_instrument_tag_swaption_does_not_match_vanilla(self):
        pattern = ContractPattern(payoff=PayoffPattern(kind="swaption_payoff"))
        result = evaluate_pattern(pattern, _vanilla_european_call_ir())
        assert result.ok is False
        assert result.mismatch_reason is not None

    def test_instrument_tag_basket_matches_basket_family(self):
        pattern = ContractPattern(payoff=PayoffPattern(kind="basket_payoff"))
        assert evaluate_pattern(pattern, _basket_european_equity_ir()).ok is True

    def test_instrument_tag_variance_matches_variance_swap(self):
        pattern = ContractPattern(payoff=PayoffPattern(kind="variance_payoff"))
        assert evaluate_pattern(pattern, _variance_swap_ir()).ok is True

    def test_instrument_tag_vanilla_matches_vanilla_family(self):
        pattern = ContractPattern(payoff=PayoffPattern(kind="vanilla_payoff"))
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True

    def test_instrument_tag_digital_matches_digital_family(self):
        pattern = ContractPattern(payoff=PayoffPattern(kind="digital_payoff"))
        assert evaluate_pattern(pattern, _cash_or_nothing_digital_ir()).ok is True

    def test_structural_vanilla_payoff_matches_vanilla_option(self):
        # max(sub(spot, strike), 0) is the canonical vanilla European call shape.
        pattern = ContractPattern(
            payoff=PayoffPattern(
                kind="max",
                args=(
                    PayoffPattern(
                        kind="sub",
                        args=(SpotPattern(), StrikePattern()),
                    ),
                    ConstantPattern(value=0.0),
                ),
            )
        )
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True

    def test_structural_vanilla_payoff_does_not_match_swaption(self):
        pattern = ContractPattern(
            payoff=PayoffPattern(
                kind="max",
                args=(
                    PayoffPattern(
                        kind="sub",
                        args=(SpotPattern(), StrikePattern()),
                    ),
                    ConstantPattern(value=0.0),
                ),
            )
        )
        result = evaluate_pattern(pattern, _swaption_bermudan_ir())
        assert result.ok is False


# ---------------------------------------------------------------------------
# Wildcard binding semantics
# ---------------------------------------------------------------------------


class TestWildcardBindings:
    def test_named_wildcard_captures_value(self):
        pattern = ContractPattern(
            exercise=ExercisePattern(style=Wildcard(name="style")),
        )
        result = evaluate_pattern(pattern, _vanilla_european_call_ir())
        assert result.ok is True
        assert result.bindings == {"style": "european"}

    def test_anonymous_wildcard_captures_nothing(self):
        pattern = ContractPattern(
            exercise=ExercisePattern(style=Wildcard()),
        )
        result = evaluate_pattern(pattern, _vanilla_european_call_ir())
        assert result.ok is True
        assert result.bindings == {}

    def test_same_name_bound_twice_to_same_value_succeeds(self):
        # Bind :X to both exercise.style and underlying.kind; this succeeds
        # only when the IR has matching values for both.
        pattern = ContractPattern(
            exercise=ExercisePattern(style=Wildcard(name="X")),
            underlying=UnderlyingPattern(kind=Wildcard(name="X")),
        )
        # Build an IR where exercise_style == model_family by construction.
        ir = ProductIR(
            instrument="synthetic",
            payoff_family="vanilla_option",
            exercise_style="european",
            model_family="european",
        )
        result = evaluate_pattern(pattern, ir)
        assert result.ok is True
        assert result.bindings == {"X": "european"}

    def test_same_name_bound_twice_to_different_values_fails(self):
        pattern = ContractPattern(
            exercise=ExercisePattern(style=Wildcard(name="X")),
            underlying=UnderlyingPattern(kind=Wildcard(name="X")),
        )
        # vanilla European call has exercise_style="european" and
        # model_family="equity_diffusion".
        result = evaluate_pattern(pattern, _vanilla_european_call_ir())
        assert result.ok is False
        assert result.mismatch_reason is not None
        assert "conflict" in result.mismatch_reason.lower() or "X" in result.mismatch_reason


# ---------------------------------------------------------------------------
# AND / OR / NOT composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_or_over_exercise_style(self):
        pattern = ContractPattern(
            exercise=ExercisePattern(
                style=OrPattern(
                    patterns=(
                        AtomPattern(value="european"),
                        AtomPattern(value="bermudan"),
                    )
                )
            )
        )
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True
        assert evaluate_pattern(pattern, _swaption_bermudan_ir()).ok is True
        assert evaluate_pattern(pattern, _american_vanilla_call_ir()).ok is False

    def test_not_over_exercise_style(self):
        pattern = ContractPattern(
            exercise=ExercisePattern(
                style=NotPattern(pattern=AtomPattern(value="american"))
            )
        )
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True
        assert evaluate_pattern(pattern, _american_vanilla_call_ir()).ok is False

    def test_and_over_multiple_top_level_fields(self):
        # Top-level ContractPattern already composes fields with AND; this
        # test confirms field-level AND for completeness.
        pattern = ContractPattern(
            exercise=ExercisePattern(
                style=AndPattern(
                    patterns=(
                        NotPattern(pattern=AtomPattern(value="american")),
                        NotPattern(pattern=AtomPattern(value="bermudan")),
                    )
                )
            )
        )
        # european passes both NOTs, bermudan fails the second.
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True
        assert evaluate_pattern(pattern, _swaption_bermudan_ir()).ok is False

    def test_or_bindings_use_first_matching_branch(self):
        # If the first branch matches, its bindings should surface.
        pattern = ContractPattern(
            exercise=ExercisePattern(
                style=OrPattern(
                    patterns=(
                        Wildcard(name="first"),
                        Wildcard(name="second"),
                    )
                )
            )
        )
        result = evaluate_pattern(pattern, _vanilla_european_call_ir())
        assert result.ok is True
        # The first branch matched, so only "first" should be bound.
        assert "first" in result.bindings
        assert "second" not in result.bindings


# ---------------------------------------------------------------------------
# Bare-string vs AtomPattern forms
# ---------------------------------------------------------------------------


class TestFieldValueForms:
    def test_bare_string_field_matches_product_ir_field(self):
        pattern = ContractPattern(underlying=UnderlyingPattern(kind="equity_diffusion"))
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True

    def test_atompattern_wrapped_field_matches_product_ir_field(self):
        pattern = ContractPattern(
            underlying=UnderlyingPattern(kind=AtomPattern(value="equity_diffusion")),
        )
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True

    def test_bare_string_and_atompattern_give_same_result(self):
        ir = _vanilla_european_call_ir()
        bare = ContractPattern(exercise=ExercisePattern(style="european"))
        wrapped = ContractPattern(
            exercise=ExercisePattern(style=AtomPattern(value="european")),
        )
        assert evaluate_pattern(bare, ir).ok == evaluate_pattern(wrapped, ir).ok is True


# ---------------------------------------------------------------------------
# Parity with string-tag dispatch for analytical_black76
# ---------------------------------------------------------------------------


def _vanilla_when_clause() -> ContractPattern:
    """``analytical_black76`` when-clause 1: payoff_family == vanilla_option."""
    return parse_contract_pattern(
        {"payoff": {"kind": "vanilla_payoff"}}
    )


def _basket_when_clause() -> ContractPattern:
    """``analytical_black76`` when-clause 2: basket + european + equity."""
    return parse_contract_pattern(
        {
            "payoff": {"kind": "basket_payoff"},
            "exercise": {"style": "european"},
            "underlying": {"kind": "equity_diffusion"},
        }
    )


def _swaption_bermudan_when_clause() -> ContractPattern:
    """``analytical_black76`` when-clause 3: swaption + bermudan."""
    return parse_contract_pattern(
        {
            "payoff": {"kind": "swaption_payoff"},
            "exercise": {"style": "bermudan"},
        }
    )


def _swaption_european_when_clause() -> ContractPattern:
    """``analytical_black76`` when-clause 4: swaption + european."""
    return parse_contract_pattern(
        {
            "payoff": {"kind": "swaption_payoff"},
            "exercise": {"style": "european"},
        }
    )


class TestAnalyticalBlack76Parity:
    """Parity with the four existing analytical_black76 when-clauses.

    For each clause we confirm that fixtures which hit the clause under
    the existing string-tag filter (:func:`_matches_condition`) also
    match the equivalent :class:`ContractPattern`, and that fixtures
    that don't hit the clause don't match.
    """

    def test_vanilla_when_clause_matches_vanilla_fixture(self):
        pattern = _vanilla_when_clause()
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True

    def test_vanilla_when_clause_rejects_swaption(self):
        pattern = _vanilla_when_clause()
        assert evaluate_pattern(pattern, _swaption_european_ir()).ok is False

    def test_basket_when_clause_matches_basket_fixture(self):
        pattern = _basket_when_clause()
        assert evaluate_pattern(pattern, _basket_european_equity_ir()).ok is True

    def test_basket_when_clause_rejects_vanilla(self):
        pattern = _basket_when_clause()
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is False

    def test_basket_when_clause_rejects_bermudan_basket(self):
        pattern = _basket_when_clause()
        bermudan_basket = ProductIR(
            instrument="basket_option_bermudan",
            payoff_family="basket_option",
            exercise_style="bermudan",
            model_family="equity_diffusion",
        )
        assert evaluate_pattern(pattern, bermudan_basket).ok is False

    def test_swaption_bermudan_matches_bermudan_fixture(self):
        pattern = _swaption_bermudan_when_clause()
        assert evaluate_pattern(pattern, _swaption_bermudan_ir()).ok is True

    def test_swaption_bermudan_rejects_european_swaption(self):
        pattern = _swaption_bermudan_when_clause()
        assert evaluate_pattern(pattern, _swaption_european_ir()).ok is False

    def test_swaption_european_matches_european_fixture(self):
        pattern = _swaption_european_when_clause()
        assert evaluate_pattern(pattern, _swaption_european_ir()).ok is True

    def test_swaption_european_rejects_bermudan(self):
        pattern = _swaption_european_when_clause()
        assert evaluate_pattern(pattern, _swaption_bermudan_ir()).ok is False

    def test_swaption_clauses_reject_non_swaptions(self):
        for pattern in [
            _swaption_bermudan_when_clause(),
            _swaption_european_when_clause(),
        ]:
            for ir in [
                _vanilla_european_call_ir(),
                _basket_european_equity_ir(),
                _variance_swap_ir(),
                _cash_or_nothing_digital_ir(),
            ]:
                assert evaluate_pattern(pattern, ir).ok is False

    @pytest.mark.parametrize(
        "when_clause,when_key,ir_factory,expect_match",
        [
            # Vanilla hits (1) and misses (2..4).
            ("vanilla", 1, _vanilla_european_call_ir, True),
            ("basket", 2, _vanilla_european_call_ir, False),
            ("swaption_bermudan", 3, _vanilla_european_call_ir, False),
            ("swaption_european", 4, _vanilla_european_call_ir, False),
            # Basket hits (2) and misses everything else.
            ("vanilla", 1, _basket_european_equity_ir, False),
            ("basket", 2, _basket_european_equity_ir, True),
            ("swaption_bermudan", 3, _basket_european_equity_ir, False),
            ("swaption_european", 4, _basket_european_equity_ir, False),
            # Bermudan swaption hits (3).
            ("vanilla", 1, _swaption_bermudan_ir, False),
            ("basket", 2, _swaption_bermudan_ir, False),
            ("swaption_bermudan", 3, _swaption_bermudan_ir, True),
            ("swaption_european", 4, _swaption_bermudan_ir, False),
            # European swaption hits (4).
            ("vanilla", 1, _swaption_european_ir, False),
            ("basket", 2, _swaption_european_ir, False),
            ("swaption_bermudan", 3, _swaption_european_ir, False),
            ("swaption_european", 4, _swaption_european_ir, True),
        ],
    )
    def test_full_parity_matrix(self, when_clause, when_key, ir_factory, expect_match):
        clause_patterns = {
            "vanilla": _vanilla_when_clause(),
            "basket": _basket_when_clause(),
            "swaption_bermudan": _swaption_bermudan_when_clause(),
            "swaption_european": _swaption_european_when_clause(),
        }
        pattern = clause_patterns[when_clause]
        ir = ir_factory()
        assert evaluate_pattern(pattern, ir).ok is expect_match


# ---------------------------------------------------------------------------
# Schedule pattern (follow-up placeholder coverage)
# ---------------------------------------------------------------------------


class TestSchedulePattern:
    def test_schedule_wildcard_frequency_matches_trivially(self):
        # QUA-917 only exposes ``SchedulePattern.frequency``; ProductIR has
        # no direct frequency field yet, so wildcard / absent schedule
        # matches trivially when the rest of the pattern matches.
        pattern = ContractPattern(
            exercise=ExercisePattern(
                style="bermudan",
                schedule=SchedulePattern(frequency=Wildcard()),
            ),
        )
        assert evaluate_pattern(pattern, _swaption_bermudan_ir()).ok is True


# ---------------------------------------------------------------------------
# MatchResult shape
# ---------------------------------------------------------------------------


class TestMatchResult:
    def test_default_match_result_is_ok_false_with_empty_bindings(self):
        result = MatchResult(ok=False)
        assert result.ok is False
        assert result.bindings == {}
        assert result.mismatch_reason is None

    def test_match_result_is_frozen(self):
        result = MatchResult(ok=True)
        with pytest.raises(Exception):
            result.ok = False  # type: ignore[misc]

    def test_match_result_exposes_bindings_and_reason(self):
        result = MatchResult(
            ok=True,
            bindings={"K": 100.0},
            mismatch_reason=None,
        )
        assert result.bindings == {"K": 100.0}


# ---------------------------------------------------------------------------
# Type safety at the public API
# ---------------------------------------------------------------------------


class TestApiTypeSafety:
    def test_non_contract_pattern_input_raises(self):
        with pytest.raises(TypeError):
            evaluate_pattern("not a pattern", _vanilla_european_call_ir())  # type: ignore[arg-type]

    def test_non_product_ir_target_raises(self):
        pattern = ContractPattern(exercise=ExercisePattern(style="european"))
        with pytest.raises(TypeError):
            evaluate_pattern(pattern, "not a ProductIR")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Miscellaneous leaf / composite smoke tests for coverage
# ---------------------------------------------------------------------------


class TestAdditionalCoverage:
    def test_top_level_or_payoff_matches_any_alternative(self):
        pattern = ContractPattern(
            payoff=OrPattern(
                patterns=(
                    PayoffPattern(kind="swaption_payoff"),
                    PayoffPattern(kind="vanilla_payoff"),
                )
            )
        )
        # Both alternatives are admissible; IR with vanilla triggers the
        # second branch.
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True
        # Swaption IR triggers the first branch.
        assert evaluate_pattern(pattern, _swaption_european_ir()).ok is True
        # Variance IR hits neither alternative.
        assert evaluate_pattern(pattern, _variance_swap_ir()).ok is False

    def test_top_level_not_payoff_inverts_match(self):
        pattern = ContractPattern(
            payoff=NotPattern(pattern=PayoffPattern(kind="swaption_payoff"))
        )
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True
        assert evaluate_pattern(pattern, _swaption_european_ir()).ok is False

    def test_top_level_and_payoff_requires_all_children(self):
        pattern = ContractPattern(
            payoff=AndPattern(
                patterns=(
                    PayoffPattern(kind="vanilla_payoff"),
                    NotPattern(pattern=PayoffPattern(kind="swaption_payoff")),
                )
            )
        )
        assert evaluate_pattern(pattern, _vanilla_european_call_ir()).ok is True
        assert evaluate_pattern(pattern, _swaption_european_ir()).ok is False

    def test_observation_wildcard_binds_primary_state_dependence(self):
        pattern = ContractPattern(
            observation=ObservationPattern(kind=Wildcard(name="obs")),
        )
        result = evaluate_pattern(pattern, _vanilla_european_call_ir())
        assert result.ok is True
        # The primary value is the raw ``state_dependence``, not the alias.
        assert result.bindings == {"obs": "terminal_markov"}
