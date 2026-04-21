"""DSL-form binding-catalog ``conditional_primitives.when`` schema tests
(QUA-921 / Phase 1.5.E).

These tests lock in the two-form ``when``-clause dispatch contract that the
backend-binding catalog exposes after QUA-921:

1. **Legacy string-tag filter form** — a mapping of trait keys
   (``payoff_family``, ``payoff_traits``, ``instrument``, ``exercise_style``,
   ``model_family``, ``schedule_dependence``) to literal or list expectations.
   Dispatch goes through
   :func:`trellis.agent.backend_bindings._matches_condition` exactly as before
   QUA-921; every existing ``backend_bindings.yaml`` clause continues to parse
   via this shape.
2. **DSL ``contract_pattern`` form** — a mapping with a single
   ``contract_pattern`` key whose value is a structured pattern payload
   consumable by :func:`trellis.agent.contract_pattern.parse_contract_pattern`
   and evaluable by
   :func:`trellis.agent.contract_pattern_eval.evaluate_pattern`.

The parser must decide which form a clause uses at parse time, and
:func:`~trellis.agent.backend_bindings._resolve_binding_primitives` must then
route legacy clauses through ``_matches_condition`` and DSL clauses through
``evaluate_pattern``.  Mixed-form clauses (both ``contract_pattern`` and
legacy trait keys in the same ``when:``) are a parse error.

This file intentionally mirrors the shape of
``tests/test_agent/test_route_registry_dsl_when.py`` (QUA-919) so the two
overlays stay in lockstep.  The QUA-921 YAML migration itself is locked by
``tests/test_agent/test_backend_bindings_black76_dsl_parity.py``.
"""

from __future__ import annotations

import pytest

from trellis.agent.backend_bindings import (
    BackendBindingSpec,
    ConditionalBindingPrimitives,
    _parse_conditional_primitives,
    _resolve_binding_primitives,
    load_backend_binding_catalog,
)
from trellis.agent.codegen_guardrails import PrimitiveRef
from trellis.agent.contract_pattern import ContractPattern, parse_contract_pattern
from trellis.agent.knowledge.schema import ProductIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binding(
    *,
    route_id: str,
    conditional_primitives: tuple,
    base_primitives: tuple = (),
) -> BackendBindingSpec:
    """Build a minimal synthetic BackendBindingSpec around a CP tuple."""
    return BackendBindingSpec(
        route_id=route_id,
        engine_family="analytical",
        route_family="analytical",
        aliases=(),
        compatibility_alias_policy="operator_visible",
        primitives=base_primitives,
        conditional_primitives=conditional_primitives,
        conditional_route_family=(),
    )


def _vanilla_ir() -> ProductIR:
    return ProductIR(
        instrument="vanilla_call",
        payoff_family="vanilla_option",
        payoff_traits=("vanilla_option",),
        exercise_style="european",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="equity_diffusion",
        candidate_engine_families=("analytical",),
    )


def _basket_ir() -> ProductIR:
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


# ---------------------------------------------------------------------------
# 1. Round-trip parsing
# ---------------------------------------------------------------------------


class TestDSLBindingWhenClauseParsing:
    """YAML round-trip: a DSL-form clause parses into a
    :class:`ConditionalBindingPrimitives` whose ``when`` is empty and whose
    ``contract_pattern`` is populated."""

    def test_parses_contract_pattern_into_new_field(self):
        raw = [
            {
                "when": {
                    "contract_pattern": {
                        "payoff": {"kind": "vanilla_payoff"},
                    },
                },
                "primitives": [
                    {
                        "module": "trellis.models.black",
                        "symbol": "black76_call",
                        "role": "pricing_kernel",
                    },
                ],
            },
        ]
        parsed = _parse_conditional_primitives(raw)
        assert len(parsed) == 1
        cp = parsed[0]
        assert isinstance(cp, ConditionalBindingPrimitives)
        assert cp.when == {}
        assert isinstance(cp.contract_pattern, ContractPattern)
        assert cp.contract_pattern.payoff is not None
        assert cp.primitives[0].symbol == "black76_call"

    def test_legacy_string_tag_form_leaves_contract_pattern_none(self):
        raw = [
            {
                "when": {
                    "payoff_family": "vanilla_option",
                    "exercise_style": ["european"],
                },
                "primitives": [
                    {
                        "module": "trellis.models.black",
                        "symbol": "black76_call",
                        "role": "pricing_kernel",
                    },
                ],
            },
        ]
        parsed = _parse_conditional_primitives(raw)
        assert len(parsed) == 1
        cp = parsed[0]
        assert cp.contract_pattern is None
        assert cp.when == {
            "payoff_family": "vanilla_option",
            "exercise_style": ["european"],
        }

    def test_default_sentinel_leaves_contract_pattern_none(self):
        raw = [
            {
                "when": "default",
                "primitives": [],
            },
        ]
        parsed = _parse_conditional_primitives(raw)
        assert len(parsed) == 1
        assert parsed[0].when == "default"
        assert parsed[0].contract_pattern is None

    def test_legacy_trait_filter_preserves_payoff_traits_key(self):
        """``payoff_traits`` is a legacy-only key today (the DSL has no
        trait predicate yet).  It must survive parsing untouched."""
        raw = [
            {
                "when": {
                    "payoff_family": "basket_option",
                    "payoff_traits": ["two_asset_terminal_basket"],
                    "exercise_style": ["european"],
                    "model_family": ["equity_diffusion"],
                },
                "primitives": [],
            },
        ]
        parsed = _parse_conditional_primitives(raw)
        assert len(parsed) == 1
        assert parsed[0].contract_pattern is None
        assert parsed[0].when == {
            "payoff_family": "basket_option",
            "payoff_traits": ["two_asset_terminal_basket"],
            "exercise_style": ["european"],
            "model_family": ["equity_diffusion"],
        }


# ---------------------------------------------------------------------------
# 2. Dispatch (DSL-only binding)
# ---------------------------------------------------------------------------


class TestDSLBindingDispatch:
    """:func:`_resolve_binding_primitives` routes DSL-form clauses through the
    evaluator and returns the clause's primitives on a pattern match."""

    def test_dsl_clause_matches_vanilla_ir_returns_clause_primitives(self):
        expected_prim = PrimitiveRef(
            module="trellis.models.black",
            symbol="black76_call",
            role="pricing_kernel",
        )
        dsl_clause = ConditionalBindingPrimitives(
            when={},
            contract_pattern=parse_contract_pattern(
                {"payoff": {"kind": "vanilla_payoff"}}
            ),
            primitives=(expected_prim,),
        )
        binding = _make_binding(
            route_id="synthetic_dsl_vanilla",
            conditional_primitives=(dsl_clause,),
        )
        resolved = _resolve_binding_primitives(binding, _vanilla_ir())
        assert resolved == (expected_prim,)

    def test_dsl_clause_does_not_match_non_target_falls_through(self):
        base_prim = PrimitiveRef(
            module="trellis.models.black",
            symbol="black76_call",
            role="pricing_kernel",
        )
        clause_prim = PrimitiveRef(
            module="trellis.models.rate_style_swaption",
            symbol="price_swaption_black76",
            role="route_helper",
        )
        dsl_clause = ConditionalBindingPrimitives(
            when={},
            contract_pattern=parse_contract_pattern(
                {"payoff": {"kind": "swaption_payoff"}}
            ),
            primitives=(clause_prim,),
        )
        binding = _make_binding(
            route_id="synthetic_dsl_swaption_only",
            conditional_primitives=(dsl_clause,),
            base_primitives=(base_prim,),
        )
        # A basket IR does NOT satisfy the swaption DSL pattern, so dispatch
        # falls through to the base primitives.
        resolved = _resolve_binding_primitives(binding, _basket_ir())
        assert resolved == (base_prim,)

    def test_dsl_clause_with_exercise_style_atom(self):
        bermudan_prim = PrimitiveRef(
            module="trellis.models.rate_style_swaption",
            symbol="price_bermudan_swaption_black76_lower_bound",
            role="route_helper",
        )
        bermudan_clause = ConditionalBindingPrimitives(
            when={},
            contract_pattern=parse_contract_pattern(
                {
                    "payoff": {"kind": "swaption_payoff"},
                    "exercise": {"style": "bermudan"},
                }
            ),
            primitives=(bermudan_prim,),
        )
        binding = _make_binding(
            route_id="synthetic_dsl_bermudan_swaption",
            conditional_primitives=(bermudan_clause,),
        )
        # European swaption does not match a bermudan exercise pattern.
        assert _resolve_binding_primitives(binding, _swaption_european_ir()) == ()


# ---------------------------------------------------------------------------
# 3. Mixed-mode (DSL + legacy clauses in the same binding)
# ---------------------------------------------------------------------------


class TestMixedModeBindingDispatch:
    """A single binding can carry both legacy and DSL clauses; each dispatches
    correctly against its target ProductIR."""

    def test_dsl_then_legacy_default_tail(self):
        vanilla_prim = PrimitiveRef(
            module="trellis.models.black",
            symbol="black76_call",
            role="pricing_kernel",
        )
        default_prim = PrimitiveRef(
            module="trellis.models.black",
            symbol="black76_default",
            role="pricing_kernel",
        )
        dsl_clause = ConditionalBindingPrimitives(
            when={},
            contract_pattern=parse_contract_pattern(
                {"payoff": {"kind": "vanilla_payoff"}}
            ),
            primitives=(vanilla_prim,),
        )
        default_clause = ConditionalBindingPrimitives(
            when="default",
            primitives=(default_prim,),
        )
        binding = _make_binding(
            route_id="synthetic_mixed_dsl_default",
            conditional_primitives=(dsl_clause, default_clause),
        )

        # DSL match on vanilla.
        assert _resolve_binding_primitives(binding, _vanilla_ir()) == (vanilla_prim,)
        # Fallthrough to default on a basket.
        assert _resolve_binding_primitives(binding, _basket_ir()) == (default_prim,)

    def test_legacy_then_dsl_dispatch_independently(self):
        legacy_prim = PrimitiveRef(
            module="trellis.models.rate_cap_floor",
            symbol="price_rate_cap_floor_strip_analytical",
            role="route_helper",
        )
        dsl_prim = PrimitiveRef(
            module="trellis.models.rate_style_swaption",
            symbol="price_swaption_black76",
            role="route_helper",
        )

        legacy_clause = ConditionalBindingPrimitives(
            when={"payoff_family": "period_rate_option_strip"},
            primitives=(legacy_prim,),
        )
        dsl_clause = ConditionalBindingPrimitives(
            when={},
            contract_pattern=parse_contract_pattern(
                {
                    "payoff": {"kind": "swaption_payoff"},
                    "exercise": {"style": "european"},
                }
            ),
            primitives=(dsl_prim,),
        )
        binding = _make_binding(
            route_id="synthetic_legacy_then_dsl",
            conditional_primitives=(legacy_clause, dsl_clause),
        )

        # Legacy path hits on a rate cap.
        cap_ir = ProductIR(
            instrument="cap",
            payoff_family="period_rate_option_strip",
            exercise_style="none",
            state_dependence="schedule_state",
            schedule_dependence=True,
            model_family="rate_style",
            candidate_engine_families=("analytical",),
        )
        assert _resolve_binding_primitives(binding, cap_ir) == (legacy_prim,)
        # DSL path hits on a european swaption.
        assert _resolve_binding_primitives(binding, _swaption_european_ir()) == (
            dsl_prim,
        )


# ---------------------------------------------------------------------------
# 4. Error paths
# ---------------------------------------------------------------------------


class TestDSLBindingWhenClauseErrors:
    """Mixed-form clauses and malformed patterns raise clear parse errors."""

    def test_mixed_contract_pattern_and_legacy_keys_raises(self):
        raw = [
            {
                "when": {
                    "contract_pattern": {
                        "payoff": {"kind": "vanilla_payoff"},
                    },
                    "payoff_family": "vanilla_option",
                },
                "primitives": [],
            },
        ]
        with pytest.raises(ValueError, match="contract_pattern"):
            _parse_conditional_primitives(raw)

    def test_invalid_contract_pattern_propagates_parse_error(self):
        """``bogus_kind`` is not a recognised payoff head tag, so
        :func:`parse_contract_pattern` raises
        :class:`~trellis.agent.contract_pattern.ContractPatternParseError`
        (itself a :class:`ValueError`)."""
        raw = [
            {
                "when": {
                    "contract_pattern": {
                        "payoff": {"kind": "bogus_kind"},
                    },
                },
                "primitives": [],
            },
        ]
        with pytest.raises(ValueError):
            _parse_conditional_primitives(raw)

    def test_dataclass_rejects_both_populated_when_and_contract_pattern(self):
        """Constructing a :class:`ConditionalBindingPrimitives` with both a
        non-empty ``when`` trait mapping AND a populated ``contract_pattern``
        is explicitly disallowed."""
        with pytest.raises(ValueError):
            ConditionalBindingPrimitives(
                when={"payoff_family": "vanilla_option"},
                contract_pattern=parse_contract_pattern(
                    {"payoff": {"kind": "vanilla_payoff"}}
                ),
                primitives=(),
            )

    def test_dataclass_rejects_string_sentinel_with_contract_pattern(self):
        """The ``"default"`` string sentinel is a fall-through marker and
        must not be combined with a structural pattern."""
        with pytest.raises(ValueError):
            ConditionalBindingPrimitives(
                when="default",
                contract_pattern=parse_contract_pattern(
                    {"payoff": {"kind": "vanilla_payoff"}}
                ),
                primitives=(),
            )


# ---------------------------------------------------------------------------
# 5. Canonical catalog well-formedness regression
# ---------------------------------------------------------------------------


class TestLegacyBindingCatalogRegression:
    """The on-disk ``backend_bindings.yaml`` parses into the two-form contract
    cleanly.  Every clause must be exactly one of the two supported forms
    (legacy string-tag or DSL ``contract_pattern``).
    """

    def test_every_existing_clause_is_exactly_one_recognised_form(self):
        """Allowlist-gate: each clause is EITHER legacy form OR DSL form.

        Legacy form: ``cp.contract_pattern is None`` and ``cp.when`` is a
        string sentinel (``"default"``) or a legacy trait-filter dict.

        DSL form: ``cp.contract_pattern`` is a parsed
        :class:`~trellis.agent.contract_pattern.ContractPattern` and
        ``cp.when`` is the empty-dict placeholder.

        Any other combination is a schema violation that would break
        dispatch in ``_conditional_binding_primitive_matches``.
        """
        catalog = load_backend_binding_catalog()
        for binding in catalog.bindings:
            for idx, cp in enumerate(binding.conditional_primitives):
                is_dsl_form = cp.contract_pattern is not None

                if is_dsl_form:
                    assert isinstance(cp.contract_pattern, ContractPattern), (
                        f"Binding {binding.route_id!r} clause[{idx}] has a "
                        f"non-ContractPattern contract_pattern: "
                        f"{cp.contract_pattern!r}"
                    )
                    assert cp.when == {}, (
                        f"Binding {binding.route_id!r} clause[{idx}] is DSL "
                        f"form but `when` is not the empty-dict placeholder: "
                        f"{cp.when!r}"
                    )
                else:
                    # Legacy form: when is string sentinel or trait-filter
                    # dict.
                    assert isinstance(cp.when, (dict, str)), (
                        f"Binding {binding.route_id!r} clause[{idx}] has a "
                        f"malformed when-clause: {cp.when!r}"
                    )

    def test_analytical_black76_uses_dsl_form_for_two_swaption_clauses(self):
        """QUA-921 migration lock: the two swaption-structural
        ``analytical_black76`` clauses in ``backend_bindings.yaml`` are DSL
        form.  The other three clauses (vanilla, period_rate_option_strip,
        basket) and the ``default`` sentinel intentionally stay legacy
        (see the module docstring in
        ``test_backend_bindings_black76_dsl_parity.py`` for the per-clause
        rationale).

        This regression lock prevents an accidental revert of the migration
        (``backend_bindings.yaml`` edited back to string-tag form) from
        silently shipping: the YAML-level forms are observable, reviewable,
        and covered by the parity test.
        """
        catalog = load_backend_binding_catalog()
        binding = next(
            (b for b in catalog.bindings if b.route_id == "analytical_black76"),
            None,
        )
        assert binding is not None, "analytical_black76 missing from catalog"

        dsl_forms = [
            cp for cp in binding.conditional_primitives if cp.contract_pattern is not None
        ]
        # Two structural DSL clauses landed by QUA-921:
        #   - swaption + bermudan
        #   - swaption + european
        assert len(dsl_forms) == 2, (
            f"analytical_black76 binding should carry exactly two DSL "
            f"structural clauses after QUA-921, found {len(dsl_forms)}"
        )

        # Spot-check the two DSL patterns are the expected swaption shapes.
        dsl_kinds = {
            cp.contract_pattern.payoff.kind for cp in dsl_forms  # type: ignore[union-attr]
        }
        assert dsl_kinds == {"swaption_payoff"}, (
            f"expected both DSL clauses to use 'swaption_payoff', got "
            f"{dsl_kinds}"
        )
        dsl_styles = set()
        for cp in dsl_forms:
            exercise = cp.contract_pattern.exercise  # type: ignore[union-attr]
            if exercise is not None:
                # Bare strings and AtomPattern are both valid style payloads;
                # see ``contract_pattern.ExercisePattern`` for the dual form.
                style_value = (
                    exercise.style
                    if isinstance(exercise.style, str)
                    else getattr(exercise.style, "value", None)
                )
                if style_value is not None:
                    dsl_styles.add(style_value)
        assert dsl_styles == {"bermudan", "european"}, (
            f"expected DSL clauses to cover {{'bermudan', 'european'}} "
            f"exercise styles, got {dsl_styles}"
        )
