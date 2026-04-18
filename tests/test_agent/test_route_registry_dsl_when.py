"""DSL-form `conditional_primitives.when` schema tests (QUA-919 / Phase 1.5.C).

These tests lock in the two-form ``when``-clause dispatch contract that the
route registry exposes after QUA-919:

1. **Legacy string-tag filter form** — a mapping of trait keys (``payoff_family``,
   ``exercise_style``, ``model_family``, ``schedule_dependence``) to literal or
   list expectations.  Dispatch goes through ``_matches_condition`` exactly as
   before QUA-919; every existing ``routes.yaml`` clause keeps this shape.
2. **DSL ``contract_pattern`` form** — a mapping with a single
   ``contract_pattern`` key whose value is a structured pattern payload
   consumable by :func:`trellis.agent.contract_pattern.parse_contract_pattern`
   and evaluable by
   :func:`trellis.agent.contract_pattern_eval.evaluate_pattern`.

The parser must decide which form a clause uses at parse time, and dispatch
must then route legacy clauses through ``_matches_condition`` and DSL clauses
through ``evaluate_pattern``.  Mixed-form clauses (both ``contract_pattern``
and legacy trait keys in the same ``when:``) are a parse error.

Forward-compat: QUA-920 migrates ``analytical_black76``'s existing four
``conditional_primitives.when`` clauses to this DSL form without changing
behaviour.  The tests below assert that the migrated shape (``payoff + exercise
+ underlying`` patterns) dispatches identically to today's string-tag form.
"""

from __future__ import annotations

import pytest

from trellis.agent.codegen_guardrails import PrimitiveRef
from trellis.agent.contract_pattern import ContractPattern, parse_contract_pattern
from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.route_registry import (
    ConditionalPrimitive,
    RouteRegistry,
    RouteSpec,
    _parse_conditional_primitives,
    _parse_route,
    load_route_registry,
    resolve_route_adapters,
    resolve_route_notes,
    resolve_route_primitives,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_route_spec(
    *,
    route_id: str,
    conditional_primitives: tuple,
    base_primitives: tuple = (),
    base_adapters: tuple = (),
    base_notes: tuple = (),
) -> RouteSpec:
    """Build a minimal synthetic RouteSpec around a conditional_primitives tuple."""
    return RouteSpec(
        id=route_id,
        engine_family="analytical",
        route_family="analytical",
        status="promoted",
        confidence=1.0,
        match_methods=("analytical",),
        match_instruments=None,
        exclude_instruments=(),
        match_exercise=None,
        exclude_exercise=(),
        match_payoff_family=None,
        match_payoff_traits=None,
        match_required_market_data=None,
        exclude_required_market_data=None,
        primitives=base_primitives,
        conditional_primitives=conditional_primitives,
        conditional_route_family=None,
        adapters=base_adapters,
        notes=base_notes,
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


class TestDSLWhenClauseParsing:
    """YAML round-trip: a DSL-form clause parses into a ConditionalPrimitive
    whose ``when`` is empty and whose ``contract_pattern`` is populated."""

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
        assert isinstance(cp, ConditionalPrimitive)
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

    def test_dsl_form_routes_through_parse_route_wrapper(self):
        """A full route payload with a DSL when-clause parses end-to-end."""
        raw_route = {
            "id": "synthetic_dsl_route",
            "engine_family": "analytical",
            "match": {"methods": ["analytical"]},
            "primitives": [],
            "conditional_primitives": [
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
            ],
        }
        spec = _parse_route(raw_route)
        assert spec.id == "synthetic_dsl_route"
        assert len(spec.conditional_primitives) == 1
        cp = spec.conditional_primitives[0]
        assert cp.when == {}
        assert isinstance(cp.contract_pattern, ContractPattern)


# ---------------------------------------------------------------------------
# 2. Dispatch (DSL-only route)
# ---------------------------------------------------------------------------


class TestDSLDispatch:
    """``resolve_route_primitives`` routes DSL-form clauses through the
    evaluator and returns the clause's primitives on a pattern match."""

    def test_dsl_clause_matches_vanilla_ir_returns_clause_primitives(self, monkeypatch):
        # Disable the backend-binding short-circuit so the conditional branch
        # is actually exercised.
        from trellis.agent import backend_bindings as backend_bindings_module

        monkeypatch.setattr(
            backend_bindings_module,
            "resolve_backend_binding_by_route_id",
            lambda *args, **kwargs: None,
        )

        expected_prim = PrimitiveRef(
            module="trellis.models.black",
            symbol="black76_call",
            role="pricing_kernel",
        )
        dsl_clause = ConditionalPrimitive(
            when={},
            contract_pattern=parse_contract_pattern(
                {"payoff": {"kind": "vanilla_payoff"}}
            ),
            primitives=(expected_prim,),
        )
        spec = _make_route_spec(
            route_id="synthetic_dsl_vanilla",
            conditional_primitives=(dsl_clause,),
        )
        resolved = resolve_route_primitives(spec, _vanilla_ir())
        assert resolved == (expected_prim,)

    def test_dsl_clause_does_not_match_non_vanilla_falls_through(self, monkeypatch):
        from trellis.agent import backend_bindings as backend_bindings_module

        monkeypatch.setattr(
            backend_bindings_module,
            "resolve_backend_binding_by_route_id",
            lambda *args, **kwargs: None,
        )

        base_prim = PrimitiveRef(
            module="trellis.models.black",
            symbol="black76_call",
            role="pricing_kernel",
        )
        clause_prim = PrimitiveRef(
            module="trellis.models.black",
            symbol="black76_basket",
            role="pricing_kernel",
        )
        dsl_clause = ConditionalPrimitive(
            when={},
            contract_pattern=parse_contract_pattern(
                {"payoff": {"kind": "basket_payoff"}}
            ),
            primitives=(clause_prim,),
        )
        spec = _make_route_spec(
            route_id="synthetic_dsl_basket",
            conditional_primitives=(dsl_clause,),
            base_primitives=(base_prim,),
        )
        # A vanilla IR should NOT match the basket DSL pattern, so resolution
        # falls through to the base primitives.
        resolved = resolve_route_primitives(spec, _vanilla_ir())
        assert resolved == (base_prim,)

    def test_dsl_clause_adapters_and_notes_propagate(self, monkeypatch):
        from trellis.agent import backend_bindings as backend_bindings_module

        monkeypatch.setattr(
            backend_bindings_module,
            "resolve_backend_binding_by_route_id",
            lambda *args, **kwargs: None,
        )

        dsl_clause = ConditionalPrimitive(
            when={},
            contract_pattern=parse_contract_pattern(
                {"payoff": {"kind": "vanilla_payoff"}}
            ),
            primitives=(
                PrimitiveRef(
                    module="trellis.models.black",
                    symbol="black76_call",
                    role="pricing_kernel",
                ),
            ),
            adapters=("vanilla_adapter",),
            notes=("use vanilla kernel",),
        )
        spec = _make_route_spec(
            route_id="synthetic_dsl_adapters",
            conditional_primitives=(dsl_clause,),
            base_adapters=("base_adapter",),
            base_notes=("base note",),
        )
        assert resolve_route_adapters(spec, _vanilla_ir()) == ("vanilla_adapter",)
        assert resolve_route_notes(spec, _vanilla_ir()) == ("use vanilla kernel",)


# ---------------------------------------------------------------------------
# 3. Mixed-mode (DSL + legacy clauses in the same route)
# ---------------------------------------------------------------------------


class TestMixedModeDispatch:
    """A single route can carry both legacy and DSL clauses; each dispatches
    correctly against its target ProductIR."""

    def test_dsl_then_legacy_default_tail(self, monkeypatch):
        from trellis.agent import backend_bindings as backend_bindings_module

        monkeypatch.setattr(
            backend_bindings_module,
            "resolve_backend_binding_by_route_id",
            lambda *args, **kwargs: None,
        )

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
        dsl_clause = ConditionalPrimitive(
            when={},
            contract_pattern=parse_contract_pattern(
                {"payoff": {"kind": "vanilla_payoff"}}
            ),
            primitives=(vanilla_prim,),
        )
        default_clause = ConditionalPrimitive(
            when="default",
            primitives=(default_prim,),
        )
        spec = _make_route_spec(
            route_id="synthetic_mixed_dsl_default",
            conditional_primitives=(dsl_clause, default_clause),
        )

        # DSL match on vanilla.
        assert resolve_route_primitives(spec, _vanilla_ir()) == (vanilla_prim,)
        # Fallthrough to default on a basket.
        assert resolve_route_primitives(spec, _basket_ir()) == (default_prim,)

    def test_legacy_then_dsl_dispatch_independently(self, monkeypatch):
        from trellis.agent import backend_bindings as backend_bindings_module

        monkeypatch.setattr(
            backend_bindings_module,
            "resolve_backend_binding_by_route_id",
            lambda *args, **kwargs: None,
        )

        legacy_prim = PrimitiveRef(
            module="trellis.models.rate_style_swaption",
            symbol="price_swaption_black76",
            role="route_helper",
        )
        dsl_prim = PrimitiveRef(
            module="trellis.models.basket_option",
            symbol="price_basket_option_analytical",
            role="route_helper",
        )

        legacy_clause = ConditionalPrimitive(
            when={
                "payoff_family": "swaption",
                "exercise_style": ["european"],
            },
            primitives=(legacy_prim,),
        )
        dsl_clause = ConditionalPrimitive(
            when={},
            contract_pattern=parse_contract_pattern(
                {
                    "payoff": {"kind": "basket_payoff"},
                    "exercise": {"style": "european"},
                    "underlying": {"kind": "equity_diffusion"},
                }
            ),
            primitives=(dsl_prim,),
        )
        spec = _make_route_spec(
            route_id="synthetic_legacy_then_dsl",
            conditional_primitives=(legacy_clause, dsl_clause),
        )

        # Legacy path hits on european swaption.
        assert resolve_route_primitives(spec, _swaption_european_ir()) == (legacy_prim,)
        # DSL path hits on basket.
        assert resolve_route_primitives(spec, _basket_ir()) == (dsl_prim,)


# ---------------------------------------------------------------------------
# 4. Error paths
# ---------------------------------------------------------------------------


class TestDSLWhenClauseErrors:
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
        # "bogus_kind" isn't a valid payoff head tag, so parse_contract_pattern
        # should raise ContractPatternParseError, which is a ValueError.
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

    def test_contract_pattern_dataclass_rejects_both_populated(self):
        """Constructing a ConditionalPrimitive with both non-empty ``when``
        and non-None ``contract_pattern`` is explicitly disallowed."""
        with pytest.raises(ValueError):
            ConditionalPrimitive(
                when={"payoff_family": "vanilla_option"},
                contract_pattern=parse_contract_pattern(
                    {"payoff": {"kind": "vanilla_payoff"}}
                ),
                primitives=(),
            )


# ---------------------------------------------------------------------------
# 5. Legacy regression — the on-disk canonical registry keeps parsing cleanly
# ---------------------------------------------------------------------------


class TestLegacyRegistryRegression:
    """Every existing ``routes.yaml`` clause is legacy form; the new parser
    must preserve that exactly."""

    def test_every_existing_clause_is_legacy_form(self):
        registry = load_route_registry()
        for route in registry.routes:
            for cp in route.conditional_primitives:
                # Legacy form: contract_pattern must be None, and the
                # `when` must be either a string sentinel or a trait dict.
                assert cp.contract_pattern is None, (
                    f"Route {route.id!r} has an unexpected DSL-form clause; "
                    "routes.yaml still owns only legacy form before QUA-920."
                )
                assert isinstance(cp.when, (dict, str)), (
                    f"Route {route.id!r} has a malformed when-clause: {cp.when!r}"
                )
