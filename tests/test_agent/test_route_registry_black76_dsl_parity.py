"""Parity test for QUA-920: analytical_black76 legacy-vs-DSL when-clause forms.

QUA-920 migrates the four existing ``conditional_primitives.when`` clauses on
``analytical_black76`` from legacy string-tag form (``{payoff_family: ...,
exercise_style: [...], model_family: [...]}``) to the DSL ``contract_pattern``
form landed by QUA-917 / QUA-918 / QUA-919.

The migration must be a pure YAML rewrite — no kernel, adapter, or note
changes.  These tests lock that by constructing two synthetic ``RouteSpec``
variants of ``analytical_black76`` that share every non-``when`` field and
differ only in the form of each ``conditional_primitives.when`` clause.  The
``default`` catch-all clause stays in legacy ``"default"`` sentinel form in
both variants — it is a fall-through marker, not a structural match, and
QUA-920 explicitly keeps it unchanged.

For each canonical fixture that exercised one of the four legacy clauses
(plus a "no clause matches → default" fixture and a ``None`` ProductIR
fixture), we assert that :func:`resolve_route_primitives`,
:func:`resolve_route_adapters`, and :func:`resolve_route_notes` return
identical tuples under both variants.  Any divergence would mean the DSL
evaluator disagrees with the legacy ``_matches_condition`` filter on a clause
the QUA-920 migration touched, and the YAML swap would silently move
dispatch.
"""

from __future__ import annotations

import pytest

from trellis.agent.codegen_guardrails import PrimitiveRef
from trellis.agent.contract_pattern import parse_contract_pattern
from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.route_registry import (
    ConditionalPrimitive,
    RouteSpec,
    resolve_route_adapters,
    resolve_route_notes,
    resolve_route_primitives,
)


# ---------------------------------------------------------------------------
# Shared primitives / adapters / notes (verbatim from routes.yaml)
# ---------------------------------------------------------------------------


_VANILLA_PRIMS: tuple[PrimitiveRef, ...] = (
    PrimitiveRef("trellis.models.black", "black76_call", "pricing_kernel"),
    PrimitiveRef("trellis.models.black", "black76_put", "pricing_kernel"),
    PrimitiveRef(
        "trellis.models.black",
        "black76_asset_or_nothing_call",
        "pricing_kernel",
        required=False,
    ),
    PrimitiveRef(
        "trellis.models.black",
        "black76_asset_or_nothing_put",
        "pricing_kernel",
        required=False,
    ),
    PrimitiveRef(
        "trellis.models.black",
        "black76_cash_or_nothing_call",
        "pricing_kernel",
        required=False,
    ),
    PrimitiveRef(
        "trellis.models.black",
        "black76_cash_or_nothing_put",
        "pricing_kernel",
        required=False,
    ),
    PrimitiveRef(
        "trellis.models.analytical",
        "terminal_vanilla_from_basis",
        "assembly_helper",
        required=False,
    ),
    PrimitiveRef("trellis.core.date_utils", "year_fraction", "time_measure"),
)
_VANILLA_ADAPTERS: tuple[str, ...] = ("map_spot_discount_and_vol_to_forward_black76",)
_VANILLA_NOTES: tuple[str, ...] = (
    "For European vanilla equity options, derive the forward from spot and discounting before calling Black-style kernels.",
    "For plain European call/put comparators, prefer direct `black76_call` / `black76_put` on the forward. Only assemble from asset-or-nothing and cash-or-nothing basis claims when the request explicitly needs the decomposition.",
    "For cash-or-nothing digital options, use the Black76 digital helpers directly instead of a vanilla call/put approximation.",
)

_BASKET_PRIMS: tuple[PrimitiveRef, ...] = (
    PrimitiveRef(
        "trellis.models.basket_option",
        "price_basket_option_analytical",
        "route_helper",
    ),
)

_SWAPTION_BERM_PRIMS: tuple[PrimitiveRef, ...] = (
    PrimitiveRef(
        "trellis.models.rate_style_swaption",
        "price_bermudan_swaption_black76_lower_bound",
        "route_helper",
    ),
)

_SWAPTION_EUR_PRIMS: tuple[PrimitiveRef, ...] = (
    PrimitiveRef(
        "trellis.models.rate_style_swaption",
        "price_swaption_black76",
        "route_helper",
    ),
)

_DEFAULT_PRIMS: tuple[PrimitiveRef, ...] = (
    PrimitiveRef("trellis.models.black", "black76_call", "pricing_kernel"),
    PrimitiveRef("trellis.models.black", "black76_put", "pricing_kernel"),
    PrimitiveRef(
        "trellis.models.black",
        "black76_asset_or_nothing_call",
        "pricing_kernel",
        required=False,
    ),
    PrimitiveRef(
        "trellis.models.black",
        "black76_asset_or_nothing_put",
        "pricing_kernel",
        required=False,
    ),
    PrimitiveRef(
        "trellis.models.black",
        "black76_cash_or_nothing_call",
        "pricing_kernel",
        required=False,
    ),
    PrimitiveRef(
        "trellis.models.black",
        "black76_cash_or_nothing_put",
        "pricing_kernel",
        required=False,
    ),
    PrimitiveRef(
        "trellis.models.analytical",
        "terminal_vanilla_from_basis",
        "assembly_helper",
        required=False,
    ),
    PrimitiveRef(
        "trellis.core.date_utils", "build_payment_timeline", "schedule_builder"
    ),
    PrimitiveRef("trellis.core.date_utils", "year_fraction", "time_measure"),
)
_DEFAULT_ADAPTERS: tuple[str, ...] = (
    "extract_forward_and_annuity_from_market_state",
)
_DEFAULT_NOTES: tuple[str, ...] = (
    "Prefer thin orchestration around existing analytical kernels.",
    "For schedule-driven analytical products such as caps and floors, prefer `build_payment_timeline(...)` so accrual fractions and model times stay explicit.",
    "Terminal vanilla payoffs should be assembled from asset-or-nothing and cash-or-nothing basis claims when the decomposition is exact.",
    "For cash-or-nothing digital options, use the Black76 digital helpers directly instead of a vanilla call/put approximation.",
)


# ---------------------------------------------------------------------------
# Shared RouteSpec scaffold: identical for both variants.
# ---------------------------------------------------------------------------


def _make_spec(
    conditional_primitives: tuple[ConditionalPrimitive, ...],
) -> RouteSpec:
    """Build an ``analytical_black76``-shaped RouteSpec around a CP tuple.

    Every field other than ``conditional_primitives`` is identical between
    the legacy and DSL variants.  The base ``primitives`` / ``adapters`` /
    ``notes`` tuples are deliberately left empty — dispatch for this route is
    fully driven by its conditional-primitives ladder (every match path plus
    ``default`` lands in a conditional clause, so the base values would never
    surface in production).
    """
    return RouteSpec(
        id="analytical_black76_dsl_parity",
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
        primitives=(),
        conditional_primitives=conditional_primitives,
        conditional_route_family=None,
        adapters=(),
        notes=(),
    )


# ---------------------------------------------------------------------------
# Legacy and DSL variants of each clause.
# ---------------------------------------------------------------------------


def _legacy_clauses() -> tuple[ConditionalPrimitive, ...]:
    """Legacy string-tag form, verbatim from pre-QUA-920 routes.yaml."""
    return (
        ConditionalPrimitive(
            when={"payoff_family": "vanilla_option"},
            primitives=_VANILLA_PRIMS,
            adapters=_VANILLA_ADAPTERS,
            notes=_VANILLA_NOTES,
        ),
        ConditionalPrimitive(
            when={
                "payoff_family": "basket_option",
                "exercise_style": ["european"],
                "model_family": ["equity_diffusion"],
            },
            primitives=_BASKET_PRIMS,
            adapters=(),
            notes=(),
        ),
        ConditionalPrimitive(
            when={
                "payoff_family": "swaption",
                "exercise_style": ["bermudan"],
            },
            primitives=_SWAPTION_BERM_PRIMS,
            adapters=(),
            notes=(),
        ),
        ConditionalPrimitive(
            when={
                "payoff_family": "swaption",
                "exercise_style": ["european"],
            },
            primitives=_SWAPTION_EUR_PRIMS,
            adapters=(),
            notes=(),
        ),
        ConditionalPrimitive(
            when="default",
            primitives=_DEFAULT_PRIMS,
            adapters=_DEFAULT_ADAPTERS,
            notes=_DEFAULT_NOTES,
        ),
    )


def _dsl_clauses() -> tuple[ConditionalPrimitive, ...]:
    """DSL ``contract_pattern`` form — the QUA-920 migration target.

    The ``default`` clause intentionally stays in legacy ``"default"`` sentinel
    form; it is a fall-through marker, not a structural match.
    """
    return (
        ConditionalPrimitive(
            when={},
            contract_pattern=parse_contract_pattern(
                {"payoff": {"kind": "vanilla_payoff"}}
            ),
            primitives=_VANILLA_PRIMS,
            adapters=_VANILLA_ADAPTERS,
            notes=_VANILLA_NOTES,
        ),
        ConditionalPrimitive(
            when={},
            contract_pattern=parse_contract_pattern(
                {
                    "payoff": {"kind": "basket_payoff"},
                    "exercise": {"style": "european"},
                    "underlying": {"kind": "equity_diffusion"},
                }
            ),
            primitives=_BASKET_PRIMS,
            adapters=(),
            notes=(),
        ),
        ConditionalPrimitive(
            when={},
            contract_pattern=parse_contract_pattern(
                {
                    "payoff": {"kind": "swaption_payoff"},
                    "exercise": {"style": "bermudan"},
                }
            ),
            primitives=_SWAPTION_BERM_PRIMS,
            adapters=(),
            notes=(),
        ),
        ConditionalPrimitive(
            when={},
            contract_pattern=parse_contract_pattern(
                {
                    "payoff": {"kind": "swaption_payoff"},
                    "exercise": {"style": "european"},
                }
            ),
            primitives=_SWAPTION_EUR_PRIMS,
            adapters=(),
            notes=(),
        ),
        # default stays legacy by design.
        ConditionalPrimitive(
            when="default",
            primitives=_DEFAULT_PRIMS,
            adapters=_DEFAULT_ADAPTERS,
            notes=_DEFAULT_NOTES,
        ),
    )


@pytest.fixture(scope="module")
def legacy_spec() -> RouteSpec:
    return _make_spec(_legacy_clauses())


@pytest.fixture(scope="module")
def dsl_spec() -> RouteSpec:
    return _make_spec(_dsl_clauses())


# ---------------------------------------------------------------------------
# Canonical ProductIR fixtures — one per clause plus a "default fall-through"
# and a "None IR" case.
# ---------------------------------------------------------------------------


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


def _cap_default_fallthrough_ir() -> ProductIR:
    """ProductIR that hits none of the four clauses → default branch.

    A cap is a schedule-driven Black76 product: payoff_family is
    ``period_rate_option_strip``, which is not covered by any of the four
    conditional clauses, so dispatch falls through to the ``default``
    catch-all.  This guarantees the default branch still resolves
    identically under both variants.
    """
    return ProductIR(
        instrument="cap",
        payoff_family="period_rate_option_strip",
        exercise_style="none",
        state_dependence="schedule_state",
        schedule_dependence=True,
        model_family="rate_style",
        candidate_engine_families=("analytical",),
    )


def _bermudan_basket_ir() -> ProductIR:
    """Bermudan-style basket — misses the basket clause (exercise != european).

    This checks the negative path: the legacy filter rejects on
    ``exercise_style`` list mismatch, and the DSL pattern rejects on the
    ``ExercisePattern.style == "european"`` atom.  Both should fall through
    to the default clause.
    """
    return ProductIR(
        instrument="basket_option_bermudan",
        payoff_family="basket_option",
        payoff_traits=("basket_payoff",),
        exercise_style="bermudan",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="equity_diffusion",
        candidate_engine_families=("analytical",),
    )


# Every fixture below is a ``(label, factory)`` tuple so parametrized test
# IDs stay human-readable in failure reports.
_FIXTURES: tuple[tuple[str, object], ...] = (
    ("vanilla_european_equity", _vanilla_ir),
    ("basket_european_equity", _basket_european_equity_ir),
    ("swaption_bermudan", _swaption_bermudan_ir),
    ("swaption_european", _swaption_european_ir),
    ("cap_default_fallthrough", _cap_default_fallthrough_ir),
    ("bermudan_basket_miss", _bermudan_basket_ir),
)


# ---------------------------------------------------------------------------
# Parity assertions
# ---------------------------------------------------------------------------


class TestBlack76LegacyVsDSLConditionalPrimitiveParity:
    """Every dispatch helper returns identical tuples across legacy and DSL."""

    @pytest.mark.parametrize(
        "label,ir_factory",
        _FIXTURES,
        ids=[label for label, _ in _FIXTURES],
    )
    def test_resolve_primitives_parity(
        self,
        legacy_spec: RouteSpec,
        dsl_spec: RouteSpec,
        label: str,
        ir_factory,
    ):
        ir = ir_factory()
        legacy_prims = resolve_route_primitives(legacy_spec, ir, binding_spec=None)
        dsl_prims = resolve_route_primitives(dsl_spec, ir, binding_spec=None)
        assert legacy_prims == dsl_prims, (
            f"primitive drift on fixture={label!r}:\n"
            f"  legacy: {[(p.module, p.symbol, p.role) for p in legacy_prims]}\n"
            f"  dsl:    {[(p.module, p.symbol, p.role) for p in dsl_prims]}"
        )

    @pytest.mark.parametrize(
        "label,ir_factory",
        _FIXTURES,
        ids=[label for label, _ in _FIXTURES],
    )
    def test_resolve_adapters_parity(
        self,
        legacy_spec: RouteSpec,
        dsl_spec: RouteSpec,
        label: str,
        ir_factory,
    ):
        ir = ir_factory()
        legacy_adapters = resolve_route_adapters(legacy_spec, ir)
        dsl_adapters = resolve_route_adapters(dsl_spec, ir)
        assert legacy_adapters == dsl_adapters, (
            f"adapter drift on fixture={label!r}:\n"
            f"  legacy: {legacy_adapters}\n"
            f"  dsl:    {dsl_adapters}"
        )

    @pytest.mark.parametrize(
        "label,ir_factory",
        _FIXTURES,
        ids=[label for label, _ in _FIXTURES],
    )
    def test_resolve_notes_parity(
        self,
        legacy_spec: RouteSpec,
        dsl_spec: RouteSpec,
        label: str,
        ir_factory,
    ):
        ir = ir_factory()
        legacy_notes = resolve_route_notes(legacy_spec, ir)
        dsl_notes = resolve_route_notes(dsl_spec, ir)
        assert legacy_notes == dsl_notes, (
            f"note drift on fixture={label!r}:\n"
            f"  legacy: {legacy_notes}\n"
            f"  dsl:    {dsl_notes}"
        )

    def test_none_product_ir_falls_through_to_default_on_both_variants(
        self, legacy_spec: RouteSpec, dsl_spec: RouteSpec
    ):
        """``None`` ProductIR must hit the shared ``default`` clause.

        The legacy path uses a string-default fallback (payoff_family=""), so
        none of the four non-default clauses match.  The DSL path short-
        circuits to ``False`` on a ``None`` product_ir inside
        ``_conditional_primitive_matches``.  Both paths then walk to the
        ``default`` sentinel and must return its primitives / adapters / notes
        verbatim.
        """
        assert resolve_route_primitives(legacy_spec, None, binding_spec=None) == (
            resolve_route_primitives(dsl_spec, None, binding_spec=None)
        )
        assert resolve_route_adapters(legacy_spec, None) == resolve_route_adapters(
            dsl_spec, None
        )
        assert resolve_route_notes(legacy_spec, None) == resolve_route_notes(
            dsl_spec, None
        )

    def test_on_disk_analytical_black76_dispatches_like_both_variants(
        self,
        legacy_spec: RouteSpec,
        dsl_spec: RouteSpec,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The on-disk ``analytical_black76`` route-level conditional_primitives
        must agree with both synthetic variants on every fixture.

        This is the load-bearing assertion: whether ``routes.yaml`` is in
        legacy form (pre-QUA-920) or DSL form (post-QUA-920), the on-disk
        route's ``conditional_primitives`` ladder must produce the same
        dispatch outputs as the two reference variants for every fixture.
        If a future edit accidentally shifts a clause's semantics, this test
        fires.

        The backend-binding catalog at
        ``trellis/agent/knowledge/canonical/backend_bindings.yaml`` has an
        independent overlay for ``analytical_black76`` that takes priority in
        :func:`resolve_route_primitives` when present. QUA-920 does not
        migrate the binding catalog (that overlay stays in legacy form), so
        we bypass the binding cache here to exercise the route-level ladder
        directly; the binding catalog is covered by its own test suite.
        """
        from trellis.agent import backend_bindings as backend_bindings_module
        from trellis.agent.route_registry import (
            find_route_by_id,
            load_route_registry,
        )

        # Force route-level dispatch by zeroing out the binding cache.
        monkeypatch.setattr(
            backend_bindings_module,
            "resolve_backend_binding_by_route_id",
            lambda *args, **kwargs: None,
        )

        registry = load_route_registry()
        on_disk_spec = find_route_by_id("analytical_black76", registry)
        assert on_disk_spec is not None, "analytical_black76 missing from registry"

        for label, ir_factory in _FIXTURES:
            ir = ir_factory()
            legacy_prims = resolve_route_primitives(legacy_spec, ir, binding_spec=None)
            dsl_prims = resolve_route_primitives(dsl_spec, ir, binding_spec=None)
            on_disk_prims = resolve_route_primitives(
                on_disk_spec, ir, binding_spec=None
            )
            assert legacy_prims == on_disk_prims == dsl_prims, (
                f"on-disk analytical_black76 primitives diverge on fixture="
                f"{label!r}:\n"
                f"  legacy:  {[(p.module, p.symbol, p.role) for p in legacy_prims]}\n"
                f"  on-disk: {[(p.module, p.symbol, p.role) for p in on_disk_prims]}\n"
                f"  dsl:     {[(p.module, p.symbol, p.role) for p in dsl_prims]}"
            )
            assert (
                resolve_route_adapters(legacy_spec, ir)
                == resolve_route_adapters(on_disk_spec, ir)
                == resolve_route_adapters(dsl_spec, ir)
            ), f"adapter drift on fixture={label!r}"
            assert (
                resolve_route_notes(legacy_spec, ir)
                == resolve_route_notes(on_disk_spec, ir)
                == resolve_route_notes(dsl_spec, ir)
            ), f"note drift on fixture={label!r}"
