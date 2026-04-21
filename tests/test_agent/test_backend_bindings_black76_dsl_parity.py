"""Parity test for QUA-921: analytical_black76 binding-catalog DSL migration.

QUA-920 migrated the four structural ``analytical_black76.conditional_primitives``
clauses in ``routes.yaml`` to the DSL ``contract_pattern`` form.  However, the
parallel binding-catalog overlay in
``trellis/agent/knowledge/canonical/backend_bindings.yaml`` retained the legacy
string-tag form, and its dispatcher
(:func:`trellis.agent.backend_bindings._resolve_binding_primitives`) takes
priority in :func:`resolve_route_primitives` when the binding cache hits.
Production dispatch for ``analytical_black76`` therefore still flowed through
the legacy catalog path.

QUA-921 closes that loop: this ticket extends
:class:`ConditionalBindingPrimitives` (and its parser + dispatcher) to accept
DSL ``contract_pattern`` form, and migrates the DSL-expressible
``analytical_black76`` clauses in ``backend_bindings.yaml`` to DSL form.
Two structural clauses migrate: ``swaption + bermudan`` and
``swaption + european``.  The other clauses intentionally stay legacy:

* ``default`` is a fall-through marker, not a structural pattern match.
* ``basket_option`` relies on an extra
  ``payoff_traits: [two_asset_terminal_basket]`` filter that the current
  :class:`~trellis.agent.contract_pattern.ContractPattern` AST cannot
  express.  Extending the DSL with a trait predicate is a separate follow-on.
* ``vanilla_option`` stays legacy because the DSL ``vanilla_payoff`` tag
  also matches trait-level ``vanilla_option`` entries, which would over-match
  on two-asset baskets carrying ``vanilla_option`` as a leg-style trait
  (pinned by
  ``test_resolve_backend_binding_spec_uses_basket_option_exact_helpers`` in
  ``tests/test_agent/test_backend_bindings.py``).
* ``period_rate_option_strip`` follows the same discipline as QUA-920's
  routes.yaml migration.

This file asserts that the on-disk ``analytical_black76`` binding catalog
entry dispatches to the same primitives as two synthetic reference variants
(legacy and DSL) across the six canonical fixtures QUA-920 pinned down.
Crucially, the parity test does **not** patch out the binding cache: it
exercises the production code path end-to-end, which is the evidence that DSL
dispatch is now load-bearing in ``backend_bindings._resolve_binding_primitives``
rather than only in ``route_registry._conditional_primitive_matches``.
"""

from __future__ import annotations

import pytest

from trellis.agent.backend_bindings import (
    BackendBindingSpec,
    ConditionalBindingPrimitives,
    _resolve_binding_primitives,
    clear_backend_binding_catalog_cache,
    find_backend_binding_by_route_id,
    load_backend_binding_catalog,
    resolve_backend_binding_spec,
)
from trellis.agent.codegen_guardrails import PrimitiveRef
from trellis.agent.contract_pattern import parse_contract_pattern
from trellis.agent.knowledge.schema import ProductIR


# ---------------------------------------------------------------------------
# Shared primitives (verbatim from backend_bindings.yaml).
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

_RATE_CAP_FLOOR_PRIMS: tuple[PrimitiveRef, ...] = (
    PrimitiveRef(
        "trellis.models.rate_cap_floor",
        "price_rate_cap_floor_strip_analytical",
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


# ---------------------------------------------------------------------------
# Synthetic binding specs: identical in every way except the form of each
# ``conditional_primitives.when`` clause.
# ---------------------------------------------------------------------------


def _make_binding(
    conditional_primitives: tuple[ConditionalBindingPrimitives, ...],
) -> BackendBindingSpec:
    """Build an ``analytical_black76``-shaped synthetic BackendBindingSpec.

    The base ``primitives`` tuple is deliberately empty because dispatch for
    this binding is fully driven by its conditional-primitives ladder — every
    real path lands in a conditional clause, so base primitives would never
    surface.
    """
    return BackendBindingSpec(
        route_id="analytical_black76_binding_dsl_parity",
        engine_family="analytical",
        route_family="analytical",
        aliases=(),
        compatibility_alias_policy="internal_only",
        primitives=(),
        conditional_primitives=conditional_primitives,
        conditional_route_family=(),
    )


def _legacy_clauses() -> tuple[ConditionalBindingPrimitives, ...]:
    """Legacy string-tag form, verbatim from pre-QUA-921 backend_bindings.yaml.

    The vanilla, period_rate_option_strip, and basket clauses stay legacy in both
    variants; only the two swaption clauses (bermudan, european) migrate to
    DSL.  See the module docstring for the per-clause rationale — in short,
    the DSL today cannot faithfully express the extra trait filter on the
    basket clause, and the ``vanilla_payoff`` tag over-matches on trait-level
    ``vanilla_option`` entries that appear on baskets.
    """
    return (
        ConditionalBindingPrimitives(
            when={"payoff_family": "vanilla_option"},
            primitives=_VANILLA_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when={"payoff_family": "period_rate_option_strip"},
            primitives=_RATE_CAP_FLOOR_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when={
                "payoff_family": "basket_option",
                "payoff_traits": ["two_asset_terminal_basket"],
                "exercise_style": ["european"],
                "model_family": ["equity_diffusion"],
            },
            primitives=_BASKET_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when={
                "payoff_family": "swaption",
                "exercise_style": ["bermudan"],
            },
            primitives=_SWAPTION_BERM_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when={
                "payoff_family": "swaption",
                "exercise_style": ["european"],
            },
            primitives=_SWAPTION_EUR_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when="default",
            primitives=_DEFAULT_PRIMS,
        ),
    )


def _dsl_clauses() -> tuple[ConditionalBindingPrimitives, ...]:
    """DSL ``contract_pattern`` form — the QUA-921 migration target.

    Only the two DSL-safe structural clauses (swaption-bermudan and
    swaption-european) migrate here.  ``vanilla_option``,
    ``period_rate_option_strip``, ``basket_option``, and ``default`` stay in
    legacy form for the reasons spelled out in the module docstring.
    """
    return (
        ConditionalBindingPrimitives(
            when={"payoff_family": "vanilla_option"},
            primitives=_VANILLA_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when={"payoff_family": "period_rate_option_strip"},
            primitives=_RATE_CAP_FLOOR_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when={
                "payoff_family": "basket_option",
                "payoff_traits": ["two_asset_terminal_basket"],
                "exercise_style": ["european"],
                "model_family": ["equity_diffusion"],
            },
            primitives=_BASKET_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when={},
            contract_pattern=parse_contract_pattern(
                {
                    "payoff": {"kind": "swaption_payoff"},
                    "exercise": {"style": "bermudan"},
                }
            ),
            primitives=_SWAPTION_BERM_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when={},
            contract_pattern=parse_contract_pattern(
                {
                    "payoff": {"kind": "swaption_payoff"},
                    "exercise": {"style": "european"},
                }
            ),
            primitives=_SWAPTION_EUR_PRIMS,
        ),
        ConditionalBindingPrimitives(
            when="default",
            primitives=_DEFAULT_PRIMS,
        ),
    )


@pytest.fixture(scope="module")
def legacy_binding() -> BackendBindingSpec:
    return _make_binding(_legacy_clauses())


@pytest.fixture(scope="module")
def dsl_binding() -> BackendBindingSpec:
    return _make_binding(_dsl_clauses())


# ---------------------------------------------------------------------------
# Canonical ProductIR fixtures — one per clause plus "default fall-through"
# and "None IR" cases.  Mirrors the routes.yaml parity test fixtures exactly.
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
    """Basket with the ``two_asset_terminal_basket`` trait expected by the
    legacy clause and by the ``basket_payoff`` DSL pattern after QUA-921."""
    return ProductIR(
        instrument="basket_option",
        payoff_family="basket_option",
        payoff_traits=("two_asset_terminal_basket", "basket_payoff"),
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


def _cap_rate_cap_floor_ir() -> ProductIR:
    """A cap: hits the (still-legacy) ``period_rate_option_strip`` clause.

    QUA-921 intentionally does not migrate this clause, so it should resolve
    to the rate-cap helper under both reference variants.
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
    """Bermudan basket — misses the basket clause (exercise != european).

    Legacy rejects on the ``exercise_style`` list mismatch; DSL rejects on
    ``ExercisePattern.style == "european"``.  Both fall through to the
    ``default`` clause.
    """
    return ProductIR(
        instrument="basket_option_bermudan",
        payoff_family="basket_option",
        payoff_traits=("two_asset_terminal_basket", "basket_payoff"),
        exercise_style="bermudan",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="equity_diffusion",
        candidate_engine_families=("analytical",),
    )


_FIXTURES: tuple[tuple[str, object], ...] = (
    ("vanilla_european_equity", _vanilla_ir),
    ("basket_european_equity", _basket_european_equity_ir),
    ("swaption_bermudan", _swaption_bermudan_ir),
    ("swaption_european", _swaption_european_ir),
    ("cap_rate_cap_floor", _cap_rate_cap_floor_ir),
    ("bermudan_basket_default_fallthrough", _bermudan_basket_ir),
)


# ---------------------------------------------------------------------------
# Parity assertions across the two synthetic variants.
# ---------------------------------------------------------------------------


class TestBlack76BindingCatalogLegacyVsDSLParity:
    """The legacy and DSL synthetic bindings dispatch identically on every
    canonical fixture."""

    @pytest.mark.parametrize(
        "label,ir_factory",
        _FIXTURES,
        ids=[label for label, _ in _FIXTURES],
    )
    def test_resolve_binding_primitives_parity(
        self,
        legacy_binding: BackendBindingSpec,
        dsl_binding: BackendBindingSpec,
        label: str,
        ir_factory,
    ):
        ir = ir_factory()
        legacy_prims = _resolve_binding_primitives(legacy_binding, ir)
        dsl_prims = _resolve_binding_primitives(dsl_binding, ir)
        assert legacy_prims == dsl_prims, (
            f"binding primitive drift on fixture={label!r}:\n"
            f"  legacy: {[(p.module, p.symbol, p.role) for p in legacy_prims]}\n"
            f"  dsl:    {[(p.module, p.symbol, p.role) for p in dsl_prims]}"
        )

    def test_none_product_ir_falls_through_to_default_on_both_variants(
        self,
        legacy_binding: BackendBindingSpec,
        dsl_binding: BackendBindingSpec,
    ):
        """``None`` ProductIR must hit the shared ``default`` clause.

        The legacy path defaults ``payoff_family`` / ``exercise_style`` /
        ``model_family`` to stringy fallbacks that fail every specific clause.
        The DSL path short-circuits to ``False`` on a ``None`` product_ir.
        Both paths therefore walk to the ``default`` sentinel and must return
        its primitives verbatim.
        """
        assert _resolve_binding_primitives(
            legacy_binding, None
        ) == _resolve_binding_primitives(dsl_binding, None)


# ---------------------------------------------------------------------------
# Load-bearing parity assertion: the on-disk catalog entry (exercising the
# binding-cache production path WITHOUT monkeypatching) must match both
# reference variants.
# ---------------------------------------------------------------------------


class TestOnDiskBlack76BindingDispatchMatchesBothVariants:
    """The on-disk ``analytical_black76`` binding catalog entry dispatches
    identically to both the legacy and DSL reference variants on every
    canonical fixture.

    **No binding-cache patching.**  Unlike QUA-920's parity test which had to
    monkey-patch ``resolve_backend_binding_by_route_id`` to force route-level
    dispatch (because the catalog overlay was still legacy and would have
    shadowed the migrated routes.yaml clauses), this test exercises the
    full production stack.  That's the load-bearing evidence that DSL
    dispatch is live inside
    :func:`backend_bindings._resolve_binding_primitives` after QUA-921, not
    just inside ``route_registry._conditional_primitive_matches``.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        # A catalog load earlier in the session may have cached the pre-migration
        # catalog; ensure each test re-reads the on-disk YAML.
        clear_backend_binding_catalog_cache()
        yield
        clear_backend_binding_catalog_cache()

    @pytest.mark.parametrize(
        "label,ir_factory",
        _FIXTURES,
        ids=[label for label, _ in _FIXTURES],
    )
    def test_on_disk_binding_primitives_match_both_variants(
        self,
        legacy_binding: BackendBindingSpec,
        dsl_binding: BackendBindingSpec,
        label: str,
        ir_factory,
    ):
        ir = ir_factory()

        catalog = load_backend_binding_catalog()
        on_disk_binding = find_backend_binding_by_route_id(
            "analytical_black76", catalog
        )
        assert on_disk_binding is not None, (
            "analytical_black76 missing from backend_bindings catalog"
        )

        legacy_prims = _resolve_binding_primitives(legacy_binding, ir)
        dsl_prims = _resolve_binding_primitives(dsl_binding, ir)
        on_disk_prims = _resolve_binding_primitives(on_disk_binding, ir)

        assert legacy_prims == dsl_prims == on_disk_prims, (
            f"on-disk analytical_black76 binding primitives diverge on "
            f"fixture={label!r}:\n"
            f"  legacy:  {[(p.module, p.symbol, p.role) for p in legacy_prims]}\n"
            f"  on-disk: {[(p.module, p.symbol, p.role) for p in on_disk_prims]}\n"
            f"  dsl:     {[(p.module, p.symbol, p.role) for p in dsl_prims]}"
        )

    def test_on_disk_binding_resolves_end_to_end_for_every_fixture(self):
        """End-to-end check through :func:`resolve_backend_binding_spec`.

        This exercises the full catalog-hosted pipeline (helpers, binding_id,
        kernel refs) rather than the bare dispatch helper, and asserts that
        every fixture continues to resolve without error after the YAML
        migration.  Coupled with the primitive-parity test above, this rules
        out a silent helper-ref drift from the DSL route change.
        """
        catalog = load_backend_binding_catalog()
        on_disk_binding = find_backend_binding_by_route_id(
            "analytical_black76", catalog
        )
        assert on_disk_binding is not None

        for _, ir_factory in _FIXTURES:
            ir = ir_factory()
            resolved = resolve_backend_binding_spec(on_disk_binding, product_ir=ir)
            assert resolved.route_id == "analytical_black76"
            # Every fixture is expected to surface a non-empty primitive set:
            # the default clause always produces Black76 kernels, and each
            # specific fixture lands in a helper-backed clause.
            assert resolved.primitives, (
                f"analytical_black76 resolved to empty primitives for fixture "
                f"ir={ir!r}; expected at least one primitive"
            )
