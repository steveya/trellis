"""Knowledge authority for the strict physical Bermudan swaption lattice route."""

from trellis.agent.backend_bindings import (
    find_backend_binding_by_route_id,
    load_backend_binding_catalog,
    resolve_backend_binding_spec,
)
from trellis.agent.knowledge.import_registry import is_valid_import
from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.route_registry import (
    find_route_by_id,
    load_route_registry,
    match_candidate_routes,
)
from trellis.agent.semantic_validation import extract_semantic_signals


STRICT_ROUTE_ID = "physical_bermudan_swaption_lattice"
STRICT_MODULE = "trellis.models.rate_swap_tail"


def _strict_product_ir() -> ProductIR:
    return ProductIR(
        instrument="physical_bermudan_swaption",
        payoff_family="physical_bermudan_swaption",
        payoff_traits=(
            "physical_settlement",
            "co_terminal_swap_tails",
            "dual_curve_projection",
            "strict_conventions",
            "named_hull_white_parameters",
            "bermudan_exercise",
        ),
        exercise_style="bermudan",
        state_dependence="schedule_state",
        schedule_dependence=True,
        model_family="interest_rate",
        candidate_engine_families=("lattice",),
        route_families=(STRICT_ROUTE_ID,),
        required_market_data=frozenset(
            {"discount_curve", "forward_curve", "model_parameters"}
        ),
    )


def test_strict_physical_bermudan_route_is_distinct_and_promoted():
    registry = load_route_registry()
    route = find_route_by_id(STRICT_ROUTE_ID, registry)

    assert route is not None
    assert route.status == "promoted"
    assert route.engine_family == "lattice"
    assert route.route_family == STRICT_ROUTE_ID
    assert route.match_instruments is None
    assert route.match_payoff_family == ("physical_bermudan_swaption",)
    assert route.match_exercise == ("bermudan",)
    assert set(route.match_required_payoff_traits or ()) == {
        "physical_settlement",
        "co_terminal_swap_tails",
        "dual_curve_projection",
        "strict_conventions",
        "named_hull_white_parameters",
        "bermudan_exercise",
    }
    assert set(route.market_data_access.required) == {
        "discount_curve",
        "forward_curve",
        "model_parameters",
    }

    matches = match_candidate_routes(
        registry,
        "rate_tree",
        _strict_product_ir(),
        skip_market_data_filters=True,
    )
    assert STRICT_ROUTE_ID in {candidate.id for candidate in matches}

    incomplete = ProductIR(
        instrument="physical_bermudan_swaption",
        payoff_family="physical_bermudan_swaption",
        payoff_traits=("physical_settlement",),
        exercise_style="bermudan",
        model_family="interest_rate",
    )
    assert match_candidate_routes(
        registry,
        "rate_tree",
        incomplete,
        skip_market_data_filters=True,
    ) == ()


def test_strict_physical_bermudan_binding_has_no_legacy_or_black_fallback():
    catalog = load_backend_binding_catalog()
    binding = find_backend_binding_by_route_id(STRICT_ROUTE_ID, catalog)

    assert binding is not None
    resolved = resolve_backend_binding_spec(
        binding,
        product_ir=_strict_product_ir(),
        method="rate_tree",
    )
    primitives = {
        (primitive.module, primitive.symbol, primitive.role)
        for primitive in resolved.primitives
    }
    assert (
        STRICT_MODULE,
        "price_physical_bermudan_swaption_lattice",
        "pricing_kernel",
    ) in primitives
    assert (
        STRICT_MODULE,
        "resolve_co_terminal_swap_tails",
        "schedule_builder",
    ) in primitives
    assert not any(module == "trellis.models.bermudan_swaption_tree" for module, _, _ in primitives)
    assert not any("black76" in symbol.lower() for _, symbol, _ in primitives)


def test_strict_physical_bermudan_primitives_are_codegen_authorized():
    for symbol in (
        "ExerciseSwapStart",
        "FixedLegConvention",
        "FloatingLegConvention",
        "NamedRateCurve",
        "PhysicalBermudanSwapTailSpec",
        "compile_physical_bermudan_swap_tail_spec",
        "resolve_co_terminal_swap_tails",
        "map_swap_tail_dates_to_lattice",
        "observe_conditional_discount_bonds",
        "build_bermudan_swaption_exercise_values",
        "price_physical_bermudan_swaption_lattice",
    ):
        assert is_valid_import(STRICT_MODULE, symbol)

    assert is_valid_import(
        "trellis.models.hull_white_parameters",
        "resolve_named_hull_white_parameter_set",
    )


def test_strict_swap_tail_module_is_classified_as_rate_lattice_code():
    signals = extract_semantic_signals(
        "from trellis.models.rate_swap_tail import "
        "price_physical_bermudan_swaption_lattice\n"
    )

    assert signals.engine_families == ("rate_lattice",)
