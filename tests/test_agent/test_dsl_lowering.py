"""Tests for semantic DSL lowering onto helper-backed route targets."""

from __future__ import annotations

from trellis.agent.dsl_algebra import ChoiceExpr, ContractAtom, ThenExpr, ControlStyle
from trellis.agent.family_lowering_ir import (
    AnalyticalBlack76IR,
    CorrelatedBasketMonteCarloIR,
    ExerciseLatticeIR,
    VanillaEquityPDEIR,
)


def test_ranked_observation_basket_lowers_to_market_binding_then_helper():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract

    contract = make_ranked_observation_basket_contract(
        description="Himalaya on AAPL, MSFT, NVDA",
        constituents=("AAPL", "MSFT", "NVDA"),
        observation_schedule=("2025-06-15", "2025-12-15", "2026-06-15"),
    )
    blueprint = compile_semantic_contract(contract)

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "correlated_basket_monte_carlo"
    assert lowering.route_family == "monte_carlo"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, CorrelatedBasketMonteCarloIR)
    assert lowering.family_ir.market_binding_symbol == "resolve_basket_semantics"
    assert lowering.family_ir.helper_symbol == "price_ranked_observation_basket_monte_carlo"
    assert lowering.family_ir.path_requirement_kind == "observation_snapshot_state"
    assert lowering.family_ir.binding_sources[-1] == (
        "correlation_matrix",
        "runtime_connector_resolution",
    )
    assert isinstance(lowering.normalized_expr, ThenExpr)
    binding, helper = lowering.normalized_expr.terms
    assert isinstance(binding, ContractAtom)
    assert isinstance(helper, ContractAtom)
    assert binding.atom_id == "correlated_basket_monte_carlo:market_binding"
    assert helper.atom_id == "correlated_basket_monte_carlo:route_helper"
    assert helper.primitive_ref == (
        "trellis.models.monte_carlo.semantic_basket."
        "price_ranked_observation_basket_monte_carlo"
    )
    assert lowering.control_styles == ()
    assert (
        "trellis.models.resolution.basket_semantics.resolve_basket_semantics"
        in lowering.helper_refs
    )


def test_callable_bond_lowers_to_explicit_issuer_choice_plus_helper_targets():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_callable_bond_contract

    contract = make_callable_bond_contract(
        description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        observation_schedule=("2026-01-15", "2027-01-15"),
    )
    blueprint = compile_semantic_contract(contract)

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "exercise_lattice"
    assert lowering.route_family == "rate_lattice"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, ExerciseLatticeIR)
    assert lowering.family_ir.helper_symbol == "price_callable_bond_tree"
    assert lowering.family_ir.control_style == "issuer_min"
    assert "coupon_accrual_fractions" in lowering.family_ir.derived_quantities
    assert isinstance(lowering.normalized_expr, ChoiceExpr)
    assert lowering.normalized_expr.style == ControlStyle.ISSUER_MIN
    assert lowering.control_styles == (ControlStyle.ISSUER_MIN,)
    assert {branch.atom_id for branch in lowering.normalized_expr.branches} == {
        "rate_lattice:continuation",
        "rate_lattice:exercise_now",
    }
    assert (
        "trellis.models.callable_bond_tree.price_callable_bond_tree"
        in lowering.helper_refs
    )
    assert "trellis.models.callable_bond_tree" in blueprint.route_modules


def test_bermudan_swaption_lowers_to_explicit_holder_choice_plus_helper_targets():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="Bermudan payer swaption with annual exercise dates",
        observation_schedule=("2026-01-15", "2027-01-15", "2028-01-15"),
        preferred_method="rate_tree",
        exercise_style="bermudan",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="rate_tree")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "exercise_lattice"
    assert lowering.route_family == "rate_lattice"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, ExerciseLatticeIR)
    assert lowering.family_ir.helper_symbol == "price_bermudan_swaption_tree"
    assert lowering.family_ir.control_style == "holder_max"
    assert "par_rate_bindings" in lowering.family_ir.derived_quantities
    assert isinstance(lowering.normalized_expr, ChoiceExpr)
    assert lowering.normalized_expr.style == ControlStyle.HOLDER_MAX
    assert lowering.control_styles == (ControlStyle.HOLDER_MAX,)
    assert (
        "trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree"
        in lowering.helper_refs
    )
    assert "trellis.models.bermudan_swaption_tree" in blueprint.route_modules


def test_vanilla_option_analytical_lowers_to_checked_in_black76_kernel():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR call on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
    )
    blueprint = compile_semantic_contract(contract)

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "analytical_black76"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, AnalyticalBlack76IR)
    assert lowering.family_ir.kernel_symbol == "black76_call"
    assert lowering.family_ir.required_input_ids == blueprint.required_market_data
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == "trellis.models.black.black76_call"
    assert "trellis.models.black" in blueprint.route_modules


def test_vanilla_put_analytical_lowers_to_put_kernel():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR put on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
    )
    blueprint = compile_semantic_contract(contract)

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert isinstance(lowering.family_ir, AnalyticalBlack76IR)
    assert lowering.family_ir.kernel_symbol == "black76_put"
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == "trellis.models.black.black76_put"


def test_vanilla_option_pde_lowers_to_checked_in_helper():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR call on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
        preferred_method="pde_solver",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="pde_solver")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "vanilla_equity_theta_pde"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, VanillaEquityPDEIR)
    assert lowering.family_ir.theta == 0.5
    assert lowering.family_ir.required_input_ids == blueprint.required_market_data
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.equity_option_pde.price_vanilla_equity_option_pde"
    )


def test_rate_style_swaption_analytical_lowers_to_black76_call_kernel():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="5Yx10Y USD payer swaption Black-76",
        observation_schedule=("2031-03-15",),
    )
    blueprint = compile_semantic_contract(contract, preferred_method="analytical")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "analytical_black76"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == "trellis.models.black.black76_call"


def test_rate_style_swaption_receiver_analytical_lowers_to_black76_put_kernel():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="5Yx10Y USD receiver swaption Black-76",
        observation_schedule=("2031-03-15",),
    )
    blueprint = compile_semantic_contract(contract, preferred_method="analytical")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == "trellis.models.black.black76_put"


def test_bermudan_swaption_analytical_lowers_to_lower_bound_helper():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="Bermudan payer swaption analytical lower bound",
        observation_schedule=("2026-01-15", "2027-01-15", "2028-01-15"),
        preferred_method="analytical",
        exercise_style="bermudan",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="analytical")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "analytical_black76"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.rate_style_swaption.price_bermudan_swaption_black76_lower_bound"
    )
