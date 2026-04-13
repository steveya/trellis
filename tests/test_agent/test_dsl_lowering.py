"""Tests for semantic DSL lowering onto helper-backed route targets."""

from __future__ import annotations

from types import SimpleNamespace

from trellis.agent.codegen_guardrails import PrimitiveRef
from trellis.agent.dsl_algebra import ChoiceExpr, ContractAtom, ThenExpr, ControlStyle
from trellis.agent.family_lowering_ir import (
    AnalyticalBlack76IR,
    CorrelatedBasketMonteCarloIR,
    CreditDefaultSwapIR,
    EventAwareMonteCarloIR,
    EventAwarePDEIR,
    ExerciseLatticeIR,
    NthToDefaultIR,
    TransformPricingIR,
    VanillaEquityPDEIR,
)


def _make_bermudan_equity_pde_contract():
    from trellis.agent.semantic_contracts import make_american_option_contract

    return make_american_option_contract(
        description="Bermudan put on AAPL with quarterly exercise dates",
        underliers=("AAPL",),
        observation_schedule=("2026-03-20", "2026-06-20", "2026-09-20", "2026-12-20"),
        preferred_method="pde_solver",
        exercise_style="bermudan",
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
    assert (
        lowering.binding_id
        == "trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo"
    )
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
    assert binding.atom_id == (
        "trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo:"
        "market_binding"
    )
    assert helper.atom_id == (
        "trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo:"
        "route_helper"
    )
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
    assert blueprint.lane_plan is not None
    assert blueprint.lane_plan.lane_family == "lattice"
    assert blueprint.lane_plan.plan_kind == "exact_target_binding"
    assert "control_style:issuer_min" in blueprint.lane_plan.control_obligations
    assert (
        "trellis.models.callable_bond_tree.price_callable_bond_tree"
        in lowering.helper_refs
    )
    assert "trellis.models.callable_bond_tree" in blueprint.route_modules


def test_quanto_lowering_prefers_binding_spec_targets_when_route_primitives_are_stale(monkeypatch):
    from trellis.agent import backend_bindings as backend_bindings_module
    from trellis.agent import dsl_lowering as dsl_lowering_module
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_quanto_option_contract

    monkeypatch.setattr(
        dsl_lowering_module,
        "resolve_route_primitives",
        lambda route, product_ir: (),
    )
    monkeypatch.setattr(
        backend_bindings_module,
        "resolve_backend_binding_by_route_id",
        lambda route_id, product_ir=None, primitive_plan=None, catalog=None: SimpleNamespace(
            primitives=(
                PrimitiveRef(
                    "trellis.models.quanto_option",
                    "price_quanto_option_analytical_from_market_state",
                    "route_helper",
                ),
            ),
            route_family="analytical",
        ),
    )

    contract = make_quanto_option_contract(
        description="Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        underliers=("SAP",),
        observation_schedule=("2025-11-15",),
        preferred_method="analytical",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="analytical")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.errors == ()
    assert lowering.helper_refs == (
        "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state",
    )


def test_callable_bond_pde_lowers_to_event_aware_helper():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_callable_bond_contract

    contract = make_callable_bond_contract(
        description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        observation_schedule=("2026-01-15", "2027-01-15"),
        preferred_method="pde_solver",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="pde_solver")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "pde_theta_1d"
    assert lowering.route_family == "pde_solver"
    assert isinstance(lowering.family_ir, EventAwarePDEIR)
    assert lowering.admissibility_errors == ()
    assert lowering.family_ir.helper_symbol == "price_callable_bond_pde"
    assert lowering.family_ir.event_transform_kinds == ("add_cashflow", "project_min")
    assert lowering.family_ir.event_dates == ("2026-01-15", "2027-01-15")
    assert lowering.normalized_expr is not None
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.callable_bond_pde.price_callable_bond_pde"
    )
    assert lowering.helper_refs == (
        "trellis.models.callable_bond_pde.price_callable_bond_pde",
    )
    assert blueprint.lane_plan is not None
    assert blueprint.lane_plan.lane_family == "pde_solver"
    assert "control_style:issuer_min" in blueprint.lane_plan.control_obligations
    assert "event_transform:add_cashflow" in blueprint.lane_plan.control_obligations
    assert "event_transform:project_min" in blueprint.lane_plan.control_obligations


def test_holder_max_equity_pde_lowers_to_event_aware_helper():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract

    contract = _make_bermudan_equity_pde_contract()
    blueprint = compile_semantic_contract(contract, preferred_method="pde_solver")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "pde_theta_1d"
    assert lowering.route_family == "pde_solver"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, EventAwarePDEIR)
    assert lowering.family_ir.helper_symbol == "price_event_aware_equity_option_pde"
    assert lowering.family_ir.event_transform_kinds == ("project_max",)
    assert lowering.normalized_expr is not None
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.equity_option_pde.price_event_aware_equity_option_pde"
    )
    assert (
        "trellis.models.equity_option_pde.price_event_aware_equity_option_pde"
        in lowering.helper_refs
    )
    assert blueprint.lane_plan is not None
    assert "control_style:holder_max" in blueprint.lane_plan.control_obligations
    assert "event_transform:project_max" in blueprint.lane_plan.control_obligations


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


def test_rate_tree_swaption_lowers_to_checked_in_helper_target():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="European payer swaption on HW tree",
        observation_schedule=("2029-11-15",),
        preferred_method="rate_tree",
        exercise_style="european",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="rate_tree")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "rate_tree_backward_induction"
    assert lowering.route_family == "rate_lattice"
    assert lowering.family_ir is None
    assert lowering.admissibility_errors == ()
    assert (
        "trellis.models.rate_style_swaption_tree.price_swaption_tree"
        in lowering.helper_refs
    )
    assert "trellis.models.rate_style_swaption_tree" in blueprint.route_modules


def test_rate_style_swaption_monte_carlo_lowers_to_event_aware_compilation_steps():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.dsl_algebra import ContractAtom
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="European payer swaption under Hull-White Monte Carlo",
        observation_schedule=("2029-11-15",),
        preferred_method="monte_carlo",
        exercise_style="european",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="monte_carlo")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "monte_carlo_paths"
    assert lowering.route_family == "monte_carlo"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, EventAwareMonteCarloIR)
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.binding_id == "trellis.models.rate_style_swaption.price_swaption_monte_carlo"
    assert lowering.normalized_expr.atom_id == (
        "trellis.models.rate_style_swaption.price_swaption_monte_carlo:route_helper"
    )
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.rate_style_swaption.price_swaption_monte_carlo"
    )
    assert blueprint.lane_plan is not None
    assert blueprint.lane_plan.lane_family == "monte_carlo"
    assert "control_style:identity" in blueprint.lane_plan.control_obligations
    assert "semantic_control_style:holder_max" in blueprint.lane_plan.control_obligations
    assert "measure_family:risk_neutral" in blueprint.lane_plan.control_obligations
    assert "event_kind:exercise" in blueprint.lane_plan.control_obligations
    assert "short_rate" in blueprint.lane_plan.state_obligations
    assert "event_replay" in blueprint.lane_plan.state_obligations


def test_vanilla_option_monte_carlo_lowers_to_terminal_only_compilation_steps():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR call on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
        preferred_method="monte_carlo",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="monte_carlo")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "monte_carlo_paths"
    assert lowering.route_family == "monte_carlo"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, EventAwareMonteCarloIR)
    assert lowering.family_ir.path_requirement_spec.requirement_kind == "terminal_only"
    assert lowering.family_ir.event_kinds == ()
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert (
        lowering.binding_id
        == "trellis.models.equity_option_monte_carlo.price_vanilla_equity_option_monte_carlo"
    )
    assert lowering.normalized_expr.atom_id == (
        "trellis.models.equity_option_monte_carlo.price_vanilla_equity_option_monte_carlo:"
        "route_helper"
    )
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.equity_option_monte_carlo.price_vanilla_equity_option_monte_carlo"
    )
    assert blueprint.lane_plan is not None
    assert "spot" in blueprint.lane_plan.state_obligations
    assert "terminal_only" in blueprint.lane_plan.state_obligations


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
    assert blueprint.lane_plan is not None
    assert blueprint.lane_plan.lane_family == "analytical"
    assert "Bind analytical market inputs via `spot_discount_vol_to_forward`." in blueprint.lane_plan.construction_steps
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == "trellis.models.black.black76_call"
    assert "trellis.models.black" in blueprint.route_modules


def test_vanilla_option_transform_lowers_to_checked_in_transform_helper():
    from dataclasses import replace

    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR call on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
        preferred_method="analytical",
    )
    transform_contract = replace(
        contract,
        methods=replace(
            contract.methods,
            candidate_methods=contract.methods.candidate_methods + ("fft_pricing",),
            reference_methods=("fft_pricing",),
            preferred_method="fft_pricing",
        ),
    )
    blueprint = compile_semantic_contract(transform_contract, preferred_method="fft_pricing")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "transform_fft"
    assert lowering.route_family == "fft_pricing"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, TransformPricingIR)
    assert lowering.family_ir.control_spec.control_style == "identity"
    assert lowering.family_ir.state_spec.state_tags == ("terminal_markov",)
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert (
        lowering.binding_id
        == "trellis.models.equity_option_transforms.price_vanilla_equity_option_transform"
    )
    assert lowering.normalized_expr.atom_id == (
        "trellis.models.equity_option_transforms.price_vanilla_equity_option_transform:"
        "route_helper"
    )
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.equity_option_transforms.price_vanilla_equity_option_transform"
    )


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
    assert isinstance(lowering.family_ir, EventAwarePDEIR)
    assert isinstance(lowering.family_ir, VanillaEquityPDEIR)
    assert lowering.family_ir.theta == 0.5
    assert lowering.family_ir.operator_spec.operator_family == "black_scholes_1d"
    assert lowering.family_ir.required_input_ids == blueprint.required_market_data
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.equity_option_pde.price_vanilla_equity_option_pde"
    )


def test_rate_style_swaption_analytical_lowers_to_black76_family_helper():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract
    from trellis.agent.dsl_algebra import ContractAtom

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
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.rate_style_swaption.price_swaption_black76"
    )


def test_rate_style_swaption_receiver_analytical_lowers_to_black76_family_helper():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract
    from trellis.agent.dsl_algebra import ContractAtom

    contract = make_rate_style_swaption_contract(
        description="5Yx10Y USD receiver swaption Black-76",
        observation_schedule=("2031-03-15",),
    )
    blueprint = compile_semantic_contract(contract, preferred_method="analytical")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.models.rate_style_swaption.price_swaption_black76"
    )


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


def test_credit_default_swap_analytical_lowers_to_schedule_then_helper():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_credit_default_swap_contract

    contract = make_credit_default_swap_contract(
        description="Single-name CDS on ACME analytical",
        observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20"),
    )
    blueprint = compile_semantic_contract(contract)

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "credit_default_swap_analytical"
    assert lowering.route_family == "credit_default_swap"
    assert lowering.admissibility_errors == ()
    assert lowering.binding_id == "trellis.models.credit_default_swap.price_cds_analytical"
    assert isinstance(lowering.family_ir, CreditDefaultSwapIR)
    assert lowering.family_ir.pricing_mode == "analytical"
    assert isinstance(lowering.normalized_expr, ThenExpr)
    schedule_builder, helper = lowering.normalized_expr.terms
    assert isinstance(schedule_builder, ContractAtom)
    assert isinstance(helper, ContractAtom)
    assert schedule_builder.atom_id == (
        "trellis.models.credit_default_swap.price_cds_analytical:schedule_builder"
    )
    assert helper.atom_id == (
        "trellis.models.credit_default_swap.price_cds_analytical:route_helper"
    )
    assert schedule_builder.primitive_ref == "trellis.models.credit_default_swap.build_cds_schedule"
    assert helper.primitive_ref == "trellis.models.credit_default_swap.price_cds_analytical"


def test_credit_default_swap_monte_carlo_lowers_to_schedule_then_helper():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_credit_default_swap_contract

    contract = make_credit_default_swap_contract(
        description="Single-name CDS on ACME Monte Carlo",
        observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20"),
        preferred_method="monte_carlo",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="monte_carlo")

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "credit_default_swap_monte_carlo"
    assert lowering.route_family == "credit_default_swap"
    assert lowering.admissibility_errors == ()
    assert lowering.binding_id == "trellis.models.credit_default_swap.price_cds_monte_carlo"
    assert isinstance(lowering.family_ir, CreditDefaultSwapIR)
    assert lowering.family_ir.pricing_mode == "monte_carlo"
    assert isinstance(lowering.normalized_expr, ThenExpr)
    schedule_builder, helper = lowering.normalized_expr.terms
    assert schedule_builder.primitive_ref == "trellis.models.credit_default_swap.build_cds_schedule"
    assert helper.primitive_ref == "trellis.models.credit_default_swap.price_cds_monte_carlo"


def test_credit_default_swap_missing_schedule_builder_reports_binding_first_error(monkeypatch):
    from trellis.agent import backend_bindings as backend_bindings_module
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_credit_default_swap_contract

    def _binding_without_schedule_builder(*args, **kwargs):
        del args, kwargs
        return backend_bindings_module.ResolvedBackendBindingSpec(
            route_id="credit_default_swap_analytical",
            engine_family="analytical",
            route_family="credit_default_swap",
            binding_id="trellis.models.credit_default_swap.price_cds_analytical",
            primitives=(
                PrimitiveRef(
                    "trellis.models.credit_default_swap",
                    "price_cds_analytical",
                    "route_helper",
                ),
            ),
            primitive_refs=("trellis.models.credit_default_swap.price_cds_analytical",),
            helper_refs=("trellis.models.credit_default_swap.price_cds_analytical",),
            exact_target_refs=("trellis.models.credit_default_swap.price_cds_analytical",),
        )

    monkeypatch.setattr(
        backend_bindings_module,
        "resolve_backend_binding_by_route_id",
        _binding_without_schedule_builder,
    )

    contract = make_credit_default_swap_contract(
        description="Single-name CDS on ACME analytical",
        observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20"),
    )
    blueprint = compile_semantic_contract(contract)

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.expr is None
    assert lowering.binding_id == "trellis.models.credit_default_swap.price_cds_analytical"
    assert lowering.errors[0].code == "missing_schedule_builder"
    assert lowering.errors[0].message == (
        "Binding 'trellis.models.credit_default_swap.price_cds_analytical' is missing "
        "the required schedule builder primitive 'build_cds_schedule'."
    )


def test_nth_to_default_lowers_to_helper_backed_credit_basket_route():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_nth_to_default_contract

    contract = make_nth_to_default_contract(
        description="First-to-default basket on ACME, BRAVO, CHARLIE, DELTA, ECHO through 2029-11-15",
        observation_schedule=("2029-11-15",),
        reference_entities=("ACME", "BRAVO", "CHARLIE", "DELTA", "ECHO"),
        trigger_rank=1,
    )
    blueprint = compile_semantic_contract(contract)

    lowering = blueprint.dsl_lowering
    assert lowering is not None
    assert lowering.route_id == "nth_to_default_monte_carlo"
    assert lowering.route_family == "nth_to_default"
    assert lowering.admissibility_errors == ()
    assert isinstance(lowering.family_ir, NthToDefaultIR)
    assert isinstance(lowering.normalized_expr, ContractAtom)
    assert lowering.normalized_expr.primitive_ref == (
        "trellis.instruments.nth_to_default.price_nth_to_default_basket"
    )


def test_unknown_dsl_route_reports_structured_lowering_error():
    from trellis.agent.dsl_lowering import lower_semantic_blueprint
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_credit_default_swap_contract

    contract = make_credit_default_swap_contract(
        description="Single-name CDS on ACME analytical",
        observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20"),
    )
    blueprint = compile_semantic_contract(contract)

    lowering = lower_semantic_blueprint(
        blueprint.contract,
        product_ir=blueprint.product_ir,
        pricing_plan=blueprint.pricing_plan,
        primitive_routes=("does_not_exist",),
        valuation_context=blueprint.valuation_context,
        market_binding_spec=blueprint.market_binding_spec,
    )

    assert lowering.expr is None
    assert lowering.route_id == "does_not_exist"
    assert lowering.errors[0].route_id == "does_not_exist"
    assert lowering.errors[0].code == "unknown_primitive_route"
    assert "Unknown primitive route" in lowering.errors[0].message
    assert lowering.admissibility_errors == (
        "Unknown primitive route for DSL lowering: 'does_not_exist'",
    )
