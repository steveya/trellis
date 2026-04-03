"""Tests for typed family lowering IR construction."""

from __future__ import annotations

from dataclasses import replace

from trellis.agent.family_lowering_ir import (
    AnalyticalBlack76IR,
    CorrelatedBasketMonteCarloIR,
    CreditDefaultSwapIR,
    ExerciseLatticeIR,
    NthToDefaultIR,
    VanillaEquityPDEIR,
)


def test_vanilla_option_compiles_to_analytical_black76_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR call on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
    )
    blueprint = compile_semantic_contract(contract, requested_outputs=["price", "vega"])

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, AnalyticalBlack76IR)
    assert family_ir.route_id == "analytical_black76"
    assert family_ir.product_instrument == "european_option"
    assert family_ir.payoff_family == "vanilla_option"
    assert family_ir.option_type == "call"
    assert family_ir.kernel_symbol == "black76_call"
    assert family_ir.market_mapping == "spot_discount_vol_to_forward"
    assert family_ir.required_input_ids == blueprint.required_market_data
    assert family_ir.requested_outputs == ("price", "vega")


def test_vanilla_option_compiles_to_pde_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR put on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
        preferred_method="pde_solver",
    )
    blueprint = compile_semantic_contract(
        contract,
        preferred_method="pde_solver",
        requested_outputs=["price"],
    )

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, VanillaEquityPDEIR)
    assert family_ir.route_id == "vanilla_equity_theta_pde"
    assert family_ir.product_instrument == "european_option"
    assert family_ir.payoff_family == "vanilla_option"
    assert family_ir.option_type == "put"
    assert family_ir.theta == 0.5
    assert family_ir.helper_symbol == "price_vanilla_equity_option_pde"
    assert family_ir.market_mapping == "equity_spot_discount_black_vol"
    assert family_ir.required_input_ids == blueprint.required_market_data
    assert family_ir.requested_outputs == ("price",)


def test_vanilla_family_ir_ignores_legacy_settlement_rule_mirror():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    analytical = make_vanilla_option_contract(
        description="EUR call on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
    )
    analytical = replace(
        analytical,
        product=replace(
            analytical.product,
            settlement_rule="legacy_rule_should_be_ignored",
            maturity_settlement_rule="legacy_rule_should_be_ignored",
        ),
    )
    analytical_bp = compile_semantic_contract(analytical)

    assert isinstance(analytical_bp.dsl_lowering.family_ir, AnalyticalBlack76IR)
    assert analytical_bp.dsl_lowering.route_id == "analytical_black76"

    pde = make_vanilla_option_contract(
        description="EUR put on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
        preferred_method="pde_solver",
    )
    pde = replace(
        pde,
        product=replace(
            pde.product,
            settlement_rule="",
            maturity_settlement_rule="",
        ),
    )
    pde_bp = compile_semantic_contract(pde, preferred_method="pde_solver")

    assert isinstance(pde_bp.dsl_lowering.family_ir, VanillaEquityPDEIR)
    assert pde_bp.dsl_lowering.route_id == "vanilla_equity_theta_pde"


def test_non_migrated_analytical_swaption_keeps_legacy_lowering_path():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="5Yx10Y USD payer swaption Black-76",
        observation_schedule=("2031-03-15",),
    )
    blueprint = compile_semantic_contract(contract, preferred_method="analytical")

    assert blueprint.dsl_lowering is not None
    assert blueprint.dsl_lowering.route_id == "analytical_black76"
    assert blueprint.dsl_lowering.family_ir is None


def test_callable_bond_compiles_to_exercise_lattice_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_callable_bond_contract

    contract = make_callable_bond_contract(
        description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        observation_schedule=("2026-01-15", "2027-01-15"),
    )
    blueprint = compile_semantic_contract(contract)

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, ExerciseLatticeIR)
    assert family_ir.route_id == "exercise_lattice"
    assert family_ir.product_instrument == "callable_bond"
    assert family_ir.control_style == "issuer_min"
    assert family_ir.helper_symbol == "price_callable_bond_tree"
    assert family_ir.observable_types == ("discount_curve", "cashflow_schedule")
    assert "coupon_accrual_fractions" in family_ir.derived_quantities


def test_bermudan_swaption_compiles_to_exercise_lattice_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="Bermudan payer swaption with annual exercise dates",
        observation_schedule=("2026-01-15", "2027-01-15", "2028-01-15"),
        preferred_method="rate_tree",
        exercise_style="bermudan",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="rate_tree")

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, ExerciseLatticeIR)
    assert family_ir.route_id == "exercise_lattice"
    assert family_ir.product_instrument == "swaption"
    assert family_ir.control_style == "holder_max"
    assert family_ir.helper_symbol == "price_bermudan_swaption_tree"
    assert "forward_rate" in family_ir.observable_types
    assert "par_rate_bindings" in family_ir.derived_quantities


def test_exercise_lattice_family_ir_ignores_legacy_settlement_rule_mirror():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import (
        make_callable_bond_contract,
        make_rate_style_swaption_contract,
    )

    callable_contract = make_callable_bond_contract(
        description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        observation_schedule=("2026-01-15", "2027-01-15"),
    )
    callable_contract = replace(
        callable_contract,
        product=replace(
            callable_contract.product,
            settlement_rule="",
            maturity_settlement_rule="",
        ),
    )
    callable_bp = compile_semantic_contract(callable_contract)

    assert isinstance(callable_bp.dsl_lowering.family_ir, ExerciseLatticeIR)
    assert callable_bp.dsl_lowering.family_ir.helper_symbol == "price_callable_bond_tree"

    swaption_contract = make_rate_style_swaption_contract(
        description="Bermudan payer swaption with annual exercise dates",
        observation_schedule=("2026-01-15", "2027-01-15", "2028-01-15"),
        preferred_method="rate_tree",
        exercise_style="bermudan",
    )
    swaption_contract = replace(
        swaption_contract,
        product=replace(
            swaption_contract.product,
            settlement_rule="wrong_legacy_mirror",
            maturity_settlement_rule="wrong_legacy_mirror",
        ),
    )
    swaption_bp = compile_semantic_contract(swaption_contract, preferred_method="rate_tree")

    assert isinstance(swaption_bp.dsl_lowering.family_ir, ExerciseLatticeIR)
    assert swaption_bp.dsl_lowering.family_ir.helper_symbol == "price_bermudan_swaption_tree"


def test_exercise_lattice_family_ir_rejects_settlement_before_decision_dates():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_callable_bond_contract

    contract = make_callable_bond_contract(
        description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        observation_schedule=("2026-01-15", "2027-01-15"),
    )
    contract = replace(
        contract,
        product=replace(
            contract.product,
            timeline=replace(
                contract.product.timeline,
                decision_dates=("2027-01-15",),
                settlement_dates=("2026-01-15",),
            ),
        ),
    )

    blueprint = compile_semantic_contract(contract)

    assert blueprint.dsl_lowering is not None
    assert blueprint.dsl_lowering.family_ir is None
    assert any(
        "settlement on or after the first decision date" in error
        for error in blueprint.dsl_lowering.admissibility_errors
    )


def test_ranked_observation_basket_compiles_to_correlated_basket_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract

    contract = make_ranked_observation_basket_contract(
        description="Himalaya on AAPL, MSFT, NVDA",
        constituents=("AAPL", "MSFT", "NVDA"),
        observation_schedule=("2025-06-15", "2025-12-15", "2026-06-15"),
    )
    blueprint = compile_semantic_contract(contract, requested_outputs=["price", "delta"])

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, CorrelatedBasketMonteCarloIR)
    assert family_ir.route_id == "correlated_basket_monte_carlo"
    assert family_ir.product_instrument == "basket_path_payoff"
    assert family_ir.payoff_family == "basket_path_payoff"
    assert family_ir.constituent_names == ("AAPL", "MSFT", "NVDA")
    assert family_ir.observable_types == ("spot_vector", "simple_return")
    assert family_ir.automatic_event_names == ("rank_and_select", "settle")
    assert family_ir.state_tags == ("pathwise_only", "remaining_pool", "locked_cashflow_state")
    assert family_ir.binding_sources == (
        ("discount_curve", "runtime_connector_resolution"),
        ("underlier_spots", "runtime_connector_resolution"),
        ("black_vol_surface", "runtime_connector_resolution"),
        ("correlation_matrix", "runtime_connector_resolution"),
    )
    assert family_ir.path_requirement_kind == "observation_snapshot_state"
    assert family_ir.required_fixing_schedule == ("2025-06-15", "2025-12-15", "2026-06-15")


def test_ranked_observation_basket_family_ir_ignores_legacy_event_transition_mirror():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract

    contract = make_ranked_observation_basket_contract(
        description="Himalaya on AAPL, MSFT, NVDA",
        constituents=("AAPL", "MSFT", "NVDA"),
        observation_schedule=("2025-06-15", "2025-12-15", "2026-06-15"),
    )
    contract = replace(
        contract,
        product=replace(
            contract.product,
            event_transitions=("legacy_wrong_event",),
        ),
    )
    blueprint = compile_semantic_contract(contract, requested_outputs=["price", "delta"])

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, CorrelatedBasketMonteCarloIR)
    assert family_ir.automatic_event_names == ("rank_and_select", "settle")
    assert "legacy_wrong_event" not in family_ir.automatic_event_names


def test_ranked_observation_basket_family_ir_rejects_missing_correlation_input():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract
    import pytest

    contract = make_ranked_observation_basket_contract(
        description="Himalaya on AAPL, MSFT",
        constituents=("AAPL", "MSFT"),
        observation_schedule=("2025-06-15", "2025-12-15"),
    )
    contract = replace(
        contract,
        market_data=replace(
            contract.market_data,
            required_inputs=tuple(
                item for item in contract.market_data.required_inputs if item.input_id != "correlation_matrix"
            ),
        ),
    )

    with pytest.raises(ValueError, match="correlation data"):
        compile_semantic_contract(contract)


def test_credit_default_swap_compiles_to_analytical_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_credit_default_swap_contract

    contract = make_credit_default_swap_contract(
        description="Single-name CDS on ACME with quarterly premium dates through 2027-06-20",
        observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20"),
    )
    blueprint = compile_semantic_contract(contract, requested_outputs=["price", "scenario_pnl"])

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, CreditDefaultSwapIR)
    assert family_ir.route_id == "credit_default_swap_analytical"
    assert family_ir.route_family == "credit_default_swap"
    assert family_ir.product_instrument == "cds"
    assert family_ir.payoff_family == "credit_default_swap"
    assert family_ir.pricing_mode == "analytical"
    assert family_ir.schedule_builder_symbol == "build_cds_schedule"
    assert family_ir.helper_symbol == "price_cds_analytical"
    assert family_ir.market_mapping == "discount_curve_credit_curve_to_cds_legs"
    assert family_ir.required_input_ids == blueprint.required_market_data
    assert family_ir.requested_outputs == ("price", "scenario_pnl")
    assert family_ir.payment_dates == ("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20")


def test_credit_default_swap_compiles_to_monte_carlo_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_credit_default_swap_contract

    contract = make_credit_default_swap_contract(
        description="Single-name CDS on ACME Monte Carlo",
        observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20"),
        preferred_method="monte_carlo",
    )
    blueprint = compile_semantic_contract(
        contract,
        preferred_method="monte_carlo",
        requested_outputs=["price"],
    )

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, CreditDefaultSwapIR)
    assert family_ir.route_id == "credit_default_swap_monte_carlo"
    assert family_ir.pricing_mode == "monte_carlo"
    assert family_ir.helper_symbol == "price_cds_monte_carlo"
    assert family_ir.payment_dates[-1] == "2027-06-20"


def test_nth_to_default_compiles_to_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_nth_to_default_contract

    contract = make_nth_to_default_contract(
        description="First-to-default basket on ACME, BRAVO, CHARLIE, DELTA, ECHO through 2029-11-15",
        observation_schedule=("2029-11-15",),
        reference_entities=("ACME", "BRAVO", "CHARLIE", "DELTA", "ECHO"),
        trigger_rank=1,
    )
    blueprint = compile_semantic_contract(contract, requested_outputs=["price", "scenario_pnl"])

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, NthToDefaultIR)
    assert family_ir.route_id == "nth_to_default_monte_carlo"
    assert family_ir.route_family == "nth_to_default"
    assert family_ir.product_instrument == "nth_to_default"
    assert family_ir.payoff_family == "nth_to_default"
    assert family_ir.helper_symbol == "price_nth_to_default_basket"
    assert family_ir.copula_symbol == "GaussianCopula"
    assert family_ir.trigger_rank == 1
    assert family_ir.reference_entities == ("ACME", "BRAVO", "CHARLIE", "DELTA", "ECHO")
    assert family_ir.required_input_ids == blueprint.required_market_data
    assert family_ir.requested_outputs == ("price", "scenario_pnl")
