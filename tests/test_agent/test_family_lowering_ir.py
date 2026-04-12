"""Tests for typed family lowering IR construction."""

from __future__ import annotations

from dataclasses import replace

from trellis.agent.backend_bindings import ResolvedBackendBindingSpec
from trellis.agent.codegen_guardrails import PrimitiveRef
from trellis.agent.family_lowering_ir import (
    AnalyticalBlack76IR,
    CorrelatedBasketMonteCarloIR,
    CreditDefaultSwapIR,
    EventAwarePDEIR,
    EventAwareMonteCarloIR,
    ExerciseLatticeIR,
    MCCalibrationBindingSpec,
    MCControlSpec,
    MCEventSpec,
    MCEventTimeSpec,
    MCMeasureSpec,
    MCPathRequirementSpec,
    MCPayoffReducerSpec,
    MCProcessSpec,
    MCStateSpec,
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


def _resolved_binding_spec(
    *primitives: PrimitiveRef,
    route_id: str,
    route_family: str,
    engine_family: str,
) -> ResolvedBackendBindingSpec:
    primitive_refs = tuple(
        dict.fromkeys(f"{primitive.module}.{primitive.symbol}" for primitive in primitives)
    )

    def refs_for_role(role: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                f"{primitive.module}.{primitive.symbol}"
                for primitive in primitives
                if primitive.role == role
            )
        )

    return ResolvedBackendBindingSpec(
        route_id=route_id,
        engine_family=engine_family,
        route_family=route_family,
        binding_id=f"{engine_family}.{route_id}",
        primitives=tuple(primitives),
        primitive_refs=primitive_refs,
        helper_refs=refs_for_role("route_helper"),
        pricing_kernel_refs=refs_for_role("pricing_kernel"),
        schedule_builder_refs=refs_for_role("schedule_builder"),
        cashflow_engine_refs=refs_for_role("cashflow_engine"),
        market_binding_refs=refs_for_role("market_binding"),
        exact_target_refs=primitive_refs,
    )


def _patch_binding(monkeypatch, binding_spec: ResolvedBackendBindingSpec) -> None:
    import trellis.agent.family_lowering_ir as family_lowering_ir

    monkeypatch.setattr(
        family_lowering_ir,
        "_resolve_family_lowering_binding",
        lambda route_id, *, product_ir=None: (
            binding_spec if route_id == binding_spec.route_id else None
        ),
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
    assert isinstance(family_ir, EventAwarePDEIR)
    assert isinstance(family_ir, VanillaEquityPDEIR)
    assert family_ir.route_id == "vanilla_equity_theta_pde"
    assert family_ir.product_instrument == "european_option"
    assert family_ir.payoff_family == "vanilla_option"
    assert family_ir.option_type == "put"
    assert family_ir.theta == 0.5
    assert family_ir.helper_symbol == "price_vanilla_equity_option_pde"
    assert family_ir.market_mapping == "equity_spot_discount_black_vol"
    assert family_ir.state_spec.state_variable == "spot"
    assert family_ir.state_spec.dimension == 1
    assert family_ir.state_spec.state_tags == ("terminal_markov", "recombining_safe")
    assert family_ir.operator_spec.operator_family == "black_scholes_1d"
    assert family_ir.operator_spec.solver_family == "theta_method"
    assert family_ir.control_spec.control_style == "identity"
    assert family_ir.event_transform_kinds == ()
    assert family_ir.boundary_spec.terminal_condition_kind == "expiry_payoff"
    assert family_ir.compatibility_wrapper == "VanillaEquityPDEIR"
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

    assert isinstance(pde_bp.dsl_lowering.family_ir, EventAwarePDEIR)
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


def test_rate_cap_floor_strip_analytical_compiles_to_black76_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_cap_floor_strip_contract

    contract = make_rate_cap_floor_strip_contract(
        description="5Y cap on SOFR under Black caplet strip pricing",
        instrument_class="cap",
        observation_schedule=("cap_schedule_placeholder",),
        preferred_method="analytical",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="analytical")

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, AnalyticalBlack76IR)
    assert family_ir.route_id == "analytical_black76"
    assert family_ir.product_instrument == "cap"
    assert family_ir.payoff_family == "rate_cap_floor_strip"
    assert family_ir.option_type == "call"
    assert family_ir.kernel_symbol == "black76_call"
    assert family_ir.market_mapping == "discount_curve_forward_curve_black_vol_to_caplet_strip"


def test_rate_style_swaption_monte_carlo_compiles_to_event_aware_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="European payer swaption under Hull-White Monte Carlo",
        observation_schedule=("2029-11-15",),
        preferred_method="monte_carlo",
        exercise_style="european",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="monte_carlo")

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, EventAwareMonteCarloIR)
    assert family_ir.route_id == "monte_carlo_paths"
    assert family_ir.product_instrument == "swaption"
    assert family_ir.payoff_family == "swaption"
    assert family_ir.state_spec.state_variable == "short_rate"
    assert family_ir.state_spec.dimension == 1
    assert "schedule_state" in family_ir.state_spec.state_tags
    assert "recombining_safe" in family_ir.state_spec.state_tags
    assert family_ir.process_spec.process_family == "hull_white_1f"
    assert family_ir.process_spec.simulation_scheme == "exact_ou"
    assert family_ir.control_spec.control_style == "identity"
    assert family_ir.control_spec.controller_role == "holder"
    assert family_ir.control_program.control_style == "holder_max"
    assert family_ir.control_program.controller_role == "holder"
    assert family_ir.control_program.decision_phase == "decision"
    assert family_ir.control_program.schedule_role == "decision_dates"
    assert family_ir.path_requirement_spec.requirement_kind == "event_replay"
    assert family_ir.path_requirement_spec.replay_mode == "deterministic_timeline"
    assert "exercise_date" in family_ir.path_requirement_spec.stored_fields
    assert "swap_rate" in family_ir.path_requirement_spec.stored_fields
    assert family_ir.payoff_reducer_spec.reducer_kind == "swaption_exercise_payoff"
    assert family_ir.event_dates == ("2029-11-15",)
    assert family_ir.event_kinds == ("observation", "exercise", "settlement")
    assert family_ir.event_program.event_dates == ("2029-11-15",)
    assert family_ir.event_program.event_kinds == ("observation", "exercise", "settlement")
    first_bucket = family_ir.event_timeline[0]
    assert first_bucket.schedule_roles == ("observation_dates", "settlement_dates")
    assert first_bucket.phase_sequence == ("observation", "settlement")
    assert tuple(event.event_name for event in first_bucket.events) == (
        "forward_swap_rate",
        "discount_curve_state",
        "price_swaption_at_exercise",
        "settle_at_exercise",
        "exercise_cash_settlement",
    )


def test_rate_cap_floor_strip_monte_carlo_compiles_to_event_aware_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_rate_cap_floor_strip_contract

    contract = make_rate_cap_floor_strip_contract(
        description="Black floorlet strip vs Hull-White Monte Carlo",
        instrument_class="floor",
        observation_schedule=("floor_schedule_placeholder",),
        preferred_method="monte_carlo",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="monte_carlo")

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, EventAwareMonteCarloIR)
    assert family_ir.route_id == "monte_carlo_paths"
    assert family_ir.product_instrument == "floor"
    assert family_ir.payoff_family == "rate_cap_floor_strip"
    assert family_ir.state_spec.state_variable == "short_rate"
    assert family_ir.process_spec.process_family == "hull_white_1f"
    assert family_ir.path_requirement_spec.requirement_kind == "event_replay"
    assert family_ir.payoff_reducer_spec.reducer_kind == "period_option_cashflow_strip"
    assert family_ir.market_mapping == "discount_curve_forward_curve_black_vol_to_rate_option_strip_mc"


def test_vanilla_option_monte_carlo_compiles_to_terminal_only_event_aware_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR call on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
        preferred_method="monte_carlo",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="monte_carlo")

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, EventAwareMonteCarloIR)
    assert family_ir.route_id == "monte_carlo_paths"
    assert family_ir.product_instrument == "european_option"
    assert family_ir.payoff_family == "vanilla_option"
    assert family_ir.helper_symbol == "price_vanilla_equity_option_monte_carlo"
    assert family_ir.state_spec.state_variable == "spot"
    assert family_ir.process_spec.process_family == "gbm_1d"
    assert family_ir.path_requirement_spec.requirement_kind == "terminal_only"


def test_vanilla_option_transform_compiles_to_bounded_transform_family_ir():
    from dataclasses import replace

    from trellis.agent.family_lowering_ir import build_family_lowering_ir
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import ControllerProtocol, make_vanilla_option_contract

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

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, TransformPricingIR)
    assert family_ir.route_id == "transform_fft"
    assert family_ir.route_family == "fft_pricing"
    assert family_ir.product_instrument == "european_option"
    assert family_ir.payoff_family == "vanilla_option"
    assert family_ir.state_spec.state_variable == "spot"
    assert family_ir.state_spec.dimension == 1
    assert family_ir.state_spec.state_tags == ("terminal_markov",)
    assert family_ir.characteristic_spec.model_family == "equity_diffusion"
    assert family_ir.characteristic_spec.characteristic_family == "gbm_log_spot"
    assert family_ir.characteristic_spec.supported_methods == ("fft", "cos")
    assert family_ir.control_spec.control_style == "identity"
    assert family_ir.control_spec.controller_role == "holder"
    assert family_ir.control_program.control_style == "holder_max"
    assert family_ir.control_program.controller_role == "holder"
    assert family_ir.control_program.schedule_role == "decision_dates"
    assert family_ir.terminal_payoff_kind == "vanilla_terminal_payoff"
    assert family_ir.strike_semantics == "vanilla_strike"
    assert family_ir.quote_semantics == "equity_black_vol_surface"
    assert family_ir.helper_symbol == "price_vanilla_equity_option_transform"
    assert family_ir.market_mapping == "single_state_diffusion_transform_inputs"
    assert family_ir.timeline_roles == blueprint.dsl_lowering.normalized_expr.signature.timeline_roles
    assert family_ir.required_input_ids == blueprint.required_market_data

    unsupported_contract = replace(
        transform_contract,
        product=replace(
            transform_contract.product,
            controller_protocol=ControllerProtocol(
                controller_style="issuer_min",
                controller_role="issuer",
                decision_phase="decision",
                schedule_role="decision_dates",
            ),
        ),
    )
    unsupported_family_ir = build_family_lowering_ir(
        unsupported_contract,
        route_id="transform_fft",
        route_family="fft_pricing",
        product_ir=blueprint.product_ir,
    )

    assert unsupported_family_ir is None


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
    assert family_ir.control_program.control_style == "issuer_min"
    assert family_ir.control_program.controller_role == "issuer"
    assert family_ir.helper_symbol == "price_callable_bond_tree"
    assert family_ir.event_program.event_dates == ("2026-01-15", "2027-01-15")
    assert "add_cashflow" in family_ir.event_program.transform_kinds
    assert "project_min" in family_ir.event_program.transform_kinds
    assert family_ir.observable_types == ("discount_curve", "cashflow_schedule")
    assert "coupon_accrual_fractions" in family_ir.derived_quantities


def test_callable_bond_pde_compiles_to_event_aware_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_callable_bond_contract

    contract = make_callable_bond_contract(
        description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        observation_schedule=("2026-01-15", "2027-01-15"),
        preferred_method="pde_solver",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="pde_solver")

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, EventAwarePDEIR)
    assert not isinstance(family_ir, VanillaEquityPDEIR)
    assert family_ir.route_id == "pde_theta_1d"
    assert family_ir.product_instrument == "callable_bond"
    assert family_ir.operator_spec.operator_family == "hull_white_1f"
    assert family_ir.state_spec.state_variable == "short_rate"
    assert family_ir.state_spec.state_tags == ("terminal_markov", "recombining_safe")
    assert family_ir.control_spec.control_style == "issuer_min"
    assert family_ir.control_program.control_style == "issuer_min"
    assert family_ir.control_program.schedule_role == "decision_dates"
    assert family_ir.event_transform_kinds == ("add_cashflow", "project_min")
    assert family_ir.event_dates == ("2026-01-15", "2027-01-15")
    assert family_ir.event_program.event_dates == ("2026-01-15", "2027-01-15")
    assert "add_cashflow" in family_ir.event_program.transform_kinds
    assert "project_min" in family_ir.event_program.transform_kinds
    assert family_ir.helper_symbol == "price_callable_bond_pde"
    first_bucket = family_ir.event_timeline[0]
    assert first_bucket.schedule_roles == ("determination_dates", "decision_dates")
    assert first_bucket.phase_sequence == ("determination", "decision")
    assert tuple(transform.transform_kind for transform in first_bucket.transforms) == (
        "add_cashflow",
        "project_min",
    )
    assert family_ir.boundary_spec.terminal_condition_kind == "cashflow_terminal_value"


def test_holder_max_equity_pde_compiles_to_event_aware_helper_family_ir():
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract

    contract = _make_bermudan_equity_pde_contract()
    blueprint = compile_semantic_contract(contract, preferred_method="pde_solver")

    family_ir = blueprint.dsl_lowering.family_ir
    assert isinstance(family_ir, EventAwarePDEIR)
    assert not isinstance(family_ir, VanillaEquityPDEIR)
    assert family_ir.route_id == "pde_theta_1d"
    assert family_ir.product_instrument == "american_option"
    assert family_ir.operator_spec.operator_family == "black_scholes_1d"
    assert family_ir.state_spec.state_variable == "spot"
    assert family_ir.control_spec.control_style == "holder_max"
    assert family_ir.control_program.control_style == "holder_max"
    assert family_ir.event_transform_kinds == ("project_max",)
    assert family_ir.helper_symbol == "price_event_aware_equity_option_pde"
    assert family_ir.event_dates == (
        "2026-03-20",
        "2026-06-20",
        "2026-09-20",
        "2026-12-20",
    )
    assert family_ir.event_program.event_dates == (
        "2026-03-20",
        "2026-06-20",
        "2026-09-20",
        "2026-12-20",
    )
    assert "project_max" in family_ir.event_program.transform_kinds


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


def test_event_aware_monte_carlo_ir_collects_typed_event_and_path_contracts():
    family_ir = EventAwareMonteCarloIR(
        route_id="monte_carlo_paths",
        route_family="monte_carlo",
        product_instrument="swaption",
        payoff_family="swaption",
        state_spec=MCStateSpec(
            state_variable="short_rate",
            dimension=1,
            state_tags=("terminal_markov", "schedule_state"),
        ),
        process_spec=MCProcessSpec(
            process_family="hull_white_1f",
            simulation_scheme="exact_ou",
        ),
        path_requirement_spec=MCPathRequirementSpec(
            requirement_kind="event_replay",
            reducer_kinds=("discounted_swap_pv",),
            stored_fields=("exercise_state",),
        ),
        payoff_reducer_spec=MCPayoffReducerSpec(
            reducer_kind="positive_part_at_exercise",
            output_semantics="swaption_exercise_payoff",
            event_dependencies=("exercise", "settlement"),
        ),
        control_spec=MCControlSpec(
            control_style="identity",
            controller_role="holder",
        ),
        measure_spec=MCMeasureSpec(
            measure_family="risk_neutral",
            numeraire_binding="discount_curve",
        ),
        calibration_binding=MCCalibrationBindingSpec(
            model_family="hull_white_1f",
            quote_family="black_swaption_vol",
            required_parameters=("mean_reversion", "sigma"),
            requires_quote_normalization=True,
        ),
        event_timeline=(
            MCEventTimeSpec(
                event_date="2027-03-15",
                schedule_roles=("observation_dates", "settlement_dates"),
                phase_sequence=("observation", "settlement"),
                events=(
                    MCEventSpec(
                        event_name="exercise",
                        event_kind="observation",
                        schedule_role="observation_dates",
                        phase="observation",
                        value_semantics="forward_swap_rate",
                    ),
                    MCEventSpec(
                        event_name="settlement",
                        event_kind="settlement",
                        schedule_role="settlement_dates",
                        phase="settlement",
                        value_semantics="cash_settlement",
                    ),
                ),
            ),
        ),
    )

    assert family_ir.state_spec.state_variable == "short_rate"
    assert family_ir.process_spec.process_family == "hull_white_1f"
    assert family_ir.path_requirement_spec.requirement_kind == "event_replay"
    assert family_ir.payoff_reducer_spec.reducer_kind == "positive_part_at_exercise"
    assert family_ir.event_kinds == ("observation", "settlement")
    assert family_ir.event_dates == ("2027-03-15",)
    assert family_ir.reducer_kinds == (
        "discounted_swap_pv",
        "positive_part_at_exercise",
    )


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


def test_black76_family_ir_dispatches_from_binding_surface_not_route_id(monkeypatch):
    from trellis.agent.family_lowering_ir import build_family_lowering_ir
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR call on AAPL, K=150, T=1y",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
    )
    blueprint = compile_semantic_contract(contract)
    synthetic_route_id = "binding_black76_terminal"
    _patch_binding(
        monkeypatch,
        _resolved_binding_spec(
            PrimitiveRef("trellis.models.black", "black76_call", "pricing_kernel"),
            PrimitiveRef("trellis.models.black", "black76_put", "pricing_kernel"),
            PrimitiveRef(
                "trellis.models.analytical",
                "terminal_vanilla_from_basis",
                "assembly_helper",
                required=False,
            ),
            PrimitiveRef("trellis.models.time", "year_fraction", "time_measure", required=False),
            route_id=synthetic_route_id,
            route_family="analytical",
            engine_family="analytical",
        ),
    )

    family_ir = build_family_lowering_ir(
        contract,
        route_id=synthetic_route_id,
        route_family="legacy_route_family_should_not_matter",
        product_ir=blueprint.product_ir,
    )

    assert isinstance(family_ir, AnalyticalBlack76IR)
    assert family_ir.route_id == synthetic_route_id
    assert family_ir.route_family == "analytical"
    assert family_ir.kernel_symbol == "black76_call"


def test_exercise_lattice_dispatches_from_binding_surface_not_route_id(monkeypatch):
    from trellis.agent.family_lowering_ir import build_family_lowering_ir
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_callable_bond_contract

    contract = make_callable_bond_contract(
        description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        observation_schedule=("2026-01-15", "2027-01-15"),
    )
    blueprint = compile_semantic_contract(contract)
    synthetic_route_id = "binding_tree_callable"
    _patch_binding(
        monkeypatch,
        _resolved_binding_spec(
            PrimitiveRef(
                "trellis.models.trees.callable_bond",
                "price_callable_bond_tree",
                "route_helper",
            ),
            PrimitiveRef(
                "trellis.models.trees.lattice",
                "build_rate_lattice",
                "lattice_builder",
            ),
            PrimitiveRef(
                "trellis.models.trees.lattice",
                "lattice_backward_induction",
                "backward_induction",
            ),
            PrimitiveRef(
                "trellis.models.trees.control",
                "resolve_lattice_exercise_policy",
                "control_policy",
            ),
            route_id=synthetic_route_id,
            route_family="lattice",
            engine_family="lattice",
        ),
    )

    family_ir = build_family_lowering_ir(
        contract,
        route_id=synthetic_route_id,
        route_family="legacy_route_family_should_not_matter",
        product_ir=blueprint.product_ir,
    )

    assert isinstance(family_ir, ExerciseLatticeIR)
    assert family_ir.route_id == synthetic_route_id
    assert family_ir.route_family == "lattice"
    assert family_ir.helper_symbol == "price_callable_bond_tree"


def test_event_aware_pde_dispatches_from_binding_surface_not_route_id(monkeypatch):
    from trellis.agent.family_lowering_ir import build_family_lowering_ir
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_callable_bond_contract

    contract = make_callable_bond_contract(
        description="Callable bond with annual coupons and issuer call dates 2026-01-15, 2027-01-15",
        observation_schedule=("2026-01-15", "2027-01-15"),
        preferred_method="pde_solver",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="pde_solver")
    synthetic_route_id = "binding_callable_pde"
    _patch_binding(
        monkeypatch,
        _resolved_binding_spec(
            PrimitiveRef("trellis.models.pde.grid", "Grid", "grid"),
            PrimitiveRef(
                "trellis.models.pde.operator",
                "BlackScholesOperator",
                "spatial_operator",
            ),
            PrimitiveRef(
                "trellis.models.pde.theta_method",
                "theta_method_1d",
                "time_stepping",
            ),
            PrimitiveRef(
                "trellis.models.callable_bond_pde",
                "price_callable_bond_pde",
                "route_helper",
            ),
            route_id=synthetic_route_id,
            route_family="pde_solver",
            engine_family="pde_solver",
        ),
    )

    family_ir = build_family_lowering_ir(
        contract,
        route_id=synthetic_route_id,
        route_family="legacy_route_family_should_not_matter",
        product_ir=blueprint.product_ir,
    )

    assert isinstance(family_ir, EventAwarePDEIR)
    assert family_ir.route_id == synthetic_route_id
    assert family_ir.route_family == "pde_solver"
    assert family_ir.helper_symbol == "price_callable_bond_pde"


def test_local_vol_monte_carlo_dispatches_from_binding_surface_not_route_id(monkeypatch):
    from trellis.agent.family_lowering_ir import build_family_lowering_ir
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_vanilla_option_contract

    contract = make_vanilla_option_contract(
        description="EUR call on AAPL under local-vol Monte Carlo",
        underliers=("AAPL",),
        observation_schedule=("2026-06-20",),
        preferred_method="monte_carlo",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="monte_carlo")
    synthetic_route_id = "binding_local_vol_mc"
    _patch_binding(
        monkeypatch,
        _resolved_binding_spec(
            PrimitiveRef(
                "trellis.models.processes.local_vol",
                "LocalVol",
                "state_process",
            ),
            PrimitiveRef(
                "trellis.models.monte_carlo.engine",
                "MonteCarloEngine",
                "path_simulation",
            ),
            PrimitiveRef(
                "trellis.models.local_vol",
                "local_vol_european_vanilla_price",
                "pricing_kernel",
            ),
            route_id=synthetic_route_id,
            route_family="monte_carlo",
            engine_family="monte_carlo",
        ),
    )

    family_ir = build_family_lowering_ir(
        contract,
        route_id=synthetic_route_id,
        route_family="legacy_route_family_should_not_matter",
        product_ir=blueprint.product_ir,
    )

    assert isinstance(family_ir, EventAwareMonteCarloIR)
    assert family_ir.route_id == synthetic_route_id
    assert family_ir.route_family == "monte_carlo"
    assert family_ir.process_spec.process_family == "local_vol_1d"
    assert family_ir.process_spec.simulation_scheme == "euler_local_vol"


def test_credit_default_swap_dispatches_from_binding_surface_not_route_id(monkeypatch):
    from trellis.agent.family_lowering_ir import build_family_lowering_ir
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_credit_default_swap_contract

    contract = make_credit_default_swap_contract(
        description="Single-name CDS on ACME Monte Carlo",
        observation_schedule=("2026-06-20", "2026-09-20", "2026-12-20", "2027-03-20", "2027-06-20"),
        preferred_method="monte_carlo",
    )
    blueprint = compile_semantic_contract(contract, preferred_method="monte_carlo")
    synthetic_route_id = "binding_cds_mc"
    _patch_binding(
        monkeypatch,
        _resolved_binding_spec(
            PrimitiveRef(
                "trellis.models.credit_schedule",
                "build_cds_schedule",
                "schedule_builder",
            ),
            PrimitiveRef(
                "trellis.models.credit_survival",
                "interval_default_probability",
                "event_probability",
            ),
            PrimitiveRef("trellis.models.cds", "price_cds_monte_carlo", "route_helper"),
            PrimitiveRef(
                "trellis.core.differentiable",
                "get_numpy",
                "array_backend",
            ),
            route_id=synthetic_route_id,
            route_family="credit_default_swap",
            engine_family="monte_carlo",
        ),
    )

    family_ir = build_family_lowering_ir(
        contract,
        route_id=synthetic_route_id,
        route_family="legacy_route_family_should_not_matter",
        product_ir=blueprint.product_ir,
    )

    assert isinstance(family_ir, CreditDefaultSwapIR)
    assert family_ir.route_id == synthetic_route_id
    assert family_ir.route_family == "credit_default_swap"
    assert family_ir.pricing_mode == "monte_carlo"
    assert family_ir.helper_symbol == "price_cds_monte_carlo"


def test_nth_to_default_dispatches_from_binding_surface_not_route_id(monkeypatch):
    from trellis.agent.family_lowering_ir import build_family_lowering_ir
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract
    from trellis.agent.semantic_contracts import make_nth_to_default_contract

    contract = make_nth_to_default_contract(
        description="First-to-default basket on ACME, BRAVO, CHARLIE, DELTA, ECHO through 2029-11-15",
        observation_schedule=("2029-11-15",),
        reference_entities=("ACME", "BRAVO", "CHARLIE", "DELTA", "ECHO"),
        trigger_rank=1,
    )
    blueprint = compile_semantic_contract(contract)
    synthetic_route_id = "binding_nth_to_default"
    _patch_binding(
        monkeypatch,
        _resolved_binding_spec(
            PrimitiveRef(
                "trellis.models.schedule",
                "generate_schedule",
                "schedule_builder",
            ),
            PrimitiveRef(
                "trellis.models.time",
                "year_fraction",
                "time_measure",
            ),
            PrimitiveRef(
                "trellis.models.copula",
                "GaussianCopula",
                "default_time_sampler",
            ),
            PrimitiveRef(
                "trellis.models.nth_to_default",
                "price_nth_to_default_basket",
                "route_helper",
            ),
            PrimitiveRef(
                "trellis.models.monte_carlo.engine",
                "MonteCarloEngine",
                "path_simulation",
            ),
            route_id=synthetic_route_id,
            route_family="nth_to_default",
            engine_family="monte_carlo",
        ),
    )

    family_ir = build_family_lowering_ir(
        contract,
        route_id=synthetic_route_id,
        route_family="legacy_route_family_should_not_matter",
        product_ir=blueprint.product_ir,
    )

    assert isinstance(family_ir, NthToDefaultIR)
    assert family_ir.route_id == synthetic_route_id
    assert family_ir.route_family == "nth_to_default"
    assert family_ir.helper_symbol == "price_nth_to_default_basket"
    assert family_ir.copula_symbol == "GaussianCopula"
