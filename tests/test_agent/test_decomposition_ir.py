"""Tests for deterministic product decomposition into ProductIR."""

from __future__ import annotations


class TestDecompositionCharacterization:
    """Freeze the current static decomposition behavior before adding ProductIR."""

    def test_callable_bond_static_decomposition(self):
        from trellis.agent.knowledge.decompose import decompose

        decomp = decompose("Callable bond with semiannual coupon and call schedule")

        assert decomp.instrument == "callable_bond"
        assert decomp.method == "rate_tree"
        assert "callable" in decomp.features
        assert "mean_reversion" in decomp.features
        assert "discount" in decomp.required_market_data
        assert "black_vol" in decomp.required_market_data

    def test_american_put_static_decomposition(self):
        from trellis.agent.knowledge.decompose import decompose

        decomp = decompose("American put option on equity")

        assert decomp.instrument == "american_option"
        assert decomp.method == "monte_carlo"
        assert "early_exercise" in decomp.features


class TestProductIR:
    def test_ir_for_european_swaption(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir("European payer swaption")

        assert ir.instrument == "swaption"
        assert ir.payoff_family == "swaption"
        assert ir.exercise_style == "european"
        assert ir.state_dependence == "schedule_dependent"
        assert ir.schedule_dependence is True
        assert ir.model_family == "interest_rate"
        assert set(ir.candidate_engine_families) >= {"analytical"}
        assert ir.supported is True
        assert ir.unresolved_primitives == ()

    def test_ir_for_asian_option(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir("Build a pricer for: Geometric Asian option: closed-form vs MC")

        assert ir.instrument == "asian_option"
        assert ir.payoff_family == "asian_option"
        assert "asian" in ir.payoff_traits
        assert ir.exercise_style == "european"
        assert ir.state_dependence == "path_dependent"
        assert "monte_carlo" in ir.candidate_engine_families
        assert ir.supported is True

    def test_ir_for_callable_bond(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir("Callable bond with semiannual coupon and call schedule")

        assert ir.instrument == "callable_bond"
        assert ir.payoff_family == "callable_fixed_income"
        assert "callable" in ir.payoff_traits
        assert ir.exercise_style == "issuer_call"
        assert ir.schedule_dependence is True
        assert ir.state_dependence == "schedule_dependent"
        assert ir.model_family == "interest_rate"
        assert set(ir.candidate_engine_families) >= {"lattice", "exercise"}
        assert "rate_lattice" in ir.route_families
        assert ir.supported is True

    def test_ir_for_bermudan_swaption(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir("Bermudan swaption: tree vs LSM MC")

        assert ir.instrument == "bermudan_swaption"
        assert ir.payoff_family == "swaption"
        assert "bermudan" in ir.payoff_traits
        assert ir.exercise_style == "bermudan"
        assert ir.state_dependence == "schedule_dependent"
        assert set(ir.candidate_engine_families) >= {"lattice", "exercise"}
        assert "rate_lattice" in ir.route_families
        assert ir.supported is True

    def test_ir_for_american_put(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir("American put option on equity")

        assert ir.instrument == "american_put"
        assert ir.payoff_family == "vanilla_option"
        assert ir.exercise_style == "american"
        assert ir.state_dependence == "terminal_markov"
        assert ir.schedule_dependence is False
        assert ir.model_family == "equity_diffusion"
        assert set(ir.candidate_engine_families) >= {"monte_carlo", "exercise"}
        assert "equity_tree" in ir.route_families
        assert ir.supported is True

    def test_ir_for_zcb_option(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir("ZCB option: Ho-Lee vs HW tree vs Jamshidian analytical")

        assert ir.instrument == "zcb_option"
        assert ir.payoff_family == "zcb_option"
        assert ir.exercise_style == "european"
        assert ir.schedule_dependence is False
        assert ir.model_family == "interest_rate"
        assert "analytical" in ir.candidate_engine_families
        assert ir.supported is True

    def test_ir_normalizes_american_option_alias(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir("American option on equity", instrument_type="american_option")

        assert ir.instrument == "american_option"
        assert ir.exercise_style == "american"
        assert "exercise" in ir.candidate_engine_families
        assert "equity_tree" in ir.route_families

    def test_ir_reports_unresolved_composite_under_heston(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "American Asian barrier option under Heston with early exercise"
        )

        assert ir.payoff_family == "composite_option"
        assert set(ir.payoff_traits) >= {"asian", "barrier"}
        assert ir.exercise_style == "american"
        assert ir.state_dependence == "path_dependent"
        assert ir.model_family == "stochastic_volatility"
        assert set(ir.candidate_engine_families) >= {"monte_carlo", "exercise"}
        assert ir.supported is False
        assert "path_dependent_early_exercise_under_stochastic_vol" in ir.unresolved_primitives
