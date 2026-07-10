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
        assert "discount_curve" in decomp.required_market_data
        assert "black_vol_surface" in decomp.required_market_data

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

    def test_ir_for_local_vol_option_uses_local_vol_model_family(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "European equity call under local vol: PDE vs MC",
            instrument_type="european_option",
        )

        assert ir.instrument == "european_option"
        assert ir.payoff_family == "vanilla_option"
        assert ir.model_family == "local_vol"
        assert {"pde", "monte_carlo"}.issubset(set(ir.candidate_engine_families))

    def test_ir_for_barrier_option_includes_promoted_analytical_support(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "European down-and-out call barrier option on SPX with strike 100, barrier 90, expiry 2025-11-15",
            instrument_type="barrier_option",
        )

        assert ir.instrument == "barrier_option"
        assert ir.payoff_family == "barrier_option"
        assert ir.exercise_style == "european"
        assert ir.model_family == "equity_diffusion"
        assert "barrier" in ir.payoff_traits
        assert "single_barrier" in ir.payoff_traits
        assert "double_barrier" not in ir.payoff_traits
        assert set(ir.candidate_engine_families) >= {"analytical", "monte_carlo", "pde"}
        assert "analytical" in ir.route_families
        assert "pde_solver" in ir.route_families
        assert ir.supported is True

    def test_ir_for_merton_jump_diffusion_keeps_vanilla_product_shape(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "Merton jump-diffusion MC vs FFT",
            instrument_type="european_option",
        )

        assert ir.instrument == "european_option"
        assert ir.payoff_family == "vanilla_option"
        assert "jump_diffusion" in ir.payoff_traits
        assert ir.model_family == "jump_diffusion"
        assert "jump_parameters" in ir.required_market_data
        assert "monte_carlo" in ir.route_families
        assert "fft_pricing" in ir.route_families

    def test_ir_for_sabr_hagan_mc_keeps_vanilla_forward_option_shape(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "SABR MC simulation vs Hagan implied vol",
            instrument_type="european_option",
        )

        assert ir.instrument == "european_option"
        assert ir.payoff_family == "vanilla_option"
        assert "sabr" in ir.payoff_traits
        assert ir.model_family == "sabr"
        assert "model_parameters" in ir.required_market_data
        assert "monte_carlo" in ir.route_families
        assert "analytical" in ir.route_families

    def test_ir_for_variance_gamma_keeps_vanilla_option_shape(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "Variance Gamma: COS vs MC",
            instrument_type="european_option",
        )

        assert ir.instrument == "european_option"
        assert ir.payoff_family == "vanilla_option"
        assert "variance_gamma" in ir.payoff_traits
        assert ir.model_family == "variance_gamma"
        assert "model_parameters" in ir.required_market_data
        assert "fft_pricing" in ir.route_families
        assert "monte_carlo" in ir.route_families

    def test_ir_for_cgmy_keeps_vanilla_option_shape(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "CGMY / tempered stable process via COS",
            instrument_type="european_option",
        )

        assert ir.instrument == "european_option"
        assert ir.payoff_family == "vanilla_option"
        assert "cgmy" in ir.payoff_traits
        assert ir.model_family == "cgmy"
        assert "model_parameters" in ir.required_market_data
        assert "fft_pricing" in ir.route_families
        assert "monte_carlo" in ir.route_families

    def test_ir_for_kou_keeps_vanilla_option_shape(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "Kou double-exponential jump: FFT vs MC",
            instrument_type="european_option",
        )

        assert ir.instrument == "european_option"
        assert ir.payoff_family == "vanilla_option"
        assert "kou" in ir.payoff_traits
        assert "double_exponential_jump" in ir.payoff_traits
        assert ir.model_family == "kou"
        assert "model_parameters" in ir.required_market_data
        assert "black_vol_surface" not in ir.required_market_data
        assert "fft_pricing" in ir.route_families
        assert "monte_carlo" in ir.route_families
        assert "analytical" in ir.route_families
        assert "trellis.models.levy_option" in ir.reusable_primitives

    def test_ir_for_bates_keeps_vanilla_option_shape(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "Bates model (Heston + jumps): FFT vs MC",
            instrument_type="european_option",
        )

        assert ir.instrument == "european_option"
        assert ir.payoff_family == "vanilla_option"
        assert "affine_jump_stochastic_vol" in ir.payoff_traits
        assert ir.model_family == "bates"
        assert "model_parameters" in ir.required_market_data
        assert "jump_parameters" in ir.required_market_data
        assert "black_vol_surface" not in ir.required_market_data
        assert "fft_pricing" in ir.route_families
        assert "monte_carlo" in ir.route_families
        assert "trellis.models.bates_option" in ir.reusable_primitives

    def test_ir_for_short_rate_bond_uses_affine_rate_model_shape(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "Vasicek bond pricing: tree vs analytical",
            instrument_type="short_rate_bond",
        )

        assert ir.instrument == "short_rate_bond"
        assert ir.payoff_family == "discount_bond"
        assert "short_rate_model" in ir.payoff_traits
        assert ir.model_family == "interest_rate"
        assert "discount_curve" in ir.required_market_data
        assert "model_parameters" in ir.required_market_data
        assert "analytical" in ir.route_families
        assert "rate_tree" in ir.route_families
        assert "trellis.models.short_rate_bond" in ir.reusable_primitives

    def test_ir_for_absorbed_analytical_exotics_uses_specific_payoff_families(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        cases = (
            (
                "Cash-or-nothing digital call on AAPL paying 1 if spot exceeds 150 at expiry",
                "digital_option",
                "digital_option",
            ),
            (
                "Fixed-strike lookback option on AAPL expiring 2025-11-15",
                "lookback_option",
                "lookback_option",
            ),
            (
                "Chooser option on AAPL strike 150 expiring 2025-11-15",
                "chooser_option",
                "chooser_option",
            ),
            (
                "Compound option on AAPL strike 150 expiring 2025-11-15",
                "compound_option",
                "compound_option",
            ),
            (
                "Cliquet option on AAPL with annual reset dates",
                "cliquet_option",
                "cliquet_option",
            ),
        )

        for description, instrument_type, expected_family in cases:
            ir = decompose_to_ir(description, instrument_type=instrument_type)
            assert ir.instrument == instrument_type
            assert ir.payoff_family == expected_family
            assert ir.exercise_style == "european"
            inferred = decompose_to_ir(description)
            assert inferred.instrument == instrument_type
            assert inferred.payoff_family == expected_family
            assert inferred.exercise_style == "european"

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

    def test_ir_for_nth_to_default_uses_specific_payoff_family(self):
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "First-to-default basket on five names with Gaussian copula",
            instrument_type="nth_to_default",
        )

        assert ir.instrument == "nth_to_default"
        assert ir.payoff_family == "nth_to_default"
        assert ir.model_family == "credit_copula"
        assert "nth_to_default" in ir.route_families
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
