from __future__ import annotations

from datetime import date

from trellis.agent.dynamic_contract_ir import DynamicContractIR
from trellis.agent.insurance_overlay_contract import InsuranceOverlayContractIR
from trellis.agent.knowledge.decompose import (
    decompose_to_dynamic_contract_ir,
    decompose_to_insurance_overlay_contract_ir,
    decompose_to_ir,
    decompose_to_static_leg_contract_ir,
)
from trellis.agent.static_leg_contract import (
    CouponLeg,
    FixedCouponFormula,
    FloatingCouponFormula,
    KnownCashflowLeg,
    OvernightRateIndex,
    PeriodRateOptionStripLeg,
    StaticLegContractIR,
    TermRateIndex,
)


class TestDecomposeStaticAndDynamicContracts:
    def test_fixed_float_swap_decomposes_route_independently_to_static_leg_ir(self):
        description = (
            "Vanilla pay fixed USD IRS notional 1000000 fixed rate 4% "
            "effective 2025-06-30 maturity 2030-06-30 fixed semiannual "
            "float quarterly index SOFR"
        )
        product_ir = decompose_to_ir(description, instrument_type="swap")

        observed = decompose_to_static_leg_contract_ir(
            description,
            instrument_type="swap",
            product_ir=product_ir,
        )

        assert isinstance(observed, StaticLegContractIR)
        assert len(observed.legs) == 2
        fixed_leg = next(leg for leg in observed.legs if isinstance(leg.leg, CouponLeg) and isinstance(leg.leg.coupon_formula, FixedCouponFormula))
        floating_leg = next(leg for leg in observed.legs if isinstance(leg.leg, CouponLeg) and isinstance(leg.leg.coupon_formula, FloatingCouponFormula))
        assert fixed_leg.direction == "pay"
        assert fixed_leg.leg.coupon_formula.rate == 0.04
        assert isinstance(floating_leg.leg.coupon_formula.rate_index, OvernightRateIndex)
        assert floating_leg.leg.coupon_formula.rate_index.name == "SOFR"

    def test_basis_swap_and_bond_are_admitted_but_quote_spread_is_not(self):
        basis_description = (
            "SOFR-FF basis swap notional 1000000 effective 2025-06-30 maturity "
            "2030-06-30 pay SOFR quarterly receive FF quarterly plus 0.25%"
        )
        bond_description = (
            "Fixed coupon bond USD face 1000000 coupon 5% issue 2025-01-15 "
            "maturity 2030-01-15 semiannual day count ACT/ACT"
        )
        quote_description = (
            "Terminal curve-spread payoff on USD_SWAP par rate 10Y minus 2Y, "
            "notional 1000000, expiry 2026-06-30"
        )

        basis = decompose_to_static_leg_contract_ir(basis_description, instrument_type="swap")
        bond = decompose_to_static_leg_contract_ir(bond_description, instrument_type="bond")
        quote = decompose_to_static_leg_contract_ir(quote_description, instrument_type="quoted_observable")

        assert isinstance(basis, StaticLegContractIR)
        assert isinstance(bond, StaticLegContractIR)
        assert any(isinstance(leg.leg, KnownCashflowLeg) for leg in bond.legs)
        assert quote is None

    def test_cap_and_floor_requests_decompose_to_period_rate_option_strip_legs(self):
        cap_description = (
            "Price a cap strip under the declared benchmark rates surface. "
            "Instrument class: cap. Strike: 0.04. Notional: 1000000.0. "
            "Start date: 2024-11-15. End date: 2029-11-15. "
            "Payment frequency: quarterly. Day count: ACT/360. "
            "Rate index: USD-SOFR-3M."
        )
        floor_description = (
            "Build a pricer for: USD SOFR floor priced as a scheduled strip. "
            "Instrument: floor. Strike: 4%. Notional: 1000000. "
            "Start date: 2025-02-15. End date: 2030-02-15. Frequency: quarterly. "
            "Day count: Act/360. Rate index: USD-SOFR-3M."
        )

        cap_contract = decompose_to_static_leg_contract_ir(
            cap_description,
            instrument_type="rate_cap_floor_strip",
        )
        floor_contract = decompose_to_static_leg_contract_ir(
            floor_description,
            instrument_type="rate_cap_floor_strip",
        )

        assert isinstance(cap_contract, StaticLegContractIR)
        assert isinstance(floor_contract, StaticLegContractIR)
        cap_leg = cap_contract.legs[0].leg
        floor_leg = floor_contract.legs[0].leg
        assert isinstance(cap_leg, PeriodRateOptionStripLeg)
        assert isinstance(floor_leg, PeriodRateOptionStripLeg)
        assert cap_leg.option_side == "call"
        assert floor_leg.option_side == "put"
        assert cap_leg.metadata["semantic_family"] == "period_rate_option_strip"
        assert floor_leg.metadata["instrument_class"] == "floor"
        assert isinstance(cap_leg.rate_index, TermRateIndex)
        assert cap_leg.rate_index.name == "USD-SOFR"
        assert cap_leg.rate_index.tenor == "3M"

    def test_caplet_request_fails_closed_for_static_leg_strip_decomposition(self):
        description = "Caplet on SOFR strike 4% expiring 2025-11-15"

        observed = decompose_to_static_leg_contract_ir(
            description,
            instrument_type="cap",
        )

        assert observed is None

    def test_callable_bond_decomposes_to_dynamic_wrapper_over_static_leg_base(self):
        description = (
            "Issuer callable fixed coupon bond USD face 1000000 coupon 5% issue "
            "2025-01-15 maturity 2030-01-15 semiannual day count ACT/ACT "
            "call dates 2027-01-15, 2028-01-15, 2029-01-15"
        )

        observed = decompose_to_dynamic_contract_ir(
            description,
            instrument_type="callable_bond",
        )

        assert isinstance(observed, DynamicContractIR)
        assert isinstance(observed.base_contract, StaticLegContractIR)
        assert observed.semantic_family == "callable_bond"
        assert observed.base_track == "static_leg"
        assert observed.control_program is not None
        assert observed.control_program.controller_role == "issuer"
        assert observed.control_program.decision_event_labels == (
            "call_2027-01-15",
            "call_2028-01-15",
            "call_2029-01-15",
        )
        assert observed.event_program.buckets[0].event_date == date(2027, 1, 15)

    def test_automatic_dynamic_families_decompose_to_stateful_wrappers(self):
        autocall_description = (
            "Phoenix autocallable note on SPX notional 1000000 coupon 8% "
            "autocall barrier 100% observation dates 2025-07-15, 2026-01-15, "
            "2026-07-15, 2027-01-15 maturity 2027-01-15"
        )
        tarn_description = (
            "TARN on EURUSD notional 1000000 coupon 2% target 10% "
            "fixing dates 2025-03-31, 2025-06-30, 2025-09-30, 2025-12-31 "
            "maturity 2025-12-31"
        )

        autocall = decompose_to_dynamic_contract_ir(
            autocall_description,
            instrument_type="autocallable",
        )
        tarn = decompose_to_dynamic_contract_ir(
            tarn_description,
            instrument_type="tarn",
        )

        assert isinstance(autocall, DynamicContractIR)
        assert autocall.semantic_family == "autocallable"
        assert autocall.base_track == "payoff_expression"
        assert autocall.control_program is None
        assert autocall.event_program.termination_rules
        assert autocall.state_schema.field_names == ("coupon_memory",)

        assert isinstance(tarn, DynamicContractIR)
        assert tarn.semantic_family == "tarn"
        assert tarn.base_track == "payoff_expression"
        assert tarn.control_program is None
        assert tarn.event_program.termination_rules
        assert tarn.state_schema.field_names == ("accrued_coupon",)

    def test_discrete_and_continuous_control_families_decompose_to_dynamic_wrappers(self):
        swing_description = (
            "Swing option on power notional 1000000 rights 3 exercise dates "
            "2025-01-31, 2025-02-28, 2025-03-31, 2025-04-30 maturity 2025-04-30"
        )
        gmwb_description = (
            "GMWB contract premium 100000 guarantee base 100000 account value 100000 "
            "withdrawal dates 2026-01-15, 2027-01-15, 2028-01-15"
        )

        swing = decompose_to_dynamic_contract_ir(
            swing_description,
            instrument_type="swing_option",
        )
        gmwb = decompose_to_dynamic_contract_ir(
            gmwb_description,
            instrument_type="gmwb",
        )

        assert isinstance(swing, DynamicContractIR)
        assert swing.semantic_family == "swing_option"
        assert swing.base_track == "payoff_expression"
        assert swing.control_program is not None
        assert swing.control_program.inventory_fields == ("remaining_rights",)

        assert isinstance(gmwb, DynamicContractIR)
        assert gmwb.semantic_family == "gmwb"
        assert gmwb.base_track == "payoff_expression"
        assert gmwb.control_program is not None
        assert gmwb.control_program.admissible_actions[0].action_domain == "continuous"
        assert gmwb.control_program.admissible_actions[0].quantity_source == "withdrawal_amount"

    def test_gmwb_overlay_bearing_descriptions_fail_closed_before_dynamic_decomposition(self):
        overlay_description = (
            "GMWB contract premium 100000 guarantee base 100000 account value 100000 "
            "withdrawal dates 2026-01-15, 2027-01-15, 2028-01-15 mortality rider fee 1%"
        )

        assert (
            decompose_to_dynamic_contract_ir(
                overlay_description,
                instrument_type="gmwb",
            )
            is None
        )

    def test_gmwb_overlay_bearing_descriptions_decompose_to_overlay_wrapper(self):
        overlay_description = (
            "GMWB contract premium 100000 guarantee base 100000 account value 100000 "
            "withdrawal dates 2026-01-15, 2027-01-15, 2028-01-15 mortality rider fee 1%"
        )

        observed = decompose_to_insurance_overlay_contract_ir(
            overlay_description,
            instrument_type="gmwb",
        )

        assert isinstance(observed, InsuranceOverlayContractIR)
        assert observed.core_contract.semantic_family == "gmwb"
        assert observed.policy_state_schema.field_names == ("policy_status",)
        assert tuple(event.label for event in observed.overlay_events) == (
            "mortality_transition",
            "rider_fee",
        )
        assert observed.overlay_parameters.parameter_names == (
            "mortality_hazard",
            "rider_fee_rate",
        )

    def test_gmwb_overlay_fee_fallback_formula_reuses_declared_parameter_name(self):
        overlay_description = (
            "GMWB contract premium 100000 guarantee base 100000 account value 100000 "
            "withdrawal dates 2026-01-15, 2027-01-15, 2028-01-15 rider fee"
        )

        observed = decompose_to_insurance_overlay_contract_ir(
            overlay_description,
            instrument_type="gmwb",
        )

        assert isinstance(observed, InsuranceOverlayContractIR)
        fee_event = next(event for event in observed.overlay_events if event.label == "rider_fee")
        assert fee_event.fee_formula == "rider_fee_rate * account_value"
        assert "rider_fee_rate" in observed.overlay_parameters.parameter_names

    def test_dynamic_decomposition_rejects_static_and_quote_only_descriptions(self):
        assert (
            decompose_to_dynamic_contract_ir(
                "Fixed coupon bond USD face 1000000 coupon 5% issue 2025-01-15 maturity 2030-01-15 semiannual day count ACT/ACT",
                instrument_type="bond",
            )
            is None
        )
        assert (
            decompose_to_dynamic_contract_ir(
                "Terminal vol-skew payoff on SPX_IV black vol 1Y 90% moneyness minus 1Y 110% moneyness, notional 100000, expiry 2026-06-30",
                instrument_type="quoted_observable",
            )
            is None
        )
