from __future__ import annotations

from datetime import date

from trellis.agent.dynamic_contract_ir import DynamicContractIR
from trellis.agent.knowledge.decompose import (
    decompose_to_dynamic_contract_ir,
    decompose_to_ir,
    decompose_to_static_leg_contract_ir,
)
from trellis.agent.static_leg_contract import (
    CouponLeg,
    FixedCouponFormula,
    FloatingCouponFormula,
    KnownCashflowLeg,
    OvernightRateIndex,
    StaticLegContractIR,
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
        assert observed.control_program is not None
        assert observed.control_program.controller_role == "issuer"
        assert observed.control_program.decision_event_labels == (
            "call_2027-01-15",
            "call_2028-01-15",
            "call_2029-01-15",
        )
        assert observed.event_program.buckets[0].event_date == date(2027, 1, 15)

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
