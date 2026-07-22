from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from trellis.agent.family_lowering_ir import FactorStateSimulationIR
from trellis.agent.static_leg_contract import (
    CouponLeg,
    CouponPeriod,
    FixedCouponFormula,
    FloatingCouponFormula,
    KnownCashflow,
    KnownCashflowLeg,
    NotionalSchedule,
    NotionalStep,
    OvernightRateIndex,
    SignedLeg,
    StaticLegContractIR,
)
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.execution import compile_static_leg_execution_ir
from trellis.execution.visitors.simulation_bridge import (
    build_future_value_cube_from_execution_ir,
    compile_factor_state_simulation_ir_from_execution_ir,
    compile_swap_spec_from_execution_ir,
)
from trellis.instruments.swap import SwapPayoff
from trellis.models.monte_carlo.simulation_substrate import (
    price_interest_rate_swap_future_value_cube,
)
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.042, max_tenor=10.0),
        forecast_curves={"SOFR": YieldCurve.flat(0.046, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
        selected_curve_names={
            "discount_curve": "usd_ois",
            "forecast_curve": "SOFR",
        },
    )


def _notional(amount: float) -> NotionalSchedule:
    return NotionalSchedule(
        (
            NotionalStep(
                start_date=date(2024, 11, 15),
                end_date=date(2027, 11, 15),
                amount=amount,
            ),
        )
    )


def _periods(*bounds: tuple[str, str]) -> tuple[CouponPeriod, ...]:
    return tuple(
        CouponPeriod(
            accrual_start=date.fromisoformat(start_day),
            accrual_end=date.fromisoformat(end_day),
            payment_date=date.fromisoformat(end_day),
            fixing_date=date.fromisoformat(start_day),
        )
        for start_day, end_day in bounds
    )


def _fixed_float_swap() -> StaticLegContractIR:
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="pay",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(1_000_000.0),
                    coupon_periods=_periods(
                        ("2024-11-15", "2025-05-15"),
                        ("2025-05-15", "2025-11-15"),
                        ("2025-11-15", "2026-05-15"),
                        ("2026-05-15", "2026-11-15"),
                        ("2026-11-15", "2027-05-15"),
                        ("2027-05-15", "2027-11-15"),
                    ),
                    coupon_formula=FixedCouponFormula(0.045),
                    day_count="30/360",
                    payment_frequency="semiannual",
                    label="fixed",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(1_000_000.0),
                    coupon_periods=_periods(
                        ("2024-11-15", "2025-02-15"),
                        ("2025-02-15", "2025-05-15"),
                        ("2025-05-15", "2025-08-15"),
                        ("2025-08-15", "2025-11-15"),
                        ("2025-11-15", "2026-02-15"),
                        ("2026-02-15", "2026-05-15"),
                        ("2026-05-15", "2026-08-15"),
                        ("2026-08-15", "2026-11-15"),
                        ("2026-11-15", "2027-02-15"),
                        ("2027-02-15", "2027-05-15"),
                        ("2027-05-15", "2027-08-15"),
                        ("2027-08-15", "2027-11-15"),
                    ),
                    coupon_formula=FloatingCouponFormula(OvernightRateIndex("SOFR")),
                    day_count="ACT/360",
                    payment_frequency="quarterly",
                    label="float",
                ),
            ),
        ),
        metadata={"semantic_family": "fixed_float_swap"},
    )


def _fixed_coupon_bond() -> StaticLegContractIR:
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency="USD",
                    notional_schedule=_notional(1_000_000.0),
                    coupon_periods=_periods(
                        ("2024-11-15", "2025-05-15"),
                        ("2025-05-15", "2025-11-15"),
                    ),
                    coupon_formula=FixedCouponFormula(0.05),
                    day_count="ACT/ACT",
                    payment_frequency="semiannual",
                    label="coupon",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=KnownCashflowLeg(
                    currency="USD",
                    cashflows=(
                        KnownCashflow(
                            payment_date=date(2025, 11, 15),
                            amount=1_000_000.0,
                            currency="USD",
                            label="principal_redemption",
                        ),
                    ),
                    label="principal",
                ),
            ),
        ),
        metadata={"semantic_family": "fixed_coupon_bond"},
    )


def _irregular_fixed_float_swap() -> StaticLegContractIR:
    contract = _fixed_float_swap()
    periods = _periods(
        ("2024-11-15", "2025-01-15"),
        ("2025-01-15", "2025-08-15"),
    )
    fixed_leg, floating_leg = contract.legs
    return replace(
        contract,
        legs=(
            replace(fixed_leg, leg=replace(fixed_leg.leg, coupon_periods=periods)),
            replace(
                floating_leg,
                leg=replace(floating_leg.leg, coupon_periods=periods),
            ),
        ),
    )


def test_fixed_float_swap_execution_ir_compiles_to_factor_state_simulation_ir():
    ir = compile_static_leg_execution_ir(_fixed_float_swap())

    family_ir = compile_factor_state_simulation_ir_from_execution_ir(ir)

    assert isinstance(family_ir, FactorStateSimulationIR)
    assert family_ir.route_id == "simulation_substrate"
    assert family_ir.route_family == "simulation"
    assert family_ir.product_instrument == "interest_rate_swap"
    assert family_ir.payoff_family == "conditional_valuation"
    assert family_ir.required_input_ids == ("discount_curve:USD", "forward_curve:SOFR")
    assert family_ir.market_data_requirements == frozenset({"discount_curve:USD", "forward_curve:SOFR"})
    assert family_ir.state_spec.dimension == 1
    assert family_ir.state_spec.state_layout == "scalar"
    assert family_ir.factor_names == ("short_rate",)
    assert family_ir.process_spec.process_family == "hull_white_1f"
    assert family_ir.projection_spec.projection_family == "hull_white_1f_rate_projection"
    assert family_ir.observation_program.observable_ids == (
        "discount_curve:USD",
        "forward_curve:SOFR",
    )
    assert family_ir.observation_program.terminal_value_symbol == "clean_future_value"
    assert family_ir.conditional_valuation.supports_exact is True
    assert family_ir.helper_symbol == "price_interest_rate_swap_future_value_cube"
    assert family_ir.event_program.event_dates[0] == "2024-11-15"
    assert family_ir.event_program.event_dates[-1] == "2027-11-15"


def test_fixed_float_swap_future_value_cube_bridge_matches_direct_substrate():
    market_state = _market_state()
    ir = compile_static_leg_execution_ir(_fixed_float_swap())

    bridged_spec = compile_swap_spec_from_execution_ir(ir)
    bridged_cube = build_future_value_cube_from_execution_ir(
        ir,
        market_state,
        position_name="payer_swap",
        n_paths=512,
        n_steps=96,
        seed=19,
    )
    direct_cube = price_interest_rate_swap_future_value_cube(
        name="payer_swap",
        spec=bridged_spec,
        market_state=market_state,
        n_paths=512,
        n_steps=96,
        seed=19,
    )

    assert bridged_spec.fixed_frequency == Frequency.SEMI_ANNUAL
    assert bridged_spec.float_frequency == Frequency.QUARTERLY
    assert bridged_spec.fixed_day_count == DayCountConvention.THIRTY_360
    assert bridged_spec.float_day_count == DayCountConvention.ACT_360
    assert bridged_spec.rate_index == "SOFR"
    assert bridged_spec.is_payer is True

    np.testing.assert_allclose(bridged_cube.values, direct_cube.values, atol=0.0, rtol=0.0)
    assert bridged_cube.position_names == ("payer_swap",)
    assert bridged_cube.compute_plan["bridge_family"] == "execution_ir"
    assert bridged_cube.compute_plan["execution_product_family"] == "fixed_float_swap"
    assert bridged_cube.position_provenance["payer_swap"]["execution_source_kind"] == "static_leg_contract_ir"

    direct_pv = SwapPayoff(bridged_spec).evaluate(market_state)
    np.testing.assert_allclose(
        bridged_cube.values_for_position("payer_swap")[0],
        np.full(bridged_cube.n_paths, direct_pv, dtype=float),
        atol=1e-10,
        rtol=1e-10,
    )


def test_future_value_bridge_fails_closed_for_unsupported_execution_family():
    ir = compile_static_leg_execution_ir(_fixed_coupon_bond())

    with pytest.raises(ValueError, match="fixed_float_swap"):
        compile_factor_state_simulation_ir_from_execution_ir(ir)


def test_future_value_bridge_rejects_irregular_coupon_obligation_execution():
    ir = compile_static_leg_execution_ir(_irregular_fixed_float_swap())

    with pytest.raises(ValueError, match="static_leg_fixed_float_swap"):
        compile_factor_state_simulation_ir_from_execution_ir(ir)
