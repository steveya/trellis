"""Verification tests for bounded smooth path-summary hybrid AD."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.agent.contract_ir import (
    ArithmeticMean,
    Constant,
    ContractIR,
    EquitySpot,
    Exercise,
    FiniteSchedule,
    Max,
    Observation,
    Singleton,
    Spot,
    Strike,
    Sub,
    Underlying,
)
from trellis.analytics import (
    HybridDerivativeRequest,
    admit_hybrid_ad_lane,
    differentiate_arithmetic_asian_path_summary,
)
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.asian_option import price_arithmetic_asian_option_analytical
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)
EXPIRY = date(2025, 11, 15)
OBSERVATIONS = (
    date(2025, 2, 15),
    date(2025, 5, 15),
    date(2025, 8, 15),
    EXPIRY,
)


@dataclass(frozen=True)
class _ArithmeticAsianSpec:
    spot: float
    strike: float
    expiry_date: date
    observation_dates: tuple[date, ...]
    option_type: str = "call"
    notional: float = 1.0
    exercise_style: str = "european"
    averaging_type: str = "arithmetic"
    dividend_yield: float = 0.0


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def _asian_spec(option_type: str = "call") -> _ArithmeticAsianSpec:
    return _ArithmeticAsianSpec(
        spot=100.0,
        strike=100.0,
        expiry_date=EXPIRY,
        observation_dates=OBSERVATIONS,
        option_type=option_type,
        notional=2.0,
    )


def _path_summary_contract_ir(option_type: str = "call") -> ContractIR:
    averaging_schedule = FiniteSchedule(OBSERVATIONS)
    intrinsic = (
        Sub(ArithmeticMean(Spot("SPX"), averaging_schedule), Strike(100.0))
        if option_type == "call"
        else Sub(Strike(100.0), ArithmeticMean(Spot("SPX"), averaging_schedule))
    )
    return ContractIR(
        payoff=Max((intrinsic, Constant(0.0))),
        exercise=Exercise("european", Singleton(EXPIRY)),
        observation=Observation("path_dependent", averaging_schedule),
        underlying=Underlying(EquitySpot("SPX", "gbm")),
    )


@pytest.mark.parametrize("option_type", ("call", "put"))
def test_hybrid_path_summary_vjp_matches_independent_flat_vol_bump(option_type):
    spec = _asian_spec(option_type)
    admission = admit_hybrid_ad_lane(
        _path_summary_contract_ir(option_type),
        product_family="arithmetic_asian_option",
        derivative_method="vjp",
    )
    request = HybridDerivativeRequest(semantic_admission=admission)

    result = differentiate_arithmetic_asian_path_summary(
        spec,
        _market_state(0.20),
        request,
        position_name=f"asian_{option_type}",
        vol_surface_name="spx_flat",
        currency="USD",
    )
    factor = tuple(result.risk_vector)[0]
    bump = 1.0e-5
    up = price_arithmetic_asian_option_analytical(_market_state(0.20 + bump), spec)
    down = price_arithmetic_asian_option_analytical(_market_state(0.20 - bump), spec)
    finite_difference = (up - down) / (2.0 * bump)

    assert result.support_status == "supported"
    assert result.value == pytest.approx(
        price_arithmetic_asian_option_analytical(_market_state(0.20), spec),
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert result.risk_vector[factor] == pytest.approx(
        finite_difference,
        rel=5.0e-6,
        abs=1.0e-6,
    )
    assert result.method_metadata["semantic_state_kind"] == "smooth_path_summary"
