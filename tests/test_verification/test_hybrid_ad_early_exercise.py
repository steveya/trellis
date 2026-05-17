"""Verification tests for bounded early-exercise hybrid AD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pytest

from trellis.agent.contract_ir import (
    Constant,
    ContinuousInterval,
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
    differentiate_vanilla_early_exercise,
)
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol, GridVolSurface


SETTLEMENT = date(2024, 11, 15)
EXPIRY = date(2025, 11, 15)
EXERCISE_DATES = (
    date(2025, 2, 15),
    date(2025, 5, 15),
    date(2025, 8, 15),
)


@dataclass(frozen=True)
class _VanillaEquitySpec:
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "put"
    notional: float = 1.0
    exercise_style: str = "american"
    exercise_dates: tuple[date, ...] = ()
    aad_tree_steps: int = 64
    early_exercise_boundary_tolerance: float = 1.0e-12


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def _grid_market_state() -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=GridVolSurface(
            expiries=(0.5, 1.5),
            strikes=(90.0, 110.0),
            vols=((0.18, 0.21), (0.24, 0.27)),
        ),
    )


def _american_put_spec() -> _VanillaEquitySpec:
    return _VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=EXPIRY,
        option_type="put",
        notional=2.0,
        exercise_style="american",
    )


def _bermudan_put_spec() -> _VanillaEquitySpec:
    return _VanillaEquitySpec(
        spot=90.0,
        strike=100.0,
        expiry_date=EXPIRY,
        option_type="put",
        exercise_style="bermudan",
        exercise_dates=EXERCISE_DATES,
    )


def _early_exercise_ir(exercise_style: str) -> ContractIR:
    observation_schedule = Singleton(EXPIRY)
    exercise_schedule = (
        ContinuousInterval(EXERCISE_DATES[0], EXPIRY)
        if exercise_style == "american"
        else FiniteSchedule(EXERCISE_DATES + (EXPIRY,))
    )
    return ContractIR(
        payoff=Max((Sub(Strike(100.0), Spot("SPX")), Constant(0.0))),
        exercise=Exercise(exercise_style, exercise_schedule),
        observation=Observation("terminal", observation_schedule),
        underlying=Underlying(EquitySpot("SPX", "gbm")),
    )


def _exercise_steps(spec: _VanillaEquitySpec, maturity: float) -> frozenset[int]:
    n_steps = int(spec.aad_tree_steps)
    if spec.exercise_style == "american":
        return frozenset(range(0, n_steps))
    if spec.exercise_style != "bermudan":
        return frozenset()
    steps: set[int] = set()
    for exercise_date in spec.exercise_dates:
        exercise_time = float(year_fraction(SETTLEMENT, exercise_date))
        if exercise_time <= 0.0 or exercise_time >= maturity:
            continue
        step = int(round(exercise_time / maturity * n_steps))
        if 0 < step < n_steps:
            steps.add(step)
    return frozenset(steps)


def _intrinsic(spec: _VanillaEquitySpec, spot: float) -> float:
    if spec.option_type == "put":
        return spec.notional * max(spec.strike - spot, 0.0)
    return spec.notional * max(spot - spec.strike, 0.0)


def _independent_tree_value(spec: _VanillaEquitySpec, vol: float) -> float:
    maturity = float(year_fraction(SETTLEMENT, spec.expiry_date))
    n_steps = int(spec.aad_tree_steps)
    dt = maturity / float(n_steps)
    discount = YieldCurve.flat(0.05)
    maturity_df = discount.discount(maturity)
    rate = -math.log(max(float(maturity_df), 1.0e-12)) / maturity
    one_step_df = math.exp(-rate * dt)
    up = math.exp(vol * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp(rate * dt)
    probability = (growth - down) / (up - down)

    values = [
        _intrinsic(spec, spec.spot * (up**node) * (down ** (n_steps - node)))
        for node in range(n_steps + 1)
    ]
    exercise_steps = _exercise_steps(spec, maturity)
    for step in range(n_steps - 1, -1, -1):
        continuation = [
            one_step_df
            * ((1.0 - probability) * values[node] + probability * values[node + 1])
            for node in range(step + 1)
        ]
        if step in exercise_steps:
            values = [
                max(
                    continuation[node],
                    _intrinsic(spec, spec.spot * (up**node) * (down ** (step - node))),
                )
                for node in range(step + 1)
            ]
        else:
            values = continuation
    return float(values[0])


@pytest.mark.parametrize(
    ("spec", "product_family", "contract_ir"),
    (
        (_american_put_spec(), "american_vanilla_option", _early_exercise_ir("american")),
        (_bermudan_put_spec(), "bermudan_vanilla_option", _early_exercise_ir("bermudan")),
    ),
)
def test_hybrid_early_exercise_vjp_matches_independent_flat_vol_bump(
    spec,
    product_family,
    contract_ir,
):
    admission = admit_hybrid_ad_lane(
        contract_ir,
        product_family=product_family,
        derivative_method="vjp",
    )
    request = HybridDerivativeRequest(semantic_admission=admission)

    result = differentiate_vanilla_early_exercise(
        spec,
        _market_state(0.20),
        request,
        position_name=f"{spec.exercise_style}_put",
        vol_surface_name="spx_flat",
        currency="USD",
    )
    factor = tuple(result.risk_vector)[0]
    bump = 1.0e-5
    finite_difference = (
        _independent_tree_value(spec, 0.20 + bump)
        - _independent_tree_value(spec, 0.20 - bump)
    ) / (2.0 * bump)

    assert result.support_status == "supported"
    assert result.value == pytest.approx(
        _independent_tree_value(spec, 0.20),
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert result.risk_vector[factor] == pytest.approx(
        finite_difference,
        rel=5.0e-5,
        abs=1.0e-5,
    )
    assert result.method_metadata["semantic_state_kind"] == "early_exercise_control"


@pytest.mark.parametrize(
    ("market_state", "spec", "expected_code"),
    (
        (
            _grid_market_state(),
            _american_put_spec(),
            "unsupported_early_exercise_vol_surface_parameterization",
        ),
        (
            _market_state(),
            _VanillaEquitySpec(
                spot=80.0,
                strike=100.0,
                expiry_date=EXPIRY,
                option_type="put",
                exercise_style="american",
                early_exercise_boundary_tolerance=1.0e9,
            ),
            "early_exercise_boundary_kink",
        ),
    ),
)
def test_hybrid_early_exercise_verification_keeps_unsupported_shapes_fail_closed(
    market_state,
    spec,
    expected_code,
):
    admission = admit_hybrid_ad_lane(
        _early_exercise_ir("american"),
        product_family="american_vanilla_option",
        derivative_method="vjp",
    )

    result = differentiate_vanilla_early_exercise(
        spec,
        market_state,
        HybridDerivativeRequest(semantic_admission=admission),
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "unsupported"
    assert result.value is None
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == expected_code

