from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.bermudan_swaption_tree import (
    build_bermudan_swaption_lattice,
    compile_bermudan_swaption_contract_spec,
    price_bermudan_swaption_on_lattice,
    price_bermudan_swaption_tree,
)
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class _Spec:
    def __init__(
        self,
        *,
        notional: float = 100.0,
        strike: float = 0.05,
        exercise_dates: tuple[date, ...] = (
            date(2025, 11, 15),
            date(2026, 11, 15),
            date(2027, 11, 15),
            date(2028, 11, 15),
            date(2029, 11, 15),
        ),
        swap_end: date = date(2030, 11, 15),
        is_payer: bool = True,
    ) -> None:
        from trellis.core.types import DayCountConvention, Frequency

        self.notional = notional
        self.strike = strike
        self.exercise_dates = exercise_dates
        self.swap_end = swap_end
        self.swap_frequency = Frequency.SEMI_ANNUAL
        self.day_count = DayCountConvention.ACT_360
        self.rate_index = None
        self.is_payer = is_payer


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=31.0),
        vol_surface=FlatVol(vol),
    )


def test_price_bermudan_swaption_tree_matches_task_reference():
    from tests.test_tasks.test_t04_bermudan_swaption import (
        EXERCISE_YEARS,
        FIXED_RATE,
        HW_A,
        NOTIONAL,
        T_SWAP_TENOR,
        _price_bermudan_swaption_on_tree,
    )

    market_state = _market_state()
    spec = _Spec(notional=NOTIONAL, strike=FIXED_RATE)
    lattice = build_bermudan_swaption_lattice(
        market_state,
        spec,
        model="hull_white",
        mean_reversion=HW_A,
        n_steps=200,
    )
    helper_price = price_bermudan_swaption_on_lattice(
        lattice,
        spec=spec,
        settlement=SETTLE,
    )
    reference_price = _price_bermudan_swaption_on_tree(
        lattice,
        EXERCISE_YEARS,
        T_SWAP_TENOR,
        FIXED_RATE,
        NOTIONAL,
    )
    assert abs(helper_price - reference_price) / max(reference_price, 1e-6) < 1e-9


def test_price_bermudan_swaption_tree_monotone_in_exercise_rights():
    market_state = _market_state()
    fewer = _Spec(exercise_dates=(date(2027, 11, 15), date(2029, 11, 15)))
    more = _Spec()

    price_fewer = price_bermudan_swaption_tree(market_state, fewer, model="hull_white")
    price_more = price_bermudan_swaption_tree(market_state, more, model="hull_white")

    assert price_more >= price_fewer


def test_price_bermudan_swaption_tree_receiver_is_non_negative():
    market_state = _market_state()
    receiver = _Spec(is_payer=False)

    price = price_bermudan_swaption_tree(market_state, receiver, model="hull_white")

    assert price >= 0.0


def test_compile_bermudan_swaption_contract_spec_matches_helper_price():
    market_state = _market_state()
    spec = _Spec()
    lattice = build_bermudan_swaption_lattice(
        market_state,
        spec,
        model="hull_white",
        n_steps=120,
    )
    contract = compile_bermudan_swaption_contract_spec(
        lattice,
        spec=spec,
        settlement=market_state.settlement,
    )

    assert price_bermudan_swaption_on_lattice(
        lattice,
        spec=spec,
        settlement=market_state.settlement,
    ) == price_bermudan_swaption_on_lattice(
        lattice,
        contract_spec=contract,
    )


def test_price_bermudan_swaption_tree_ignores_expired_exercise_dates_after_settlement():
    market_state = MarketState(
        as_of=date(2026, 1, 15),
        settlement=date(2026, 1, 15),
        discount=YieldCurve.flat(0.05, max_tenor=31.0),
        vol_surface=FlatVol(0.20),
    )
    full_schedule = _Spec(
        exercise_dates=(
            date(2025, 11, 15),
            date(2026, 11, 15),
            date(2027, 11, 15),
            date(2028, 11, 15),
            date(2029, 11, 15),
        ),
    )
    remaining_schedule = _Spec(
        exercise_dates=(
            date(2026, 11, 15),
            date(2027, 11, 15),
            date(2028, 11, 15),
            date(2029, 11, 15),
        ),
    )

    full_price = price_bermudan_swaption_tree(
        market_state,
        full_schedule,
        model="hull_white",
        n_steps=120,
    )
    remaining_price = price_bermudan_swaption_tree(
        market_state,
        remaining_schedule,
        model="hull_white",
        n_steps=120,
    )

    assert full_price == pytest.approx(remaining_price, rel=1e-9)


def test_price_bermudan_swaption_tree_uses_market_state_hull_white_params():
    market_state = _market_state()
    spec = _Spec(exercise_dates=(date(2026, 11, 15),), swap_end=date(2031, 11, 15))
    calibrated_state = replace(
        market_state,
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.03,
            "sigma": 0.004,
        },
    )

    via_market_state = price_bermudan_swaption_tree(
        calibrated_state,
        spec,
        model="hull_white",
        n_steps=120,
    )
    via_explicit = price_bermudan_swaption_tree(
        market_state,
        spec,
        model="hull_white",
        mean_reversion=0.03,
        sigma=0.004,
        n_steps=120,
    )

    assert via_market_state == pytest.approx(via_explicit, rel=1e-10)
