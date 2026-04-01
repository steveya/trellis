from __future__ import annotations

from datetime import date

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.bermudan_swaption_tree import (
    build_bermudan_swaption_lattice,
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
        exercise_dates: str = "2025-11-15,2026-11-15,2027-11-15,2028-11-15,2029-11-15",
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
    fewer = _Spec(exercise_dates="2027-11-15,2029-11-15")
    more = _Spec()

    price_fewer = price_bermudan_swaption_tree(market_state, fewer, model="hull_white")
    price_more = price_bermudan_swaption_tree(market_state, more, model="hull_white")

    assert price_more >= price_fewer


def test_price_bermudan_swaption_tree_receiver_is_non_negative():
    market_state = _market_state()
    receiver = _Spec(is_payer=False)

    price = price_bermudan_swaption_tree(market_state, receiver, model="hull_white")

    assert price >= 0.0
