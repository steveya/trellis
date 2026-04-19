from __future__ import annotations

from datetime import date, timedelta

import pytest

from trellis.agent.contract_ir import (
    Annuity,
    ArithmeticMean,
    CompositeUnderlying,
    Constant,
    ContractIR,
    ContinuousInterval,
    EquitySpot,
    Exercise,
    FiniteSchedule,
    ForwardRate,
    Gt,
    Indicator,
    LinearBasket,
    Lt,
    Max,
    Mul,
    Observation,
    Scaled,
    Singleton,
    Spot,
    Strike,
    Sub,
    SwapRate,
    Underlying,
    VarianceObservable,
    canonicalize,
)
from trellis.agent.knowledge.decompose import decompose_to_contract_ir, decompose_to_ir


def _singleton(day: str) -> Singleton:
    return Singleton(date(*map(int, day.split("-"))))


def _interval(start_day: str, end_day: str) -> ContinuousInterval:
    return ContinuousInterval(
        date(*map(int, start_day.split("-"))),
        date(*map(int, end_day.split("-"))),
    )


def _month_end_schedule(year: int) -> FiniteSchedule:
    month_ends = (
        date(year, 1, 31),
        date(year, 2, 28),
        date(year, 3, 31),
        date(year, 4, 30),
        date(year, 5, 31),
        date(year, 6, 30),
        date(year, 7, 31),
        date(year, 8, 31),
        date(year, 9, 30),
        date(year, 10, 31),
        date(year, 11, 30),
        date(year, 12, 31),
    )
    return FiniteSchedule(month_ends)


def _weekly_schedule(start_day: str, end_day: str) -> FiniteSchedule:
    start = date(*map(int, start_day.split("-")))
    end = date(*map(int, end_day.split("-")))
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current = current + timedelta(days=7)
    if dates[-1] != end:
        dates.append(end)
    return FiniteSchedule(tuple(dates))


def _swap_schedule(expiry_day: str, tenor_years: int) -> FiniteSchedule:
    expiry = date(*map(int, expiry_day.split("-")))
    dates = tuple(date(expiry.year + offset, expiry.month, expiry.day) for offset in range(1, tenor_years + 1))
    return FiniteSchedule(dates)


def _contracts():
    swap_schedule = _swap_schedule("2025-11-15", 5)
    asian_monthly = _month_end_schedule(2025)
    asian_weekly = _weekly_schedule("2025-01-03", "2025-01-31")
    return [
        (
            "European call on AAPL strike 150 expiring 2025-11-15",
            "european_option",
            ContractIR(
                payoff=Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0))),
                exercise=Exercise(style="european", schedule=_singleton("2025-11-15")),
                observation=Observation(kind="terminal", schedule=_singleton("2025-11-15")),
                underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
            ),
        ),
        (
            "European put on SPX strike 4500 expiring 2025-11-15",
            "european_option",
            ContractIR(
                payoff=Max((Sub(Strike(4500.0), Spot("SPX")), Constant(0.0))),
                exercise=Exercise(style="european", schedule=_singleton("2025-11-15")),
                observation=Observation(kind="terminal", schedule=_singleton("2025-11-15")),
                underlying=Underlying(spec=EquitySpot("SPX", "gbm")),
            ),
        ),
        (
            "European call on AAPL strike 0 expiring 2025-11-15",
            "european_option",
            ContractIR(
                payoff=Max((Sub(Spot("AAPL"), Strike(0.0)), Constant(0.0))),
                exercise=Exercise(style="european", schedule=_singleton("2025-11-15")),
                observation=Observation(kind="terminal", schedule=_singleton("2025-11-15")),
                underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
            ),
        ),
        (
            "European payer swaption on 5Y USD IRS strike 5% expiring 2025-11-15",
            "swaption",
            ContractIR(
                payoff=Scaled(
                    Annuity("USD-IRS-5Y", swap_schedule),
                    Max((Sub(SwapRate("USD-IRS-5Y", swap_schedule), Strike(0.05)), Constant(0.0))),
                ),
                exercise=Exercise(style="european", schedule=_singleton("2025-11-15")),
                observation=Observation(kind="terminal", schedule=_singleton("2025-11-15")),
                underlying=Underlying(spec=ForwardRate("USD-IRS-5Y", "lognormal_forward")),
            ),
        ),
        (
            "European receiver swaption on 5Y USD IRS strike 5% expiring 2025-11-15",
            "swaption",
            ContractIR(
                payoff=Scaled(
                    Annuity("USD-IRS-5Y", swap_schedule),
                    Max((Sub(Strike(0.05), SwapRate("USD-IRS-5Y", swap_schedule)), Constant(0.0))),
                ),
                exercise=Exercise(style="european", schedule=_singleton("2025-11-15")),
                observation=Observation(kind="terminal", schedule=_singleton("2025-11-15")),
                underlying=Underlying(spec=ForwardRate("USD-IRS-5Y", "lognormal_forward")),
            ),
        ),
        (
            "European basket call on {SPX 50%, NDX 50%} strike 4500 expiring 2025-11-15",
            "basket_option",
            ContractIR(
                payoff=Max(
                    (
                        Sub(
                            LinearBasket(((0.5, Spot("SPX")), (0.5, Spot("NDX")))),
                            Strike(4500.0),
                        ),
                        Constant(0.0),
                    )
                ),
                exercise=Exercise(style="european", schedule=_singleton("2025-11-15")),
                observation=Observation(kind="terminal", schedule=_singleton("2025-11-15")),
                underlying=Underlying(
                    spec=CompositeUnderlying((EquitySpot("SPX", "gbm"), EquitySpot("NDX", "gbm")))
                ),
            ),
        ),
        (
            "Equity variance swap on SPX, variance strike 0.04, notional 10000, expiry 2025-11-15",
            "variance_swap",
            ContractIR(
                payoff=Scaled(
                    Constant(10000.0),
                    Sub(
                        VarianceObservable("SPX", _interval("2025-01-01", "2025-11-15")),
                        Strike(0.04),
                    ),
                ),
                exercise=Exercise(style="european", schedule=_singleton("2025-11-15")),
                observation=Observation(kind="terminal", schedule=_singleton("2025-11-15")),
                underlying=Underlying(spec=EquitySpot("SPX", "gbm")),
            ),
        ),
        (
            "Cash-or-nothing digital call on AAPL paying $2 if spot > 150 at expiry 2025-11-15",
            "digital_option",
            ContractIR(
                payoff=Mul((Constant(2.0), Indicator(Gt(Spot("AAPL"), Strike(150.0))))),
                exercise=Exercise(style="european", schedule=_singleton("2025-11-15")),
                observation=Observation(kind="terminal", schedule=_singleton("2025-11-15")),
                underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
            ),
        ),
        (
            "Asset-or-nothing digital put on AAPL if spot < 150 at expiry 2025-11-15",
            "digital_option",
            ContractIR(
                payoff=Mul((Spot("AAPL"), Indicator(Lt(Spot("AAPL"), Strike(150.0))))),
                exercise=Exercise(style="european", schedule=_singleton("2025-11-15")),
                observation=Observation(kind="terminal", schedule=_singleton("2025-11-15")),
                underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
            ),
        ),
        (
            "Arithmetic Asian call on SPX monthly average over 2025 strike 4500",
            "asian_option",
            ContractIR(
                payoff=Max(
                    (
                        Sub(ArithmeticMean(Spot("SPX"), asian_monthly), Strike(4500.0)),
                        Constant(0.0),
                    )
                ),
                exercise=Exercise(style="european", schedule=Singleton(date(2025, 12, 31))),
                observation=Observation(kind="schedule", schedule=asian_monthly),
                underlying=Underlying(spec=EquitySpot("SPX", "gbm")),
            ),
        ),
        (
            "Arithmetic Asian put on SPX weekly average from 2025-01-03 to 2025-01-31 strike 4500",
            "asian_option",
            ContractIR(
                payoff=Max(
                    (
                        Sub(Strike(4500.0), ArithmeticMean(Spot("SPX"), asian_weekly)),
                        Constant(0.0),
                    )
                ),
                exercise=Exercise(style="european", schedule=Singleton(date(2025, 1, 31))),
                observation=Observation(kind="schedule", schedule=asian_weekly),
                underlying=Underlying(spec=EquitySpot("SPX", "gbm")),
            ),
        ),
    ]


class TestDecomposeContractIR:
    @pytest.mark.parametrize(
        "description,instrument_type,expected",
        _contracts(),
    )
    def test_supported_descriptions_emit_expected_contract_ir(
        self,
        description: str,
        instrument_type: str,
        expected: ContractIR,
    ):
        product_ir = decompose_to_ir(description, instrument_type=instrument_type)
        observed = decompose_to_contract_ir(
            description,
            instrument_type=instrument_type,
            product_ir=product_ir,
        )
        assert observed == expected
        assert observed is not None
        assert canonicalize(observed.payoff) == observed.payoff

    @pytest.mark.parametrize(
        "description,instrument_type",
        [
            ("American put on AAPL strike 150 expiring 2025-11-15", "american_option"),
            ("Barrier option with 200 knock-out on AAPL expiring 2025-11-15", "barrier_option"),
            ("Lookback option on AAPL expiring 2025-11-15", "lookback_option"),
            ("Chooser option on AAPL strike 150 expiring 2025-11-15", "chooser_option"),
            ("Callable bond with semiannual coupon and call schedule", "callable_bond"),
            ("CDS on Ford 5Y", "cds"),
            ("Caplet on SOFR strike 4% expiring 2025-11-15", "cap"),
        ],
    )
    def test_out_of_family_descriptions_return_none(
        self,
        description: str,
        instrument_type: str,
    ):
        product_ir = decompose_to_ir(description, instrument_type=instrument_type)
        assert (
            decompose_to_contract_ir(
                description,
                instrument_type=instrument_type,
                product_ir=product_ir,
            )
            is None
        )
