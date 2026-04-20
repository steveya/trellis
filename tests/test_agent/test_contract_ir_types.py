from __future__ import annotations

from datetime import date

import pytest

from trellis.agent.contract_ir import (
    Add,
    And,
    Annuity,
    ArithmeticMean,
    CompositeUnderlying,
    Constant,
    ContractIR,
    ContractIRWellFormednessError,
    ContinuousInterval,
    CurveQuote,
    EquitySpot,
    Exercise,
    FiniteSchedule,
    ForwardRate,
    ForwardRateInterval,
    Gt,
    Indicator,
    LinearBasket,
    Max,
    Mul,
    Observation,
    PayoffEvalEnv,
    ParRateTenor,
    QuoteCurve,
    QuoteSurface,
    Scaled,
    Singleton,
    Spot,
    Strike,
    Sub,
    SurfaceQuote,
    SwapRate,
    Underlying,
    VarianceObservable,
    VolDeltaPoint,
    VolPoint,
    ZeroRateTenor,
    evaluate_payoff_expr,
)


def _singleton(day: str) -> Singleton:
    year, month, day_value = map(int, day.split("-"))
    return Singleton(date(year, month, day_value))


def _finite_schedule(*days: str) -> FiniteSchedule:
    dates = []
    for day in days:
        year, month, day_value = map(int, day.split("-"))
        dates.append(date(year, month, day_value))
    return FiniteSchedule(tuple(dates))


def _continuous_interval(start_day: str, end_day: str) -> ContinuousInterval:
    start_year, start_month, start_date = map(int, start_day.split("-"))
    end_year, end_month, end_date = map(int, end_day.split("-"))
    return ContinuousInterval(
        date(start_year, start_month, start_date),
        date(end_year, end_month, end_date),
    )


def _equity_call_fixture() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0))),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
    )


def _variance_fixture() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Scaled(
            Constant(10000.0),
            Sub(
                VarianceObservable("SPX", _continuous_interval("2025-01-01", "2025-11-15")),
                Strike(0.04),
            ),
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("SPX", "gbm")),
    )


def _quoted_curve_fixture() -> ContractIR:
    expiry = _singleton("2026-06-30")
    return ContractIR(
        payoff=Scaled(
            Constant(1_000_000.0),
            Sub(
                CurveQuote("USD_SWAP", ParRateTenor("10Y"), "par_rate"),
                CurveQuote("USD_SWAP", ParRateTenor("2Y"), "par_rate"),
            ),
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=QuoteCurve("USD_SWAP")),
    )


def _quoted_surface_fixture() -> ContractIR:
    expiry = _singleton("2026-06-30")
    return ContractIR(
        payoff=Scaled(
            Constant(100_000.0),
            Sub(
                SurfaceQuote("SPX_IV", VolPoint("1Y", 0.90, "moneyness"), "black_vol"),
                SurfaceQuote("SPX_IV", VolPoint("1Y", 1.10, "moneyness"), "black_vol"),
            ),
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=QuoteSurface("SPX_IV")),
    )


def _digital_fixture() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Mul(
            (
                Constant(1.0),
                Indicator(Gt(Spot("AAPL"), Strike(150.0))),
            )
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
    )


def _asian_fixture() -> ContractIR:
    expiry = _singleton("2025-12-31")
    averaging = _finite_schedule(
        "2025-01-01",
        "2025-02-01",
        "2025-03-01",
        "2025-04-01",
    )
    return ContractIR(
        payoff=Max(
            (
                Sub(ArithmeticMean(Spot("SPX"), averaging), Strike(4500.0)),
                Constant(0.0),
            )
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="schedule", schedule=averaging),
        underlying=Underlying(spec=EquitySpot("SPX", "gbm")),
    )


class TestContractIRTypes:
    def test_four_phase_two_family_fixtures_are_structurally_equal(self):
        assert _equity_call_fixture() == _equity_call_fixture()
        assert _variance_fixture() == _variance_fixture()
        assert _digital_fixture() == _digital_fixture()
        assert _asian_fixture() == _asian_fixture()
        assert _quoted_curve_fixture() == _quoted_curve_fixture()
        assert _quoted_surface_fixture() == _quoted_surface_fixture()

    def test_rule_1_duplicate_underlying_names_are_rejected(self):
        expiry = _singleton("2025-11-15")
        with pytest.raises(ContractIRWellFormednessError, match="unique"):
            ContractIR(
                payoff=Max((Sub(Spot("SPX"), Strike(4500.0)), Constant(0.0))),
                exercise=Exercise(style="european", schedule=expiry),
                observation=Observation(kind="terminal", schedule=expiry),
                underlying=Underlying(
                    spec=CompositeUnderlying(
                        (
                            EquitySpot("SPX", "gbm"),
                            EquitySpot("SPX", "gbm"),
                        )
                    )
                ),
            )

    def test_composite_underlying_rejects_non_leaf_parts(self):
        with pytest.raises(ContractIRWellFormednessError, match="UnderlyingSpecLeaf"):
            CompositeUnderlying(
                (
                    EquitySpot("SPX", "gbm"),
                    "not-a-leaf",  # type: ignore[arg-type]
                )
            )

    def test_rule_2_unknown_underlier_id_is_rejected(self):
        expiry = _singleton("2025-11-15")
        with pytest.raises(ContractIRWellFormednessError, match="underlier"):
            ContractIR(
                payoff=Max((Sub(Spot("MISSING"), Strike(150.0)), Constant(0.0))),
                exercise=Exercise(style="european", schedule=expiry),
                observation=Observation(kind="terminal", schedule=expiry),
                underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
            )

    def test_rule_2_unknown_quote_object_id_is_rejected(self):
        expiry = _singleton("2026-06-30")
        with pytest.raises(ContractIRWellFormednessError, match="underlier"):
            ContractIR(
                payoff=Scaled(
                    Constant(1_000_000.0),
                    Sub(
                        CurveQuote("MISSING", ParRateTenor("10Y"), "par_rate"),
                        CurveQuote("MISSING", ParRateTenor("2Y"), "par_rate"),
                    ),
                ),
                exercise=Exercise(style="european", schedule=expiry),
                observation=Observation(kind="terminal", schedule=expiry),
                underlying=Underlying(spec=QuoteCurve("USD_SWAP")),
            )

    def test_rule_3_payoff_schedules_must_be_schedule_values(self):
        expiry = _singleton("2025-11-15")
        with pytest.raises(ContractIRWellFormednessError, match="schedule"):
            ContractIR(
                payoff=Scaled(
                    Annuity("USD-IRS-5Y", "bad-schedule"),  # type: ignore[arg-type]
                    Max(
                        (
                            Sub(
                                SwapRate("USD-IRS-5Y", _finite_schedule("2026-11-15", "2027-11-15")),
                                Strike(0.05),
                            ),
                            Constant(0.0),
                        )
                    ),
                ),
                exercise=Exercise(style="european", schedule=expiry),
                observation=Observation(kind="terminal", schedule=expiry),
                underlying=Underlying(spec=ForwardRate("USD-IRS-5Y", "lognormal_forward")),
            )

    def test_rule_4_finite_schedule_must_be_non_empty_and_increasing(self):
        with pytest.raises(ContractIRWellFormednessError, match="strictly increasing"):
            FiniteSchedule((date(2025, 2, 1), date(2025, 1, 1)))
        with pytest.raises(ContractIRWellFormednessError, match="non-empty"):
            FiniteSchedule(())

    def test_rule_5_node_local_schedule_discipline_is_enforced(self):
        expiry = _singleton("2025-11-15")
        with pytest.raises(ContractIRWellFormednessError, match="ArithmeticMean"):
            ContractIR(
                payoff=Max(
                    (
                        Sub(
                            ArithmeticMean(Spot("SPX"), expiry),  # type: ignore[arg-type]
                            Strike(4500.0),
                        ),
                        Constant(0.0),
                    )
                ),
                exercise=Exercise(style="european", schedule=expiry),
                observation=Observation(kind="terminal", schedule=expiry),
                underlying=Underlying(spec=EquitySpot("SPX", "gbm")),
            )

    def test_rule_6_european_requires_singleton_exercise_schedule(self):
        with pytest.raises(ContractIRWellFormednessError, match="european"):
            Exercise(
                style="european",
                schedule=_finite_schedule("2025-11-15", "2025-12-15"),
            )

    def test_rule_7_terminal_observation_requires_singleton_schedule(self):
        with pytest.raises(ContractIRWellFormednessError, match="terminal"):
            Observation(
                kind="terminal",
                schedule=_finite_schedule("2025-11-15", "2025-12-15"),
            )

    def test_rule_8_arity_mismatch_is_rejected(self):
        with pytest.raises(ContractIRWellFormednessError, match="Max"):
            Max(())
        with pytest.raises(ContractIRWellFormednessError, match="Add"):
            Add((Constant(1.0),))
        with pytest.raises(ContractIRWellFormednessError, match="Mul"):
            Mul((Constant(1.0),))

    def test_rule_9_linear_basket_must_be_non_empty(self):
        with pytest.raises(ContractIRWellFormednessError, match="LinearBasket"):
            LinearBasket(())

    def test_american_uses_continuous_interval(self):
        american = Exercise(
            style="american",
            schedule=_continuous_interval("2025-01-01", "2025-12-31"),
        )
        assert american.style == "american"

    def test_path_dependent_observation_accepts_interval(self):
        observation = Observation(
            kind="path_dependent",
            schedule=_continuous_interval("2025-01-01", "2025-12-31"),
        )
        assert observation.kind == "path_dependent"

    def test_quote_coordinate_dataclasses_validate_their_shape(self):
        assert ParRateTenor("10Y").tenor == "10Y"
        assert ZeroRateTenor("5Y").tenor == "5Y"
        assert ForwardRateInterval("3M", "6M").start_tenor == "3M"
        assert VolPoint("1Y", 0.9, "moneyness").strike_style == "moneyness"
        assert VolDeltaPoint("1Y", 0.25, "spot_delta").delta_style == "spot_delta"

        with pytest.raises(ContractIRWellFormednessError, match="non-empty"):
            ParRateTenor("")
        with pytest.raises(ContractIRWellFormednessError, match="numeric"):
            VolPoint("1Y", "bad", "moneyness")  # type: ignore[arg-type]

    def test_quote_leaves_evaluate_against_explicit_quote_keys(self):
        curve_env = PayoffEvalEnv(
            values={
                ("curve_quote", "USD_SWAP", ParRateTenor("10Y"), "par_rate"): 0.041,
                ("curve_quote", "USD_SWAP", ParRateTenor("2Y"), "par_rate"): 0.032,
            }
        )
        surface_env = PayoffEvalEnv(
            values={
                ("surface_quote", "SPX_IV", VolPoint("1Y", 0.90, "moneyness"), "black_vol"): 0.24,
                ("surface_quote", "SPX_IV", VolPoint("1Y", 1.10, "moneyness"), "black_vol"): 0.19,
            }
        )

        assert evaluate_payoff_expr(_quoted_curve_fixture().payoff, curve_env) == pytest.approx(9000.0)
        assert evaluate_payoff_expr(_quoted_surface_fixture().payoff, surface_env) == pytest.approx(5000.0)
