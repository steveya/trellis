from __future__ import annotations

from datetime import date
import math

from hypothesis import HealthCheck, given, settings, strategies as st

from trellis.agent.contract_ir import (
    Add,
    Annuity,
    ArithmeticMean,
    Constant,
    ContinuousInterval,
    FiniteSchedule,
    Gt,
    Indicator,
    LinearBasket,
    Max,
    Min,
    Mul,
    PayoffEvalEnv,
    Scaled,
    Singleton,
    Spot,
    Strike,
    Sub,
    SwapRate,
    VarianceObservable,
    canonicalize,
    evaluate_payoff_expr,
)


def _singleton(day: str) -> Singleton:
    year, month, day_value = map(int, day.split("-"))
    return Singleton(date(year, month, day_value))


def _finite_schedule(*days: str) -> FiniteSchedule:
    return FiniteSchedule(tuple(date(*map(int, day.split("-"))) for day in days))


def _interval(start_day: str, end_day: str) -> ContinuousInterval:
    return ContinuousInterval(
        date(*map(int, start_day.split("-"))),
        date(*map(int, end_day.split("-"))),
    )


def _environment() -> PayoffEvalEnv:
    jan = date(2025, 1, 1)
    feb = date(2025, 2, 1)
    mar = date(2025, 3, 1)
    apr = date(2025, 4, 1)
    swap_schedule = _finite_schedule("2026-11-15", "2027-11-15")
    return PayoffEvalEnv(
        values={
            ("spot", "AAPL"): 162.0,
            ("spot", "SPX"): 4525.0,
            ("spot", "NDX"): 18100.0,
            ("spot", "SPX", jan): 4410.0,
            ("spot", "SPX", feb): 4475.0,
            ("spot", "SPX", mar): 4510.0,
            ("spot", "SPX", apr): 4555.0,
            ("swap_rate", "USD-IRS-5Y", swap_schedule.key()): 0.047,
            ("annuity", "USD-IRS-5Y", swap_schedule.key()): 4.25,
            ("variance_observable", "SPX", date(2025, 1, 1), date(2025, 11, 15)): 0.052,
        }
    )


_SWAP_SCHEDULE = _finite_schedule("2026-11-15", "2027-11-15")
_ASIAN_SCHEDULE = _finite_schedule("2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01")
_VARIANCE_INTERVAL = _interval("2025-01-01", "2025-11-15")


def _finite_float(min_value: float, max_value: float):
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )


def _payoff_expr_strategy():
    leaf = st.one_of(
        _finite_float(-5.0, 5.0).map(Constant),
        st.sampled_from(("AAPL", "SPX", "NDX")).map(Spot),
        st.sampled_from((-10.0, 0.0, 1.0, 2.0, 150.0, 4500.0)).map(Strike),
        st.just(SwapRate("USD-IRS-5Y", _SWAP_SCHEDULE)),
        st.just(Annuity("USD-IRS-5Y", _SWAP_SCHEDULE)),
        st.just(VarianceObservable("SPX", _VARIANCE_INTERVAL)),
        st.just(ArithmeticMean(Spot("SPX"), _ASIAN_SCHEDULE)),
    )

    scalar = st.one_of(
        _finite_float(-2.0, 2.0).map(Constant),
        st.just(Annuity("USD-IRS-5Y", _SWAP_SCHEDULE)),
    )

    return st.recursive(
        leaf,
        lambda children: st.one_of(
            st.tuples(children, children).map(Add),
            st.tuples(children, children).map(Mul),
            st.tuples(children, children).map(lambda pair: Max(tuple(pair))),
            st.tuples(children, children).map(lambda pair: Min(tuple(pair))),
            st.tuples(children, children).map(lambda pair: Sub(pair[0], pair[1])),
            st.tuples(scalar, children).map(lambda pair: Scaled(pair[0], pair[1])),
            st.tuples(children, children).map(
                lambda pair: Indicator(Gt(pair[0], pair[1]))
            ),
            st.just(LinearBasket(((0.5, Spot("SPX")), (0.5, Spot("NDX"))))),
        ),
        max_leaves=8,
    )


@st.composite
def _environment_strategy(draw):
    jan = date(2025, 1, 1)
    feb = date(2025, 2, 1)
    mar = date(2025, 3, 1)
    apr = date(2025, 4, 1)
    return PayoffEvalEnv(
        values={
            ("spot", "AAPL"): draw(_finite_float(50.0, 250.0)),
            ("spot", "SPX"): draw(_finite_float(3000.0, 6000.0)),
            ("spot", "NDX"): draw(_finite_float(10000.0, 25000.0)),
            ("spot", "SPX", jan): draw(_finite_float(3000.0, 6000.0)),
            ("spot", "SPX", feb): draw(_finite_float(3000.0, 6000.0)),
            ("spot", "SPX", mar): draw(_finite_float(3000.0, 6000.0)),
            ("spot", "SPX", apr): draw(_finite_float(3000.0, 6000.0)),
            ("swap_rate", "USD-IRS-5Y", _SWAP_SCHEDULE.key()): draw(_finite_float(-0.05, 0.15)),
            ("annuity", "USD-IRS-5Y", _SWAP_SCHEDULE.key()): draw(_finite_float(0.1, 10.0)),
            (
                "variance_observable",
                "SPX",
                date(2025, 1, 1),
                date(2025, 11, 15),
            ): draw(_finite_float(0.0, 0.5)),
        }
    )


def _property_settings(*, max_examples: int):
    return settings(
        max_examples=max_examples,
        deadline=None,
        derandomize=True,
        suppress_health_check=[HealthCheck.too_slow],
    )


class TestContractIRSimplify:
    def test_call_shape_sorts_zero_to_second_argument(self):
        expr = Max((Constant(0.0), Sub(Spot("AAPL"), Strike(150.0))))
        assert canonicalize(expr) == Max(
            (Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0))
        )

    def test_negative_short_call_is_not_rewritten_into_put_orientation(self):
        short_call = Scaled(
            Constant(-1.0),
            Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0))),
        )
        canonical = canonicalize(short_call)
        assert canonical == short_call
        assert canonical != Max((Sub(Strike(150.0), Spot("AAPL")), Constant(0.0)))

    def test_linear_basket_zero_and_singleton_rules(self):
        assert canonicalize(
            LinearBasket(((0.0, Spot("SPX")), (0.0, Spot("NDX"))))
        ) == Constant(0.0)
        assert canonicalize(
            LinearBasket(((2.0, Spot("SPX")),))
        ) == Scaled(Constant(2.0), Spot("SPX"))

    def test_factor_common_positive_scalar_out_of_max(self):
        annuity = Annuity(
            "USD-IRS-5Y",
            _finite_schedule("2026-11-15", "2027-11-15"),
        )
        expr = Max(
            (
                Scaled(
                    annuity,
                    Sub(
                        SwapRate("USD-IRS-5Y", _finite_schedule("2026-11-15", "2027-11-15")),
                        Strike(0.05),
                    ),
                ),
                Constant(0.0),
            )
        )
        assert canonicalize(expr) == Scaled(
            annuity,
            Max(
                (
                    Sub(
                        SwapRate("USD-IRS-5Y", _finite_schedule("2026-11-15", "2027-11-15")),
                        Strike(0.05),
                    ),
                    Constant(0.0),
                )
            ),
        )

    def test_positive_scaled_sum_stays_factorized(self):
        expr = Scaled(Constant(2.0), Add((Spot("AAPL"), Constant(1.0))))
        assert canonicalize(expr) == Scaled(
            Constant(2.0),
            Add((Spot("AAPL"), Constant(1.0))),
        )

    def test_family_templates_canonicalize_to_expected_forms(self):
        assert canonicalize(
            Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0)))
        ) == Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0)))

        assert canonicalize(
            Scaled(
                Constant(10000.0),
                Sub(
                    VarianceObservable("SPX", _interval("2025-01-01", "2025-11-15")),
                    Strike(0.04),
                ),
            )
        ) == Scaled(
            Constant(10000.0),
            Sub(
                VarianceObservable("SPX", _interval("2025-01-01", "2025-11-15")),
                Strike(0.04),
            ),
        )

        assert canonicalize(
            Mul((Constant(2.0), Indicator(Gt(Spot("AAPL"), Strike(150.0)))))
        ) == Mul((Constant(2.0), Indicator(Gt(Spot("AAPL"), Strike(150.0)))))

        assert canonicalize(
            Max(
                (
                    Sub(
                        ArithmeticMean(
                            Spot("SPX"),
                            _finite_schedule(
                                "2025-01-01",
                                "2025-02-01",
                                "2025-03-01",
                                "2025-04-01",
                            ),
                        ),
                        Strike(4500.0),
                    ),
                    Constant(0.0),
                )
            )
        ) == Max(
            (
                Sub(
                    ArithmeticMean(
                        Spot("SPX"),
                        _finite_schedule(
                            "2025-01-01",
                            "2025-02-01",
                            "2025-03-01",
                            "2025-04-01",
                        ),
                    ),
                    Strike(4500.0),
                ),
                Constant(0.0),
            )
        )

    @_property_settings(max_examples=1000)
    @given(_payoff_expr_strategy())
    def test_canonicalize_is_idempotent_property(self, expr):
        once = canonicalize(expr)
        twice = canonicalize(once)
        assert twice == once

    @_property_settings(max_examples=500)
    @given(_payoff_expr_strategy(), _payoff_expr_strategy(), _payoff_expr_strategy())
    def test_canonicalize_agrees_on_equivalent_orderings(self, a, b, c):
        assert canonicalize(Max((a, b, c))) == canonicalize(Max((c, a, b)))
        assert canonicalize(Add((a, b, c))) == canonicalize(Add((b, c, a)))
        assert canonicalize(Mul((a, b, c))) == canonicalize(Mul((c, b, a)))

    @_property_settings(max_examples=500)
    @given(_payoff_expr_strategy(), _environment_strategy())
    def test_canonicalize_preserves_numeric_semantics(self, expr, env):
        before = evaluate_payoff_expr(expr, env)
        after = evaluate_payoff_expr(canonicalize(expr), env)
        assert math.isclose(after, before, rel_tol=1e-12, abs_tol=5e-12)

    def test_canonicalize_preserves_numeric_semantics_on_reference_environment(self):
        expr = Add(
            (
                Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0))),
                Scaled(
                    Constant(2.0),
                    Mul((Constant(2.0), Indicator(Gt(Spot("AAPL"), Strike(150.0))))),
                ),
            )
        )
        before = evaluate_payoff_expr(expr, _environment())
        after = evaluate_payoff_expr(canonicalize(expr), _environment())
        assert math.isclose(after, before, rel_tol=1e-12, abs_tol=5e-12)
