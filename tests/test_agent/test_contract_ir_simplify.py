from __future__ import annotations

from datetime import date
import random

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
    return FiniteSchedule(
        tuple(date(*map(int, day.split("-"))) for day in days)
    )


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


def _random_leaf(rng: random.Random):
    leaf_type = rng.choice(
        (
            "constant",
            "spot",
            "strike",
            "swap_rate",
            "annuity",
            "variance",
            "arithmetic_mean",
        )
    )
    if leaf_type == "constant":
        return Constant(rng.choice((-3.0, -1.0, 0.0, 1.0, 2.0, 5.0)))
    if leaf_type == "spot":
        return Spot(rng.choice(("AAPL", "SPX", "NDX")))
    if leaf_type == "strike":
        return Strike(rng.choice((0.0, 1.0, 2.0, 150.0, 4500.0)))
    if leaf_type == "swap_rate":
        return SwapRate(
            "USD-IRS-5Y",
            _finite_schedule("2026-11-15", "2027-11-15"),
        )
    if leaf_type == "annuity":
        return Annuity(
            "USD-IRS-5Y",
            _finite_schedule("2026-11-15", "2027-11-15"),
        )
    if leaf_type == "variance":
        return VarianceObservable(
            "SPX",
            _interval("2025-01-01", "2025-11-15"),
        )
    return ArithmeticMean(
        Spot("SPX"),
        _finite_schedule("2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"),
    )


def _random_expr(rng: random.Random, depth: int):
    if depth <= 0:
        return _random_leaf(rng)

    kind = rng.choice(
        (
            "leaf",
            "add",
            "max",
            "mul",
            "sub",
            "scaled",
            "indicator",
            "basket",
        )
    )
    if kind == "leaf":
        return _random_leaf(rng)
    if kind == "add":
        return Add((_random_expr(rng, depth - 1), _random_expr(rng, depth - 1)))
    if kind == "max":
        return Max((_random_expr(rng, depth - 1), _random_expr(rng, depth - 1)))
    if kind == "mul":
        return Mul((_random_expr(rng, depth - 1), _random_expr(rng, depth - 1)))
    if kind == "sub":
        return Sub(_random_expr(rng, depth - 1), _random_expr(rng, depth - 1))
    if kind == "scaled":
        scalar = rng.choice(
            (
                Constant(rng.choice((-2.0, -1.0, 0.0, 1.0, 2.0))),
                Annuity(
                    "USD-IRS-5Y",
                    _finite_schedule("2026-11-15", "2027-11-15"),
                ),
            )
        )
        return Scaled(scalar, _random_expr(rng, depth - 1))
    if kind == "indicator":
        return Indicator(
            Gt(_random_expr(rng, depth - 1), _random_expr(rng, depth - 1))
        )
    return LinearBasket(
        (
            (0.5, Spot("SPX")),
            (0.5, Spot("NDX")),
        )
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
                Scaled(annuity, Sub(SwapRate("USD-IRS-5Y", _finite_schedule("2026-11-15", "2027-11-15")), Strike(0.05))),
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

    def test_canonicalize_is_idempotent_on_deterministic_random_trees(self):
        rng = random.Random(917)
        for _ in range(200):
            expr = _random_expr(rng, depth=3)
            once = canonicalize(expr)
            twice = canonicalize(once)
            assert twice == once

    def test_canonicalize_agrees_on_equivalent_orderings(self):
        rng = random.Random(918)
        for _ in range(150):
            a = _random_expr(rng, depth=2)
            b = _random_expr(rng, depth=2)
            c = _random_expr(rng, depth=2)
            assert canonicalize(Max((a, b, c))) == canonicalize(Max((c, a, b)))
            assert canonicalize(Add((a, b, c))) == canonicalize(Add((b, c, a)))
            assert canonicalize(Mul((a, b, c))) == canonicalize(Mul((c, b, a)))

    def test_canonicalize_preserves_numeric_semantics(self):
        rng = random.Random(919)
        env = _environment()
        for _ in range(150):
            expr = _random_expr(rng, depth=3)
            before = evaluate_payoff_expr(expr, env)
            after = evaluate_payoff_expr(canonicalize(expr), env)
            assert abs(after - before) <= 5e-12
