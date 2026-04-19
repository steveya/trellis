from __future__ import annotations

from datetime import date

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
)
from trellis.agent.contract_pattern import (
    ContractPattern,
    ExercisePattern,
    PayoffPattern,
    UnderlyingPattern,
    dump_contract_pattern,
    parse_contract_pattern,
)
from trellis.agent.contract_pattern_eval import evaluate_pattern


def _singleton(day: str) -> Singleton:
    return Singleton(date(*map(int, day.split("-"))))


def _finite_schedule(*days: str) -> FiniteSchedule:
    return FiniteSchedule(tuple(date(*map(int, day.split("-"))) for day in days))


def _vanilla_call_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0))),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
    )


def _basket_call_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Max(
            (
                Sub(
                    LinearBasket(((0.5, Spot("SPX")), (0.5, Spot("NDX")))),
                    Strike(4500.0),
                ),
                Constant(0.0),
            )
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(
            spec=CompositeUnderlying((EquitySpot("SPX", "gbm"), EquitySpot("NDX", "gbm")))
        ),
    )


def _swaption_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    schedule = _finite_schedule(
        "2026-11-15",
        "2027-11-15",
        "2028-11-15",
        "2029-11-15",
        "2030-11-15",
    )
    return ContractIR(
        payoff=Scaled(
            Annuity("USD-IRS-5Y", schedule),
            Max((Sub(SwapRate("USD-IRS-5Y", schedule), Strike(0.05)), Constant(0.0))),
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=ForwardRate("USD-IRS-5Y", "lognormal_forward")),
    )


def _variance_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Scaled(
            Constant(10000.0),
            Sub(
                VarianceObservable(
                    "SPX",
                    ContinuousInterval(date(2025, 1, 1), date(2025, 11, 15)),
                ),
                Strike(0.04),
            ),
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("SPX", "gbm")),
    )


def _digital_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Mul((Constant(2.0), Indicator(Gt(Spot("AAPL"), Strike(150.0))))),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
    )


def _asian_contract_ir() -> ContractIR:
    averaging = _finite_schedule(
        "2025-01-31",
        "2025-02-28",
        "2025-03-31",
        "2025-04-30",
    )
    expiry = Singleton(date(2025, 4, 30))
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


class TestContractPatternEvalContractIR:
    def test_new_ir_heads_round_trip_through_parser(self):
        payload = {
            "payoff": {
                "kind": "scaled",
                "args": [
                    {"kind": "annuity", "args": ["_u", "_schedule"]},
                    {
                        "kind": "max",
                        "args": [
                            {
                                "kind": "sub",
                                "args": [
                                    {"kind": "swap_rate", "args": ["_u", "_schedule"]},
                                    {"kind": "strike", "value": "_K"},
                                ],
                            },
                            {"kind": "constant", "value": 0},
                        ],
                    },
                ],
            }
        }

        pattern = parse_contract_pattern(payload)

        assert parse_contract_pattern(dump_contract_pattern(pattern)) == pattern

    def test_structural_vanilla_pattern_matches_contract_ir_and_binds(self):
        pattern = parse_contract_pattern(
            {
                "payoff": {
                    "kind": "max",
                    "args": [
                        {
                            "kind": "sub",
                            "args": [
                                {"kind": "spot", "underlier": "_u"},
                                {"kind": "strike", "value": "_k"},
                            ],
                        },
                        {"kind": "constant", "value": 0},
                    ],
                }
            }
        )

        result = evaluate_pattern(pattern, _vanilla_call_contract_ir())

        assert result.ok is True
        assert result.bindings["u"] == "AAPL"
        assert result.bindings["k"] == 150.0

    def test_swaption_leaf_heads_match_contract_ir_and_bind_schedule(self):
        pattern = parse_contract_pattern(
            {
                "payoff": {
                    "kind": "scaled",
                    "args": [
                        {"kind": "annuity", "args": ["_u", "_schedule"]},
                        {
                            "kind": "max",
                            "args": [
                                {
                                    "kind": "sub",
                                    "args": [
                                        {"kind": "swap_rate", "args": ["_u", "_schedule"]},
                                        {"kind": "strike", "value": "_K"},
                                    ],
                                },
                                {"kind": "constant", "value": 0},
                            ],
                        },
                    ],
                }
            }
        )

        result = evaluate_pattern(pattern, _swaption_contract_ir())

        assert result.ok is True
        assert result.bindings["u"] == "USD-IRS-5Y"
        assert isinstance(result.bindings["schedule"], FiniteSchedule)
        assert result.bindings["K"] == 0.05

    def test_zero_arity_family_tags_match_phase_two_contract_ir_fixtures(self):
        fixtures = {
            "vanilla_payoff": _vanilla_call_contract_ir(),
            "basket_payoff": _basket_call_contract_ir(),
            "swaption_payoff": _swaption_contract_ir(),
            "variance_payoff": _variance_contract_ir(),
            "digital_payoff": _digital_contract_ir(),
            "asian_payoff": _asian_contract_ir(),
        }

        for tag, fixture in fixtures.items():
            result = evaluate_pattern(
                ContractPattern(payoff=PayoffPattern(kind=tag)),
                fixture,
            )
            assert result.ok is True

    def test_analytical_black76_style_patterns_preserve_contract_ir_parity(self):
        vanilla = ContractPattern(payoff=PayoffPattern(kind="vanilla_payoff"))
        basket = ContractPattern(
            payoff=PayoffPattern(kind="basket_payoff"),
            exercise=ExercisePattern(style="european"),
            underlying=UnderlyingPattern(kind="equity_diffusion"),
        )
        swaption_european = ContractPattern(
            payoff=PayoffPattern(kind="swaption_payoff"),
            exercise=ExercisePattern(style="european"),
        )

        assert evaluate_pattern(vanilla, _vanilla_call_contract_ir()).ok is True
        assert evaluate_pattern(vanilla, _swaption_contract_ir()).ok is False

        assert evaluate_pattern(basket, _basket_call_contract_ir()).ok is True
        assert evaluate_pattern(basket, _vanilla_call_contract_ir()).ok is False

        assert evaluate_pattern(swaption_european, _swaption_contract_ir()).ok is True
        assert evaluate_pattern(swaption_european, _vanilla_call_contract_ir()).ok is False
