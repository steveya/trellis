from __future__ import annotations

from dataclasses import replace
from datetime import date

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
)
from trellis.agent.contract_ir_solver_compiler import (
    ContractIRSolverBindingError,
    ContractIRSolverNoMatchError,
    build_contract_ir_term_environment,
    compile_contract_ir_solver,
    execute_contract_ir_solver_decision,
)
from trellis.agent.platform_requests import compile_build_request
from trellis.agent.semantic_contract_compiler import compile_semantic_contract
from trellis.agent.semantic_contracts import (
    make_rate_style_swaption_contract,
    make_vanilla_option_contract,
)
from trellis.agent.valuation_context import build_valuation_context
from trellis.core.types import DayCountConvention, Frequency
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.basket_option import price_basket_option_analytical
from trellis.models.black import (
    black76_asset_or_nothing_call,
    black76_asset_or_nothing_put,
    black76_call,
    black76_cash_or_nothing_put,
    black76_put,
    black76_cash_or_nothing_call,
)
from trellis.models.rate_style_swaption import price_swaption_black76
from trellis.models.analytical.equity_exotics import price_equity_variance_swap_analytical
from trellis.models.vol_surface import FlatVol


def _singleton(day: str) -> Singleton:
    return Singleton(date.fromisoformat(day))


def _finite_schedule(*days: str) -> FiniteSchedule:
    return FiniteSchedule(tuple(date.fromisoformat(day) for day in days))


def _vanilla_call_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0))),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
    )


def _digital_cash_call_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Mul((Constant(2.0), Indicator(Gt(Spot("AAPL"), Strike(150.0))))),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
    )


def _digital_cash_put_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Mul((Constant(2.0), Indicator(Lt(Spot("AAPL"), Strike(150.0))))),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
    )


def _digital_asset_call_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Mul((Spot("AAPL"), Indicator(Gt(Spot("AAPL"), Strike(150.0))))),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
    )


def _digital_asset_put_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Mul((Spot("AAPL"), Indicator(Lt(Spot("AAPL"), Strike(150.0))))),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
    )


def _vanilla_put_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Max((Sub(Strike(150.0), Spot("AAPL")), Constant(0.0))),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=EquitySpot("AAPL", "gbm")),
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


def _receiver_swaption_contract_ir() -> ContractIR:
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
            Max((Sub(Strike(0.05), SwapRate("USD-IRS-5Y", schedule)), Constant(0.0))),
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=ForwardRate("USD-IRS-5Y", "lognormal_forward")),
    )


def _forward_starting_swaption_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    schedule = _finite_schedule(
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


def _single_payment_swaption_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    schedule = _finite_schedule("2030-11-15")
    return ContractIR(
        payoff=Scaled(
            Annuity("USD-IRS-5Y", schedule),
            Max((Sub(SwapRate("USD-IRS-5Y", schedule), Strike(0.05)), Constant(0.0))),
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=ForwardRate("USD-IRS-5Y", "lognormal_forward")),
    )


def _forward_starting_eom_swaption_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    schedule = _finite_schedule(
        "2027-04-30",
        "2027-10-31",
        "2028-04-30",
        "2028-10-31",
    )
    return ContractIR(
        payoff=Scaled(
            Annuity("USD-IRS-3Y", schedule),
            Max((Sub(SwapRate("USD-IRS-3Y", schedule), Strike(0.05)), Constant(0.0))),
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=ForwardRate("USD-IRS-3Y", "lognormal_forward")),
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


def _basket_put_contract_ir() -> ContractIR:
    expiry = _singleton("2025-11-15")
    return ContractIR(
        payoff=Max(
            (
                Sub(
                    Strike(4500.0),
                    LinearBasket(((0.5, Spot("SPX")), (0.5, Spot("NDX")))),
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


def _equity_market_state(*, underlier: str = "AAPL", spot: float = 165.0, rate: float = 0.03, vol: float = 0.2) -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(rate),
        vol_surface=FlatVol(vol),
        spot=spot,
        underlier_spots={underlier: spot},
    )


def _swaption_market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.20),
    )


def _swaption_market_state_with_named_forecast_curve() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.20),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.05)},
    )


def _basket_market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.20),
        underlier_spots={"SPX": 4500.0, "NDX": 15000.0},
        model_parameters={
            "correlation_matrix": ((1.0, 0.35), (0.35, 1.0)),
            "underlier_carry_rates": {"SPX": 0.0, "NDX": 0.0},
        },
    )


def _variance_market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.02),
        vol_surface=FlatVol(0.25),
        spot=5000.0,
        underlier_spots={"SPX": 5000.0},
    )


class TestContractIRSolverCompiler:
    def test_vanilla_black76_call_is_route_mask_invariant(self):
        contract = _vanilla_call_contract_ir()
        market_state = _equity_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        decision = compile_contract_ir_solver(
            contract,
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )
        masked = compile_contract_ir_solver(
            contract,
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        assert decision.declaration_id == "black76_vanilla_call"
        assert decision.callable_ref == "trellis.models.black.black76_call"
        assert masked.declaration_id == decision.declaration_id
        assert masked.call_kwargs == decision.call_kwargs
        assert masked.value_scale == decision.value_scale

        price = execute_contract_ir_solver_decision(decision)
        T = float(decision.call_kwargs["T"])
        strike = float(decision.call_kwargs["K"])
        df = market_state.discount.discount(T)
        forward = float(market_state.spot) / max(float(df), 1e-12)
        expected = float(df) * black76_call(forward, strike, 0.2, T)
        assert price == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_vanilla_black76_put_binds_correctly(self):
        contract = _vanilla_put_contract_ir()
        market_state = _equity_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        decision = compile_contract_ir_solver(
            contract,
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        assert decision.declaration_id == "black76_vanilla_put"
        price = execute_contract_ir_solver_decision(decision)
        T = float(decision.call_kwargs["T"])
        strike = float(decision.call_kwargs["K"])
        df = market_state.discount.discount(T)
        forward = float(market_state.spot) / max(float(df), 1e-12)
        expected = float(df) * black76_put(forward, strike, 0.2, T)
        assert price == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_digital_basis_kernels_bind_correctly(self):
        market_state = _equity_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        cash_call_decision = compile_contract_ir_solver(
            _digital_cash_call_contract_ir(),
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )
        cash_put_decision = compile_contract_ir_solver(
            _digital_cash_put_contract_ir(),
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )
        asset_call_decision = compile_contract_ir_solver(
            _digital_asset_call_contract_ir(),
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )
        asset_put_decision = compile_contract_ir_solver(
            _digital_asset_put_contract_ir(),
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        assert cash_call_decision.declaration_id == "black76_cash_digital_call"
        assert cash_put_decision.declaration_id == "black76_cash_digital_put"
        assert asset_call_decision.declaration_id == "black76_asset_digital_call"
        assert asset_put_decision.declaration_id == "black76_asset_digital_put"

        cash_call_price = execute_contract_ir_solver_decision(cash_call_decision)
        cash_put_price = execute_contract_ir_solver_decision(cash_put_decision)
        asset_call_price = execute_contract_ir_solver_decision(asset_call_decision)
        asset_put_price = execute_contract_ir_solver_decision(asset_put_decision)

        T = float(cash_call_decision.call_kwargs["T"])
        K = float(cash_call_decision.call_kwargs["K"])
        df = float(market_state.discount.discount(T))
        forward = float(market_state.spot) / max(df, 1e-12)
        expected_cash_call = 2.0 * df * black76_cash_or_nothing_call(forward, K, 0.2, T)
        expected_cash_put = 2.0 * df * black76_cash_or_nothing_put(forward, K, 0.2, T)
        expected_asset_call = df * black76_asset_or_nothing_call(forward, K, 0.2, T)
        expected_asset_put = df * black76_asset_or_nothing_put(forward, K, 0.2, T)
        assert cash_call_price == pytest.approx(expected_cash_call, rel=1e-12, abs=1e-12)
        assert cash_put_price == pytest.approx(expected_cash_put, rel=1e-12, abs=1e-12)
        assert asset_call_price == pytest.approx(expected_asset_call, rel=1e-12, abs=1e-12)
        assert asset_put_price == pytest.approx(expected_asset_put, rel=1e-12, abs=1e-12)

    def test_swaption_helper_binding_matches_exact_helper(self):
        contract = _swaption_contract_ir()
        market_state = _swaption_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        decision = compile_contract_ir_solver(
            contract,
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        assert decision.declaration_id == "helper_swaption_payer_black76"
        spec = decision.call_kwargs["spec"]
        assert spec.swap_frequency == Frequency.SEMI_ANNUAL
        assert spec.day_count == DayCountConvention.ACT_360
        assert execute_contract_ir_solver_decision(decision) == pytest.approx(
            price_swaption_black76(market_state, spec),
            rel=1e-12,
            abs=1e-12,
        )

    def test_receiver_swaption_helper_binding_matches_exact_helper(self):
        contract = _receiver_swaption_contract_ir()
        market_state = _swaption_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        decision = compile_contract_ir_solver(
            contract,
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        assert decision.declaration_id == "helper_swaption_receiver_black76"
        spec = decision.call_kwargs["spec"]
        assert spec.is_payer is False
        assert execute_contract_ir_solver_decision(decision) == pytest.approx(
            price_swaption_black76(market_state, spec),
            rel=1e-12,
            abs=1e-12,
        )

    def test_swaption_helper_respects_explicit_swap_start_term_field(self):
        market_state = _swaption_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))
        semantic_contract = make_rate_style_swaption_contract(
            description="European payer swaption with explicit forward-start date",
            observation_schedule=("2025-11-15",),
            preferred_method="analytical",
            term_fields={
                "swap_start": date(2026, 5, 15),
                "swap_frequency": "semi_annual",
            },
        )

        decision = compile_contract_ir_solver(
            _swaption_contract_ir(),
            term_environment=build_contract_ir_term_environment(semantic_contract),
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        spec = decision.call_kwargs["spec"]
        assert spec.swap_start == date(2026, 5, 15)
        assert execute_contract_ir_solver_decision(decision) == pytest.approx(
            price_swaption_black76(market_state, spec),
            rel=1e-12,
            abs=1e-12,
        )

    def test_swaption_helper_accepts_start_date_alias_and_explicit_swap_end(self):
        market_state = _swaption_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))
        semantic_contract = make_rate_style_swaption_contract(
            description="European payer swaption with aliased schedule anchors",
            observation_schedule=("2025-11-15",),
            preferred_method="analytical",
            term_fields={
                "start_date": date(2026, 5, 15),
                "end_date": date(2031, 5, 15),
                "frequency": "semi_annual",
            },
        )

        decision = compile_contract_ir_solver(
            _swaption_contract_ir(),
            term_environment=build_contract_ir_term_environment(semantic_contract),
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        spec = decision.call_kwargs["spec"]
        assert spec.swap_start == date(2026, 5, 15)
        assert spec.swap_end == date(2031, 5, 15)
        assert spec.swap_frequency == Frequency.SEMI_ANNUAL

    def test_swaption_helper_infers_forward_start_from_schedule_when_term_field_missing(self):
        market_state = _swaption_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        decision = compile_contract_ir_solver(
            _forward_starting_swaption_contract_ir(),
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        spec = decision.call_kwargs["spec"]
        assert spec.swap_start == date(2026, 11, 15)
        assert execute_contract_ir_solver_decision(decision) == pytest.approx(
            price_swaption_black76(market_state, spec),
            rel=1e-12,
            abs=1e-12,
        )

    def test_swaption_helper_preserves_end_of_month_inferred_start(self):
        market_state = _swaption_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        decision = compile_contract_ir_solver(
            _forward_starting_eom_swaption_contract_ir(),
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        spec = decision.call_kwargs["spec"]
        assert spec.swap_start == date(2026, 10, 31)

    def test_swaption_helper_fails_closed_without_authoritative_start_for_single_payment_schedule(self):
        market_state = _swaption_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        with pytest.raises(
            ContractIRSolverBindingError,
            match="requires explicit swap_start or a multi-date schedule for structural binding",
        ) as exc_info:
            compile_contract_ir_solver(
                _single_payment_swaption_contract_ir(),
                valuation_context=context,
                market_state=market_state,
                preferred_method="analytical",
            )

        assert exc_info.value.best_diagnostic.declaration_id == "helper_swaption_payer_black76"
        assert exc_info.value.best_diagnostic.error_type == "ValueError"

    def test_swaption_helper_uses_forecast_curve_name_when_rate_index_missing(self):
        market_state = _swaption_market_state_with_named_forecast_curve()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))
        semantic_contract = make_rate_style_swaption_contract(
            description="European payer swaption with explicit forecast curve",
            observation_schedule=("2025-11-15",),
            preferred_method="analytical",
            term_fields={"forecast_curve_name": "USD-SOFR-3M"},
        )

        decision = compile_contract_ir_solver(
            _swaption_contract_ir(),
            term_environment=build_contract_ir_term_environment(semantic_contract),
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        spec = decision.call_kwargs["spec"]
        selected_price = execute_contract_ir_solver_decision(decision)
        fallback_price = price_swaption_black76(market_state, replace(spec, rate_index=None))

        assert spec.rate_index == "USD-SOFR-3M"
        assert "forecast_curve:USD-SOFR-3M" in decision.resolved_market_coordinates
        assert selected_price == pytest.approx(
            price_swaption_black76(market_state, spec),
            rel=1e-12,
            abs=1e-12,
        )
        assert selected_price != pytest.approx(fallback_price, rel=1e-9, abs=1e-9)

    def test_two_asset_basket_helper_binding_matches_exact_helper(self):
        contract = _basket_call_contract_ir()
        market_state = _basket_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        decision = compile_contract_ir_solver(
            contract,
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        assert decision.declaration_id == "helper_basket_option_call"
        spec = decision.call_kwargs["spec"]
        assert spec.basket_style == "weighted_sum"
        assert execute_contract_ir_solver_decision(decision) == pytest.approx(
            price_basket_option_analytical(market_state, spec),
            rel=1e-12,
            abs=1e-12,
        )

    def test_two_asset_basket_put_helper_binding_matches_exact_helper(self):
        contract = _basket_put_contract_ir()
        market_state = _basket_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        decision = compile_contract_ir_solver(
            contract,
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        assert decision.declaration_id == "helper_basket_option_put"
        spec = decision.call_kwargs["spec"]
        assert spec.option_type == "put"
        assert execute_contract_ir_solver_decision(decision) == pytest.approx(
            price_basket_option_analytical(market_state, spec),
            rel=1e-12,
            abs=1e-12,
        )

    def test_variance_helper_binding_matches_exact_helper(self):
        contract = _variance_contract_ir()
        market_state = _variance_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        decision = compile_contract_ir_solver(
            contract,
            valuation_context=context,
            market_state=market_state,
            preferred_method="analytical",
        )

        assert decision.declaration_id == "helper_equity_variance_swap"
        spec = decision.call_kwargs["spec"]
        assert execute_contract_ir_solver_decision(decision) == pytest.approx(
            price_equity_variance_swap_analytical(market_state, spec),
            rel=1e-12,
            abs=1e-12,
        )

    def test_unsupported_outputs_fail_closed(self):
        market_state = _equity_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price", "delta"))

        with pytest.raises(ContractIRSolverNoMatchError):
            compile_contract_ir_solver(
                _vanilla_call_contract_ir(),
                valuation_context=context,
                market_state=market_state,
                preferred_method="analytical",
                requested_outputs=("price", "delta"),
            )

    def test_asian_contract_remains_explicitly_unmigrated(self):
        market_state = _variance_market_state()
        context = build_valuation_context(market_snapshot=market_state, requested_outputs=("price",))

        with pytest.raises(ContractIRSolverNoMatchError):
            compile_contract_ir_solver(
                _asian_contract_ir(),
                valuation_context=context,
                market_state=market_state,
                preferred_method="analytical",
            )

    def test_semantic_blueprint_attaches_contract_ir_shadow_when_market_is_bound(self):
        market_state = _equity_market_state()
        contract = make_vanilla_option_contract(
            description="European call on AAPL strike 150 expiring 2025-11-15",
            underliers=("AAPL",),
            observation_schedule=("2025-11-15",),
            preferred_method="analytical",
        )

        blueprint = compile_semantic_contract(
            contract,
            valuation_context=build_valuation_context(
                market_snapshot=market_state,
                requested_outputs=("price",),
            ),
            preferred_method="analytical",
        )

        assert blueprint.contract_ir_solver_shadow is not None
        assert blueprint.contract_ir_solver_shadow.declaration_id == "black76_vanilla_call"
        assert blueprint.contract_ir_solver_shadow.legacy_route_id == "analytical_black76"

    def test_compiled_request_metadata_surfaces_contract_ir_shadow(self):
        market_state = _equity_market_state()
        compiled = compile_build_request(
            "European call on AAPL strike 150 expiring 2025-11-15",
            instrument_type="european_option",
            market_snapshot=market_state,
            preferred_method="analytical",
        )

        shadow = compiled.request.metadata["semantic_blueprint"]["contract_ir_solver_shadow"]
        assert shadow["declaration_id"] == "black76_vanilla_call"
        assert shadow["legacy_route_id"] == "analytical_black76"

    def test_term_environment_reads_generic_term_groups_from_semantic_contract(self):
        contract = make_rate_style_swaption_contract(
            description="European payer swaption on 5Y USD IRS strike 5% expiring 2025-11-15",
            observation_schedule=("2025-11-15",),
            preferred_method="analytical",
            term_fields={
                "notional": 2_000_000,
                "rate_index": "USD-SOFR-3M",
                "swap_frequency": "semi_annual",
                "fixed_leg_day_count": "ACT_360",
                "replication_strikes": "0.04,0.05,0.06",
                "replication_volatilities": "0.18,0.20,0.22",
            },
        )

        terms = build_contract_ir_term_environment(contract)

        assert terms.cash_settlement.notional == pytest.approx(2_000_000)
        assert terms.floating_rate_reference.rate_index == "USD-SOFR-3M"
        assert terms.accrual_conventions.payment_frequency == Frequency.SEMI_ANNUAL
        assert terms.quote_grid.replication_strikes == pytest.approx((0.04, 0.05, 0.06))

    def test_term_environment_preserves_zero_notional(self):
        contract = make_rate_style_swaption_contract(
            description="Zero-notional swaption fixture",
            observation_schedule=("2025-11-15",),
            preferred_method="analytical",
            term_fields={"notional": 0.0},
        )

        terms = build_contract_ir_term_environment(contract)

        assert terms.cash_settlement.notional == pytest.approx(0.0)
