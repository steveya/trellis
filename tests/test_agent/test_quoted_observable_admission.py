from __future__ import annotations

from datetime import date

import pytest

from trellis.agent.contract_ir import (
    Constant,
    ContractIR,
    CurveQuote,
    Exercise,
    Max,
    Observation,
    ParRateTenor,
    QuoteCurve,
    QuoteSurface,
    Scaled,
    Singleton,
    Strike,
    Sub,
    SurfaceQuote,
    Underlying,
    VolPoint,
)
from trellis.agent.contract_ir_solver_compiler import ContractIRSolverNoMatchError
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.agent.quoted_observable_admission import (
    default_quoted_observable_admission_registry,
    select_quoted_observable_lowering,
)


def _singleton(day: str) -> Singleton:
    return Singleton(date.fromisoformat(day))


def _curve_spread_contract_ir() -> ContractIR:
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


def _surface_spread_contract_ir() -> ContractIR:
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


def _curve_spread_option_contract_ir() -> ContractIR:
    expiry = _singleton("2026-06-30")
    return ContractIR(
        payoff=Max(
            (
                Sub(
                    Sub(
                        CurveQuote("USD_SWAP", ParRateTenor("10Y"), "par_rate"),
                        CurveQuote("USD_SWAP", ParRateTenor("2Y"), "par_rate"),
                    ),
                    Strike(0.0025),
                ),
                Constant(0.0),
            )
        ),
        exercise=Exercise(style="european", schedule=expiry),
        observation=Observation(kind="terminal", schedule=expiry),
        underlying=Underlying(spec=QuoteCurve("USD_SWAP")),
    )


class TestQuotedObservableAdmission:
    def test_curve_spread_selection_uses_bounded_quote_admission_registry(self):
        selection = select_quoted_observable_lowering(_curve_spread_contract_ir())
        assert selection.declaration_id == "quoted_observable_curve_spread_linear"
        assert selection.callable_ref == "trellis.models.quoted_observable.price_curve_quote_spread_analytical"
        assert selection.required_coordinate_kinds == ("curve_quote",)
        assert selection.consumed_term_groups == (
            "cash_settlement",
            "accrual_conventions",
        )

    def test_surface_spread_selection_uses_bounded_quote_admission_registry(self):
        selection = select_quoted_observable_lowering(_surface_spread_contract_ir())
        assert selection.declaration_id == "quoted_observable_surface_spread_linear"
        assert selection.callable_ref == "trellis.models.quoted_observable.price_surface_quote_spread_analytical"
        assert selection.required_coordinate_kinds == ("surface_quote",)
        assert selection.consumed_term_groups == (
            "cash_settlement",
            "accrual_conventions",
        )

    def test_option_on_quote_spread_is_not_admitted_until_a_checked_lowering_exists(self):
        with pytest.raises(ContractIRSolverNoMatchError):
            select_quoted_observable_lowering(_curve_spread_option_contract_ir())

    def test_registry_is_validated_through_the_shared_phase_three_registry_substrate(self):
        registry = default_quoted_observable_admission_registry()
        declaration_ids = tuple(item.declaration_id for item in registry.selection_order())
        assert declaration_ids == (
            "quoted_observable_curve_spread_linear",
            "quoted_observable_surface_spread_linear",
        )

    def test_default_structural_registry_admits_bounded_quoted_observable_helpers(self):
        from trellis.agent.contract_ir_solver_compiler import select_contract_ir_solver

        selection = select_contract_ir_solver(
            _surface_spread_contract_ir(),
            preferred_method="analytical",
        )

        assert selection.declaration_id == "quoted_observable_surface_spread_linear"
        assert selection.callable_ref == (
            "trellis.models.quoted_observable.price_surface_quote_spread_analytical"
        )

    def test_curve_spread_lowering_executes_through_checked_helper(self):
        from trellis.agent.contract_ir_solver_compiler import (
            build_contract_ir_term_environment,
            compile_contract_ir_solver,
            execute_contract_ir_solver_decision,
        )

        market_state = MarketState(
            as_of=date(2025, 1, 1),
            settlement=date(2025, 1, 1),
            discount=YieldCurve([0.0, 2.0, 10.0], [0.02, 0.025, 0.03]),
            selected_curve_names={"discount_curve": "USD_SWAP"},
        )
        decision = compile_contract_ir_solver(
            _curve_spread_contract_ir(),
            term_environment=build_contract_ir_term_environment(None),
            market_state=market_state,
            preferred_method="analytical",
            registry=default_quoted_observable_admission_registry(),
        )

        assert decision.declaration_id == "quoted_observable_curve_spread_linear"
        assert execute_contract_ir_solver_decision(decision) != 0.0
