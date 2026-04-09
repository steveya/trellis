"""Tests for single-method reference oracles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol
from trellis.models.zcb_option import price_zcb_option_jamshidian


SETTLE = date(2024, 11, 15)


@dataclass(frozen=True)
class _ZCBSpec:
    notional: float = 100.0
    strike: float = 63.0
    expiry_date: date = date(2027, 11, 15)
    bond_maturity_date: date = date(2032, 11, 15)
    day_count: DayCountConvention = DayCountConvention.ACT_365
    option_type: str = "call"


def _zcb_market_state(rate=0.05, vol=0.20) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(rate, max_tenor=31.0),
        vol_surface=FlatVol(vol),
    )


def test_select_reference_oracle_for_analytical_swaption_returns_exact_helper():
    from trellis.agent.reference_oracles import select_reference_oracle

    oracle = select_reference_oracle(
        instrument_type="swaption",
        method="analytical",
    )

    assert oracle is not None
    assert oracle.oracle_id == "swaption_black76_exact"
    assert oracle.relation == "within_tolerance"
    assert oracle.source == "trellis.models.rate_style_swaption.price_swaption_black76"


def test_select_reference_oracle_for_puttable_bond_returns_bound_helper():
    from trellis.agent.reference_oracles import select_reference_oracle

    oracle = select_reference_oracle(
        instrument_type="puttable_bond",
        method="rate_tree",
    )

    assert oracle is not None
    assert oracle.oracle_id == "puttable_bond_straight_bond_bound"
    assert oracle.relation == ">="


def test_select_reference_oracle_prefers_more_specific_product_family_over_generic_request_family():
    from trellis.agent.reference_oracles import select_reference_oracle

    oracle = select_reference_oracle(
        instrument_type="european_option",
        method="analytical",
        product_ir=SimpleNamespace(instrument="zcb_option"),
    )

    assert oracle is not None
    assert oracle.oracle_id == "zcb_option_jamshidian_exact"
    assert oracle.instrument_type == "zcb_option"


def test_execute_reference_oracle_passes_for_helper_backed_zcb_option():
    from trellis.agent.reference_oracles import execute_reference_oracle, select_reference_oracle

    class HelperBackedZCBPayoff:
        def __init__(self, spec):
            self._spec = spec

        @property
        def requirements(self):
            return {"discount_curve", "black_vol_surface"}

        def evaluate(self, market_state):
            return float(price_zcb_option_jamshidian(market_state, self._spec, mean_reversion=0.1))

    oracle = select_reference_oracle(instrument_type="zcb_option", method="analytical")
    execution = execute_reference_oracle(
        oracle,
        payoff_factory=lambda: HelperBackedZCBPayoff(_ZCBSpec()),
        market_state_factory=_zcb_market_state,
    )

    assert execution is not None
    assert execution.passed is True
    assert execution.failure_message is None
    assert execution.sampled_prices


def test_execute_reference_oracle_detects_zcb_magnitude_error():
    from trellis.agent.reference_oracles import execute_reference_oracle, select_reference_oracle

    class BrokenZCBPayoff:
        def __init__(self, spec):
            self._spec = spec

        @property
        def requirements(self):
            return {"discount_curve", "black_vol_surface"}

        def evaluate(self, market_state):
            return float(
                price_zcb_option_jamshidian(market_state, self._spec, mean_reversion=0.1)
                / self._spec.notional
            )

    oracle = select_reference_oracle(instrument_type="zcb_option", method="analytical")
    execution = execute_reference_oracle(
        oracle,
        payoff_factory=lambda: BrokenZCBPayoff(_ZCBSpec()),
        market_state_factory=_zcb_market_state,
    )

    assert execution is not None
    assert execution.passed is False
    assert execution.failure_message is not None
    assert execution.max_abs_deviation is not None


def test_execute_reference_oracle_threads_explicit_swaption_comparison_kwargs(monkeypatch):
    from trellis.agent.reference_oracles import execute_reference_oracle, select_reference_oracle
    from trellis.agent.valuation_context import (
        EngineModelSpec,
        PotentialSpec,
        RatesCurveRoleSpec,
        SourceSpec,
    )

    captured_kwargs: dict[str, object] = {}

    class _SwaptionSpec:
        notional = 1_000_000.0
        strike = 0.03
        expiry_date = date(2025, 11, 15)
        swap_start = date(2025, 11, 15)
        swap_end = date(2030, 11, 15)
        swap_frequency = "SEMI_ANNUAL"
        day_count = "THIRTY_360"
        rate_index = "USD-SOFR-3M"
        is_payer = True

    class _Payoff:
        def __init__(self):
            self._spec = _SwaptionSpec()

        @property
        def requirements(self):
            return {"discount_curve", "forward_curve", "black_vol_surface"}

        def evaluate(self, market_state):
            return 1.23

    def _fake_price_swaption_black76(market_state, spec, **kwargs):
        captured_kwargs.update(kwargs)
        return 1.23

    monkeypatch.setattr(
        "trellis.models.rate_style_swaption.price_swaption_black76",
        _fake_price_swaption_black76,
    )

    oracle = select_reference_oracle(instrument_type="swaption", method="analytical")
    semantic_blueprint = SimpleNamespace(
        valuation_context=SimpleNamespace(
            engine_model_spec=EngineModelSpec(
                model_family="rates",
                model_name="hull_white_1f",
                state_semantics=("short_rate",),
                potential=PotentialSpec(discount_term="risk_free_rate"),
                sources=(SourceSpec(source_kind="coupon_stream"),),
                calibration_requirements=("bootstrap_curve", "fit_hw_strip"),
                backend_hints=("analytical",),
                parameter_overrides={"mean_reversion": 0.05, "sigma": 0.01},
                rates_curve_roles=RatesCurveRoleSpec(
                    discount_curve_role="discount_curve",
                    forecast_curve_role="forward_curve",
                ),
            )
        )
    )

    execution = execute_reference_oracle(
        oracle,
        payoff_factory=_Payoff,
        market_state_factory=_zcb_market_state,
        semantic_blueprint=semantic_blueprint,
    )

    assert execution is not None
    assert execution.passed is True
    assert captured_kwargs == {
        "mean_reversion": 0.05,
        "sigma": 0.01,
    }
