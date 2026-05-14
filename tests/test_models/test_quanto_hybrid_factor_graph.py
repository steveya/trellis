from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.resolution.quanto import (
    HybridFactorGraph,
    build_quanto_hybrid_factor_graph,
    resolve_quanto_inputs,
)
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)


@dataclass(frozen=True)
class _QuantoSpec:
    strike: float = 100.0
    expiry_date: date = date(2025, 11, 15)
    fx_pair: str = "EURUSD"
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    day_count: object = DayCountConvention.ACT_365
    quanto_correlation_key: str | None = "sx5e_eurusd"


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.03)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        spot=100.0,
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        model_parameters={"sx5e_eurusd": {"kind": "explicit", "value": 0.25}},
        selected_curve_names={"discount_curve": "USD-OIS", "forecast_curve": "EUR-DISC"},
    )


def _resolved_payload(resolved) -> tuple[float, ...]:
    return (
        float(resolved.spot),
        float(resolved.fx_spot),
        float(resolved.T),
        float(resolved.domestic_df),
        float(resolved.foreign_df),
        float(resolved.sigma_underlier),
        float(resolved.sigma_fx),
        float(resolved.corr),
    )


def test_quanto_resolver_default_path_does_not_attach_hybrid_graph():
    resolved = resolve_quanto_inputs(_market_state(), _QuantoSpec())

    assert resolved.hybrid_factor_graph is None


def test_quanto_resolver_can_attach_hybrid_factor_graph_without_numeric_change():
    market_state = _market_state()
    spec = _QuantoSpec()
    baseline = resolve_quanto_inputs(market_state, spec)
    resolved = resolve_quanto_inputs(
        market_state,
        spec,
        include_hybrid_factor_graph=True,
    )

    assert _resolved_payload(resolved) == pytest.approx(_resolved_payload(baseline))
    assert isinstance(resolved.hybrid_factor_graph, HybridFactorGraph)

    graph = resolved.hybrid_factor_graph.validate()
    roles = {
        role
        for node in graph.nodes
        for role in node.metadata.get("resolved_inputs", ())
    }

    assert graph.graph_id == "quanto:EUR:USD:EURUSD"
    assert roles >= {
        "underlier_spot",
        "fx_spot",
        "domestic_curve",
        "foreign_curve",
        "underlier_vol",
        "fx_vol",
        "correlation",
    }
    assert graph.unsupported_dependencies == ()
    assert any(
        coordinate.factor_id.coordinate_type == "correlation"
        for coordinate in graph.coordinates
    )
    assert any(
        coordinate.factor_id.coordinate_type == "zero_rate"
        for coordinate in graph.coordinates
    )
    assert any(
        coordinate.factor_id.coordinate_type == "flat_vol"
        for coordinate in graph.coordinates
    )


def test_quanto_hybrid_factor_graph_builder_marks_missing_foreign_curve_unsupported():
    market_state = _market_state()
    market_state = MarketState(
        as_of=market_state.as_of,
        settlement=market_state.settlement,
        discount=market_state.discount,
        fx_rates=market_state.fx_rates,
        spot=market_state.spot,
        underlier_spots=market_state.underlier_spots,
        vol_surface=market_state.vol_surface,
        model_parameters=market_state.model_parameters,
    )
    spec = _QuantoSpec()
    resolved = resolve_quanto_inputs(
        _market_state(),
        spec,
        include_hybrid_factor_graph=False,
    )

    graph = build_quanto_hybrid_factor_graph(market_state, spec, resolved)

    assert "foreign_curve_unresolved" in graph.unsupported_reasons
    assert any(
        node.support_status == "held_fixed"
        for node in graph.nodes
        if "foreign_curve" in node.metadata.get("resolved_inputs", ())
    )
