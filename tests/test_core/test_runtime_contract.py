"""Tests for the shared helper-facing runtime contract."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.runtime_contract import (
    ContractState,
    ContractViolation,
    ResolvedInputs,
    wrap_market_state_with_contract,
)
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def test_contract_state_separates_event_state_and_contract_memory():
    state = ContractState(
        event_state={"remaining_pool": "pathwise"},
        contract_memory={"selection_rule": "best_of_remaining"},
        phase="observation",
    )

    assert state.require_event("remaining_pool") == "pathwise"
    assert state.require_memory("selection_rule") == "best_of_remaining"
    assert state.phase == "observation"

    updated = state.with_event_state(settlement="pending")
    assert updated.require_event("settlement") == "pending"
    with pytest.raises(KeyError):
        state.require_event("settlement")


def test_resolved_inputs_require_bindings_without_mutable_state():
    resolved = ResolvedInputs(
        bindings={"constituent_spots": (100.0, 95.0), "domestic_df": 0.97},
        requirements=("spot", "discount_curve"),
        source_kind="basket_semantics",
    )

    assert resolved.require("constituent_spots") == (100.0, 95.0)
    assert resolved.get("domestic_df") == 0.97
    assert resolved.requirements == ("spot", "discount_curve")
    assert resolved.source_kind == "basket_semantics"

    with pytest.raises(KeyError):
        resolved.require("missing_binding")


def test_market_state_contract_proxy_raises_for_missing_required_scalar_field():
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        underlier_spots={"AAPL": 100.0, "MSFT": 95.0},
    )
    proxied = wrap_market_state_with_contract(
        market_state,
        requirements=("spot",),
        context="DemoPayoff",
    )

    with pytest.raises(ContractViolation) as excinfo:
        _ = proxied.spot

    assert excinfo.value.kind == "missing_market_field"
    assert excinfo.value.field == "spot"
    assert excinfo.value.requirement == "spot"
    assert "DemoPayoff" in str(excinfo.value)


def test_market_state_contract_proxy_raises_for_missing_mapping_key():
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        underlier_spots={"AAPL": 100.0, "MSFT": 95.0},
    )
    proxied = wrap_market_state_with_contract(
        market_state,
        requirements=("spot",),
        context="TickerLookupPayoff",
    )

    with pytest.raises(ContractViolation) as excinfo:
        proxied.underlier_spots["GOOG"]

    assert excinfo.value.kind == "missing_market_key"
    assert excinfo.value.field == "underlier_spots"
    assert excinfo.value.missing_key == "GOOG"
    assert excinfo.value.available_keys == ("AAPL", "MSFT")


def test_market_state_contract_proxy_wraps_named_vol_surface_lookups():
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        vol_surface=FlatVol(0.20),
        vol_surfaces={
            "sx5e_implied_vol": FlatVol(0.20),
            "eurusd_implied_vol": FlatVol(0.12),
        },
    )
    proxied = wrap_market_state_with_contract(
        market_state,
        requirements=("black_vol_surface",),
        context="QuantoPayoff",
    )

    assert proxied.vol_surfaces["eurusd_implied_vol"].black_vol(1.0, 1.1) == pytest.approx(0.12)
    with pytest.raises(ContractViolation) as excinfo:
        proxied.vol_surfaces["missing_surface"]

    assert excinfo.value.kind == "missing_market_key"
    assert excinfo.value.field == "vol_surfaces"
    assert excinfo.value.missing_key == "missing_surface"


def test_price_payoff_surfaces_structured_contract_violation_for_missing_mapping_key():
    class MissingTickerPayoff:
        @property
        def requirements(self):
            return {"spot"}

        def evaluate(self, market_state):
            return float(market_state.underlier_spots["GOOG"])

    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        underlier_spots={"AAPL": 100.0, "MSFT": 95.0},
        discount=YieldCurve.flat(0.05),
    )

    with pytest.raises(ContractViolation) as excinfo:
        price_payoff(MissingTickerPayoff(), market_state)

    assert excinfo.value.kind == "missing_market_key"
    assert excinfo.value.field == "underlier_spots"
    assert excinfo.value.missing_key == "GOOG"
    assert "MissingTickerPayoff" in str(excinfo.value)


def test_price_payoff_preserves_optional_none_probe_for_undeclared_field():
    class OptionalProbePayoff:
        @property
        def requirements(self):
            return set()

        def evaluate(self, market_state):
            return 1.0 if market_state.discount is None else 2.0

    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
    )

    assert price_payoff(OptionalProbePayoff(), market_state) == 1.0
