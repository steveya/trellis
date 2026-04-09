from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class _Spec:
    notional = 100.0
    spot = 100.0
    strike = 100.0
    expiry_date = date(2025, 11, 15)
    option_type = "call"
    n_paths = 12_000
    n_steps = 64
    seed = 7


def _market_state(vol: float = 0.20, rate: float = 0.05) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(rate, max_tenor=5.0),
        vol_surface=FlatVol(vol),
    )


def test_resolve_single_state_terminal_claim_monte_carlo_inputs_reads_market_and_controls():
    from trellis.models.monte_carlo.single_state_diffusion import (
        resolve_single_state_terminal_claim_monte_carlo_inputs,
    )

    resolved = resolve_single_state_terminal_claim_monte_carlo_inputs(
        _market_state(vol=0.30, rate=0.04),
        _Spec(),
        scheme="log_euler",
        variance_reduction="antithetic",
        n_paths=8000,
        n_steps=96,
        seed=11,
    )

    assert resolved.spot == pytest.approx(100.0)
    assert resolved.rate == pytest.approx(0.04)
    assert resolved.sigma == pytest.approx(0.30)
    assert resolved.scheme == "log_euler"
    assert resolved.variance_reduction == "antithetic"
    assert resolved.n_paths == 8000
    assert resolved.n_steps == 96
    assert resolved.seed == 11


def test_build_single_state_terminal_claim_monte_carlo_problem_uses_terminal_only_event_aware_shape():
    from trellis.models.monte_carlo.single_state_diffusion import (
        build_single_state_terminal_claim_monte_carlo_problem_from_resolved,
        resolve_single_state_terminal_claim_monte_carlo_inputs,
    )

    resolved = resolve_single_state_terminal_claim_monte_carlo_inputs(
        _market_state(),
        _Spec(),
    )
    problem = build_single_state_terminal_claim_monte_carlo_problem_from_resolved(
        resolved,
        terminal_payoff=lambda terminal: terminal,
    )

    assert problem.path_requirement.full_path is False
    assert problem.path_requirement.snapshot_steps == ()
    assert problem.maturity == pytest.approx(1.0)
    assert problem.discount_rate == pytest.approx(resolved.rate)
    assert problem.simulation_method == "exact"


def test_equity_monte_carlo_wrapper_delegates_through_single_state_family_helper(monkeypatch):
    from trellis.models import equity_option_monte_carlo as module

    calls: list[tuple[object, object]] = []

    class FakeResult:
        price = 123.45

    def fake_result(market_state, spec, **kwargs):
        calls.append((market_state, spec))
        return FakeResult()

    monkeypatch.setattr(
        module,
        "price_single_state_terminal_claim_monte_carlo_result",
        fake_result,
    )

    price = module.price_vanilla_equity_option_monte_carlo(_market_state(), _Spec())

    assert calls
    assert price == pytest.approx(123.45)
