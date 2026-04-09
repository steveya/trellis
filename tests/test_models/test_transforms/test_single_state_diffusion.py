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


def _market_state(vol: float = 0.20, rate: float = 0.05) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(rate, max_tenor=5.0),
        vol_surface=FlatVol(vol),
    )


def test_resolve_single_state_terminal_claim_transform_inputs_reads_market_and_controls():
    from trellis.models.transforms.single_state_diffusion import (
        resolve_single_state_terminal_claim_transform_inputs,
    )

    resolved = resolve_single_state_terminal_claim_transform_inputs(
        _market_state(vol=0.30, rate=0.04),
        _Spec(),
        method="cos",
        fft_alpha=2.0,
        fft_points=2048,
        fft_eta=0.2,
        cos_points=128,
        cos_truncation=12.0,
    )

    assert resolved.spot == pytest.approx(100.0)
    assert resolved.rate == pytest.approx(0.04)
    assert resolved.sigma == pytest.approx(0.30)
    assert resolved.method == "cos"
    assert resolved.fft_alpha == pytest.approx(2.0)
    assert resolved.fft_points == 2048
    assert resolved.fft_eta == pytest.approx(0.2)
    assert resolved.cos_points == 128
    assert resolved.cos_truncation == pytest.approx(12.0)


def test_price_single_state_terminal_claim_transform_result_prices_intrinsic_at_expiry():
    from trellis.models.transforms.single_state_diffusion import (
        price_single_state_terminal_claim_transform_result,
    )

    spec = _Spec()
    spec.expiry_date = SETTLE
    result = price_single_state_terminal_claim_transform_result(
        _market_state(),
        spec,
        intrinsic_fn=lambda terminal, resolved: max(float(terminal) - resolved.strike, 0.0),
        fft_log_spot_char_fn=lambda resolved: lambda u: 0.0,
        cos_log_ratio_char_fn=lambda resolved: lambda u: 0.0,
        put_from_call_parity_fn=lambda call_price, resolved: float(call_price),
    )

    assert result.price == pytest.approx(0.0)
    assert result.maturity == pytest.approx(0.0)


def test_equity_transform_wrapper_delegates_through_single_state_family_helper(monkeypatch):
    from trellis.models import equity_option_transforms as module

    calls: list[tuple[object, object]] = []

    class FakeResult:
        price = 321.0

    def fake_result(market_state, spec, **kwargs):
        calls.append((market_state, spec))
        return FakeResult()

    monkeypatch.setattr(
        module,
        "price_single_state_terminal_claim_transform_result",
        fake_result,
    )

    price = module.price_vanilla_equity_option_transform(_market_state(), _Spec())

    assert calls
    assert price == pytest.approx(321.0)
