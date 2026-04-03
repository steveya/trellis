"""Regression tests for the resolved-input Garman-Kohlhagen helper."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trellis.core.differentiable import gradient
from trellis.models.analytical.fx import (
    ResolvedGarmanKohlhagenInputs,
    garman_kohlhagen_call_raw,
    garman_kohlhagen_price_raw,
    garman_kohlhagen_put_raw,
)
from trellis.models.analytical.support import (
    discounted_value,
    forward_from_discount_factors,
    terminal_vanilla_from_basis,
)
from trellis.models.black import (
    black76_asset_or_nothing_call,
    black76_cash_or_nothing_call,
    garman_kohlhagen_call,
    garman_kohlhagen_put,
)


def _finite_difference(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


def _resolved_inputs() -> ResolvedGarmanKohlhagenInputs:
    return ResolvedGarmanKohlhagenInputs(
        spot=1.10,
        strike=1.05,
        sigma=0.22,
        T=2.0,
        df_domestic=0.91,
        df_foreign=0.96,
    )


def test_raw_call_and_put_match_public_wrappers():
    resolved = _resolved_inputs()

    assert garman_kohlhagen_call_raw(resolved) == pytest.approx(
        garman_kohlhagen_call(
            resolved.spot,
            resolved.strike,
            resolved.sigma,
            resolved.T,
            resolved.df_domestic,
            resolved.df_foreign,
        ),
        abs=1e-12,
    )
    assert garman_kohlhagen_put_raw(resolved) == pytest.approx(
        garman_kohlhagen_put(
            resolved.spot,
            resolved.strike,
            resolved.sigma,
            resolved.T,
            resolved.df_domestic,
            resolved.df_foreign,
        ),
        abs=1e-12,
    )


def test_raw_kernel_matches_explicit_support_basis_assembly():
    resolved = _resolved_inputs()
    forward = forward_from_discount_factors(
        spot=resolved.spot,
        domestic_df=resolved.df_domestic,
        foreign_df=resolved.df_foreign,
    )
    asset_value = black76_asset_or_nothing_call(
        forward,
        resolved.strike,
        resolved.sigma,
        resolved.T,
    )
    cash_value = black76_cash_or_nothing_call(
        forward,
        resolved.strike,
        resolved.sigma,
        resolved.T,
    )
    expected = discounted_value(
        terminal_vanilla_from_basis(
            "call",
            asset_value=asset_value,
            cash_value=cash_value,
            strike=resolved.strike,
        ),
        resolved.df_domestic,
    )

    assert garman_kohlhagen_price_raw("call", resolved) == pytest.approx(expected, abs=1e-12)


def test_raw_put_call_parity_matches_discounted_fx_forward_identity():
    resolved = _resolved_inputs()
    call = garman_kohlhagen_call_raw(resolved)
    put = garman_kohlhagen_put_raw(resolved)

    assert call - put == pytest.approx(
        resolved.spot * resolved.df_foreign - resolved.strike * resolved.df_domestic,
        abs=1e-10,
    )


def test_raw_call_delta_matches_finite_difference():
    resolved = _resolved_inputs()

    autodiff_delta = gradient(
        lambda spot: garman_kohlhagen_call_raw(replace(resolved, spot=spot))
    )(resolved.spot)
    fd_delta = _finite_difference(
        lambda spot: garman_kohlhagen_call_raw(replace(resolved, spot=spot)),
        resolved.spot,
    )

    assert autodiff_delta == pytest.approx(fd_delta, rel=1e-6, abs=1e-8)


def test_raw_call_vega_matches_finite_difference():
    resolved = _resolved_inputs()

    autodiff_vega = gradient(
        lambda sigma: garman_kohlhagen_call_raw(replace(resolved, sigma=sigma))
    )(resolved.sigma)
    fd_vega = _finite_difference(
        lambda sigma: garman_kohlhagen_call_raw(replace(resolved, sigma=sigma)),
        resolved.sigma,
    )

    assert autodiff_vega == pytest.approx(fd_vega, rel=1e-6, abs=1e-8)
