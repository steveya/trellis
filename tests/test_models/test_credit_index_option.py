"""Tests for bounded credit-index spread option helpers."""

from __future__ import annotations

import pytest


def test_credit_index_option_black_and_mc_agree_on_stable_fixture():
    from trellis.models.credit_index_option import (
        CreditIndexOptionSpec,
        price_credit_index_option_black_on_spread,
        price_credit_index_option_monte_carlo,
    )

    spec = CreditIndexOptionSpec(
        notional=10_000_000.0,
        forward_spread=0.0125,
        strike_spread=0.0100,
        spread_volatility=0.30,
        maturity_years=1.25,
        index_annuity=4.2,
        discount_rate=0.04,
        option_type="call",
    )

    black_price = price_credit_index_option_black_on_spread(None, spec)
    mc_price = price_credit_index_option_monte_carlo(
        None,
        spec,
        n_paths=65_536,
        seed=55,
    )

    assert black_price > 0.0
    assert mc_price == pytest.approx(black_price, rel=0.015)


def test_credit_index_option_loss_given_default_convention_scales_notional():
    from trellis.models.credit_index_option import (
        CreditIndexOptionSpec,
        price_credit_index_option_black_on_spread,
    )

    spread_annuity = CreditIndexOptionSpec(loss_convention="spread_annuity")
    lgd = CreditIndexOptionSpec(loss_convention="loss_given_default", recovery_rate=0.40)

    spread_price = price_credit_index_option_black_on_spread(None, spread_annuity)
    lgd_price = price_credit_index_option_black_on_spread(None, lgd)

    assert lgd_price == pytest.approx(spread_price * 0.60)


def test_credit_index_option_spec_fails_closed_on_invalid_spread_inputs():
    from trellis.models.credit_index_option import CreditIndexOptionSpec

    with pytest.raises(ValueError, match="forward_spread"):
        CreditIndexOptionSpec(forward_spread=-0.01)

    with pytest.raises(ValueError, match="option_type"):
        CreditIndexOptionSpec(option_type="payer")
