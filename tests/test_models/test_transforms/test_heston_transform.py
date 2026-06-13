from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as raw_np
import pytest

from trellis.core.types import DayCountConvention
from trellis.models.processes.heston import (
    Heston,
    build_heston_parameter_payload,
)
from trellis.models.transforms.fft_pricer import fft_price


class FlatDiscountCurve:
    def __init__(self, rate: float) -> None:
        self.rate = float(rate)

    def zero_rate(self, t: float) -> float:
        return self.rate

    def discount(self, t: float) -> float:
        return float(raw_np.exp(-self.rate * t))


def _market_state(*, v0: float = 0.04):
    return SimpleNamespace(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=FlatDiscountCurve(0.05),
        vol_surface=None,
        selected_curve_names={},
        model_parameters=build_heston_parameter_payload(
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=v0,
            parameter_set_name="heston_equity",
        ),
    )


def _spec(*, option_type: str = "call"):
    return SimpleNamespace(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type=option_type,
        day_count=DayCountConvention.ACT_365,
        dividend_yield=0.0,
    )


def test_heston_fft_helper_matches_direct_characteristic_function_price():
    from trellis.models.transforms.heston import price_heston_option_transform_result

    market_state = _market_state()
    spec = _spec()

    result = price_heston_option_transform_result(market_state, spec, method="fft")

    process = Heston(mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    direct = fft_price(
        lambda u: process.characteristic_function(u, 1.0, log_spot=raw_np.log(100.0)),
        100.0,
        100.0,
        1.0,
        0.05,
    )
    assert result.price == pytest.approx(direct, rel=1e-12)
    assert result.method == "fft"
    assert result.characteristic_family == "heston_log_spot"
    assert result.validation_bundle == "heston:transform"
    assert result.runtime_binding["model_parameters"]["model_family"] == "heston"


def test_heston_transform_helper_uses_model_parameters_without_black_vol_surface():
    from trellis.models.transforms.heston import price_heston_option_transform_result

    result = price_heston_option_transform_result(_market_state(), _spec(), method="cos")

    assert result.price > 0.0
    assert result.method == "cos"


def test_heston_transform_helper_treats_none_notional_as_unit_notional():
    from trellis.models.transforms.heston import price_heston_option_transform_result

    unit = price_heston_option_transform_result(_market_state(), _spec(), method="fft")
    none_notional = price_heston_option_transform_result(
        _market_state(),
        SimpleNamespace(**{**vars(_spec()), "notional": None}),
        method="fft",
    )

    assert none_notional.price == pytest.approx(unit.price)


def test_heston_transform_helper_accepts_task_spec_aliases_without_market_payload():
    from trellis.models.transforms.heston import price_heston_option_transform_result

    market_state = SimpleNamespace(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=FlatDiscountCurve(0.05),
        vol_surface=None,
        selected_curve_names={},
        model_parameters=None,
        model_parameter_sets=None,
    )
    spec = SimpleNamespace(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        option_type="call",
        day_count=DayCountConvention.ACT_365,
        dividend_yield=0.0,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        initial_variance=0.04,
    )

    result = price_heston_option_transform_result(market_state, spec, method="cos")

    assert result.price > 0.0
    assert result.model_parameters["xi"] == pytest.approx(0.3)
    assert result.model_parameters["v0"] == pytest.approx(0.04)


def test_heston_transform_helper_accepts_surface_maturity_grid_alias():
    from trellis.models.transforms.heston import price_heston_option_transform_result

    spec = SimpleNamespace(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        surface_maturities=(date(2025, 11, 15),),
        option_type="call",
        day_count=DayCountConvention.ACT_365,
        dividend_yield=0.0,
    )

    result = price_heston_option_transform_result(_market_state(), spec, method="cos")

    assert result.price > 0.0
    assert result.maturity == pytest.approx(1.0)


def test_heston_transform_helper_selects_representative_surface_strike():
    from trellis.models.transforms.heston import resolve_heston_transform_inputs

    spec = SimpleNamespace(
        notional=1.0,
        spot=100.0,
        surface_strikes=(80.0, 100.0, 120.0),
        surface_maturities=(date(2025, 11, 15),),
        option_type="call",
        day_count=DayCountConvention.ACT_365,
        dividend_yield=0.0,
    )

    resolved = resolve_heston_transform_inputs(_market_state(), spec, method="fft")

    assert resolved.strike == pytest.approx(100.0)
    assert resolved.spot == pytest.approx(100.0)


def test_heston_transform_helper_accepts_scalar_surface_aliases():
    from trellis.models.transforms.heston import resolve_heston_transform_inputs

    spec = SimpleNamespace(
        notional=1.0,
        spot=100.0,
        surface_strikes=100.0,
        surface_maturities=1.0,
        option_type="call",
        day_count=DayCountConvention.ACT_365,
        dividend_yield=0.0,
    )

    resolved = resolve_heston_transform_inputs(_market_state(), spec, method="fft")

    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity == pytest.approx(1.0)


def test_heston_transform_helper_accepts_comma_delimited_surface_aliases():
    from trellis.models.transforms.heston import resolve_heston_transform_inputs

    spec = SimpleNamespace(
        notional=1.0,
        spot=100.0,
        surface_strikes="80, 100, 120",
        surface_maturities="0.5, 1.0, 2.0",
        option_type="call",
        day_count=DayCountConvention.ACT_365,
        dividend_yield=0.0,
    )

    resolved = resolve_heston_transform_inputs(_market_state(), spec, method="cos")

    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity == pytest.approx(0.5)


def test_heston_transform_helper_uses_bounded_surface_fallbacks_for_smile_specs():
    from trellis.models.transforms.heston import resolve_heston_transform_inputs

    class HestonSmileImpliedVolSurfaceSpec:
        """Heston smile implied vol surface representative scalar spec."""

        notional = 1.0
        spot = 100.0
        option_type = "call"
        day_count = DayCountConvention.ACT_365
        dividend_yield = 0.0

    resolved = resolve_heston_transform_inputs(
        _market_state(),
        HestonSmileImpliedVolSurfaceSpec(),
        method="fft",
    )

    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity == pytest.approx(1.0)


def test_heston_transform_price_is_sensitive_to_heston_parameters():
    from trellis.models.transforms.heston import price_heston_option_transform

    base_price = price_heston_option_transform(_market_state(v0=0.04), _spec(), method="fft")
    bumped_model_price = price_heston_option_transform(
        _market_state(v0=0.09),
        _spec(),
        method="fft",
    )

    assert abs(bumped_model_price - base_price) > 1.0


def test_heston_transform_reports_unsupported_laguerre_kernel():
    from trellis.models.transforms.heston import (
        UnsupportedHestonTransformMethod,
        price_heston_option_transform_result,
    )

    with pytest.raises(UnsupportedHestonTransformMethod) as exc_info:
        price_heston_option_transform_result(
            _market_state(),
            _spec(),
            method="gauss_laguerre",
        )

    packet = exc_info.value.repair_packet
    assert packet["packet_type"] == "missing_heston_gauss_laguerre_transform_kernel"
    assert packet["missing_primitive"] == "heston_gauss_laguerre_transform_kernel"
    assert packet["unsupported_class"] == "heston_gauss_laguerre_transform"
