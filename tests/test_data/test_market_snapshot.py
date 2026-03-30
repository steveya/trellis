"""Tests for the canonical market snapshot schema."""

from datetime import date

import pytest

from trellis.curves.credit_curve import CreditCurve
from trellis.core.state_space import StateSpace
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def test_market_snapshot_to_market_state_uses_named_defaults():
    from trellis.data.schema import MarketSnapshot

    discount = YieldCurve.flat(0.05)
    forecast = YieldCurve.flat(0.051)
    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="unit",
        discount_curves={"usd_ois": discount},
        forecast_curves={"USD-SOFR-3M": forecast},
        vol_surfaces={"usd_atm": FlatVol(0.20)},
        credit_curves={"corp": CreditCurve.flat(0.02)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        default_discount_curve="usd_ois",
        default_vol_surface="usd_atm",
        default_credit_curve="corp",
        provenance={
            "source": "unit",
            "source_kind": "explicit_input",
            "source_ref": "test_market_snapshot_to_market_state_uses_named_defaults",
        },
    )

    market_state = snapshot.to_market_state(settlement=SETTLE)

    assert market_state.as_of == SETTLE
    assert market_state.settlement == SETTLE
    assert market_state.discount is discount
    assert market_state.vol_surface is snapshot.vol_surface()
    assert market_state.credit_curve is snapshot.credit_curve()
    assert market_state.forecast_curves == {"USD-SOFR-3M": forecast}
    assert market_state.selected_curve_names == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
        "credit_curve": "corp",
    }
    assert market_state.market_provenance["source_kind"] == "explicit_input"
    assert market_state.market_provenance["source_ref"] == (
        "test_market_snapshot_to_market_state_uses_named_defaults"
    )
    assert market_state.fx_rates["EURUSD"].spot == pytest.approx(1.10)
    assert market_state.available_capabilities >= {
        "discount_curve",
        "forward_curve",
        "black_vol_surface",
        "credit_curve",
        "fx_rates",
    }


def test_market_snapshot_selected_forecast_curve_becomes_runtime_forward_curve():
    from trellis.data.schema import MarketSnapshot

    usd = YieldCurve.flat(0.05)
    eur = YieldCurve.flat(0.03)
    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="unit",
        discount_curves={"usd_ois": usd},
        forecast_curves={"EUR-DISC": eur},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        default_discount_curve="usd_ois",
    )

    market_state = snapshot.to_market_state(
        settlement=SETTLE,
        forecast_curve="EUR-DISC",
        fx_rate="EURUSD",
    )

    assert set(market_state.forecast_curves) == {"EUR-DISC"}
    assert market_state.forward_curve is not None
    assert market_state.forecast_forward_curve("EUR-DISC").forward_rate(0.5, 1.0) == pytest.approx(
        market_state.forward_curve.forward_rate(0.5, 1.0)
    )
    assert market_state.selected_curve_names == {
        "discount_curve": "usd_ois",
        "forecast_curve": "EUR-DISC",
    }
    assert set(market_state.fx_rates) == {"EURUSD"}
    assert market_state.spot == pytest.approx(1.10)
    assert market_state.underlier_spots["EURUSD"] == pytest.approx(1.10)
    assert market_state.available_capabilities >= {"forward_curve", "fx_rates", "spot"}


def test_market_snapshot_requires_named_default_when_multiple_discount_curves():
    from trellis.data.schema import MarketSnapshot

    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="unit",
        discount_curves={
            "usd_ois": YieldCurve.flat(0.05),
            "eur_ois": YieldCurve.flat(0.03),
        },
    )

    with pytest.raises(ValueError, match="default discount curve"):
        snapshot.to_market_state(settlement=SETTLE)


def test_market_snapshot_default_discount_curve_accessor():
    from trellis.data.schema import MarketSnapshot

    usd = YieldCurve.flat(0.05)
    eur = YieldCurve.flat(0.03)
    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="unit",
        discount_curves={"usd_ois": usd, "eur_ois": eur},
        default_discount_curve="eur_ois",
    )

    assert snapshot.discount_curve() is eur
    assert snapshot.discount_curve("usd_ois") is usd


def test_market_snapshot_to_market_state_records_selected_curve_names():
    from trellis.data.schema import MarketSnapshot

    usd = YieldCurve.flat(0.05)
    eur = YieldCurve.flat(0.03)
    forecast = YieldCurve.flat(0.051)
    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="unit",
        discount_curves={"usd_ois": usd, "eur_ois": eur},
        forecast_curves={"USD-SOFR-3M": forecast},
        default_discount_curve="eur_ois",
    )

    market_state = snapshot.to_market_state(settlement=SETTLE)

    assert market_state.selected_curve_name("discount_curve") == "eur_ois"
    assert market_state.selected_curve_name("forecast_curve") == "USD-SOFR-3M"
    assert market_state.selected_curve_names == {
        "discount_curve": "eur_ois",
        "forecast_curve": "USD-SOFR-3M",
    }


def test_market_snapshot_to_market_state_includes_spots_and_parameter_packs():
    from trellis.data.schema import MarketSnapshot

    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="unit",
        discount_curves={"usd_ois": YieldCurve.flat(0.05)},
        vol_surfaces={"usd_atm": FlatVol(0.20)},
        underlier_spots={"AAPL": 203.5},
        local_vol_surfaces={"equity_lv": lambda s, t: 0.24},
        jump_parameter_sets={
            "merton_eq": {
                "lam": 0.35,
                "jump_mean": -0.08,
                "jump_vol": 0.22,
                "sigma": 0.25,
            }
        },
        model_parameter_sets={
            "heston_eq": {
                "kappa": 1.8,
                "theta": 0.04,
                "xi": 0.55,
                "rho": -0.65,
                "v0": 0.04,
            }
        },
        default_discount_curve="usd_ois",
        default_vol_surface="usd_atm",
        default_underlier_spot="AAPL",
        default_local_vol_surface="equity_lv",
        default_jump_parameters="merton_eq",
        default_model_parameters="heston_eq",
    )

    market_state = snapshot.to_market_state(settlement=SETTLE)

    assert market_state.spot == pytest.approx(203.5)
    assert market_state.underlier_spots == {"AAPL": 203.5}
    assert market_state.local_vol_surface is not None
    assert market_state.local_vol_surface(100.0, 1.0) == pytest.approx(0.24)
    assert market_state.jump_parameters == snapshot.jump_parameters()
    assert market_state.model_parameters == snapshot.model_parameters()
    assert market_state.available_capabilities >= {
        "spot",
        "local_vol_surface",
        "jump_parameters",
        "model_parameters",
    }


def test_market_snapshot_to_market_state_can_build_state_space_from_factory():
    from trellis.data.schema import MarketSnapshot

    def build_state_space(base_state, snapshot, settlement):
        return StateSpace(states={"base": (1.0, base_state)})

    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="unit",
        discount_curves={"usd_ois": YieldCurve.flat(0.05)},
        state_spaces={"macro": build_state_space},
        default_discount_curve="usd_ois",
        default_state_space="macro",
    )

    market_state = snapshot.to_market_state(settlement=SETTLE)

    assert market_state.state_space is not None
    assert market_state.state_space.probability("base") == pytest.approx(1.0)
    assert "state_space" in market_state.available_capabilities
