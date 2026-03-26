"""Tests for trellis.data.resolver."""

from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from trellis.data.resolver import resolve_curve, resolve_market_snapshot
from trellis.instruments.fx import FXRate
from trellis.models.vol_surface import FlatVol, GridVolSurface


SAMPLE_YIELDS = {
    0.25: 0.045,
    0.5: 0.046,
    1.0: 0.047,
    2.0: 0.048,
    5.0: 0.045,
    10.0: 0.044,
    30.0: 0.046,
}


class TestResolveCurve:
    """resolve_curve with mocked providers."""

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_treasury_gov_returns_curve(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        curve = resolve_curve(as_of=date(2024, 11, 15), source="treasury_gov")
        assert len(curve.tenors) == len(SAMPLE_YIELDS)
        mock.fetch_yields.assert_called_once_with(date(2024, 11, 15))

    @patch("trellis.data.fred.FredDataProvider")
    def test_fred_returns_curve(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        curve = resolve_curve(as_of=date(2024, 11, 15), source="fred")
        assert len(curve.tenors) == len(SAMPLE_YIELDS)

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_latest_resolves_to_today(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        resolve_curve(as_of="latest")
        mock.fetch_yields.assert_called_once_with(date.today())

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_empty_yields_raises(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = {}
        with pytest.raises(RuntimeError, match="No yield data"):
            resolve_curve(as_of=date(2024, 11, 15))

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown data source"):
            resolve_curve(source="bloomberg")

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_string_date_parsed(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        resolve_curve(as_of="2024-11-15")
        mock.fetch_yields.assert_called_once_with(date(2024, 11, 15))


class TestResolveMarketSnapshot:
    """Generalized market snapshot resolver."""

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_returns_market_snapshot_with_default_discount_curve(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS

        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="treasury_gov",
        )

        assert snapshot.as_of == date(2024, 11, 15)
        assert snapshot.source == "treasury_gov"
        assert snapshot.default_discount_curve == "discount"
        assert len(snapshot.discount_curve().tenors) == len(SAMPLE_YIELDS)
        mock.fetch_yields.assert_called_once_with(date(2024, 11, 15))

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_passes_through_optional_market_components(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        vol = FlatVol(0.20)
        state_space = object()

        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="treasury_gov",
            vol_surface=vol,
            forecast_curves={"USD-SOFR-3M": MagicMock(name="forecast_curve")},
            fx_rates={"EURUSD": MagicMock(name="eurusd")},
            state_space=state_space,
        )

        assert snapshot.vol_surface() is vol
        assert "USD-SOFR-3M" in snapshot.forecast_curves
        assert "EURUSD" in snapshot.fx_rates
        assert snapshot.state_space() is state_space

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_supports_named_vol_surface_sets(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        smile = GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=((0.25, 0.22), (0.27, 0.24)),
        )
        atm = FlatVol(0.20)

        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="treasury_gov",
            vol_surfaces={"atm": atm, "smile": smile},
            default_vol_surface="smile",
        )

        assert snapshot.vol_surface() is smile
        assert snapshot.vol_surface("atm") is atm

    def test_mock_source_merges_explicit_overrides(self):
        override_vol = FlatVol(0.35)
        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="mock",
            vol_surface=override_vol,
            fx_rates={"AUDUSD": FXRate(0.66, domestic="USD", foreign="AUD")},
        )

        assert snapshot.vol_surface() is override_vol
        assert "USD-SOFR-3M" in snapshot.forecast_curves
        assert "EUR-DISC" in snapshot.forecast_curves
        assert "EURUSD" in snapshot.fx_rates
        assert snapshot.fx_rates["AUDUSD"].spot == pytest.approx(0.66)

    def test_mock_source_preserves_spots_and_parameter_packs(self):
        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="mock",
        )

        assert snapshot.state_space() is not None
        assert snapshot.underlier_spot() == pytest.approx(snapshot.underlier_spots["SPX"])
        assert snapshot.local_vol_surface() is not None
        assert "merton_equity" in snapshot.jump_parameter_sets
        assert "heston_equity" in snapshot.model_parameter_sets
