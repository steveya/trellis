"""Tests for trellis.data.resolver."""

from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from trellis.conventions.day_count import DayCountConvention
from trellis.core.types import Frequency
from trellis.data.resolver import resolve_curve, resolve_market_snapshot
from trellis.curves.bootstrap import BootstrapConventionBundle, BootstrapCurveInputBundle, BootstrapInstrument
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
        assert snapshot.provenance["source_kind"] == "direct_quote"
        assert snapshot.provenance["source_ref"] == "fetch_yields"
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
    def test_supports_named_bootstrap_curve_sources(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        discount_bundle = BootstrapCurveInputBundle(
            curve_name="usd_ois_boot",
            currency="USD",
            rate_index="USD-SOFR-3M",
            conventions=BootstrapConventionBundle(
                deposit_day_count=DayCountConvention.ACT_360,
                swap_fixed_frequency=Frequency.ANNUAL,
                swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
                swap_float_frequency=Frequency.QUARTERLY,
                swap_float_day_count=DayCountConvention.ACT_360,
            ),
            instruments=(
                BootstrapInstrument(tenor=1.0, quote=0.05, instrument_type="deposit", label="DEP1Y"),
            ),
        )
        forecast_bundle = BootstrapCurveInputBundle(
            curve_name="USD-SOFR-3M",
            currency="USD",
            rate_index="USD-SOFR-3M",
            conventions=BootstrapConventionBundle(
                deposit_day_count=DayCountConvention.ACT_360,
                swap_fixed_frequency=Frequency.SEMI_ANNUAL,
                swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
                swap_float_frequency=Frequency.QUARTERLY,
                swap_float_day_count=DayCountConvention.ACT_360,
            ),
            instruments=(
                BootstrapInstrument(tenor=1.0, quote=0.052, instrument_type="deposit", label="SOFR1Y"),
            ),
        )

        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="treasury_gov",
            discount_curve_bootstraps={
                "usd_ois_boot": discount_bundle,
            },
            forecast_curve_bootstraps={
                "USD-SOFR-3M": forecast_bundle,
            },
        )

        assert snapshot.default_discount_curve == "discount"
        assert "usd_ois_boot" in snapshot.discount_curves
        assert "USD-SOFR-3M" in snapshot.forecast_curves
        assert snapshot.provenance["source_kind"] == "mixed"
        assert snapshot.provenance["source_ref"] == "resolver.merged_snapshot"
        assert snapshot.provenance["bootstrap_inputs"]["discount_curves"]["usd_ois_boot"]["currency"] == "USD"
        assert (
            snapshot.provenance["bootstrap_runs"]["discount_curves"]["usd_ois_boot"]["solver_provenance"][
                "backend"
            ]["backend_id"]
            == "scipy"
        )
        assert (
            snapshot.provenance["bootstrap_runs"]["forecast_curves"]["USD-SOFR-3M"]["diagnostics"][
                "jacobian_rank"
            ]
            == 1
        )
        assert (
            snapshot.provenance["bootstrap_inputs"]["discount_curves"]["usd_ois_boot"]["conventions"][
                "swap_fixed_frequency"
            ]
            == "ANNUAL"
        )
        assert (
            snapshot.provenance["bootstrap_inputs"]["discount_curves"]["usd_ois_boot"]["instruments"][0][
                "label"
            ]
            == "DEP1Y"
        )
        assert (
            snapshot.provenance["bootstrap_inputs"]["forecast_curves"]["USD-SOFR-3M"]["conventions"][
                "swap_float_frequency"
            ]
            == "QUARTERLY"
        )
        assert float(snapshot.discount_curve("usd_ois_boot").discount(1.0)) == pytest.approx(
            1.0 / 1.05
        )

        market_state = snapshot.to_market_state(
            settlement=date(2024, 11, 15),
            discount_curve="usd_ois_boot",
            forecast_curve="USD-SOFR-3M",
        )

        assert market_state.selected_curve_names == {
            "discount_curve": "usd_ois_boot",
            "forecast_curve": "USD-SOFR-3M",
        }

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_rejects_duplicate_bootstrap_curve_names(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS

        with pytest.raises(ValueError, match="Duplicate discount curve sources"):
            resolve_market_snapshot(
                as_of=date(2024, 11, 15),
                source="treasury_gov",
                discount_curve_bootstraps={
                    "discount": [
                        BootstrapInstrument(tenor=1.0, quote=0.05, instrument_type="deposit"),
                    ],
                },
            )

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

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_single_vol_surface_respects_explicit_default_name(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        override = FlatVol(0.24)

        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="treasury_gov",
            vol_surface=override,
            default_vol_surface="atm_override",
        )

        assert snapshot.default_vol_surface == "atm_override"
        assert snapshot.vol_surface() is override
        assert snapshot.vol_surface("atm_override") is override

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
        assert snapshot.provenance["source_kind"] == "mixed"

    def test_mock_source_named_vol_override_does_not_clobber_provider_default_surface(self):
        override_vol = FlatVol(0.35)
        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="mock",
            vol_surface=override_vol,
            default_vol_surface="override_atm",
        )

        assert snapshot.default_vol_surface == "override_atm"
        assert snapshot.vol_surface() is override_vol
        assert snapshot.vol_surface("override_atm") is override_vol
        assert snapshot.vol_surface("usd_rates_smile") is not override_vol
        assert snapshot.provenance["prior_family"] == "embedded_market_regime"
        assert snapshot.provenance["prior_parameters"]["regime"] == "easing_cycle"

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
