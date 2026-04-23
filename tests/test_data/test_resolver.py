"""Tests for trellis.data.resolver."""

from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from trellis.conventions.day_count import DayCountConvention
from trellis.conventions.schedule import StubType
from trellis.core.date_utils import build_period_schedule, year_fraction
from trellis.core.types import Frequency
from trellis.data.resolver import resolve_curve, resolve_market_snapshot
from trellis.curves.bootstrap import (
    BootstrapConventionBundle,
    BootstrapCurveInputBundle,
    BootstrapInstrument,
    DatedBootstrapCurveInputBundle,
    DatedBootstrapInstrument,
    MultiCurveBootstrapProgram,
)
from trellis.curves.yield_curve import YieldCurve
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


def _synthetic_heston_market_vols(*, spot: float, rate: float, expiry_years: float, strikes):
    from trellis.models.calibration.implied_vol import implied_vol
    from trellis.models.processes.heston import Heston
    from trellis.models.transforms.fft_pricer import fft_price

    process = Heston(
        mu=rate,
        kappa=1.8,
        theta=0.04,
        xi=0.35,
        rho=-0.6,
        v0=0.05,
    )
    return [
        implied_vol(
            fft_price(
                lambda u: process.characteristic_function(u, expiry_years, log_spot=np.log(spot)),
                spot,
                strike,
                expiry_years,
                rate,
                N=1024,
                eta=0.1,
            ),
            spot,
            strike,
            expiry_years,
            rate,
            option_type="call",
        )
        for strike in strikes
    ]


def _dated_deposit_quote(curve: YieldCurve, *, start_date: date, end_date: date) -> float:
    start_years = year_fraction(date(2024, 11, 15), start_date, DayCountConvention.ACT_360)
    end_years = year_fraction(date(2024, 11, 15), end_date, DayCountConvention.ACT_360)
    accrual = year_fraction(start_date, end_date, DayCountConvention.ACT_360)
    return (float(curve.discount(start_years)) / float(curve.discount(end_years)) - 1.0) / accrual


def _dated_swap_quote(
    *,
    discount_curve: YieldCurve,
    forecast_curve: YieldCurve,
    start_date: date,
    end_date: date,
) -> float:
    fixed_schedule = build_period_schedule(
        start_date,
        end_date,
        Frequency.ANNUAL,
        day_count=DayCountConvention.THIRTY_360_US,
        time_origin=date(2024, 11, 15),
        stub=StubType.SHORT_LAST,
    )
    float_schedule = build_period_schedule(
        start_date,
        end_date,
        Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
        time_origin=date(2024, 11, 15),
        stub=StubType.SHORT_LAST,
    )
    annuity = sum(
        float(period.accrual_fraction) * float(discount_curve.discount(float(period.t_payment)))
        for period in fixed_schedule.periods
    )
    float_pv = 0.0
    for period in float_schedule.periods:
        accrual = float(period.accrual_fraction)
        forward_rate = (
            float(forecast_curve.discount(max(float(period.t_start), 0.0)))
            / float(forecast_curve.discount(float(period.t_end)))
            - 1.0
        ) / accrual
        float_pv += forward_rate * accrual * float(discount_curve.discount(float(period.t_payment)))
    return float_pv / annuity


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
    def test_supports_dependency_aware_multi_curve_bootstrap_program(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        settle = date(2024, 11, 15)
        ois_true = YieldCurve.flat(0.040)
        sofr_true = YieldCurve.flat(0.042)

        program = MultiCurveBootstrapProgram(
            settlement_date=settle,
            curve_inputs=(
                DatedBootstrapCurveInputBundle(
                    curve_name="usd_ois_dated",
                    currency="USD",
                    rate_index="USD-OIS",
                    curve_role="discount_curve",
                    conventions=BootstrapConventionBundle(
                        swap_fixed_frequency=Frequency.ANNUAL,
                        swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
                        swap_float_frequency=Frequency.QUARTERLY,
                        swap_float_day_count=DayCountConvention.ACT_360,
                    ),
                    instruments=(
                        DatedBootstrapInstrument(
                            start_date=settle,
                            end_date=date(2025, 2, 15),
                            quote=_dated_deposit_quote(
                                ois_true,
                                start_date=settle,
                                end_date=date(2025, 2, 15),
                            ),
                            instrument_type="deposit",
                            day_count=DayCountConvention.ACT_360,
                            label="OIS_DEP3M",
                        ),
                    ),
                ),
                DatedBootstrapCurveInputBundle(
                    curve_name="USD-SOFR-3M_dated",
                    currency="USD",
                    rate_index="USD-SOFR-3M",
                    curve_role="forecast_curve",
                    dependency_names={"discount_curve": "usd_ois_dated"},
                    conventions=BootstrapConventionBundle(
                        swap_fixed_frequency=Frequency.ANNUAL,
                        swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
                        swap_float_frequency=Frequency.QUARTERLY,
                        swap_float_day_count=DayCountConvention.ACT_360,
                    ),
                    instruments=(
                        DatedBootstrapInstrument(
                            start_date=settle,
                            end_date=date(2025, 2, 15),
                            quote=_dated_deposit_quote(
                                sofr_true,
                                start_date=settle,
                                end_date=date(2025, 2, 15),
                            ),
                            instrument_type="deposit",
                            day_count=DayCountConvention.ACT_360,
                            label="SOFR_DEP3M",
                        ),
                        DatedBootstrapInstrument(
                            start_date=settle,
                            end_date=date(2027, 1, 20),
                            quote=_dated_swap_quote(
                                discount_curve=ois_true,
                                forecast_curve=sofr_true,
                                start_date=settle,
                                end_date=date(2027, 1, 20),
                            ),
                            instrument_type="swap",
                            stub_type=StubType.SHORT_LAST,
                            label="SOFR_SWAP_STUB",
                        ),
                    ),
                ),
            ),
        )

        snapshot = resolve_market_snapshot(
            as_of=settle,
            source="treasury_gov",
            multi_curve_bootstrap_program=program,
        )

        assert "usd_ois_dated" in snapshot.discount_curves
        assert "USD-SOFR-3M_dated" in snapshot.forecast_curves
        assert snapshot.provenance["bootstrap_runs"]["multi_curve_program"]["dependency_order"] == [
            "usd_ois_dated",
            "USD-SOFR-3M_dated",
        ]
        assert (
            snapshot.provenance["bootstrap_runs"]["multi_curve_program"]["dependency_graph"]["USD-SOFR-3M_dated"][
                "discount_curve"
            ]
            == "usd_ois_dated"
        )
        assert snapshot.provenance["bootstrap_inputs"]["multi_curve_program"]["settlement_date"] == settle.isoformat()

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

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_supports_direct_quote_model_parameter_sources(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS

        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="treasury_gov",
            model_parameter_sources={
                "quanto_direct": {
                    "source_kind": "direct_quote",
                    "source_ref": "unit_test.direct_quote_feed",
                    "parameters": {
                        "quanto_correlation": 0.35,
                        "vol_fx": 0.12,
                    },
                }
            },
            default_model_parameters="quanto_direct",
        )

        assert snapshot.model_parameters("quanto_direct")["quanto_correlation"] == pytest.approx(0.35)
        assert snapshot.default_model_parameters == "quanto_direct"
        source_spec = snapshot.provenance["market_parameter_sources"]["quanto_direct"]
        assert source_spec["source_kind"] == "direct_quote"
        assert source_spec["source_ref"] == "unit_test.direct_quote_feed"
        assert source_spec["parameters"]["vol_fx"] == pytest.approx(0.12)

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_supports_bootstrap_model_parameter_sources(self, MockProvider):
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

        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="treasury_gov",
            discount_curve_bootstraps={"usd_ois_boot": discount_bundle},
            model_parameter_sources={
                "curve_bootstrap_pack": {
                    "source_kind": "bootstrap",
                    "source_ref": "unit_test.bootstrap_curve_samples",
                    "bootstrap_inputs": {
                        "entries": (
                            {
                                "parameter": "zero_1y",
                                "curve_family": "discount_curves",
                                "curve_name": "usd_ois_boot",
                                "measure": "zero_rate",
                                "tenor": 1.0,
                            },
                            {
                                "parameter": "df_1y",
                                "curve_family": "discount_curves",
                                "curve_name": "usd_ois_boot",
                                "measure": "discount_factor",
                                "tenor": 1.0,
                            },
                        )
                    },
                }
            },
            default_model_parameters="curve_bootstrap_pack",
        )

        curve = snapshot.discount_curve("usd_ois_boot")
        params = snapshot.model_parameters("curve_bootstrap_pack")
        assert params["zero_1y"] == pytest.approx(curve.zero_rate(1.0))
        assert params["df_1y"] == pytest.approx(curve.discount(1.0))
        assert snapshot.provenance["market_parameter_sources"]["curve_bootstrap_pack"]["source_kind"] == "bootstrap"
        assert (
            snapshot.provenance["bootstrap_inputs"]["model_parameters"]["curve_bootstrap_pack"]["entries"][0][
                "curve_name"
            ]
            == "usd_ois_boot"
        )

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_supports_empirical_model_parameter_sources(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS

        observations = {
            "SPX": [0.01, 0.02, -0.01, 0.00],
            "EURUSD": [0.015, 0.025, -0.005, 0.005],
        }
        expected_corr = float(
            np.corrcoef(
                np.array(
                    [
                        observations["SPX"],
                        observations["EURUSD"],
                    ],
                    dtype=float,
                ),
                rowvar=True,
            )[0, 1]
        )

        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="treasury_gov",
            model_parameter_sources={
                "empirical_quanto": {
                    "source_kind": "empirical",
                    "source_ref": "unit_test.empirical_history",
                    "empirical_inputs": {
                        "observations": observations,
                        "window": {"lookback_days": 60, "frequency": "daily"},
                        "source_paths": {
                            "SPX": "hist/SPX.csv",
                            "EURUSD": "hist/EURUSD.csv",
                        },
                    },
                    "entries": (
                        {
                            "parameter": "quanto_correlation",
                            "measure": "pairwise_correlation",
                            "series_names": ("SPX", "EURUSD"),
                            "estimator": "sample_pearson",
                            "descriptor": True,
                        },
                    ),
                }
            },
            default_model_parameters="empirical_quanto",
        )

        params = snapshot.model_parameters("empirical_quanto")
        quanto_corr = params["quanto_correlation"]
        assert quanto_corr["kind"] == "empirical"
        assert quanto_corr["value"] == pytest.approx(expected_corr)
        assert quanto_corr["sample_size"] == 4
        assert quanto_corr["estimator"] == "sample_pearson"
        assert quanto_corr["source_ref"] == "unit_test.empirical_history"
        assert quanto_corr["parameters"]["series_names"] == ["SPX", "EURUSD"]
        assert quanto_corr["parameters"]["window"]["lookback_days"] == 60

        source_spec = snapshot.provenance["market_parameter_sources"]["empirical_quanto"]
        assert source_spec["source_kind"] == "empirical"
        assert source_spec["source_ref"] == "unit_test.empirical_history"
        assert source_spec["empirical_inputs"]["window"]["lookback_days"] == 60
        assert source_spec["empirical_outputs"]["quanto_correlation"]["sample_size"] == 4
        assert source_spec["empirical_outputs"]["quanto_correlation"]["series_names"] == [
            "SPX",
            "EURUSD",
        ]
        assert source_spec["empirical_outputs"]["quanto_correlation"]["source_paths"] == {
            "SPX": "hist/SPX.csv",
            "EURUSD": "hist/EURUSD.csv",
        }

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_supports_calibration_model_parameter_sources(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS

        spot = 100.0
        rate = 0.02
        expiry_years = 1.0
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        market_vols = _synthetic_heston_market_vols(
            spot=spot,
            rate=rate,
            expiry_years=expiry_years,
            strikes=strikes,
        )

        snapshot = resolve_market_snapshot(
            as_of=date(2024, 11, 15),
            source="treasury_gov",
            model_parameter_sources={
                "heston_surface_fit": {
                    "source_kind": "calibration",
                    "source_ref": "unit_test.option_surface",
                    "calibration_inputs": {
                        "workflow": "heston_smile",
                        "surface": {
                            "spot": spot,
                            "rate": rate,
                            "expiry_years": expiry_years,
                            "strikes": strikes,
                            "market_vols": market_vols,
                            "surface_name": "equity_1y_smile",
                        },
                        "options": {
                            "parameter_set_name": "heston_surface_fit",
                            "warm_start": (1.2, 0.05, 0.25, -0.3, 0.04),
                        },
                    },
                }
            },
            default_model_parameters="heston_surface_fit",
        )

        params = snapshot.model_parameters("heston_surface_fit")
        assert params["model_family"] == "heston"
        assert params["source_kind"] == "calibration_workflow"
        assert params["theta"] == pytest.approx(0.04, abs=0.02)

        source_spec = snapshot.provenance["market_parameter_sources"]["heston_surface_fit"]
        assert source_spec["source_kind"] == "calibration"
        assert source_spec["source_ref"] == "unit_test.option_surface"
        assert source_spec["calibration_source"]["workflow"] == "heston_smile"
        assert source_spec["calibration_result"]["source_kind"] == "calibrated_surface"
        assert source_spec["calibration_result"]["calibration_target"]["quote_map"]["quote_family"] == (
            "implied_vol"
        )
        assert source_spec["calibration_result"]["calibration_target"]["quote_map"]["quote_subject"] == (
            "equity_option"
        )
        assert source_spec["calibration_result"]["calibration_target"]["quote_map"]["quote_unit"] == (
            "decimal_volatility"
        )
        assert source_spec["calibration_result"]["fit_diagnostics"]["point_count"] == len(strikes)

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_rejects_unsupported_model_parameter_source_combinations(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS

        with pytest.raises(ValueError, match="Unsupported model-parameter source kind"):
            resolve_market_snapshot(
                as_of=date(2024, 11, 15),
                source="treasury_gov",
                model_parameter_sources={
                    "bad_pack": {
                        "source_kind": "calibrated_surface",
                        "parameters": {"rho": 0.25},
                    }
                },
            )

        with pytest.raises(ValueError, match="model_parameter_sources=.*not both"):
            resolve_market_snapshot(
                as_of=date(2024, 11, 15),
                source="treasury_gov",
                model_parameter_sources={
                    "direct_pack": {
                        "source_kind": "direct_quote",
                        "parameters": {"rho": 0.25},
                    }
                },
                model_parameter_sets={"legacy_pack": {"rho": 0.10}},
            )

        with pytest.raises(ValueError, match="requires a mapping payload under 'empirical_inputs'"):
            resolve_market_snapshot(
                as_of=date(2024, 11, 15),
                source="treasury_gov",
                model_parameter_sources={
                    "bad_empirical": {
                        "source_kind": "empirical",
                        "entries": (
                            {
                                "parameter": "quanto_correlation",
                                "measure": "pairwise_correlation",
                                "series_names": ("SPX", "EURUSD"),
                            },
                        ),
                    }
                },
            )

        with pytest.raises(ValueError, match="Unsupported empirical estimator"):
            resolve_market_snapshot(
                as_of=date(2024, 11, 15),
                source="treasury_gov",
                model_parameter_sources={
                    "bad_empirical": {
                        "source_kind": "empirical",
                        "empirical_inputs": {
                            "observations": {
                                "SPX": [0.01, 0.02, -0.01, 0.00],
                                "EURUSD": [0.015, 0.025, -0.005, 0.005],
                            },
                        },
                        "entries": (
                            {
                                "parameter": "quanto_correlation",
                                "measure": "pairwise_correlation",
                                "series_names": ("SPX", "EURUSD"),
                                "estimator": "kendall_tau",
                            },
                        ),
                    }
                },
            )

        with pytest.raises(ValueError, match="Unsupported calibration workflow"):
            resolve_market_snapshot(
                as_of=date(2024, 11, 15),
                source="treasury_gov",
                model_parameter_sources={
                    "bad_calibration": {
                        "source_kind": "calibration",
                        "calibration_inputs": {
                            "workflow": "svi_surface",
                            "surface": {},
                        },
                    }
                },
            )
