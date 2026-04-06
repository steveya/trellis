"""Tests for trellis.data.mock — MockDataProvider."""

from datetime import date

import pytest

from trellis.models.calibration.credit import (
    CreditHazardCalibrationQuote,
    calibrate_single_name_credit_curve_workflow,
)
from trellis.models.credit_default_swap import build_cds_schedule, price_cds_analytical
from trellis.core.types import DayCountConvention, Frequency
from trellis.data.mock import MockDataProvider, SNAPSHOTS, _TENOR_GRID
from trellis.data.resolver import resolve_curve, resolve_market_snapshot


class TestMockDataProvider:

    def test_fetch_yields_returns_dict(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields()
        assert isinstance(yields, dict)
        assert len(yields) == len(_TENOR_GRID)

    def test_all_yields_reasonable(self):
        provider = MockDataProvider()
        for d in provider.available_dates:
            yields = provider.fetch_yields(d)
            for tenor, rate in yields.items():
                assert 0 <= rate <= 0.10, f"rate {rate} at tenor {tenor} on {d}"

    def test_exact_date_match(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields(date(2024, 11, 15))
        assert yields == SNAPSHOTS[date(2024, 11, 15)]

    def test_nearest_prior_date(self):
        """A date between snapshots returns the most recent prior."""
        provider = MockDataProvider()
        # 2024-06-01 is between 2023-10-15 and 2024-11-15
        yields = provider.fetch_yields(date(2024, 6, 1))
        assert yields == SNAPSHOTS[date(2023, 10, 15)]

    def test_future_date_returns_latest(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields(date(2030, 1, 1))
        assert yields == SNAPSHOTS[date(2024, 11, 15)]

    def test_before_all_snapshots_returns_empty(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields(date(2000, 1, 1))
        assert yields == {}

    def test_none_returns_latest(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields(None)
        assert yields == SNAPSHOTS[date(2024, 11, 15)]

    def test_custom_overrides(self):
        custom = {date(2025, 1, 1): {1.0: 0.05, 10.0: 0.06}}
        provider = MockDataProvider(overrides=custom)
        yields = provider.fetch_yields(date(2025, 1, 1))
        assert yields == {1.0: 0.05, 10.0: 0.06}
        # Built-in snapshots still accessible
        yields_old = provider.fetch_yields(date(2019, 9, 15))
        assert yields_old == SNAPSHOTS[date(2019, 9, 15)]

    def test_from_dict_no_builtins(self):
        custom = {date(2025, 6, 1): {5.0: 0.04}}
        provider = MockDataProvider.from_dict(custom)
        # Only the custom data exists
        assert provider.available_dates == [date(2025, 6, 1)]
        assert provider.fetch_yields(date(2025, 6, 1)) == {5.0: 0.04}
        # Date before the only snapshot returns empty
        assert provider.fetch_yields(date(2024, 11, 15)) == {}
        # Date after returns the snapshot
        assert provider.fetch_yields(date(2026, 1, 1)) == {5.0: 0.04}

    def test_returns_copy(self):
        """Mutations to returned dict don't affect internal state."""
        provider = MockDataProvider()
        y1 = provider.fetch_yields(date(2024, 11, 15))
        y1[999.0] = 0.99
        y2 = provider.fetch_yields(date(2024, 11, 15))
        assert 999.0 not in y2

    def test_available_dates(self):
        provider = MockDataProvider()
        dates = provider.available_dates
        assert len(dates) == len(SNAPSHOTS)
        assert dates == sorted(dates)

    def test_fetch_market_snapshot_returns_full_named_snapshot(self):
        provider = MockDataProvider()
        snapshot = provider.fetch_market_snapshot(date(2024, 11, 15))
        assert snapshot.default_discount_curve == "usd_ois"
        assert snapshot.default_vol_surface == "usd_rates_smile"
        assert snapshot.default_fixing_history == "USD-SOFR-3M"
        assert snapshot.default_credit_curve == "usd_ig"
        assert snapshot.default_state_space == "macro_regime"
        assert snapshot.default_underlier_spot == "SPX"
        assert snapshot.default_local_vol_surface == "spx_local_vol"
        assert snapshot.default_jump_parameters == "merton_equity"
        assert snapshot.default_model_parameters == "heston_equity"
        assert "EUR-DISC" in snapshot.forecast_curves
        assert "USD-SOFR-3M" in snapshot.forecast_curves
        assert "EURUSD" in snapshot.fx_rates
        assert "macro_regime" in snapshot.state_spaces
        assert "SPX" in snapshot.underlier_spots
        assert "USD-SOFR-3M" in snapshot.fixing_histories
        assert "spx_local_vol" in snapshot.local_vol_surfaces
        assert "merton_equity" in snapshot.jump_parameter_sets
        assert "heston_equity" in snapshot.model_parameter_sets
        assert snapshot.provenance["source_kind"] == "synthetic_snapshot"
        assert snapshot.provenance["prior_family"] == "embedded_market_regime"
        assert snapshot.provenance["prior_parameters"]["regime"] == "easing_cycle"
        assert snapshot.provenance["prior_seed"] > 0
        assert snapshot.vol_surface().black_vol(1.0, 0.05) > 0
        assert snapshot.credit_curve() is not None
        assert snapshot.fixing_history()[date(2024, 11, 14)] > 0.0

    def test_fetch_market_snapshot_exposes_model_consistency_contract(self):
        provider = MockDataProvider()
        snapshot = provider.fetch_market_snapshot(date(2024, 11, 15))

        contract = snapshot.provenance["prior_parameters"]["model_consistency_contract"]
        assert contract["version"] == "v1"
        assert contract["seed"] == snapshot.provenance["prior_seed"]
        assert contract["rates"]["curve_roles"]["discount_curve"] == "usd_ois"
        assert contract["rates"]["curve_roles"]["forecast_curve"] == "USD-SOFR-3M"
        assert contract["credit"]["workflow"] == "calibrate_single_name_credit_curve_workflow"
        assert contract["volatility"]["workflow"] == "calibration_surface_bundle"
        assert "heston_equity" in contract["volatility"]["model_parameter_sets"]

    def test_fetch_market_snapshot_exposes_synthetic_generation_contract(self):
        provider = MockDataProvider()
        snapshot = provider.fetch_market_snapshot(date(2024, 11, 15))

        contract = snapshot.provenance["prior_parameters"]["synthetic_generation_contract"]
        assert contract["version"] == "v2"
        assert contract["seed"] == snapshot.provenance["prior_seed"]
        assert contract["model_packs"]["rates"]["family"] == "shifted_curve_bundle"
        assert contract["model_packs"]["credit"]["family"] == "reduced_form_spread_grid"
        assert contract["model_packs"]["volatility"]["family"] == "regime_surface_bundle"
        assert contract["quote_bundles"]["credit"]["quote_families"] == ["spread", "hazard"]
        assert "usd_ois" in contract["runtime_targets"]["discount_curves"]
        assert "USD-SOFR-3M" in contract["runtime_targets"]["forecast_curves"]

    def test_synthetic_generation_contract_is_deterministic_for_same_request(self):
        provider = MockDataProvider()

        first = provider.fetch_market_snapshot(date(2024, 11, 15))
        second = provider.fetch_market_snapshot(date(2024, 11, 15))

        assert (
            first.provenance["prior_parameters"]["synthetic_generation_contract"]
            == second.provenance["prior_parameters"]["synthetic_generation_contract"]
        )

    def test_model_consistency_contract_is_derived_from_synthetic_generation_contract(self):
        provider = MockDataProvider()
        snapshot = provider.fetch_market_snapshot(date(2024, 11, 15))

        consistency = snapshot.provenance["prior_parameters"]["model_consistency_contract"]
        generation = snapshot.provenance["prior_parameters"]["synthetic_generation_contract"]

        assert consistency["rates"]["forecast_basis_bps"] == generation["quote_bundles"]["rates"]["forecast_basis_bps"]
        assert consistency["credit"]["spread_inputs_decimal"] == generation["quote_bundles"]["credit"]["spread_inputs_decimal"]
        assert consistency["volatility"]["rate_vol_surfaces"] == generation["runtime_targets"]["vol_surfaces"]
        assert consistency["volatility"]["local_vol_surfaces"] == generation["runtime_targets"]["local_vol_surfaces"]
        assert sorted(consistency["volatility"]["model_parameter_sets"]) == sorted(
            generation["runtime_targets"]["model_parameter_sets"]
        )

    def test_model_consistency_contract_preserves_multi_curve_basis(self):
        provider = MockDataProvider()
        snapshot = provider.fetch_market_snapshot(date(2024, 11, 15))
        contract = snapshot.provenance["prior_parameters"]["model_consistency_contract"]
        basis_bps = contract["rates"]["forecast_basis_bps"]["USD-SOFR-3M"]

        discount = snapshot.discount_curve("usd_ois")
        forecast = snapshot.forecast_curves["USD-SOFR-3M"]
        for tenor in (0.25, 1.0, 5.0):
            observed_basis_bps = (forecast.zero_rate(tenor) - discount.zero_rate(tenor)) * 10000.0
            assert observed_basis_bps == pytest.approx(basis_bps)

    def test_model_consistency_contract_credit_spreads_match_hazard_curve_knots(self):
        provider = MockDataProvider()
        snapshot = provider.fetch_market_snapshot(date(2024, 11, 15))
        contract = snapshot.provenance["prior_parameters"]["model_consistency_contract"]
        recovery = contract["credit"]["recovery"]
        spread_grid = contract["credit"]["spread_inputs_decimal"]["usd_ig"]
        ig_curve = snapshot.credit_curves["usd_ig"]

        for tenor_text, spread in spread_grid.items():
            tenor = float(tenor_text)
            expected_hazard = float(spread) / (1.0 - recovery)
            assert ig_curve.hazard_rate(tenor) == pytest.approx(expected_hazard)

    def test_model_consistency_contract_credit_inputs_drive_calibration_handoff(self):
        provider = MockDataProvider()
        snapshot = provider.fetch_market_snapshot(date(2024, 11, 15))
        contract = snapshot.provenance["prior_parameters"]["model_consistency_contract"]
        state = snapshot.to_market_state(
            settlement=date(2024, 11, 15),
            discount_curve="usd_ois",
            forecast_curve="USD-SOFR-3M",
            credit_curve="usd_ig",
        )
        quotes = tuple(
            CreditHazardCalibrationQuote(
                maturity_years=float(tenor_text),
                quote=float(spread),
                quote_kind="spread",
                label=f"synthetic_{tenor_text}y",
            )
            for tenor_text, spread in contract["credit"]["spread_inputs_decimal"]["usd_ig"].items()
        )

        result = calibrate_single_name_credit_curve_workflow(
            quotes,
            state,
            recovery=float(contract["credit"]["recovery"]),
            curve_name="synthetic_ig_credit",
        )
        calibrated_state = result.apply_to_market_state(state)
        record = calibrated_state.materialized_calibrated_object(object_kind="credit_curve")

        assert record is not None
        assert record["object_name"] == "synthetic_ig_credit"
        assert record["selected_curve_roles"]["discount_curve"] == "usd_ois"
        assert result.max_abs_hazard_residual == pytest.approx(0.0)

        schedule = build_cds_schedule(
            date(2024, 11, 15),
            date(2029, 11, 15),
            Frequency.QUARTERLY,
            DayCountConvention.ACT_360,
        )
        observed = price_cds_analytical(
            notional=1_000_000.0,
            spread_quote=contract["credit"]["spread_inputs_decimal"]["usd_ig"]["5.0"],
            recovery=float(contract["credit"]["recovery"]),
            schedule=schedule,
            credit_curve=calibrated_state.credit_curve,
            discount_curve=calibrated_state.discount,
        )

        assert observed == pytest.approx(
            price_cds_analytical(
                notional=1_000_000.0,
                spread_quote=contract["credit"]["spread_inputs_decimal"]["usd_ig"]["5.0"],
                recovery=float(contract["credit"]["recovery"]),
                schedule=schedule,
                credit_curve=snapshot.credit_curve("usd_ig"),
                discount_curve=snapshot.discount_curve("usd_ois"),
            )
        )

    def test_user_supplied_snapshot_keeps_synthetic_generation_contract_absent(self):
        provider = MockDataProvider.from_dict({date(2025, 6, 1): {1.0: 0.04, 5.0: 0.042}})
        snapshot = provider.fetch_market_snapshot(date(2025, 6, 1))

        assert "synthetic_generation_contract" not in snapshot.provenance["prior_parameters"]


class TestResolverMockSource:
    """Integration: resolve_curve(source='mock') works without mocking."""

    def test_mock_source_returns_curve(self):
        curve = resolve_curve(as_of=date(2024, 11, 15), source="mock")
        assert len(curve.tenors) == len(_TENOR_GRID)
        # Rates should be continuously compounded (converted from BEY)
        assert all(r > 0 for r in curve.rates)

    def test_mock_source_latest(self):
        curve = resolve_curve(as_of="latest", source="mock")
        assert len(curve.tenors) == len(_TENOR_GRID)

    def test_mock_source_returns_full_market_snapshot(self):
        snapshot = resolve_market_snapshot(as_of=date(2024, 11, 15), source="mock")
        assert snapshot.discount_curve() is not None
        assert snapshot.vol_surface() is not None
        assert snapshot.credit_curve() is not None
        assert snapshot.fixing_history() is not None
        assert snapshot.state_space() is not None
        assert "USD-SOFR-3M" in snapshot.forecast_curves
        assert "EURUSD" in snapshot.fx_rates
        assert snapshot.underlier_spot() == pytest.approx(snapshot.underlier_spots["SPX"])
        assert snapshot.local_vol_surface() is not None
        assert snapshot.jump_parameters() is not None
        assert snapshot.model_parameters() is not None
        assert snapshot.provenance["prior_family"] == "embedded_market_regime"
        assert snapshot.provenance["prior_parameters"]["snapshot_date"] == "2024-11-15"
