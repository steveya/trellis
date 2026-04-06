"""Tests for calibration: implied vol, SABR fit, rates vol, Dupire local vol."""

from dataclasses import replace
from datetime import date

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.bootstrap import (
    BootstrapConventionBundle,
    BootstrapCurveInputBundle,
    BootstrapInstrument,
)
from trellis.curves.yield_curve import YieldCurve
from trellis.data.resolver import resolve_market_snapshot
from trellis.data.schema import MarketSnapshot
from trellis.instruments._agent.swaption import SwaptionPayoff, SwaptionSpec
from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
from trellis.models.bermudan_swaption_tree import (
    BermudanSwaptionTreeSpec,
    price_bermudan_swaption_tree,
)
from trellis.models.calibration.rates import (
    HullWhiteCalibrationInstrument,
    HullWhiteCalibrationResult,
    RatesCalibrationResult,
    calibrate_hull_white,
    calibrate_cap_floor_black_vol,
    calibrate_swaption_black_vol,
    swaption_terms,
)
from trellis.models.calibration.implied_vol import implied_vol, implied_vol_jaeckel, _bs_price
from trellis.models.calibration.sabr_fit import (
    SABRSmileCalibrationResult,
    build_sabr_smile_surface,
    calibrate_sabr_smile_workflow,
    calibrate_sabr,
    fit_sabr_smile_surface,
)
from trellis.models.calibration.local_vol import (
    LocalVolCalibrationResult,
    calibrate_local_vol_surface_workflow,
    dupire_local_vol,
    dupire_local_vol_result,
)
from trellis.models.calibration.heston_fit import (
    HestonSmileCalibrationResult,
    build_heston_smile_surface,
    calibrate_heston_smile_workflow,
    fit_heston_smile_surface,
)
from trellis.models.calibration.quote_maps import (
    QuoteMapSpec,
    build_identity_quote_map,
    build_implied_vol_quote_map,
    supported_quote_map_surface,
)
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)
BOOTSTRAPPED_SOFR_CURVE = "USD-SOFR-3M-BOOT"


# ---------------------------------------------------------------------------
# implied_vol round-trip
# ---------------------------------------------------------------------------


class TestImpliedVol:
    def test_round_trip_call(self):
        """Compute BS call price, then recover vol."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        price = _bs_price(S, K, T, r, sigma, "call")
        recovered = implied_vol(price, S, K, T, r, option_type="call")
        assert recovered == pytest.approx(sigma, abs=1e-6)

    def test_round_trip_put(self):
        S, K, T, r, sigma = 100.0, 110.0, 1.0, 0.05, 0.30
        price = _bs_price(S, K, T, r, sigma, "put")
        recovered = implied_vol(price, S, K, T, r, option_type="put")
        assert recovered == pytest.approx(sigma, abs=1e-6)

    def test_round_trip_otm_call(self):
        S, K, T, r, sigma = 100.0, 120.0, 0.5, 0.03, 0.25
        price = _bs_price(S, K, T, r, sigma, "call")
        recovered = implied_vol(price, S, K, T, r, option_type="call")
        assert recovered == pytest.approx(sigma, abs=1e-6)


# ---------------------------------------------------------------------------
# implied_vol_jaeckel round-trip
# ---------------------------------------------------------------------------


class TestImpliedVolJaeckel:
    def test_round_trip_call(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        price = _bs_price(S, K, T, r, sigma, "call")
        recovered = implied_vol_jaeckel(price, S, K, T, r, option_type="call")
        assert recovered == pytest.approx(sigma, abs=1e-4)

    def test_matches_brent_method(self):
        """Jaeckel and Brent should give the same result."""
        S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.25
        price = _bs_price(S, K, T, r, sigma, "call")
        vol_brent = implied_vol(price, S, K, T, r, option_type="call")
        vol_jaeckel = implied_vol_jaeckel(price, S, K, T, r, option_type="call")
        assert vol_jaeckel == pytest.approx(vol_brent, abs=1e-4)


class TestQuoteMaps:
    def test_supported_quote_map_surface_covers_bounded_variants(self):
        specs = supported_quote_map_surface()
        variants = {(spec.quote_family, spec.convention) for spec in specs}
        assert ("price", "") in variants
        assert ("implied_vol", "black") in variants
        assert ("implied_vol", "normal") in variants
        assert ("par_rate", "") in variants
        assert ("spread", "") in variants
        assert ("hazard", "") in variants

    def test_identity_quote_map_round_trips_price_quotes(self):
        quote_map = build_identity_quote_map(
            QuoteMapSpec(quote_family="price"),
            source_ref="unit_test",
        )
        target = quote_map.target_price(123.45)
        model = quote_map.model_quote(123.45)
        assert target.failure is None
        assert model.failure is None
        assert target.value == pytest.approx(123.45)
        assert model.value == pytest.approx(123.45)

    def test_implied_vol_quote_map_surfaces_inverse_failures(self):
        quote_map = build_implied_vol_quote_map(
            convention="black",
            quote_to_price_fn=lambda quote: 10.0 + float(quote),
            price_to_quote_fn=lambda _price: (_ for _ in ()).throw(ValueError("solver failed")),
            source_ref="unit_test",
        )
        target = quote_map.target_price(0.25)
        model = quote_map.model_quote(10.25)
        assert target.failure is None
        assert target.value == pytest.approx(10.25)
        assert model.failure is not None
        assert "solver failed" in model.failure

    def test_quote_map_spec_rejects_unsupported_quote_family(self):
        with pytest.raises(ValueError, match="unsupported quote_family"):
            QuoteMapSpec(quote_family="variance_swap")


# ---------------------------------------------------------------------------
# SABR calibration
# ---------------------------------------------------------------------------


class TestSABRCalibration:
    def test_build_sabr_smile_surface_reports_alignment_and_bracketing_warnings(self):
        surface = build_sabr_smile_surface(
            100.0,
            1.0,
            [80.0, 85.0, 90.0, 95.0],
            [0.28, 0.25, 0.23, 0.22],
            beta=0.5,
            surface_name="usd_swaption_smile",
        )

        assert surface.surface_name == "usd_swaption_smile"
        assert surface.labels == ("strike_80", "strike_85", "strike_90", "strike_95")
        assert surface.weights == (1.0, 1.0, 1.0, 1.0)
        assert surface.payload["strike_count"] == 4
        assert surface.payload["atm_strike"] == pytest.approx(95.0)
        assert "nearest strike" in surface.warnings[0].lower()
        assert "does not bracket" in surface.warnings[1].lower()

    def test_fit_sabr_smile_surface_returns_reusable_result_artifacts(self):
        from trellis.models.processes.sabr import SABRProcess

        F, T = 100.0, 1.0
        alpha_true, beta, rho_true, nu_true = 0.20, 0.5, -0.3, 0.4
        sabr_true = SABRProcess(alpha_true, beta, rho_true, nu_true)
        strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
        market_vols = [sabr_true.implied_vol(F, K, T) for K in strikes]
        surface = build_sabr_smile_surface(
            F,
            T,
            strikes,
            market_vols,
            beta=beta,
            surface_name="usd_rates_smile",
        )

        result = fit_sabr_smile_surface(surface)

        assert isinstance(result, SABRSmileCalibrationResult)
        assert result.surface.surface_name == "usd_rates_smile"
        assert result.solve_request.request_id == "sabr_smile_least_squares"
        assert result.solve_result.success is True
        assert result.solver_provenance.backend["backend_id"] == "scipy"
        assert result.diagnostics.point_count == len(strikes)
        assert result.diagnostics.max_abs_vol_error < 0.005
        assert result.diagnostics.warning_count == 0
        assert result.summary["surface_name"] == "usd_rates_smile"
        assert result.provenance["fit_diagnostics"]["point_count"] == len(strikes)
        assert result.provenance["calibration_target"]["quote_map"]["quote_family"] == "implied_vol"
        assert result.provenance["calibration_target"]["quote_map"]["convention"] == "black"
        assert result.provenance["warnings"] == []

    def test_calibrate_sabr_smile_workflow_returns_supported_result(self):
        from trellis.models.processes.sabr import SABRProcess

        F, T = 100.0, 1.0
        alpha_true, beta, rho_true, nu_true = 0.20, 0.5, -0.3, 0.4
        sabr_true = SABRProcess(alpha_true, beta, rho_true, nu_true)
        strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
        market_vols = [sabr_true.implied_vol(F, K, T) for K in strikes]

        result = calibrate_sabr_smile_workflow(
            F,
            T,
            strikes,
            market_vols,
            beta=beta,
            labels=[f"pt_{index}" for index in range(len(strikes))],
            weights=[1.0, 1.0, 1.5, 2.0, 1.5, 1.0, 1.0],
            surface_name="usd_rates_1y_smile",
        )

        assert isinstance(result, SABRSmileCalibrationResult)
        assert result.surface.surface_name == "usd_rates_1y_smile"
        assert result.surface.labels[0] == "pt_0"
        assert result.surface.weights[3] == pytest.approx(2.0)
        assert result.provenance["source_ref"] == "calibrate_sabr_smile_workflow"
        assert result.summary["surface_name"] == "usd_rates_1y_smile"
        assert result.warnings == ()

    def test_calibrated_vols_match_market(self):
        """Generate market vols from known SABR params, then calibrate back."""
        from trellis.models.processes.sabr import SABRProcess

        F, T = 100.0, 1.0
        alpha_true, beta, rho_true, nu_true = 0.20, 0.5, -0.3, 0.4
        sabr_true = SABRProcess(alpha_true, beta, rho_true, nu_true)

        strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
        market_vols = [sabr_true.implied_vol(F, K, T) for K in strikes]

        sabr_fit = calibrate_sabr(F, T, strikes, market_vols, beta=beta)

        for K, mv in zip(strikes, market_vols):
            fitted_vol = sabr_fit.implied_vol(F, K, T)
            assert fitted_vol == pytest.approx(mv, abs=0.005)
        assert sabr_fit.calibration_provenance["source_kind"] == "calibrated_surface"
        assert sabr_fit.calibration_provenance["calibration_target"]["strike_count"] == len(strikes)
        assert sabr_fit.calibration_provenance["calibration_target"]["beta"] == pytest.approx(beta)
        assert sabr_fit.calibration_provenance["solve_request"]["problem_kind"] == "least_squares"
        assert sabr_fit.calibration_provenance["solve_request"]["objective"]["labels"] == [
            f"strike_{float(strike):g}" for strike in strikes
        ]
        assert sabr_fit.calibration_provenance["solve_result"]["metadata"]["backend_id"] == "scipy"
        assert sabr_fit.calibration_provenance["solver_provenance"]["backend"]["backend_id"] == "scipy"
        assert sabr_fit.calibration_provenance["solver_provenance"]["termination"]["success"] is True
        assert sabr_fit.calibration_provenance["solver_replay_artifact"]["request"]["request_id"] == (
            "sabr_smile_least_squares"
        )
        assert sabr_fit.calibration_provenance["fit_diagnostics"]["point_count"] == len(strikes)
        assert sabr_fit.calibration_provenance["fit_diagnostics"]["max_abs_vol_error"] < 0.005
        assert sabr_fit.calibration_provenance["calibration_target"]["quote_map"]["quote_family"] == "implied_vol"
        assert sabr_fit.calibration_provenance["calibration_target"]["quote_map"]["convention"] == "black"
        assert sabr_fit.calibration_provenance["warnings"] == []
        assert sabr_fit.calibration_summary["optimizer_success"] is True

    def test_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError, match="same length"):
            calibrate_sabr(100.0, 1.0, [90.0, 100.0, 110.0], [0.2, 0.21], beta=0.5)


# ---------------------------------------------------------------------------
# Dupire local vol
# ---------------------------------------------------------------------------


class TestDupireLocalVol:
    def test_dupire_local_vol_result_reports_stable_flat_surface(self):
        sigma_flat = 0.20
        S0, r = 100.0, 0.05
        strikes = raw_np.linspace(60, 150, 30)
        expiries = raw_np.linspace(0.1, 3.0, 15)
        implied_vols = raw_np.full((len(expiries), len(strikes)), sigma_flat)

        result = dupire_local_vol_result(strikes, expiries, implied_vols, S0, r)

        assert isinstance(result, LocalVolCalibrationResult)
        assert result.diagnostics.unstable_point_count == 0
        assert result.warnings == ()
        assert result.provenance["fit_diagnostics"]["unstable_point_count"] == 0

    def test_flat_vol_surface_gives_constant_local_vol(self):
        """If implied vol is flat (constant sigma), local vol = sigma everywhere."""
        sigma_flat = 0.20
        S0, r = 100.0, 0.05

        strikes = raw_np.linspace(60, 150, 30)
        expiries = raw_np.linspace(0.1, 3.0, 15)
        # Flat surface
        implied_vols = raw_np.full((len(expiries), len(strikes)), sigma_flat)

        local_vol_fn = dupire_local_vol(strikes, expiries, implied_vols, S0, r)

        # Check local vol at several (S, t) points near the center
        for S in [90.0, 100.0, 110.0]:
            for t in [0.5, 1.0, 2.0]:
                lv = local_vol_fn(S, t)
                assert lv == pytest.approx(sigma_flat, abs=0.02)
        assert local_vol_fn.calibration_provenance["source_kind"] == "calibrated_surface"
        assert local_vol_fn.calibration_target["surface_shape"] == (len(expiries), len(strikes))
        assert local_vol_fn.calibration_summary["spot"] == pytest.approx(S0)
        assert local_vol_fn.calibration_diagnostics["unstable_point_count"] == 0
        assert local_vol_fn.calibration_warnings == []

    def test_dupire_local_vol_result_flags_unstable_regions(self):
        strikes = raw_np.array([70.0, 90.0, 110.0, 130.0])
        expiries = raw_np.array([0.25, 0.5, 1.0, 2.0])
        implied_vols = raw_np.array(
            [
                [0.65, 0.18, 0.70, 0.22],
                [0.62, 0.16, 0.68, 0.20],
                [0.58, 0.14, 0.64, 0.18],
                [0.54, 0.12, 0.60, 0.16],
            ],
            dtype=float,
        )

        result = dupire_local_vol_result(strikes, expiries, implied_vols, 100.0, 0.03)
        payload = result.to_payload()

        assert result.diagnostics.unstable_point_count > 0
        assert result.diagnostics.unstable_point_count >= len(result.diagnostics.sample_unstable_points)
        assert result.diagnostics.sample_unstable_points
        assert result.warnings
        assert "unstable" in result.warnings[0].lower()
        assert payload["fit_diagnostics"]["unstable_point_count"] == result.diagnostics.unstable_point_count
        assert payload["warnings"] == list(result.warnings)
        assert payload["calibration_target"]["quote_map"]["quote_family"] == "implied_vol"
        assert payload["calibration_target"]["quote_map"]["convention"] == "black"
        assert result.local_vol_surface(100.0, 1.0) >= 0.0

    def test_rejects_mismatched_surface_shape(self):
        with pytest.raises(ValueError, match="shape"):
            dupire_local_vol(
                raw_np.array([60.0, 70.0, 80.0, 90.0]),
                raw_np.array([0.25, 0.5, 1.0, 2.0]),
                raw_np.ones((3, 4)),
                100.0,
                0.05,
            )

    def test_local_vol_workflow_applies_surface_to_market_state(self):
        sigma_flat = 0.20
        strikes = raw_np.linspace(60, 150, 30)
        expiries = raw_np.linspace(0.1, 3.0, 15)
        implied_vols = raw_np.full((len(expiries), len(strikes)), sigma_flat)
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            spot=100.0,
            discount=YieldCurve.flat(0.05),
        )

        result = calibrate_local_vol_surface_workflow(
            strikes,
            expiries,
            implied_vols,
            100.0,
            0.05,
            surface_name="equity_local_vol",
        )
        enriched_state = result.apply_to_market_state(market_state)

        assert result.surface_name == "equity_local_vol"
        assert result.provenance["source_ref"] == "calibrate_local_vol_surface_workflow"
        assert "equity_local_vol" in enriched_state.local_vol_surfaces
        assert enriched_state.local_vol_surface(100.0, 1.0) == pytest.approx(sigma_flat, abs=0.02)
        local_vol_materialization = enriched_state.materialized_calibrated_object(object_kind="local_vol_surface")
        assert local_vol_materialization is not None
        assert local_vol_materialization["object_name"] == "equity_local_vol"
        assert local_vol_materialization["source_ref"] == "calibrate_local_vol_surface_workflow"
        assert local_vol_materialization["metadata"]["surface_shape"] == [15, 30]


class TestHestonCalibration:
    def test_build_heston_smile_surface_uses_forward_for_atm_diagnostics(self):
        surface = build_heston_smile_surface(
            spot=100.0,
            rate=0.05,
            dividend_yield=0.0,
            expiry_years=1.0,
            strikes=[95.0, 100.0, 105.0, 110.0, 115.0],
            market_vols=[0.26, 0.24, 0.22, 0.23, 0.25],
            surface_name="equity_heston_smile",
        )

        assert surface.payload["atm_strike"] == pytest.approx(105.0)
        assert surface.payload["atm_market_vol"] == pytest.approx(0.22)
        assert "nearest strike" in surface.warnings[0].lower()
        assert "forward" in surface.warnings[0].lower()

    @staticmethod
    def _market_vols_from_heston(*, spot, rate, expiry_years, strikes, params):
        from trellis.models.calibration.implied_vol import implied_vol
        from trellis.models.processes.heston import Heston
        from trellis.models.transforms.fft_pricer import fft_price

        process = Heston(
            mu=rate,
            kappa=params["kappa"],
            theta=params["theta"],
            xi=params["xi"],
            rho=params["rho"],
            v0=params["v0"],
        )
        return [
            implied_vol(
                fft_price(
                    lambda u: process.characteristic_function(u, expiry_years, log_spot=raw_np.log(spot)),
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

    def test_fit_heston_smile_surface_returns_runtime_binding_and_replay_artifacts(self):
        spot = 100.0
        rate = 0.02
        expiry_years = 1.0
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        true_params = {
            "kappa": 1.8,
            "theta": 0.04,
            "xi": 0.35,
            "rho": -0.6,
            "v0": 0.05,
        }
        market_vols = self._market_vols_from_heston(
            spot=spot,
            rate=rate,
            expiry_years=expiry_years,
            strikes=strikes,
            params=true_params,
        )

        surface = build_heston_smile_surface(
            spot,
            expiry_years,
            strikes,
            market_vols,
            rate=rate,
            surface_name="equity_1y_smile",
        )
        result = fit_heston_smile_surface(
            surface,
            initial_guess=(1.2, 0.05, 0.25, -0.3, 0.04),
            parameter_set_name="heston_equity",
        )
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            spot=spot,
            discount=YieldCurve.flat(rate),
        )
        enriched_state = result.apply_to_market_state(market_state)

        assert isinstance(result, HestonSmileCalibrationResult)
        assert result.surface.surface_name == "equity_1y_smile"
        assert result.solve_request.request_id == "heston_smile_least_squares"
        assert result.solve_result.success is True
        assert result.solver_provenance.backend["backend_id"] == "scipy"
        assert result.runtime_binding.parameter_set_name == "heston_equity"
        assert result.runtime_binding.process.rho == pytest.approx(true_params["rho"], abs=0.05)
        assert result.diagnostics.point_count == len(strikes)
        assert result.diagnostics.max_abs_vol_error < 0.005
        assert result.model_parameters["model_family"] == "heston"
        assert enriched_state.model_parameter_sets["heston_equity"]["model_family"] == "heston"
        heston_materialization = enriched_state.materialized_calibrated_object(object_kind="model_parameter_set")
        assert heston_materialization is not None
        assert heston_materialization["object_name"] == "heston_equity"
        assert heston_materialization["source_kind"] == "calibrated_surface"
        assert heston_materialization["metadata"]["model_family"] == "heston"
        assert result.provenance["fit_diagnostics"]["point_count"] == len(strikes)
        assert result.provenance["calibration_target"]["quote_map"]["quote_family"] == "implied_vol"
        assert result.provenance["calibration_target"]["quote_map"]["convention"] == "black"

    def test_calibrate_heston_smile_workflow_returns_supported_result(self):
        spot = 100.0
        rate = 0.02
        expiry_years = 1.0
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        true_params = {
            "kappa": 1.8,
            "theta": 0.04,
            "xi": 0.35,
            "rho": -0.6,
            "v0": 0.05,
        }
        market_vols = self._market_vols_from_heston(
            spot=spot,
            rate=rate,
            expiry_years=expiry_years,
            strikes=strikes,
            params=true_params,
        )

        result = calibrate_heston_smile_workflow(
            spot,
            expiry_years,
            strikes,
            market_vols,
            rate=rate,
            surface_name="equity_heston_workflow",
            parameter_set_name="heston_equity",
            warm_start=(1.2, 0.05, 0.25, -0.3, 0.04),
        )

        assert isinstance(result, HestonSmileCalibrationResult)
        assert result.provenance["source_ref"] == "calibrate_heston_smile_workflow"
        assert result.summary["surface_name"] == "equity_heston_workflow"
        assert result.provenance["calibration_target"]["quote_map"]["quote_family"] == "implied_vol"
        assert result.provenance["calibration_target"]["quote_map"]["convention"] == "black"
        assert result.solve_request.warm_start is not None
        assert result.solve_request.warm_start.parameter_values == pytest.approx((1.2, 0.05, 0.25, -0.3, 0.04))
        assert result.runtime_binding.process.theta == pytest.approx(true_params["theta"], abs=0.02)


# ---------------------------------------------------------------------------
# Rates Black-vol calibration
# ---------------------------------------------------------------------------


def _multi_curve_state() -> tuple[MarketSnapshot, object]:
    """Return a multi-curve snapshot and its compiled MarketState."""
    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="test",
        discount_curves={
            "usd_ois": YieldCurve.flat(0.050),
            "usd_ois_alt": YieldCurve.flat(0.045),
        },
        forecast_curves={
            "USD-SOFR-3M": YieldCurve.flat(0.052),
            "USD-LIBOR-3M": YieldCurve.flat(0.054),
        },
        provenance={
            "source": "test",
            "source_kind": "explicit_input",
            "source_ref": "_multi_curve_state",
        },
    )
    market_state = snapshot.to_market_state(
        settlement=SETTLE,
        discount_curve="usd_ois",
        forecast_curve="USD-SOFR-3M",
    )
    return snapshot, market_state


def _bootstrap_bundle(
    *,
    curve_name: str,
    rate_index: str,
    deposit_quote: float,
    swap_2y_quote: float,
    swap_5y_quote: float,
) -> BootstrapCurveInputBundle:
    return BootstrapCurveInputBundle(
        curve_name=curve_name,
        currency="USD",
        rate_index=rate_index,
        conventions=BootstrapConventionBundle(
            swap_fixed_frequency=Frequency.ANNUAL,
            swap_float_frequency=Frequency.QUARTERLY,
            swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
            swap_float_day_count=DayCountConvention.ACT_360,
        ),
        instruments=(
            BootstrapInstrument(tenor=0.25, quote=deposit_quote, instrument_type="deposit", label=f"{curve_name}_DEP3M"),
            BootstrapInstrument(tenor=2.0, quote=swap_2y_quote, instrument_type="swap", label=f"{curve_name}_SWAP2Y"),
            BootstrapInstrument(tenor=5.0, quote=swap_5y_quote, instrument_type="swap", label=f"{curve_name}_SWAP5Y"),
        ),
    )


def _bootstrapped_multi_curve_state():
    snapshot = resolve_market_snapshot(
        as_of=SETTLE,
        source="mock",
        discount_curve_bootstraps={
            "usd_ois_boot": _bootstrap_bundle(
                curve_name="usd_ois_boot",
                rate_index=BOOTSTRAPPED_SOFR_CURVE,
                deposit_quote=0.040,
                swap_2y_quote=0.045,
                swap_5y_quote=0.048,
            ),
        },
        forecast_curve_bootstraps={
            BOOTSTRAPPED_SOFR_CURVE: _bootstrap_bundle(
                curve_name=BOOTSTRAPPED_SOFR_CURVE,
                rate_index=BOOTSTRAPPED_SOFR_CURVE,
                deposit_quote=0.041,
                swap_2y_quote=0.046,
                swap_5y_quote=0.049,
            ),
        },
    )
    market_state = snapshot.to_market_state(
        settlement=SETTLE,
        discount_curve="usd_ois_boot",
        forecast_curve=BOOTSTRAPPED_SOFR_CURVE,
    )
    return snapshot, market_state


class TestRatesCalibration:
    @pytest.mark.parametrize("kind,payoff_cls", [("cap", CapPayoff), ("floor", FloorPayoff)])
    def test_cap_floor_round_trip_preserves_curve_provenance(self, kind, payoff_cls):
        """Cap/floor implied-vol calibration should round-trip under multi-curve inputs."""
        _snapshot, market_state = _multi_curve_state()
        true_vol = 0.215
        spec = CapFloorSpec(
            notional=1_000_000.0,
            strike=0.05,
            start_date=date(2025, 2, 15),
            end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
            day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
        )
        target_state = replace(market_state, vol_surface=FlatVol(true_vol))
        target_price = payoff_cls(spec).evaluate(target_state)

        result = calibrate_cap_floor_black_vol(
            spec,
            market_state,
            target_price,
            kind=kind,
            vol_surface_name="rates_cap_surface",
            correlation_source="not_used",
        )

        assert isinstance(result, RatesCalibrationResult)
        assert result.calibrated_vol == pytest.approx(true_vol, abs=1e-6)
        assert result.repriced_price == pytest.approx(target_price, abs=1e-5)
        assert result.residual == pytest.approx(0.0, abs=1e-5)
        assert result.provenance["selected_curve_names"] == {
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
        }
        assert result.provenance["rate_index"] == "USD-SOFR-3M"
        assert result.provenance["vol_surface_name"] == "rates_cap_surface"
        assert result.provenance["correlation_source"] == "not_used"
        assert result.provenance["quote_map"]["quote_family"] == "implied_vol"
        assert result.provenance["quote_map"]["convention"] == "black"
        assert result.provenance["solve_request"]["problem_kind"] == "root_scalar"
        assert result.provenance["solve_request"]["objective"]["labels"] == ["price_residual"]
        assert result.provenance["solve_result"]["metadata"]["backend_id"] == "scipy"
        assert result.provenance["solver_provenance"]["backend"]["backend_id"] == "scipy"
        assert result.provenance["solver_provenance"]["options"]["tol"] == pytest.approx(1e-8)
        assert result.provenance["solver_provenance"]["termination"]["success"] is True
        assert result.provenance["solver_replay_artifact"]["request"]["request_id"] == (
            "rates_flat_black_vol_root"
        )
        assert result.provenance["market_provenance"]["source_kind"] == "explicit_input"
        assert result.summary["period_count"] > 0

    def test_swaption_round_trip_preserves_curve_provenance(self):
        """Swaption implied-vol calibration should round-trip under multi-curve inputs."""
        _snapshot, market_state = _multi_curve_state()
        true_vol = 0.180
        spec = SwaptionSpec(
            notional=5_000_000.0,
            strike=0.05,
            expiry_date=date(2026, 2, 15),
            swap_start=date(2026, 2, 15),
            swap_end=date(2031, 2, 15),
            swap_frequency=Frequency.SEMI_ANNUAL,
            day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
            is_payer=True,
        )
        target_state = replace(market_state, vol_surface=FlatVol(true_vol))
        target_price = SwaptionPayoff(spec).evaluate(target_state)
        T, annuity, forward_swap_rate, payment_count = swaption_terms(spec, market_state)
        assert T > 0.0
        assert annuity > 0.0
        assert forward_swap_rate > 0.0
        assert payment_count > 0

        result = calibrate_swaption_black_vol(
            spec,
            market_state,
            target_price,
            vol_surface_name="rates_swaption_surface",
            correlation_source="corr_pack_A",
        )

        assert isinstance(result, RatesCalibrationResult)
        assert result.calibrated_vol == pytest.approx(true_vol, abs=1e-6)
        assert result.repriced_price == pytest.approx(target_price, abs=1e-5)
        assert result.residual == pytest.approx(0.0, abs=1e-5)
        assert result.provenance["selected_curve_names"] == {
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
        }
        assert result.provenance["rate_index"] == "USD-SOFR-3M"
        assert result.provenance["vol_surface_name"] == "rates_swaption_surface"
        assert result.provenance["correlation_source"] == "corr_pack_A"
        assert result.provenance["quote_map"]["quote_family"] == "implied_vol"
        assert result.provenance["quote_map"]["convention"] == "black"
        assert result.provenance["solve_request"]["problem_kind"] == "root_scalar"
        assert result.provenance["solve_request"]["objective"]["labels"] == ["price_residual"]
        assert result.provenance["solve_result"]["metadata"]["backend_id"] == "scipy"
        assert result.provenance["solver_provenance"]["backend"]["backend_id"] == "scipy"
        assert result.provenance["solver_provenance"]["termination"]["success"] is True
        assert result.provenance["solver_replay_artifact"]["request"]["request_id"] == (
            "rates_flat_black_vol_root"
        )
        assert result.provenance["market_provenance"]["source_ref"] == "_multi_curve_state"
        assert result.summary["annuity"] > 0.0
        assert result.summary["forward_swap_rate"] > 0.0

    def test_hull_white_calibration_round_trip_preserves_bootstrap_provenance(self):
        _snapshot, market_state = _bootstrapped_multi_curve_state()
        true_mean_reversion = 0.08
        true_sigma = 0.006
        tree_specs = (
            BermudanSwaptionTreeSpec(
                notional=1_000_000.0,
                strike=0.047,
                exercise_dates=(date(2025, 11, 15),),
                swap_end=date(2030, 11, 15),
                swap_frequency=Frequency.SEMI_ANNUAL,
                day_count=DayCountConvention.ACT_360,
                rate_index=BOOTSTRAPPED_SOFR_CURVE,
                is_payer=True,
            ),
            BermudanSwaptionTreeSpec(
                notional=1_000_000.0,
                strike=0.048,
                exercise_dates=(date(2026, 11, 15),),
                swap_end=date(2031, 11, 15),
                swap_frequency=Frequency.SEMI_ANNUAL,
                day_count=DayCountConvention.ACT_360,
                rate_index=BOOTSTRAPPED_SOFR_CURVE,
                is_payer=True,
            ),
        )
        target_prices = tuple(
            price_bermudan_swaption_tree(
                market_state,
                spec,
                model="hull_white",
                mean_reversion=true_mean_reversion,
                sigma=true_sigma,
                n_steps=80,
            )
            for spec in tree_specs
        )
        instruments = tuple(
            HullWhiteCalibrationInstrument(
                label=f"ATM_{index + 1}",
                notional=spec.notional,
                strike=spec.strike,
                exercise_date=spec.exercise_dates[0],
                swap_end=spec.swap_end,
                quote=target_price,
                quote_kind="price",
                swap_frequency=spec.swap_frequency,
                day_count=spec.day_count,
                rate_index=spec.rate_index,
                is_payer=spec.is_payer,
            )
            for index, (spec, target_price) in enumerate(zip(tree_specs, target_prices))
        )

        result = calibrate_hull_white(
            instruments,
            market_state,
            n_steps=80,
            tol=1e-10,
            mean_reversion_bounds=(0.01, 0.30),
            sigma_bounds=(0.001, 0.02),
            parameter_set_name="hw_calibrated",
        )

        assert isinstance(result, HullWhiteCalibrationResult)
        assert result.mean_reversion == pytest.approx(true_mean_reversion, rel=0.05)
        assert result.sigma == pytest.approx(true_sigma, rel=0.05)
        assert result.max_abs_price_residual < 1e-5
        assert result.provenance["market_provenance"]["bootstrap_runs"]["discount_curves"]["usd_ois_boot"][
            "solver_provenance"
        ]["backend"]["backend_id"] == "scipy"
        assert result.provenance["solve_request"]["problem_kind"] == "least_squares"
        assert result.provenance["solve_result"]["metadata"]["backend_id"] == "scipy"
        assert result.provenance["solver_provenance"]["backend"]["backend_id"] == "scipy"
        assert result.provenance["calibration_target"]["quote_maps"][0]["quote_family"] == "price"
        assert result.provenance["calibration_target"]["quote_maps"][0]["multi_curve_roles"]["discount_curve"]
        assert result.provenance["calibration_target"]["quote_maps"][0]["multi_curve_roles"]["forecast_curve"]

        calibrated_state = result.apply_to_market_state(market_state)
        assert calibrated_state.model_parameters["model_family"] == "hull_white"
        assert calibrated_state.model_parameter_sets["hw_calibrated"]["sigma"] == pytest.approx(result.sigma)
        hw_materialization = calibrated_state.materialized_calibrated_object(object_kind="model_parameter_set")
        assert hw_materialization is not None
        assert hw_materialization["object_name"] == "hw_calibrated"
        assert hw_materialization["metadata"]["instrument_family"] == "rates"
        assert hw_materialization["selected_curve_roles"]["discount_curve"] == "usd_ois_boot"
        assert hw_materialization["selected_curve_roles"]["forecast_curve"] == BOOTSTRAPPED_SOFR_CURVE
        assert price_bermudan_swaption_tree(
            calibrated_state,
            tree_specs[0],
            model="hull_white",
            n_steps=80,
        ) == pytest.approx(target_prices[0], rel=1e-4)
