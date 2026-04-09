from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import patch

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.data.resolver import resolve_market_snapshot
from trellis.instruments.fx import FXRate
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _quanto_market_state(
    *,
    corr: float = 0.35,
    forecast_curves: dict[str, object] | None = None,
    model_parameters: dict[str, object] | None = None,
    selected_curve_names: dict[str, str] | None = None,
    market_provenance: dict[str, object] | None = None,
) -> MarketState:
    domestic = YieldCurve.flat(0.05)
    foreign = YieldCurve.flat(0.03)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=domestic,
        forecast_curves=forecast_curves if forecast_curves is not None else {"EUR-DISC": foreign},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        spot=100.0,
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        model_parameters=(
            model_parameters if model_parameters is not None else {"quanto_correlation": corr}
        ),
        selected_curve_names=selected_curve_names,
        market_provenance=market_provenance,
    )


def _himalaya_market_state(*, corr: float = 0.35) -> MarketState:
    from trellis.curves.forward_curve import ForwardCurve

    domestic = YieldCurve.flat(0.05)
    carry_a = YieldCurve.flat(0.03)
    carry_b = YieldCurve.flat(0.025)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=domestic,
        forecast_curves={
            "SPX-DISC": ForwardCurve(carry_a),
            "NDX-DISC": ForwardCurve(carry_b),
        },
        spot=100.0,
        underlier_spots={"SPX": 100.0, "NDX": 101.5},
        vol_surface=FlatVol(0.20),
        model_parameters={"correlation_matrix": corr},
    )


def _single_state_market_state(*, rate: float = 0.05, vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(rate, max_tenor=5.0),
        vol_surface=FlatVol(vol),
    )


def test_resolve_single_state_diffusion_inputs_reads_market_state_contract():
    from trellis.models.resolution.single_state_diffusion import (
        resolve_single_state_diffusion_inputs,
    )

    spec = SimpleNamespace(
        notional=100.0,
        spot=105.0,
        strike=110.0,
        expiry_date=date(2025, 11, 15),
        option_type="put",
    )

    resolved = resolve_single_state_diffusion_inputs(
        _single_state_market_state(rate=0.04, vol=0.30),
        spec,
    )

    assert resolved.notional == pytest.approx(100.0)
    assert resolved.spot == pytest.approx(105.0)
    assert resolved.strike == pytest.approx(110.0)
    assert resolved.maturity == pytest.approx(1.0)
    assert resolved.rate == pytest.approx(0.04)
    assert resolved.sigma == pytest.approx(0.30)
    assert resolved.dividend_yield == pytest.approx(0.0)
    assert resolved.option_type == "put"


def test_short_rate_comparison_regime_builds_typed_flat_market_objects():
    from trellis.models.resolution.short_rate_claims import ShortRateComparisonRegime

    regime = ShortRateComparisonRegime(
        regime_name="t01_short_rate_comparison",
        flat_discount_rate=0.05,
        flat_sigma=0.01,
        hull_white_mean_reversion=0.1,
    )

    discount_curve = regime.build_discount_curve(max_tenor=12.0)
    vol_surface = regime.build_vol_surface()

    assert discount_curve.zero_rate(3.0) == pytest.approx(0.05)
    assert vol_surface.black_vol(3.0, 0.63) == pytest.approx(0.01)
    payload = regime.to_payload()
    assert payload["regime_family"] == "short_rate"
    assert payload["vol_surface"]["quote_subject"] == "discount_bond_option"
    assert payload["vol_surface"]["quote_unit"] == "decimal_volatility"
    assert payload["vol_surface"]["quote_semantics"]["quote_subject"] == "discount_bond_option"


def test_resolve_short_rate_regime_uses_model_specific_mean_reversion_from_comparison_regime():
    from trellis.models.resolution.short_rate_claims import (
        ShortRateComparisonRegime,
        resolve_short_rate_regime,
    )

    regime = ShortRateComparisonRegime(
        regime_name="t01_short_rate_comparison",
        flat_discount_rate=0.05,
        flat_sigma=0.01,
        hull_white_mean_reversion=0.1,
        ho_lee_mean_reversion=0.0,
    )
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=regime.build_discount_curve(max_tenor=12.0),
        vol_surface=regime.build_vol_surface(),
        market_provenance={"comparison_regime": regime.to_payload()},
    )

    hull_white = resolve_short_rate_regime(
        market_state,
        model="hull_white",
        maturity=3.0,
        strike=0.63,
    )
    ho_lee = resolve_short_rate_regime(
        market_state,
        model="ho_lee",
        maturity=3.0,
        strike=0.63,
    )

    assert hull_white.sigma == pytest.approx(0.01)
    assert hull_white.mean_reversion == pytest.approx(0.1)
    assert ho_lee.sigma == pytest.approx(0.01)
    assert ho_lee.mean_reversion == pytest.approx(0.0)


def test_single_state_diffusion_gbm_characteristic_functions_match_known_moments():
    from trellis.models.resolution.single_state_diffusion import (
        ResolvedSingleStateDiffusionInputs,
        gbm_log_ratio_char_fn,
        gbm_log_spot_char_fn,
        put_from_call_parity,
    )

    resolved = ResolvedSingleStateDiffusionInputs(
        notional=100.0,
        spot=100.0,
        strike=105.0,
        maturity=1.25,
        rate=0.04,
        dividend_yield=0.01,
        sigma=0.30,
        option_type="put",
    )

    log_spot_phi = gbm_log_spot_char_fn(resolved)
    log_ratio_phi = gbm_log_ratio_char_fn(resolved)
    expected_forward = resolved.spot * np.exp(
        (resolved.rate - resolved.dividend_yield) * resolved.maturity
    )

    assert log_spot_phi(0.0) == pytest.approx(1.0)
    assert log_ratio_phi(0.0) == pytest.approx(1.0)
    assert log_spot_phi(-1j) == pytest.approx(expected_forward)
    assert log_ratio_phi(-1j) == pytest.approx(expected_forward / resolved.spot)

    call_price = 12.5
    expected_put = call_price - resolved.spot * np.exp(
        -resolved.dividend_yield * resolved.maturity
    ) + resolved.strike * np.exp(-resolved.rate * resolved.maturity)
    assert put_from_call_parity(call_price, resolved) == pytest.approx(expected_put)


def test_resolve_quanto_inputs_matches_expected_market_binding():
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    spec = QuantoOptionSpec(
        notional=250_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        underlier_currency="EUR",
        domestic_currency="USD",
    )

    resolved = resolve_quanto_inputs(_quanto_market_state(), spec)

    assert resolved.spot == pytest.approx(100.0)
    assert resolved.fx_spot == pytest.approx(1.10)
    assert resolved.domestic_df > 0.0
    assert resolved.foreign_df > 0.0
    assert resolved.sigma_underlier == pytest.approx(0.20)
    assert resolved.sigma_fx == pytest.approx(0.20)
    assert resolved.corr == pytest.approx(0.35)
    assert resolved.provenance["underlier_spot"]["source_kind"] == "underlier_spot"
    assert resolved.provenance["fx_spot"]["source_kind"] == "fx_rate"
    assert resolved.provenance["domestic_curve"]["source_kind"] == "discount_curve"
    assert resolved.provenance["foreign_curve"]["source_kind"] == "forecast_curve"
    assert resolved.provenance["foreign_curve"]["source_parameters"]["binding_kind"] == (
        "canonical_foreign_curve"
    )
    assert resolved.provenance["underlier_vol"]["source_kind"] == "surface_lookup"
    assert resolved.provenance["fx_vol"]["source_kind"] == "surface_lookup"
    assert resolved.provenance["correlation"]["source_family"] == "explicit"
    assert resolved.provenance["correlation"]["source_kind"] == "explicit_scalar"
    assert resolved.valuation_date == SETTLE


def test_resolve_quanto_inputs_requires_correlation():
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    market_state = _quanto_market_state()
    market_state = MarketState(
        as_of=market_state.as_of,
        settlement=market_state.settlement,
        discount=market_state.discount,
        forecast_curves=market_state.forecast_curves,
        fx_rates=market_state.fx_rates,
        spot=market_state.spot,
        underlier_spots=market_state.underlier_spots,
        vol_surface=market_state.vol_surface,
        model_parameters={},
    )
    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
    )

    with pytest.raises(ValueError, match="underlier/FX correlation"):
        resolve_quanto_inputs(market_state, spec)


@patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
def test_resolve_quanto_inputs_accepts_empirical_model_parameter_pack(MockProvider):
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    mock = MockProvider.return_value
    mock.fetch_yields.return_value = {
        0.25: 0.045,
        0.5: 0.046,
        1.0: 0.047,
        2.0: 0.048,
        5.0: 0.045,
        10.0: 0.044,
        30.0: 0.046,
    }
    observations = {
        "EUR": [0.01, 0.02, -0.01, 0.00],
        "EURUSD": [0.015, 0.025, -0.005, 0.005],
    }
    expected_corr = float(
        np.corrcoef(
            np.array(
                [
                    observations["EUR"],
                    observations["EURUSD"],
                ],
                dtype=float,
            ),
            rowvar=True,
        )[0, 1]
    )

    snapshot = resolve_market_snapshot(
        as_of=SETTLE,
        source="treasury_gov",
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.03)},
        model_parameter_sources={
            "empirical_quanto": {
                "source_kind": "empirical",
                "source_ref": "unit_test.empirical_history",
                "empirical_inputs": {
                    "observations": observations,
                    "window": {"lookback_days": 60, "frequency": "daily"},
                },
                "entries": (
                    {
                        "parameter": "quanto_correlation",
                        "measure": "pairwise_correlation",
                        "series_names": ("EUR", "EURUSD"),
                        "descriptor": True,
                    },
                ),
            }
        },
        default_model_parameters="empirical_quanto",
    )
    market_state = snapshot.to_market_state(
        settlement=SETTLE,
        model_parameters="empirical_quanto",
    )
    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        underlier_currency="EUR",
        domestic_currency="USD",
    )

    resolved = resolve_quanto_inputs(market_state, spec)

    assert resolved.corr == pytest.approx(float(np.clip(expected_corr, -0.999, 0.999)))
    assert resolved.provenance["correlation"]["source_family"] == "empirical"
    assert resolved.provenance["correlation"]["source_kind"] == "empirical_scalar"
    assert resolved.provenance["correlation"]["source_estimator"] == "sample_pearson"
    assert resolved.provenance["correlation"]["source_parameters"]["sample_size"] == 4


def test_resolve_quanto_inputs_rejects_noncanonical_single_forecast_curve_without_policy():
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    market_state = _quanto_market_state(
        forecast_curves={"foreign_proxy": YieldCurve.flat(0.03)},
    )
    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        underlier_currency="EUR",
        domestic_currency="USD",
    )

    with pytest.raises(ValueError, match="foreign carry/discount curve bound"):
        resolve_quanto_inputs(market_state, spec)


def test_resolve_quanto_inputs_rejects_implicit_domestic_discount_bridge():
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    market_state = _quanto_market_state(
        forecast_curves={},
    )
    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        underlier_currency="EUR",
        domestic_currency="USD",
    )

    with pytest.raises(ValueError, match="quanto_foreign_curve_policy"):
        resolve_quanto_inputs(market_state, spec)


def test_resolve_quanto_inputs_supports_selected_forecast_curve_bridge_policy():
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    market_state = _quanto_market_state(
        forecast_curves={"foreign_proxy": YieldCurve.flat(0.03)},
        selected_curve_names={"discount_curve": "usd_ois", "forecast_curve": "foreign_proxy"},
        market_provenance={
            "source": "unit",
            "source_kind": "user_supplied_snapshot",
            "source_ref": "selected_forecast_curve_bridge",
            "quanto_foreign_curve_policy": {"kind": "selected_forecast_curve"},
        },
    )
    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        underlier_currency="EUR",
        domestic_currency="USD",
    )

    resolved = resolve_quanto_inputs(market_state, spec)

    assert resolved.foreign_df > 0.0
    assert resolved.provenance["foreign_curve"]["source_family"] == "derived"
    assert resolved.provenance["foreign_curve"]["source_kind"] == (
        "selected_forecast_curve_bridge"
    )
    assert resolved.provenance["foreign_curve"]["source_key"] == "foreign_proxy"
    assert resolved.provenance["foreign_curve"]["source_parameters"]["policy_key"] == (
        "market_provenance.quanto_foreign_curve_policy"
    )


def test_resolve_quanto_inputs_supports_explicit_domestic_discount_bridge_policy():
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    market_state = _quanto_market_state(
        forecast_curves={},
        selected_curve_names={"discount_curve": "usd_ois"},
        model_parameters={
            "quanto_correlation": 0.35,
            "quanto_foreign_curve_policy": {"kind": "domestic_discount_curve"},
        },
    )
    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        underlier_currency="EUR",
        domestic_currency="USD",
    )

    resolved = resolve_quanto_inputs(market_state, spec)

    assert resolved.foreign_df == pytest.approx(resolved.domestic_df)
    assert resolved.provenance["foreign_curve"]["source_kind"] == "domestic_discount_bridge"
    assert resolved.provenance["foreign_curve"]["source_parameters"]["policy_key"] == (
        "model_parameters.quanto_foreign_curve_policy"
    )


def test_resolved_quanto_inputs_support_generated_code_aliases():
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        underlier_currency="EUR",
        domestic_currency="USD",
    )

    resolved = resolve_quanto_inputs(_quanto_market_state(), spec)

    assert resolved["underlier_spot"] == pytest.approx(resolved.spot)
    assert resolved["time_to_expiry"] == pytest.approx(resolved.T)
    assert resolved["domestic_discount_factor"] == pytest.approx(resolved.domestic_df)
    assert resolved["foreign_discount_factor"] == pytest.approx(resolved.foreign_df)
    assert resolved["underlier_vol"] == pytest.approx(resolved.sigma_underlier)
    assert resolved["fx_vol"] == pytest.approx(resolved.sigma_fx)
    assert resolved["correlation"] == pytest.approx(resolved.corr)
    assert resolved["quanto_correlation"] == pytest.approx(resolved.corr)
    assert resolved["valuation_date"] == SETTLE


def test_quanto_analytical_route_is_differentiable_in_underlier_spot():
    from trellis.core.differentiable import gradient
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec
    from trellis.models.analytical.quanto import price_quanto_option_analytical
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        underlier_currency="EUR",
        domestic_currency="USD",
    )

    def price_from_spot(spot):
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            forecast_curves={"EUR-DISC": YieldCurve.flat(0.03)},
            fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
            spot=spot,
            underlier_spots={"EUR": spot},
            vol_surface=FlatVol(0.20),
            model_parameters={"quanto_correlation": 0.35},
        )
        resolved = resolve_quanto_inputs(market_state, spec)
        return price_quanto_option_analytical(spec, resolved)

    autodiff_delta = gradient(price_from_spot)(100.0)
    finite_difference = (price_from_spot(100.01) - price_from_spot(99.99)) / 0.02

    assert autodiff_delta == pytest.approx(finite_difference, rel=1e-6)


def test_resolve_himalaya_inputs_matches_expected_market_binding():
    from trellis.instruments._agent.himalayaoptionmontecarlo import HimalayaOptionSpec
    from trellis.models.resolution.himalaya import resolve_himalaya_inputs

    spec = HimalayaOptionSpec(
        notional=250_000,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX",
        observation_dates=(date(2025, 2, 15), date(2025, 5, 15)),
    )

    resolved = resolve_himalaya_inputs(_himalaya_market_state(), spec)

    assert resolved.constituent_names == ("SPX", "NDX")
    assert resolved.constituent_spots == pytest.approx((100.0, 101.5))
    assert resolved.constituent_carry == pytest.approx((0.03, 0.025))
    assert resolved.correlation_matrix[0][1] == pytest.approx(0.35)
    assert resolved.observation_dates == (date(2025, 2, 15), date(2025, 5, 15))
    assert resolved.domestic_df > 0.0
    assert resolved.valuation_date == SETTLE
    assert resolved["constituents"] == ("SPX", "NDX")
    assert resolved["correlation"][1][0] == pytest.approx(0.35)
    assert resolved.spots == pytest.approx((100.0, 101.5))
    assert resolved.vols == pytest.approx((0.2, 0.2))
    assert resolved.carries == pytest.approx((0.03, 0.025))
    assert resolved.div_yields == pytest.approx((0.03, 0.025))
    assert resolved.dividends == pytest.approx((0.03, 0.025))
    assert resolved.rates[0] == pytest.approx(resolved.risk_free_rates[0])
    assert resolved.risk_free_rates[0] == pytest.approx(resolved.risk_free_rates[1])


def test_resolve_basket_semantics_matches_expected_market_binding():
    from trellis.instruments._agent.himalayaoptionmontecarlo import HimalayaOptionSpec
    from trellis.models.resolution.basket_semantics import resolve_basket_semantics

    spec = HimalayaOptionSpec(
        notional=250_000,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX",
        observation_dates=(date(2025, 2, 15), date(2025, 5, 15)),
    )

    resolved = resolve_basket_semantics(_himalaya_market_state(), spec)

    assert resolved.constituent_names == ("SPX", "NDX")
    assert resolved.constituent_spots == pytest.approx((100.0, 101.5))
    assert resolved.constituent_carry == pytest.approx((0.03, 0.025))
    assert resolved.correlation_matrix[0][1] == pytest.approx(0.35)
    assert resolved.observation_dates == (date(2025, 2, 15), date(2025, 5, 15))
    assert resolved.domestic_df > 0.0
    assert resolved.valuation_date == SETTLE
    assert resolved["constituents"] == ("SPX", "NDX")
    assert resolved["correlation"][1][0] == pytest.approx(0.35)
    assert resolved.correlation_preflight is not None
    assert resolved.correlation_preflight.correlation_status == "accepted"


def test_resolve_basket_semantics_regularizes_non_pd_correlation_matrix():
    from trellis.curves.forward_curve import ForwardCurve
    from trellis.instruments._agent.himalayaoptionmontecarlo import HimalayaOptionSpec
    from trellis.models.monte_carlo.ranked_observation_payoffs import build_ranked_observation_basket_process
    from trellis.models.resolution.basket_semantics import resolve_basket_semantics

    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        forecast_curves={
            "SPX-DISC": ForwardCurve(YieldCurve.flat(0.03)),
            "NDX-DISC": ForwardCurve(YieldCurve.flat(0.025)),
            "RTY-DISC": ForwardCurve(YieldCurve.flat(0.02)),
        },
        spot=100.0,
        underlier_spots={"SPX": 100.0, "NDX": 101.5, "RTY": 99.0},
        vol_surface=FlatVol(0.20),
        model_parameters={
            "correlation_matrix": [
                [1.0, -0.9, -0.9],
                [-0.9, 1.0, -0.9],
                [-0.9, -0.9, 1.0],
            ]
        },
    )
    spec = HimalayaOptionSpec(
        notional=250_000,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX,RTY",
        observation_dates=(date(2025, 2, 15), date(2025, 5, 15)),
    )

    resolved = resolve_basket_semantics(market_state, spec)
    corr = np.asarray(resolved.correlation_matrix, dtype=float)

    assert resolved.correlation_preflight is not None
    assert resolved.correlation_preflight.source_kind == "explicit_matrix"
    assert resolved.correlation_preflight.correlation_status == "regularized"
    assert resolved.correlation_preflight.was_regularized is True
    assert resolved.correlation_preflight.min_eigenvalue_before < 0.0
    assert resolved.correlation_preflight.min_eigenvalue_after > 0.0
    assert np.all(np.isfinite(corr))
    assert np.allclose(corr, corr.T, atol=1e-12, rtol=0.0)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-12, rtol=0.0)
    assert np.all(np.linalg.eigvalsh(corr) > 0.0)

    process = build_ranked_observation_basket_process(resolved)
    assert process.cholesky_factor.shape == (3, 3)


def test_resolve_basket_semantics_estimates_empirical_correlation_from_history():
    from trellis.curves.forward_curve import ForwardCurve
    from trellis.instruments._agent.himalayaoptionmontecarlo import HimalayaOptionSpec
    from trellis.models.resolution.basket_semantics import resolve_basket_semantics

    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        forecast_curves={
            "SPX-DISC": ForwardCurve(YieldCurve.flat(0.03)),
            "NDX-DISC": ForwardCurve(YieldCurve.flat(0.025)),
        },
        spot=100.0,
        underlier_spots={"SPX": 100.0, "NDX": 101.5},
        vol_surface=FlatVol(0.20),
        model_parameters={
            "correlation_source": {
                "kind": "empirical",
                "observations": {
                    "SPX": [0.01, 0.02, -0.01, 0.00],
                    "NDX": [0.015, 0.025, -0.005, 0.005],
                },
                "estimator": "sample_pearson",
            }
        },
    )
    spec = HimalayaOptionSpec(
        notional=250_000,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX",
        observation_dates=(date(2025, 2, 15), date(2025, 5, 15)),
    )

    resolved = resolve_basket_semantics(market_state, spec)
    expected = np.corrcoef(
        np.array(
            [
                [0.01, 0.015],
                [0.02, 0.025],
                [-0.01, -0.005],
                [0.00, 0.005],
            ],
            dtype=float,
        ),
        rowvar=False,
    )

    assert resolved.correlation_preflight is not None
    assert resolved.correlation_preflight.source_family == "empirical"
    assert resolved.correlation_preflight.source_kind == "empirical_observations"
    assert resolved.correlation_preflight.source_estimator == "sample_pearson"
    assert resolved.correlation_preflight.sample_size == 4
    assert np.allclose(np.asarray(resolved.correlation_matrix), expected, atol=1e-12)


def test_resolve_basket_semantics_supports_implied_and_synthetic_correlation_sources():
    from trellis.curves.forward_curve import ForwardCurve
    from trellis.instruments._agent.himalayaoptionmontecarlo import HimalayaOptionSpec
    from trellis.models.resolution.basket_semantics import resolve_basket_semantics

    implied_market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        forecast_curves={
            "SPX-DISC": ForwardCurve(YieldCurve.flat(0.03)),
            "NDX-DISC": ForwardCurve(YieldCurve.flat(0.025)),
        },
        spot=100.0,
        underlier_spots={"SPX": 100.0, "NDX": 101.5},
        vol_surface=FlatVol(0.20),
        model_parameters={
            "correlation_source": {
                "kind": "implied",
                "value": 0.25,
                "source_ref": "option_surface_fit",
            }
        },
    )
    synthetic_market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        forecast_curves={
            "SPX-DISC": ForwardCurve(YieldCurve.flat(0.03)),
            "NDX-DISC": ForwardCurve(YieldCurve.flat(0.025)),
        },
        spot=100.0,
        underlier_spots={"SPX": 100.0, "NDX": 101.5},
        vol_surface=FlatVol(0.20),
        market_provenance={
            "source_kind": "synthetic_snapshot",
            "source_ref": "embedded_regime_snapshot",
            "prior_family": "embedded_market_regime",
            "prior_seed": 1337,
            "prior_parameters": {"regime": "easing_cycle"},
        },
        model_parameters={},
    )
    spec = HimalayaOptionSpec(
        notional=250_000,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX",
        observation_dates=(date(2025, 2, 15), date(2025, 5, 15)),
    )

    implied_resolved = resolve_basket_semantics(implied_market_state, spec)
    synthetic_resolved = resolve_basket_semantics(synthetic_market_state, spec)

    assert implied_resolved.correlation_preflight is not None
    assert implied_resolved.correlation_preflight.source_family == "implied"
    assert implied_resolved.correlation_preflight.source_kind == "implied_scalar"
    assert implied_resolved.correlation_preflight.source_key == "option_surface_fit"
    assert implied_resolved.correlation_matrix[0][1] == pytest.approx(0.25)

    assert synthetic_resolved.correlation_preflight is not None
    assert synthetic_resolved.correlation_preflight.source_family == "synthetic"
    assert synthetic_resolved.correlation_preflight.source_kind == "identity_default"
    assert synthetic_resolved.correlation_preflight.source_seed == 1337
    assert synthetic_resolved.correlation_preflight.source_parameters["prior_family"] == "embedded_market_regime"
    assert np.allclose(np.asarray(synthetic_resolved.correlation_matrix), np.eye(2), atol=1e-12)


@pytest.mark.parametrize(
    "bad_matrix, message",
    [
        (
            [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3]],
            "shape",
        ),
        (
            [[1.0, float("nan")], [float("nan"), 1.0]],
            "finite",
        ),
    ],
)
def test_resolve_basket_semantics_rejects_malformed_correlation_matrix(bad_matrix, message):
    from trellis.instruments._agent.himalayaoptionmontecarlo import HimalayaOptionSpec
    from trellis.models.resolution.basket_semantics import resolve_basket_semantics

    market_state = _himalaya_market_state()
    market_state = MarketState(
        as_of=market_state.as_of,
        settlement=market_state.settlement,
        discount=market_state.discount,
        forecast_curves=market_state.forecast_curves,
        spot=market_state.spot,
        underlier_spots=market_state.underlier_spots,
        vol_surface=market_state.vol_surface,
        model_parameters={"correlation_matrix": bad_matrix},
    )
    spec = HimalayaOptionSpec(
        notional=250_000,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX",
        observation_dates=(date(2025, 2, 15), date(2025, 5, 15)),
    )

    from trellis.models.resolution.basket_semantics import CorrelationPreflightError

    with pytest.raises(CorrelationPreflightError, match=message) as excinfo:
        resolve_basket_semantics(market_state, spec)

    assert excinfo.value.report.correlation_status == "rejected"
