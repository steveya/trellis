"""Tests for bounded basket-credit tranche-implied correlation calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.models.calibration.basket_credit import (
    BasketCreditTrancheQuote,
    calibrate_homogeneous_basket_tranche_correlation_workflow,
)
from trellis.models.calibration.materialization import materialize_credit_curve
from trellis.models.credit_basket_copula import (
    price_credit_basket_tranche_result,
    resolve_credit_basket_correlation,
)


SETTLE = date(2024, 11, 15)


@dataclass(frozen=True)
class _TrancheSpec:
    notional: float
    n_names: int
    attachment: float
    detachment: float
    end_date: date
    recovery: float = 0.4
    correlation: float | None = None


@dataclass(frozen=True)
class _SurfaceBackedTrancheSpec:
    notional: float
    n_names: int
    attachment: float
    detachment: float
    end_date: date
    recovery: float = 0.4


def _base_market_state() -> MarketState:
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.03, max_tenor=10.0),
        selected_curve_names={"discount_curve": "usd_ois"},
        market_provenance={"source_kind": "explicit_input", "source_ref": "unit_test"},
    )
    return materialize_credit_curve(
        market_state,
        curve_name="acme_single_name_credit",
        credit_curve=CreditCurve.flat(0.025, max_tenor=10.0),
        source_kind="calibrated_curve",
        source_ref="calibrate_single_name_credit_curve_workflow",
        selected_curve_roles={"discount_curve": "usd_ois", "credit_curve": "acme_single_name_credit"},
        metadata={"recovery": 0.4},
    )


def _tranche_quote(
    market_state: MarketState,
    *,
    attachment: float,
    detachment: float,
    correlation: float,
    maturity_years: float = 5.0,
    quote_style: str = "expected_loss_fraction",
) -> BasketCreditTrancheQuote:
    maturity = date(SETTLE.year + int(maturity_years), SETTLE.month, SETTLE.day)
    spec = _TrancheSpec(
        notional=100_000_000.0,
        n_names=100,
        attachment=attachment,
        detachment=detachment,
        end_date=maturity,
        correlation=correlation,
    )
    priced = price_credit_basket_tranche_result(market_state, spec)
    if quote_style == "fair_spread_bp":
        quote_value = priced.fair_spread_bp
        quote_family = "spread"
    elif quote_style == "present_value":
        quote_value = priced.price
        quote_family = "price"
    else:
        quote_value = priced.expected_loss_fraction
        quote_family = "price"
    return BasketCreditTrancheQuote(
        maturity_years=maturity_years,
        attachment=attachment,
        detachment=detachment,
        quote_value=quote_value,
        quote_family=quote_family,
        quote_style=quote_style,
        label=f"{attachment:.2f}_{detachment:.2f}_{correlation:.2f}",
    )


def test_tranche_quote_normalizes_quote_axes_and_rejects_invalid_bounds():
    spread_quote = BasketCreditTrancheQuote(
        maturity_years=5,
        attachment=0.03,
        detachment=0.07,
        quote_value=250.0,
        quote_family="fair_spread",
        label="mezz",
    )
    price_quote = BasketCreditTrancheQuote(
        maturity_years=3,
        attachment=0.0,
        detachment=0.03,
        quote_value=1_250_000.0,
        quote_family="tranche_price",
    )

    assert spread_quote.maturity_years == pytest.approx(5.0)
    assert spread_quote.quote_family == "spread"
    assert spread_quote.quote_style == "fair_spread_bp"
    assert spread_quote.label == "mezz"
    assert price_quote.quote_family == "price"
    assert price_quote.quote_style == "present_value"

    with pytest.raises(ValueError, match="attachment/detachment"):
        BasketCreditTrancheQuote(5.0, 0.07, 0.03, 0.01)
    with pytest.raises(ValueError, match="attachment/detachment"):
        BasketCreditTrancheQuote(5.0, -0.01, 0.03, 0.01)
    with pytest.raises(ValueError, match="quote_value"):
        BasketCreditTrancheQuote(5.0, 0.0, 0.03, -0.01)


def test_calibrates_synthetic_tranche_implied_correlations_from_known_surface():
    market_state = _base_market_state()
    quotes = (
        _tranche_quote(market_state, attachment=0.0, detachment=0.03, correlation=0.18),
        _tranche_quote(market_state, attachment=0.03, detachment=0.07, correlation=0.32),
        _tranche_quote(market_state, attachment=0.07, detachment=0.10, correlation=0.46),
    )

    result = calibrate_homogeneous_basket_tranche_correlation_workflow(
        quotes,
        market_state,
        n_names=100,
        recovery=0.4,
        notional=100_000_000.0,
        surface_name="synthetic_tranche_corr",
    )

    assert result.summary["support_boundary"] == "homogeneous_representative_curve"
    assert result.summary["representative_credit_curve"]["object_name"] == "acme_single_name_credit"
    assert result.diagnostics.root_failures == ()
    assert result.diagnostics.max_abs_quote_residual == pytest.approx(0.0, abs=1e-9)
    assert [point.correlation for point in result.surface.points] == pytest.approx(
        [0.18, 0.32, 0.46],
        abs=2e-6,
    )
    assert result.surface.correlation_for(5.0, 0.03, 0.07) == pytest.approx(0.32, abs=2e-6)
    assert result.provenance["calibration_target"]["quote_maps"][0]["quote_subject"] == (
        "basket_credit_tranche"
    )
    assert result.provenance["calibration_target"]["quote_maps"][0]["quote_axes"][1]["axis_name"] == (
        "attachment"
    )


def test_impossible_quote_records_root_failure_when_warn_policy_is_selected():
    market_state = _base_market_state()
    impossible_quote = BasketCreditTrancheQuote(
        maturity_years=5.0,
        attachment=0.03,
        detachment=0.07,
        quote_value=0.25,
        quote_family="price",
        quote_style="expected_loss_fraction",
        label="impossible_mezz",
    )

    with pytest.raises(ValueError, match="impossible_mezz"):
        calibrate_homogeneous_basket_tranche_correlation_workflow(
            (impossible_quote,),
            market_state,
            n_names=100,
            recovery=0.4,
            notional=100_000_000.0,
        )

    result = calibrate_homogeneous_basket_tranche_correlation_workflow(
        (impossible_quote,),
        market_state,
        n_names=100,
        recovery=0.4,
        notional=100_000_000.0,
        root_failure_policy="warn",
    )

    assert len(result.diagnostics.root_failures) == 1
    assert result.diagnostics.root_failures[0].label == "impossible_mezz"
    assert "outside model quote range" in result.diagnostics.root_failures[0].reason
    assert any("impossible_mezz" in warning for warning in result.diagnostics.warnings)
    assert result.surface.points == ()


def test_diagnostics_warn_on_nonmonotone_and_unsmooth_correlation_surface():
    market_state = _base_market_state()
    quotes = (
        _tranche_quote(market_state, attachment=0.0, detachment=0.03, correlation=0.64),
        _tranche_quote(market_state, attachment=0.03, detachment=0.07, correlation=0.12),
        _tranche_quote(market_state, attachment=0.07, detachment=0.10, correlation=0.72),
    )

    result = calibrate_homogeneous_basket_tranche_correlation_workflow(
        quotes,
        market_state,
        n_names=100,
        recovery=0.4,
        notional=100_000_000.0,
        smoothness_jump_threshold=0.20,
    )

    assert result.diagnostics.root_failures == ()
    assert result.diagnostics.monotonicity_warnings
    assert result.diagnostics.smoothness_warnings
    assert any("decreases" in warning for warning in result.diagnostics.monotonicity_warnings)
    assert any("jump" in warning for warning in result.diagnostics.smoothness_warnings)


def test_materialized_correlation_surface_is_consumed_by_downstream_tranche_helper():
    market_state = _base_market_state()
    quote = _tranche_quote(
        market_state,
        attachment=0.03,
        detachment=0.07,
        correlation=0.41,
        quote_style="fair_spread_bp",
    )
    result = calibrate_homogeneous_basket_tranche_correlation_workflow(
        (quote,),
        market_state,
        n_names=100,
        recovery=0.4,
        notional=100_000_000.0,
        surface_name="downstream_tranche_corr",
    )

    enriched = result.apply_to_market_state(market_state)
    record = enriched.materialized_calibrated_object(object_kind="correlation_surface")
    assert record is not None
    assert record["object_name"] == "downstream_tranche_corr"
    assert record["metadata"]["support_boundary"] == "homogeneous_representative_curve"
    assert enriched.correlation_surface is result.surface
    assert enriched.correlation_surfaces["downstream_tranche_corr"] is result.surface
    assert "correlation_surface" in enriched.available_capabilities

    surface_backed_spec = _SurfaceBackedTrancheSpec(
        notional=100_000_000.0,
        n_names=100,
        attachment=0.03,
        detachment=0.07,
        end_date=date(2029, 11, 15),
    )
    explicit_spec = _TrancheSpec(
        notional=100_000_000.0,
        n_names=100,
        attachment=0.03,
        detachment=0.07,
        end_date=date(2029, 11, 15),
        correlation=0.41,
    )

    assert resolve_credit_basket_correlation(enriched, surface_backed_spec) == pytest.approx(
        0.41,
        abs=2e-6,
    )
    surface_backed_price = price_credit_basket_tranche_result(enriched, surface_backed_spec)
    explicit_price = price_credit_basket_tranche_result(market_state, explicit_spec)
    assert surface_backed_price.expected_loss_fraction == pytest.approx(
        explicit_price.expected_loss_fraction,
        abs=1e-10,
    )
