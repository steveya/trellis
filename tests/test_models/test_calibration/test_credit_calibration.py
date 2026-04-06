"""Tests for typed single-name credit calibration."""

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.models.calibration.credit import (
    CreditHazardCalibrationQuote,
    CreditHazardCalibrationResult,
    calibrate_single_name_credit_curve_workflow,
)
from trellis.models.credit_default_swap import (
    build_cds_schedule,
    normalize_cds_running_spread,
    price_cds_analytical,
)


SETTLE = date(2024, 11, 15)
MATURITY = date(2029, 11, 15)


def _credit_market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.03),
        selected_curve_names={"discount_curve": "usd_ois"},
        market_provenance={"source_kind": "explicit_input", "source_ref": "unit_test"},
    )


def _schedule():
    return build_cds_schedule(
        SETTLE,
        MATURITY,
        Frequency.QUARTERLY,
        DayCountConvention.ACT_360,
    )


def test_calibrates_single_name_credit_curve_from_spreads_and_materializes_runtime_binding():
    market_state = _credit_market_state()
    quotes = (
        CreditHazardCalibrationQuote(1.0, 120.0, "spread", label="spread_1y"),
        CreditHazardCalibrationQuote(5.0, 180.0, "spread", label="spread_5y"),
    )

    result = calibrate_single_name_credit_curve_workflow(
        quotes,
        market_state,
        recovery=0.4,
        curve_name="acme_credit",
    )

    assert isinstance(result, CreditHazardCalibrationResult)
    assert result.target_hazards[0] == pytest.approx(normalize_cds_running_spread(120.0) / 0.6)
    assert result.target_hazards[1] == pytest.approx(normalize_cds_running_spread(180.0) / 0.6)
    assert result.provenance["potential_binding"]["discount_curve_name"] == "usd_ois"
    assert result.provenance["calibration_target"]["quote_maps"][0]["quote_family"] == "spread"
    assert result.max_abs_hazard_residual == pytest.approx(0.0)
    assert result.max_abs_quote_residual == pytest.approx(0.0)

    enriched = result.apply_to_market_state(market_state)
    assert enriched.credit_curve is not None
    record = enriched.materialized_calibrated_object(object_kind="credit_curve")
    assert record is not None
    assert record["object_name"] == "acme_credit"
    assert record["selected_curve_roles"]["discount_curve"] == "usd_ois"
    assert record["selected_curve_roles"]["credit_curve"] == "acme_credit"
    assert record["metadata"]["potential_binding"]["default_curve_name"] == "acme_credit"
    assert enriched.credit_curve.survival_probability(5.0) == pytest.approx(
        result.credit_curve.survival_probability(5.0)
    )


def test_credit_calibration_handoff_prices_cds_with_mixed_hazard_and_spread_quotes():
    market_state = _credit_market_state()
    quotes = (
        CreditHazardCalibrationQuote(1.0, 0.02, "hazard", label="hazard_1y"),
        CreditHazardCalibrationQuote(5.0, 120.0, "spread", label="spread_5y"),
    )

    result = calibrate_single_name_credit_curve_workflow(
        quotes,
        market_state,
        recovery=0.4,
        curve_name="mixed_credit",
    )
    calibrated_state = result.apply_to_market_state(market_state)
    schedule = _schedule()
    observed = price_cds_analytical(
        notional=1_000_000.0,
        spread_quote=120.0,
        recovery=0.4,
        schedule=schedule,
        credit_curve=calibrated_state.credit_curve,
        discount_curve=calibrated_state.discount,
    )
    expected_curve = CreditCurve(
        (1.0, 5.0),
        (
            0.02,
            normalize_cds_running_spread(120.0) / 0.6,
        ),
    )
    expected = price_cds_analytical(
        notional=1_000_000.0,
        spread_quote=120.0,
        recovery=0.4,
        schedule=schedule,
        credit_curve=expected_curve,
        discount_curve=market_state.discount,
    )

    assert observed == pytest.approx(expected)
    assert result.summary["quote_families"] == ["hazard", "spread"]
    assert result.provenance["calibration_target"]["quote_maps"][0]["quote_family"] == "hazard"
    assert result.provenance["calibration_target"]["quote_maps"][1]["quote_family"] == "spread"


def test_credit_calibration_rejects_missing_discount_curve_binding():
    quotes = (
        CreditHazardCalibrationQuote(1.0, 120.0, "spread", label="spread_1y"),
        CreditHazardCalibrationQuote(5.0, 180.0, "spread", label="spread_5y"),
    )
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=None,
    )

    with pytest.raises(ValueError, match="requires market_state.discount"):
        calibrate_single_name_credit_curve_workflow(
            quotes,
            market_state,
            recovery=0.4,
            curve_name="missing_discount_credit",
        )
