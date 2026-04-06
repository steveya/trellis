"""Tests for typed calibrated-object materialization onto MarketState."""

from datetime import date

from trellis.core.market_state import MarketState
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.models.calibration.materialization import (
    CalibratedObjectMaterialization,
    materialize_black_vol_surface,
    materialize_credit_curve,
    materialize_local_vol_surface,
    materialize_model_parameter_set,
    resolve_materialized_object,
)
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _base_market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.03),
        selected_curve_names={"discount_curve": "usd_ois", "forecast_curve": "USD-SOFR-3M"},
        market_provenance={"source_kind": "explicit_input", "source_ref": "unit_test"},
    )


def test_model_parameter_materialization_preserves_compatibility_fields():
    market_state = _base_market_state()
    calibrated = materialize_model_parameter_set(
        market_state,
        parameter_set_name="heston_equity",
        model_parameters={"model_family": "heston", "rho": -0.6},
        source_kind="calibrated_surface",
        source_ref="fit_heston_smile_surface",
        selected_curve_roles={"discount_curve": "usd_ois", "forecast_curve": "USD-SOFR-3M"},
        metadata={"instrument_family": "equity_vol"},
    )

    assert calibrated.model_parameters["model_family"] == "heston"
    assert calibrated.model_parameter_sets["heston_equity"]["rho"] == -0.6

    record = resolve_materialized_object(calibrated, object_kind="model_parameter_set")
    assert record is not None
    assert record["object_name"] == "heston_equity"
    assert record["source_kind"] == "calibrated_surface"
    assert record["target_fields"] == ["model_parameters", "model_parameter_sets"]
    assert record["selected_curve_roles"]["discount_curve"] == "usd_ois"
    assert record["selected_curve_roles"]["forecast_curve"] == "USD-SOFR-3M"
    assert record["metadata"]["instrument_family"] == "equity_vol"
    assert calibrated.materialized_calibrated_object(object_kind="model_parameter_set") == record


def test_local_vol_and_credit_materialization_records_kind_specific_entries():
    market_state = _base_market_state()
    local_vol = lambda _spot, _time: 0.20  # noqa: E731
    with_local = materialize_local_vol_surface(
        market_state,
        surface_name="equity_local_vol",
        local_vol_surface=local_vol,
        source_kind="calibrated_surface",
        source_ref="calibrate_local_vol_surface_workflow",
        metadata={"surface_shape": (10, 20)},
    )
    with_credit = materialize_credit_curve(
        with_local,
        curve_name="single_name_5y",
        credit_curve=CreditCurve.flat(0.02),
        source_kind="bootstrap",
        source_ref="fit_credit_curve",
        metadata={"recovery": 0.4},
    )

    local_record = resolve_materialized_object(with_credit, object_kind="local_vol_surface")
    assert local_record is not None
    assert local_record["object_name"] == "equity_local_vol"
    assert with_credit.local_vol_surfaces["equity_local_vol"] is local_vol

    credit_record = resolve_materialized_object(with_credit, object_kind="credit_curve")
    assert credit_record is not None
    assert credit_record["object_name"] == "single_name_5y"
    assert credit_record["source_kind"] == "bootstrap"
    assert credit_record["metadata"]["recovery"] == 0.4
    assert with_credit.credit_curve is not None


def test_black_vol_materialization_records_surface_binding():
    market_state = _base_market_state()
    calibrated = materialize_black_vol_surface(
        market_state,
        surface_name="rates_cap_surface",
        vol_surface=FlatVol(0.22),
        source_kind="calibrated_surface",
        source_ref="calibrate_cap_floor_black_vol",
    )

    record = resolve_materialized_object(calibrated, object_kind="black_vol_surface")
    assert record is not None
    assert record["object_name"] == "rates_cap_surface"
    assert record["target_fields"] == ["vol_surface"]
    assert calibrated.vol_surface.black_vol(1.0, 0.05) == 0.22


def test_materialization_record_rejects_empty_object_name():
    try:
        CalibratedObjectMaterialization(
            object_kind="model_parameter_set",
            object_name="",
            target_fields=("model_parameters",),
            source_kind="calibrated",
        )
    except ValueError as exc:
        assert "object_name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty object_name")
