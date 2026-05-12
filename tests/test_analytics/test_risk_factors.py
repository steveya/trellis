"""Tests for canonical risk-factor identity and sparse risk vectors."""

import json

import pytest

from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    RiskFactorRegistry,
    SparseRiskVector,
    UnsupportedRiskFactorObject,
)
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol, GridVolSurface


def _curve_factor(tenor: float, *, object_name: str = "usd_ois") -> RiskFactorId:
    return RiskFactorId(
        object_type="curve",
        coordinate_type="zero_rate",
        object_name=object_name,
        currency="USD",
        axes={"tenor_years": tenor},
        provenance_namespace="base",
    )


def test_risk_factor_id_uses_explicit_stable_fields_not_object_identity():
    first = _curve_factor(5.0)
    second = RiskFactorId(
        object_type="curve",
        coordinate_type="zero_rate",
        object_name="usd_ois",
        currency="USD",
        axes={"tenor_years": 5.0},
        provenance_namespace="base",
    )
    different_curve = _curve_factor(5.0, object_name="usd_sofr_3m")

    assert first == second
    assert hash(first) == hash(second)
    assert first.key == (
        "type=curve|object=usd_ois|coordinate=zero_rate|currency=USD|"
        "tenor_years=5|namespace=base"
    )
    assert different_curve.key != first.key


def test_risk_factor_id_payload_round_trips_and_is_json_friendly():
    factor = _curve_factor(10.0)
    payload = factor.to_payload()

    assert json.loads(json.dumps(payload)) == payload
    assert RiskFactorId.from_payload(payload) == factor
    assert payload["key"] == factor.key


def test_risk_factor_coordinate_carries_buckets_and_round_trips():
    factor = _curve_factor(2.0)
    coordinate = RiskFactorCoordinate(
        factor_id=factor,
        object_path="market_state.curves.usd_ois",
        display_name="USD OIS 2Y zero rate",
        unit="rate",
        transform="identity",
        reporting_buckets={"currency": "USD", "tenor": "2Y", "risk_class": "rates"},
    )

    payload = coordinate.to_payload()

    assert json.loads(json.dumps(payload)) == payload
    assert RiskFactorCoordinate.from_payload(payload) == coordinate
    assert coordinate.bucket("currency") == "USD"
    assert coordinate.bucket("missing", default="unbucketed") == "unbucketed"


def test_sparse_risk_vector_aggregates_sorts_scales_and_filters():
    one_year = _curve_factor(1.0)
    five_year = _curve_factor(5.0)

    vector = SparseRiskVector.from_items(
        [
            (five_year, 3.0),
            (one_year, 1.5),
            (five_year, 2.0),
            (one_year, 0.0),
        ]
    )

    assert list(vector) == [one_year, five_year]
    assert vector[one_year] == pytest.approx(1.5)
    assert vector[five_year] == pytest.approx(5.0)
    assert vector.scale(2.0)[five_year] == pytest.approx(10.0)
    assert (vector + SparseRiskVector.from_items([(one_year, -0.5)]))[one_year] == pytest.approx(1.0)
    assert dict(vector.filter({five_year}).items()) == {five_year: 5.0}


def test_sparse_risk_vector_payload_round_trips_and_buckets():
    one_year = _curve_factor(1.0)
    five_year = _curve_factor(5.0)
    vector = SparseRiskVector.from_items([(five_year, 5.0), (one_year, 1.5)])
    coordinates = [
        RiskFactorCoordinate(
            factor_id=one_year,
            reporting_buckets={"currency": "USD", "tenor": "1Y"},
        ),
        RiskFactorCoordinate(
            factor_id=five_year,
            reporting_buckets={"currency": "USD", "tenor": "5Y"},
        ),
    ]

    payload = vector.to_payload()

    assert json.loads(json.dumps(payload)) == payload
    assert SparseRiskVector.from_payload(payload) == vector
    assert vector.bucket_totals(coordinates, "currency") == {"USD": pytest.approx(6.5)}
    assert vector.bucket_totals(coordinates, "tenor") == {
        "1Y": pytest.approx(1.5),
        "5Y": pytest.approx(5.0),
    }


def test_registry_discovers_stable_yield_curve_coordinates_without_object_identity():
    registry = RiskFactorRegistry()
    first_curve = YieldCurve([1.0, 5.0], [0.04, 0.045])
    second_curve = YieldCurve([1.0, 5.0], [0.04, 0.045])

    first = registry.discover_yield_curve(
        first_curve,
        object_name="usd_ois",
        currency="USD",
        provenance_namespace="base",
    )
    second = registry.discover_yield_curve(
        second_curve,
        object_name="usd_ois",
        currency="USD",
        provenance_namespace="base",
    )
    different_namespace = registry.discover_yield_curve(
        first_curve,
        object_name="usd_ois",
        currency="USD",
        provenance_namespace="stress_up",
    )

    assert [coordinate.factor_id for coordinate in first] == [
        coordinate.factor_id for coordinate in second
    ]
    assert [coordinate.factor_id.key for coordinate in first] == [
        "type=curve|object=usd_ois|coordinate=zero_rate|currency=USD|"
        "tenor_years=1|namespace=base",
        "type=curve|object=usd_ois|coordinate=zero_rate|currency=USD|"
        "tenor_years=5|namespace=base",
    ]
    assert first[0].support_status == "supported"
    assert first[0].bucket("risk_class") == "rates"
    assert different_namespace[0].factor_id != first[0].factor_id


def test_registry_registers_coordinates_and_payloads_deterministically():
    registry = RiskFactorRegistry()
    curve = YieldCurve([1.0, 5.0], [0.04, 0.045])

    populated = registry.with_yield_curve(
        curve,
        object_name="usd_ois",
        currency="USD",
        provenance_namespace="base",
    )
    duplicate = populated.with_coordinates(populated.coordinates)
    payload = duplicate.to_payload()

    assert duplicate == populated
    assert json.loads(json.dumps(payload)) == payload
    assert RiskFactorRegistry.from_payload(payload) == populated
    assert [coordinate.factor_id.key for coordinate in populated.coordinates] == sorted(
        coordinate.factor_id.key for coordinate in populated.coordinates
    )


def test_registry_discovery_only_lanes_cover_credit_vol_and_model_parameters():
    registry = RiskFactorRegistry()

    credit = registry.discover_credit_curve(
        CreditCurve([1.0, 3.0], [0.01, 0.015]),
        object_name="acme_cds",
        issuer="ACME",
        currency="USD",
    )
    flat_vol = registry.discover_flat_vol_surface(
        FlatVol(0.2),
        object_name="spx_flat",
        currency="USD",
    )
    grid_vol = registry.discover_grid_vol_surface(
        GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=((0.2, 0.22), (0.24, 0.26)),
        ),
        object_name="spx_grid",
        currency="USD",
    )
    model_params = registry.discover_scalar_model_parameters(
        {"kappa": 1.2, "theta": 0.04},
        parameter_set_name="heston_equity",
        model_family="heston",
        provenance_namespace="calibrated",
    )

    assert [coordinate.support_status for coordinate in credit] == [
        "discovery_only",
        "discovery_only",
    ]
    assert flat_vol[0].factor_id.key == (
        "type=vol_surface|object=spx_flat|coordinate=flat_vol|currency=USD"
    )
    assert len(grid_vol) == 4
    assert grid_vol[0].bucket("risk_class") == "volatility"
    assert [coordinate.factor_id.axes for coordinate in model_params] == [
        (("parameter", "kappa"),),
        (("parameter", "theta"),),
    ]


def test_registry_fails_closed_for_unknown_market_objects():
    registry = RiskFactorRegistry()

    with pytest.raises(UnsupportedRiskFactorObject) as excinfo:
        registry.discover_market_object(object(), object_name="unknown")

    payload = excinfo.value.to_payload()
    assert payload["reason"] == "unsupported_market_object"
    assert payload["object_type"] == "object"
