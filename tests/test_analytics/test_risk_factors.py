"""Tests for canonical risk-factor identity and sparse risk vectors."""

import json

import pytest

from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    SparseRiskVector,
)


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
