"""Tests for typed portfolio-AAD request and result contracts."""

import json

import pytest

from trellis.analytics.derivative_methods import derivative_method_payload
from trellis.analytics.portfolio_aad import (
    PortfolioAADRequest,
    PortfolioAADResult,
    UnsupportedAADPosition,
)
from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    SparseRiskVector,
)


def _curve_factor(tenor: float) -> RiskFactorId:
    return RiskFactorId(
        object_type="curve",
        object_name="usd_ois",
        coordinate_type="zero_rate",
        currency="USD",
        axes={"tenor_years": tenor},
        provenance_namespace="base",
    )


def _coordinate(tenor: float) -> RiskFactorCoordinate:
    factor = _curve_factor(tenor)
    tenor_label = f"{tenor:g}Y"
    return RiskFactorCoordinate(
        factor_id=factor,
        object_path=f"curves.usd_ois.rates[{tenor_label}]",
        reporting_buckets={"currency": "USD", "tenor": tenor_label},
    )


def test_portfolio_aad_request_filters_selected_factors_and_round_trips():
    one_year = _curve_factor(1.0)
    five_year = _curve_factor(5.0)
    vector = SparseRiskVector.from_items([(one_year, 1.0), (five_year, 5.0)])
    request = PortfolioAADRequest(selected_factors=(five_year,))

    payload = request.to_payload()

    assert request.selects_all_factors is False
    assert request.filter_vector(vector) == SparseRiskVector.from_items([(five_year, 5.0)])
    assert json.loads(json.dumps(payload)) == payload
    assert PortfolioAADRequest.from_payload(payload) == request


def test_portfolio_aad_request_all_factors_is_default():
    one_year = _curve_factor(1.0)
    vector = SparseRiskVector.from_items([(one_year, 1.0)])
    request = PortfolioAADRequest()

    assert request.selects_all_factors is True
    assert request.filter_vector(vector) == vector
    assert request.to_payload()["selected_factors"] == "all"


def test_unsupported_position_is_typed_and_json_friendly():
    factor = _curve_factor(1.0)
    position = UnsupportedAADPosition(
        position_name="swap_1",
        instrument_type="InterestRateSwap",
        reason="unsupported_instrument_type",
        requested_factors=(factor,),
        included_in_value=True,
        included_in_risk=False,
        fallback_method=None,
    )

    payload = position.to_payload()

    assert json.loads(json.dumps(payload)) == payload
    assert UnsupportedAADPosition.from_payload(payload) == position
    assert payload["included_in_value"] is True
    assert payload["included_in_risk"] is False


def test_portfolio_aad_result_payload_and_legacy_axis_view_round_trip():
    one_year = _curve_factor(1.0)
    five_year = _curve_factor(5.0)
    vector = SparseRiskVector.from_items([(one_year, 1.25), (five_year, 5.5)])
    result = PortfolioAADResult(
        portfolio_value=101.5,
        risk_vector=vector,
        coordinates=(_coordinate(1.0), _coordinate(5.0)),
        unsupported_positions=(
            UnsupportedAADPosition(
                position_name="unsupported",
                instrument_type="object",
                reason="unsupported_instrument_type",
            ),
        ),
        method_metadata=derivative_method_payload(
            "portfolio_aad_vjp",
            method_support="partial",
            backend_id="autograd",
        ),
        diagnostics=({"code": "partial_book", "severity": "info"},),
    )

    payload = result.to_payload()

    assert json.loads(json.dumps(payload)) == payload
    assert PortfolioAADResult.from_payload(payload) == result
    assert result.support_status == "partial"
    assert result.values_by_axis("tenor_years") == {
        1.0: pytest.approx(1.25),
        5.0: pytest.approx(5.5),
    }
    assert payload["metadata"]["resolved_derivative_method"] == "portfolio_aad_vjp"
    assert payload["unsupported_positions"][0]["position_name"] == "unsupported"


def test_portfolio_aad_result_applies_request_filter():
    one_year = _curve_factor(1.0)
    five_year = _curve_factor(5.0)
    result = PortfolioAADResult(
        risk_vector=SparseRiskVector.from_items([(one_year, 1.25), (five_year, 5.5)]),
        coordinates=(_coordinate(1.0), _coordinate(5.0)),
        method_metadata=derivative_method_payload("portfolio_aad_vjp"),
    )

    filtered = result.apply_request(PortfolioAADRequest(selected_factors=(five_year,)))

    assert filtered.risk_vector == SparseRiskVector.from_items([(five_year, 5.5)])
    assert [coordinate.factor_id for coordinate in filtered.coordinates] == [five_year]
