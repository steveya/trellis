"""Tests for typed portfolio-AAD request and result contracts."""

import json
from dataclasses import dataclass
from datetime import date

import pytest

from trellis.analytics.derivative_methods import derivative_method_payload
from trellis.analytics.portfolio_aad import (
    AADSupportDecision,
    BondCurveAADAdapter,
    BondCurveAADMarketContext,
    DefaultUnsupportedAADPolicy,
    PortfolioAADRequest,
    PortfolioAADResult,
    TradeAADAdapter,
    UnsupportedAADPosition,
    VanillaEquityOptionVolAADAdapter,
    VanillaEquityOptionVolAADMarketContext,
)
from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    SparseRiskVector,
)
from trellis.book import Book, portfolio_aad_equity_option_vol_risk
from trellis.curves.yield_curve import YieldCurve
from trellis.core.market_state import MarketState
from trellis.instruments.bond import Bond
from trellis.models.vol_surface import FlatVol


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


@dataclass(frozen=True)
class _VanillaEquitySpec:
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "call"
    notional: float = 1.0
    exercise_style: str = "european"


def _equity_market_state(vol: float = 0.2) -> MarketState:
    settlement = date(2024, 11, 15)
    return MarketState(
        as_of=settlement,
        settlement=settlement,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
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


class _FakeAdapter:
    def support_decision(self, position_name, instrument, market_context, request):
        return AADSupportDecision(
            supported=True,
            reason="supported_fake",
            factor_dependencies=request.selected_factors,
            diagnostics=({"code": "ok"},),
        )

    def factor_dependencies(self, instrument, market_context, request):
        return request.selected_factors

    def value(self, instrument, market_context, request):
        return 1.0

    def vjp(self, instrument, market_context, request, weight=1.0):
        return SparseRiskVector.from_items(
            (factor, weight)
            for factor in request.selected_factors
        )


def test_trade_aad_adapter_protocol_and_support_decision_payload():
    factor = _curve_factor(1.0)
    request = PortfolioAADRequest(selected_factors=(factor,))
    adapter = _FakeAdapter()

    decision = adapter.support_decision("bond", object(), object(), request)
    payload = decision.to_payload()

    assert isinstance(adapter, TradeAADAdapter)
    assert json.loads(json.dumps(payload)) == payload
    assert AADSupportDecision.from_payload(payload) == decision
    assert adapter.vjp(object(), object(), request, weight=2.0)[factor] == pytest.approx(2.0)


def test_default_unsupported_policy_never_includes_position_in_aad_risk():
    factor = _curve_factor(5.0)
    request = PortfolioAADRequest(selected_factors=(factor,))
    policy = DefaultUnsupportedAADPolicy(include_value_when_priced=True)

    position = policy.record(
        position_name="unsupported",
        instrument=object(),
        reason="unsupported_instrument_type",
        request=request,
        priced_value_available=True,
    )

    assert position.position_name == "unsupported"
    assert position.instrument_type == "object"
    assert position.reason == "unsupported_instrument_type"
    assert position.requested_factors == (factor,)
    assert position.included_in_value is True
    assert position.included_in_risk is False
    assert position.fallback_method is None


def test_bond_curve_aad_adapter_reports_support_and_dependencies():
    curve = YieldCurve([1.0, 2.0, 5.0], [0.04, 0.042, 0.045])
    context = BondCurveAADMarketContext(
        curve=curve,
        settlement=date(2024, 11, 15),
        curve_name="usd_ois",
        currency="USD",
    )
    adapter = BondCurveAADAdapter()
    bond = Bond(
        face=100.0,
        coupon=0.05,
        maturity_date=date(2030, 11, 15),
        maturity=6,
        frequency=2,
    )

    decision = adapter.support_decision("bond", bond, context, PortfolioAADRequest())
    dependencies = adapter.factor_dependencies(bond, context, PortfolioAADRequest())

    assert decision.supported is True
    assert decision.reason == "supported_bond_curve_aad"
    assert dependencies == decision.factor_dependencies
    assert [factor.key for factor in dependencies] == [
        "type=curve|object=usd_ois|coordinate=zero_rate|currency=USD|"
        "tenor_years=1|namespace=portfolio_aad",
        "type=curve|object=usd_ois|coordinate=zero_rate|currency=USD|"
        "tenor_years=2|namespace=portfolio_aad",
        "type=curve|object=usd_ois|coordinate=zero_rate|currency=USD|"
        "tenor_years=5|namespace=portfolio_aad",
    ]


def test_bond_curve_aad_adapter_vjp_returns_sparse_curve_risk():
    curve = YieldCurve([1.0, 2.0, 5.0], [0.04, 0.042, 0.045])
    context = BondCurveAADMarketContext(curve=curve, settlement=date(2024, 11, 15))
    adapter = BondCurveAADAdapter()
    bond = Bond(
        face=100.0,
        coupon=0.05,
        maturity_date=date(2030, 11, 15),
        maturity=6,
        frequency=2,
    )

    vector = adapter.vjp(bond, context, PortfolioAADRequest(), weight=1_000_000)

    assert len(vector) == len(curve.tenors)
    assert all(factor.object_type == "curve" for factor in vector)
    assert any(abs(value) > 0.0 for _, value in vector.items())


def test_bond_curve_aad_adapter_rejects_unsupported_inputs():
    adapter = BondCurveAADAdapter()
    context = BondCurveAADMarketContext(
        curve=object(),
        settlement=date(2024, 11, 15),
    )

    decision = adapter.support_decision("not_bond", object(), context, PortfolioAADRequest())

    assert decision.supported is False
    assert decision.reason == "unsupported_instrument_type"


def test_vanilla_equity_option_vol_adapter_reports_flat_vol_dependency():
    adapter = VanillaEquityOptionVolAADAdapter()
    context = VanillaEquityOptionVolAADMarketContext(
        market_state=_equity_market_state(),
        vol_surface_name="spx_flat",
        currency="USD",
    )
    spec = _VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
    )

    decision = adapter.support_decision("call", spec, context, PortfolioAADRequest())
    dependencies = adapter.factor_dependencies(spec, context, PortfolioAADRequest())

    assert decision.supported is True
    assert decision.reason == "supported_vanilla_equity_flat_vol_aad"
    assert dependencies == decision.factor_dependencies
    assert [factor.key for factor in dependencies] == [
        "type=vol_surface|object=spx_flat|coordinate=flat_vol|currency=USD|"
        "namespace=portfolio_aad"
    ]
    assert context.coordinates()[0].support_status == "supported"


def test_vanilla_equity_option_vol_adapter_vjp_returns_sparse_vol_risk():
    adapter = VanillaEquityOptionVolAADAdapter()
    context = VanillaEquityOptionVolAADMarketContext(market_state=_equity_market_state())
    spec = _VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        notional=2.0,
    )

    vector = adapter.vjp(spec, context, PortfolioAADRequest(), weight=5.0)

    assert len(vector) == 1
    factor, sensitivity = vector.items()[0]
    assert factor.object_type == "vol_surface"
    assert factor.coordinate_type == "flat_vol"
    assert sensitivity > 0.0


def test_vanilla_equity_option_vol_adapter_rejects_non_european_options():
    adapter = VanillaEquityOptionVolAADAdapter()
    context = VanillaEquityOptionVolAADMarketContext(market_state=_equity_market_state())
    spec = _VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        exercise_style="american",
    )

    decision = adapter.support_decision("american", spec, context, PortfolioAADRequest())

    assert decision.supported is False
    assert decision.reason == "unsupported_exercise_style"


def test_portfolio_aad_equity_option_vol_risk_aggregates_shared_flat_vol_factor():
    context = VanillaEquityOptionVolAADMarketContext(
        market_state=_equity_market_state(),
        vol_surface_name="spx_flat",
        currency="USD",
    )
    call = _VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
    )
    put = _VanillaEquitySpec(
        spot=100.0,
        strike=95.0,
        expiry_date=date(2025, 11, 15),
        option_type="put",
    )
    book = Book({"call": call, "put": put}, notionals={"call": 10.0, "put": 4.0})

    result = portfolio_aad_equity_option_vol_risk(book, context)

    assert result.support_status == "supported"
    assert result.portfolio_value > 0.0
    assert len(result.risk_vector) == 1
    factor, sensitivity = result.risk_vector.items()[0]
    assert factor.object_type == "vol_surface"
    assert factor.object_name == "spx_flat"
    assert factor.coordinate_type == "flat_vol"
    assert sensitivity > 0.0
    assert result.method_metadata["parameterization"] == "shared_flat_vol"
    assert result.method_metadata["supported_position_names"] == ["call", "put"]
    assert result.method_metadata["risk_bucket_totals"]["bucket_names"] == [
        "risk_class",
        "currency",
        "object_name",
    ]
    assert (
        result.method_metadata["risk_bucket_totals"]["totals"][0]["buckets"]["risk_class"]
        == "volatility"
    )


def test_portfolio_aad_equity_option_vol_risk_applies_selected_factor_filter():
    context = VanillaEquityOptionVolAADMarketContext(
        market_state=_equity_market_state(),
        vol_surface_name="spx_flat",
        currency="USD",
    )
    factor = context.coordinates()[0].factor_id
    missing_factor = RiskFactorId(
        object_type="vol_surface",
        object_name="other_flat",
        coordinate_type="flat_vol",
        currency="USD",
    )
    spec = _VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
    )
    book = Book({"call": spec}, notionals={"call": 10.0})

    selected = portfolio_aad_equity_option_vol_risk(
        book,
        context,
        request=PortfolioAADRequest(selected_factors=(factor,)),
    )
    missing = portfolio_aad_equity_option_vol_risk(
        book,
        context,
        request=PortfolioAADRequest(selected_factors=(missing_factor,)),
    )

    assert tuple(selected.risk_vector) == (factor,)
    assert tuple(coordinate.factor_id for coordinate in selected.coordinates) == (factor,)
    assert len(missing.risk_vector) == 0
    assert missing.missing_selected_factors(
        PortfolioAADRequest(selected_factors=(missing_factor,))
    ) == (missing_factor,)


def test_portfolio_aad_equity_option_vol_risk_reports_unsupported_shapes():
    context = VanillaEquityOptionVolAADMarketContext(market_state=_equity_market_state())
    european = _VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
    )
    american = _VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        exercise_style="american",
    )
    book = Book({"european": european, "american": american})

    result = portfolio_aad_equity_option_vol_risk(book, context)

    assert result.support_status == "partial"
    assert result.method_metadata["supported_position_names"] == ["european"]
    assert result.method_metadata["unsupported_position_count"] == 1
    assert result.unsupported_positions[0].position_name == "american"
    assert result.unsupported_positions[0].reason == "unsupported_exercise_style"
