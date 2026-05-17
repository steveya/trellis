"""Tests for bounded smooth path-summary hybrid AD runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import trellis.analytics as analytics
from trellis.agent.contract_ir import (
    ArithmeticMean,
    Constant,
    ContractIR,
    EquitySpot,
    Exercise,
    FiniteSchedule,
    Max,
    Observation,
    Singleton,
    Spot,
    Strike,
    Sub,
    Underlying,
)
from trellis.analytics import (
    HybridDerivativeRequest,
    admit_hybrid_ad_lane,
    differentiate_arithmetic_asian_path_summary,
)
from trellis.analytics.risk_factors import RiskFactorId
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)
EXPIRY = date(2025, 11, 15)
OBSERVATIONS = (
    date(2025, 2, 15),
    date(2025, 5, 15),
    date(2025, 8, 15),
    EXPIRY,
)


@dataclass(frozen=True)
class _ArithmeticAsianSpec:
    spot: float
    strike: float
    expiry_date: date
    observation_dates: tuple[date, ...]
    option_type: str = "call"
    notional: float = 1.0
    exercise_style: str = "european"
    averaging_type: str = "arithmetic"
    dividend_yield: float = 0.0


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def _asian_spec() -> _ArithmeticAsianSpec:
    return _ArithmeticAsianSpec(
        spot=100.0,
        strike=100.0,
        expiry_date=EXPIRY,
        observation_dates=OBSERVATIONS,
        option_type="call",
        notional=2.0,
    )


def _path_summary_contract_ir() -> ContractIR:
    averaging_schedule = FiniteSchedule(OBSERVATIONS)
    return ContractIR(
        payoff=Max(
            (
                Sub(ArithmeticMean(Spot("SPX"), averaging_schedule), Strike(100.0)),
                Constant(0.0),
            )
        ),
        exercise=Exercise("european", Singleton(EXPIRY)),
        observation=Observation("path_dependent", averaging_schedule),
        underlying=Underlying(EquitySpot("SPX", "gbm")),
    )


def test_public_api_exports_path_summary_helper():
    assert (
        analytics.differentiate_arithmetic_asian_path_summary
        is differentiate_arithmetic_asian_path_summary
    )
    assert "differentiate_arithmetic_asian_path_summary" in analytics.__all__


def test_arithmetic_asian_path_summary_vjp_returns_hybrid_result():
    admission = admit_hybrid_ad_lane(
        _path_summary_contract_ir(),
        product_family="arithmetic_asian_option",
        derivative_method="vjp",
    )
    request = HybridDerivativeRequest(semantic_admission=admission)

    result = differentiate_arithmetic_asian_path_summary(
        _asian_spec(),
        _market_state(),
        request,
        position_name="asian_call",
        vol_surface_name="spx_flat",
        currency="USD",
    )

    factor = tuple(result.risk_vector)[0]

    assert result.support_status == "supported"
    assert result.value is not None
    assert len(result.risk_vector) == 1
    assert factor.object_type == "vol_surface"
    assert factor.object_name == "spx_flat"
    assert factor.coordinate_type == "flat_vol"
    assert factor.provenance_namespace == "hybrid_ad"
    assert result.method_metadata["resolved_derivative_method"] == "hybrid_path_summary_vjp"
    assert result.method_metadata["backend_operator"] == "vjp"
    assert result.method_metadata["path_derivative_policy"] == (
        "lognormal_moment_matching_smooth_path_summary"
    )
    assert result.method_metadata["semantic_state_policy"]["support_status"] == (
        "supported"
    )
    assert result.graph.metadata["path_summary_type"] == "arithmetic_mean"
    assert result.graph.nodes[0].coordinate_chart is not None


def test_path_summary_runtime_fail_closes_unsupported_semantic_admission():
    admission = admit_hybrid_ad_lane(
        _path_summary_contract_ir(),
        product_family="quanto_option",
        derivative_method="vjp",
    )
    request = HybridDerivativeRequest(semantic_admission=admission)

    result = differentiate_arithmetic_asian_path_summary(
        _asian_spec(),
        _market_state(),
        request,
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "path_dependent_hybrid_state_pending"
    assert result.method_metadata["semantic_state_policy"]["state_kind"] == (
        "smooth_path_summary"
    )


def test_path_summary_selected_factor_fail_closed_policy():
    missing_factor = RiskFactorId(
        object_type="vol_surface",
        object_name="other_surface",
        coordinate_type="flat_vol",
        provenance_namespace="hybrid_ad",
    )
    request = HybridDerivativeRequest(
        selected_factors=(missing_factor,),
        unsupported_selected_factor_policy="fail_closed",
        semantic_admission=admit_hybrid_ad_lane(
            _path_summary_contract_ir(),
            product_family="arithmetic_asian_option",
            derivative_method="vjp",
        ),
    )

    result = differentiate_arithmetic_asian_path_summary(
        _asian_spec(),
        _market_state(),
        request,
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "selected_factors_unavailable"
    assert result.diagnostics[0]["missing_factor_keys"] == [missing_factor.key]
