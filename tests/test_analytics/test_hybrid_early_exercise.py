"""Tests for bounded early-exercise smooth-interior hybrid AD helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import trellis.analytics as analytics
from trellis.agent.contract_ir import (
    ArithmeticMean,
    Constant,
    ContinuousInterval,
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
    differentiate_vanilla_early_exercise,
)
from trellis.analytics.risk_factors import RiskFactorId
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol, GridVolSurface


SETTLEMENT = date(2024, 11, 15)
EXPIRY = date(2025, 11, 15)
EXERCISE_DATES = (
    date(2025, 2, 15),
    date(2025, 5, 15),
    date(2025, 8, 15),
)


@dataclass(frozen=True)
class _VanillaEquitySpec:
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "put"
    notional: float = 1.0
    exercise_style: str = "american"
    exercise_dates: tuple[date, ...] = ()
    aad_tree_steps: int = 64
    early_exercise_boundary_tolerance: float = 1.0e-12


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def _grid_market_state() -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=GridVolSurface(
            expiries=(0.5, 1.5),
            strikes=(90.0, 110.0),
            vols=((0.18, 0.21), (0.24, 0.27)),
        ),
    )


def _american_put_spec() -> _VanillaEquitySpec:
    return _VanillaEquitySpec(
        spot=100.0,
        strike=100.0,
        expiry_date=EXPIRY,
        option_type="put",
        notional=2.0,
        exercise_style="american",
    )


def _bermudan_put_spec() -> _VanillaEquitySpec:
    return _VanillaEquitySpec(
        spot=85.0,
        strike=100.0,
        expiry_date=EXPIRY,
        option_type="put",
        exercise_style="bermudan",
        exercise_dates=EXERCISE_DATES,
    )


def _american_put_ir() -> ContractIR:
    observation_schedule = Singleton(EXPIRY)
    exercise_schedule = ContinuousInterval(EXERCISE_DATES[0], EXPIRY)
    return ContractIR(
        payoff=Max((Sub(Strike(100.0), Spot("SPX")), Constant(0.0))),
        exercise=Exercise("american", exercise_schedule),
        observation=Observation("terminal", observation_schedule),
        underlying=Underlying(EquitySpot("SPX", "gbm")),
    )


def _bermudan_put_ir() -> ContractIR:
    observation_schedule = Singleton(EXPIRY)
    exercise_schedule = FiniteSchedule(EXERCISE_DATES + (EXPIRY,))
    return ContractIR(
        payoff=Max((Sub(Strike(100.0), Spot("SPX")), Constant(0.0))),
        exercise=Exercise("bermudan", exercise_schedule),
        observation=Observation("terminal", observation_schedule),
        underlying=Underlying(EquitySpot("SPX", "gbm")),
    )


def _path_summary_contract_ir() -> ContractIR:
    averaging_schedule = FiniteSchedule(EXERCISE_DATES + (EXPIRY,))
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


def test_public_api_exports_early_exercise_helper():
    assert (
        analytics.differentiate_vanilla_early_exercise
        is differentiate_vanilla_early_exercise
    )
    assert "differentiate_vanilla_early_exercise" in analytics.__all__


def test_american_early_exercise_vjp_returns_hybrid_result():
    admission = admit_hybrid_ad_lane(
        _american_put_ir(),
        product_family="american_vanilla_option",
        derivative_method="vjp",
    )
    request = HybridDerivativeRequest(semantic_admission=admission)

    result = differentiate_vanilla_early_exercise(
        _american_put_spec(),
        _market_state(),
        request,
        position_name="american_put",
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
    assert result.method_metadata["resolved_derivative_method"] == (
        "hybrid_early_exercise_vjp"
    )
    assert result.method_metadata["backend_operator"] == "vjp"
    assert result.method_metadata["early_exercise_policy"] == (
        "hard_exercise_projection_smooth_interior"
    )
    assert result.method_metadata["semantic_state_policy"]["support_status"] == (
        "supported"
    )
    assert result.graph.metadata["exercise_style"] == "american"
    assert result.graph.nodes[0].coordinate_chart is not None


def test_bermudan_early_exercise_vjp_returns_supported_state_metadata():
    admission = admit_hybrid_ad_lane(
        _bermudan_put_ir(),
        product_family="bermudan_vanilla_option",
        derivative_method="vjp",
    )

    result = differentiate_vanilla_early_exercise(
        _bermudan_put_spec(),
        _market_state(),
        HybridDerivativeRequest(semantic_admission=admission),
        position_name="bermudan_put",
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "supported"
    assert len(result.risk_vector) == 1
    assert result.method_metadata["semantic_state_policy"]["metadata"][
        "exercise_style"
    ] == "bermudan"
    assert result.graph.metadata["exercise_style"] == "bermudan"


def test_early_exercise_runtime_fail_closes_wrong_semantic_admission():
    wrong_admission = admit_hybrid_ad_lane(
        _american_put_ir(),
        product_family="quanto_option",
        derivative_method="vjp",
    )

    result = differentiate_vanilla_early_exercise(
        _american_put_spec(),
        _market_state(),
        HybridDerivativeRequest(semantic_admission=wrong_admission),
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "early_exercise_hybrid_state_pending"
    assert result.method_metadata["semantic_state_policy"]["state_kind"] == (
        "early_exercise_control"
    )


def test_early_exercise_runtime_fail_closes_wrong_supported_semantic_lane():
    wrong_admission = admit_hybrid_ad_lane(
        _path_summary_contract_ir(),
        product_family="arithmetic_asian_option",
        derivative_method="vjp",
    )

    result = differentiate_vanilla_early_exercise(
        _american_put_spec(),
        _market_state(),
        HybridDerivativeRequest(semantic_admission=wrong_admission),
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "unsupported"
    assert result.diagnostics[0]["code"] == "semantic_admission_lane_unavailable"
    assert "runtime helper" in result.diagnostics[0]["message"]
    assert "quanto" not in result.diagnostics[0]["message"]
    assert result.method_metadata["semantic_admission"]["support_status"] == "supported"


def test_early_exercise_selected_factor_fail_closed_policy():
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
            _american_put_ir(),
            product_family="american_vanilla_option",
            derivative_method="vjp",
        ),
    )

    result = differentiate_vanilla_early_exercise(
        _american_put_spec(),
        _market_state(),
        request,
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "unsupported"
    assert len(result.risk_vector) == 0
    assert result.diagnostics[0]["code"] == "selected_factors_unavailable"
    assert result.diagnostics[0]["missing_factor_keys"] == [missing_factor.key]


def test_early_exercise_runtime_rejects_grid_vol_without_value():
    admission = admit_hybrid_ad_lane(
        _american_put_ir(),
        product_family="american_vanilla_option",
        derivative_method="vjp",
    )

    result = differentiate_vanilla_early_exercise(
        _american_put_spec(),
        _grid_market_state(),
        HybridDerivativeRequest(semantic_admission=admission),
        vol_surface_name="spx_grid",
    )

    assert result.support_status == "unsupported"
    assert result.value is None
    assert result.diagnostics[0]["code"] == (
        "unsupported_early_exercise_vol_surface_parameterization"
    )


def test_early_exercise_runtime_rejects_boundary_kink_without_value():
    admission = admit_hybrid_ad_lane(
        _american_put_ir(),
        product_family="american_vanilla_option",
        derivative_method="vjp",
    )
    spec = _VanillaEquitySpec(
        spot=80.0,
        strike=100.0,
        expiry_date=EXPIRY,
        option_type="put",
        exercise_style="american",
        early_exercise_boundary_tolerance=1.0e9,
    )

    result = differentiate_vanilla_early_exercise(
        spec,
        _market_state(),
        HybridDerivativeRequest(semantic_admission=admission),
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "unsupported"
    assert result.value is None
    assert result.diagnostics[0]["code"] == "early_exercise_boundary_kink"
    assert result.method_metadata["fallback_reason"]["code"] == (
        "early_exercise_boundary_kink"
    )


def test_early_exercise_runtime_rejects_hvp_with_state_policy_metadata():
    admission = admit_hybrid_ad_lane(
        _american_put_ir(),
        product_family="american_vanilla_option",
        derivative_method="hvp",
    )

    result = differentiate_vanilla_early_exercise(
        _american_put_spec(),
        _market_state(),
        HybridDerivativeRequest(
            derivative_method="hvp",
            semantic_admission=admission,
        ),
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "unsupported"
    assert result.diagnostics[0]["code"] == "early_exercise_hvp_pending"
    assert result.method_metadata["semantic_state_policy"]["support_status"] == (
        "planned"
    )


def test_early_exercise_runtime_rejects_jvp_fail_closed():
    result = differentiate_vanilla_early_exercise(
        _american_put_spec(),
        _market_state(),
        HybridDerivativeRequest(derivative_method="jvp"),
        vol_surface_name="spx_flat",
    )

    assert result.support_status == "unsupported"
    assert result.diagnostics[0]["code"] == "hybrid_jvp_backend_unsupported"
