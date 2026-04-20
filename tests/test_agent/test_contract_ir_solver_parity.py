from __future__ import annotations

from datetime import date

from trellis.agent.contract_ir_solver_parity import build_contract_ir_solver_parity_report
from trellis.agent.platform_requests import compile_build_request
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


def _equity_market_state(*, underlier: str = "AAPL", spot: float = 165.0, rate: float = 0.03, vol: float = 0.2) -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(rate),
        vol_surface=FlatVol(vol),
        spot=spot,
        underlier_spots={underlier: spot},
    )


def _swaption_market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.20),
    )


def _basket_market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.20),
        underlier_spots={"SPX": 4500.0, "NDX": 15000.0},
        model_parameters={
            "correlation_matrix": ((1.0, 0.35), (0.35, 1.0)),
            "underlier_carry_rates": {"SPX": 0.0, "NDX": 0.0},
        },
    )


def _variance_market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.02),
        vol_surface=FlatVol(0.25),
        spot=5000.0,
        underlier_spots={"SPX": 5000.0},
    )


def test_compile_build_request_records_request_decomposition_shadow_for_digital_option():
    compiled = compile_build_request(
        "Cash-or-nothing digital call on AAPL paying $2 if spot > 150 at expiry 2025-11-15",
        instrument_type="digital_option",
        market_snapshot=_equity_market_state(),
        preferred_method="analytical",
    )

    compiler = compiled.request.metadata["contract_ir_compiler"]

    assert compiled.semantic_blueprint is None
    assert compiler["source"] == "request_decomposition"
    assert compiler["shadow_status"] == "bound"
    assert compiler["contract_ir_solver_shadow"]["declaration_id"] == "black76_cash_digital_call"
    assert compiler["shadow_error"] is None


def test_compile_build_request_keeps_terminal_basket_request_off_vanilla_semantic_path_and_binds_shadow():
    compiled = compile_build_request(
        "European basket call on {SPX 50%, NDX 50%} strike 4500 expiring 2025-11-15",
        instrument_type="basket_option",
        market_snapshot=_basket_market_state(),
        preferred_method="analytical",
    )

    compiler = compiled.request.metadata["contract_ir_compiler"]

    assert compiled.semantic_contract is None
    assert compiled.semantic_blueprint is None
    assert compiled.product_ir is not None
    assert compiled.product_ir.instrument == "basket_option"
    assert compiled.product_ir.payoff_family == "basket_option"
    assert "two_asset_terminal_basket" in compiled.product_ir.payoff_traits
    assert compiled.generation_plan is not None
    assert compiled.generation_plan.backend_binding_id == (
        "trellis.models.basket_option.price_basket_option_analytical"
    )
    assert compiler["source"] == "request_decomposition"
    assert compiler["shadow_status"] == "bound"
    assert compiler["contract_ir_solver_shadow"]["declaration_id"] == "helper_basket_option_call"
    authority = compiled.request.metadata["route_binding_authority"]
    assert authority["route_id"] == "analytical_black76"
    assert authority["authority_kind"] == "exact_backend_fit"
    assert authority["backend_binding"]["exact_target_refs"] == [
        "trellis.models.basket_option.price_basket_option_analytical"
    ]


def test_compile_build_request_records_request_decomposition_shadow_for_variance_swap():
    compiled = compile_build_request(
        "Equity variance swap on SPX, variance strike 0.04, notional 10000, expiry 2025-11-15",
        instrument_type="variance_swap",
        market_snapshot=_variance_market_state(),
        preferred_method="analytical",
    )

    compiler = compiled.request.metadata["contract_ir_compiler"]

    assert compiled.semantic_blueprint is None
    assert compiler["source"] == "request_decomposition"
    assert compiler["shadow_status"] == "bound"
    assert compiler["contract_ir_solver_shadow"]["declaration_id"] == "helper_equity_variance_swap"
    assert compiler["shadow_error"] is None


def test_compile_build_request_records_explicit_asian_no_match_blocker():
    compiled = compile_build_request(
        "Arithmetic Asian call on SPX monthly average over 2025 strike 4500",
        instrument_type="asian_option",
        market_snapshot=_variance_market_state(),
        preferred_method="analytical",
    )

    compiler = compiled.request.metadata["contract_ir_compiler"]

    assert compiled.semantic_blueprint is None
    assert compiler["source"] == "request_decomposition"
    assert compiler["shadow_status"] == "no_match"
    assert compiler["contract_ir_solver_shadow"] is None
    assert compiler["shadow_error"]["error_type"] == "ContractIRSolverNoMatchError"


def test_contract_ir_solver_parity_report_flags_asian_blocker_and_candidate_families():
    report = build_contract_ir_solver_parity_report()

    families = {entry["family_id"]: entry for entry in report["families"]}

    assert families["basket_option"]["exact_authority_closed"] is True
    assert families["basket_option"]["phase4_candidate"] is True
    assert families["vanilla_option"]["parity_closed"] is True
    assert families["digital_option"]["parity_closed"] is True
    assert families["rate_style_swaption"]["parity_closed"] is True
    assert families["variance_swap"]["parity_closed"] is True
    assert families["asian_option"]["blocked"] is True
    assert families["asian_option"]["phase4_candidate"] is False
    assert report["totals"]["phase4_candidates"] == 5
    assert any("Arithmetic Asians remain an explicit Phase 3 blocker" in note for note in families["asian_option"]["notes"])
