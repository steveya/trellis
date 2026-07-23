from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

from trellis.agent.contract_ir_solver_parity import (
    _swaption_reference,
    build_contract_ir_solver_parity_report,
)
from trellis.agent.platform_requests import compile_build_request
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.rate_style_swaption import ResolvedSwaptionBlack76Inputs
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


def test_swaption_parity_reference_preserves_raw_kernel_degenerate_zero_contract():
    resolved = ResolvedSwaptionBlack76Inputs(
        expiry_date=date(2025, 11, 15),
        expiry_years=1.0,
        annuity=1.0,
        forward_swap_rate=0.06,
        strike=0.05,
        vol=0.20,
        notional=1.0,
        is_payer=True,
        payment_count=10,
    )
    degenerate_cases = (
        replace(resolved, expiry_years=0.0),
        replace(resolved, annuity=-1.0),
        replace(resolved, payment_count=0),
    )

    for case in degenerate_cases:
        decision = SimpleNamespace(call_kwargs={"resolved": case})
        assert _swaption_reference(decision, _swaption_market_state(), None) == 0.0


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
        "trellis.models.analytical.terminal_basket."
        "two_asset_terminal_basket_gauss_hermite"
    )
    assert compiler["source"] == "request_decomposition"
    assert compiler["shadow_status"] == "bound"
    assert compiler["contract_ir_solver_selection"]["declaration_id"] == (
        "terminal_basket_call_raw_kernel"
    )
    assert compiler["contract_ir_solver_shadow"]["declaration_id"] == (
        "terminal_basket_call_raw_kernel"
    )
    authority = compiled.request.metadata["route_binding_authority"]
    assert authority["route_id"] is None
    assert authority["authority_kind"] == "exact_backend_fit"
    assert authority["backend_binding"]["exact_target_refs"] == [
        "trellis.models.analytical.terminal_basket."
        "two_asset_terminal_basket_gauss_hermite"
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
    assert compiled.generation_plan is not None
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "analytical_black76"
    assert not any(
        "variance_swap_analytical" in ref
        for ref in compiled.generation_plan.backend_exact_target_refs
    )
    authority = compiled.request.metadata["route_binding_authority"]
    assert authority["authority_kind"] == "route_registry_binding"


def test_compile_build_request_records_bounded_asian_analytical_shadow():
    compiled = compile_build_request(
        "Arithmetic Asian call on SPX monthly average over 2025 strike 4500",
        instrument_type="asian_option",
        market_snapshot=_variance_market_state(),
        preferred_method="analytical",
    )

    compiler = compiled.request.metadata["contract_ir_compiler"]

    assert compiled.semantic_blueprint is None
    assert compiler["source"] == "request_decomposition"
    assert compiler["shadow_status"] == "bound"
    assert compiler["contract_ir_solver_shadow"]["declaration_id"] == (
        "compose_arithmetic_asian_analytical_call"
    )
    assert compiler["shadow_error"] is None


def test_compile_build_request_records_bounded_asian_put_analytical_shadow():
    compiled = compile_build_request(
        "Arithmetic Asian put on SPX weekly average from 2025-01-03 to 2025-01-31 strike 4500",
        instrument_type="asian_option",
        market_snapshot=_variance_market_state(),
        preferred_method="analytical",
    )

    compiler = compiled.request.metadata["contract_ir_compiler"]

    assert compiled.semantic_blueprint is None
    assert compiler["source"] == "request_decomposition"
    assert compiler["shadow_status"] == "bound"
    assert compiler["contract_ir_solver_shadow"]["declaration_id"] == (
        "compose_arithmetic_asian_analytical_put"
    )
    assert compiler["shadow_error"] is None


def test_compile_build_request_records_bounded_asian_monte_carlo_shadow():
    compiled = compile_build_request(
        "Arithmetic Asian call on SPX monthly average over 2025 strike 4500",
        instrument_type="asian_option",
        market_snapshot=_variance_market_state(),
        preferred_method="monte_carlo",
    )

    compiler = compiled.request.metadata["contract_ir_compiler"]

    assert compiled.semantic_blueprint is None
    assert compiler["source"] == "request_decomposition"
    assert compiler["shadow_status"] == "bound"
    assert compiler["contract_ir_solver_shadow"]["declaration_id"] == (
        "compose_arithmetic_asian_monte_carlo_call"
    )
    assert compiler["shadow_error"] is None


def test_contract_ir_solver_parity_report_promotes_bounded_asian_family():
    report = build_contract_ir_solver_parity_report()

    families = {entry["family_id"]: entry for entry in report["families"]}

    assert families["vanilla_option"]["exact_authority_closed"] is True
    assert families["basket_option"]["exact_authority_closed"] is True
    assert families["basket_option"]["phase4_candidate"] is True
    assert all(case["route_free_authority"] is True for case in families["basket_option"]["cases"])
    assert families["vanilla_option"]["parity_closed"] is True
    assert families["digital_option"]["parity_closed"] is True
    assert families["rate_style_swaption"]["parity_closed"] is True
    assert families["variance_swap"]["parity_closed"] is True
    assert families["variance_swap"]["exact_authority_closed"] is False
    assert families["variance_swap"]["phase4_candidate"] is False
    assert any(
        "comparison-only" in note
        for note in families["variance_swap"]["notes"]
    )
    assert families["asian_option"]["blocked"] is False
    assert families["asian_option"]["parity_closed"] is True
    assert families["asian_option"]["exact_authority_closed"] is True
    assert families["asian_option"]["phase4_candidate"] is True
    assert all(case["route_free_authority"] is True for case in families["asian_option"]["cases"])
    assert any(case["case_id"] == "asian_call_analytical" for case in families["asian_option"]["cases"])
    assert any(case["case_id"] == "asian_call_monte_carlo" for case in families["asian_option"]["cases"])
    assert any(case["case_id"] == "asian_put_analytical" for case in families["asian_option"]["cases"])
    assert any(case["case_id"] == "asian_put_monte_carlo" for case in families["asian_option"]["cases"])
    put_monte_carlo = next(
        case
        for case in families["asian_option"]["cases"]
        if case["case_id"] == "asian_put_monte_carlo"
    )
    assert put_monte_carlo["reference_price"] > 10.0
    assert put_monte_carlo["rel_diff"] <= 0.02
    assert report["totals"]["phase4_candidates"] == 5
    assert any("bounded analytical approximation" in note for note in families["asian_option"]["notes"])
