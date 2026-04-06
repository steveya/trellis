"""Unit tests for governed MCP pricing service provenance semantics."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace


def test_compiled_pricing_request_keeps_selected_model_method_family():
    from trellis.platform.services.pricing_service import PricingService
    from trellis.platform.services.trade_service import TradeParseResult

    parsed_trade = TradeParseResult(
        parse_status="parsed",
        semantic_id="vanilla_option",
        semantic_version="1.0.0",
        trade_type="european_option",
        parsed_contract={"description": "European option"},
        candidate_methods=(),
    )
    market_snapshot = SimpleNamespace(
        as_of="2026-04-04",
        default_discount_curve="discount",
        default_vol_surface="default",
    )
    compiled = PricingService._compiled_pricing_request(
        request_id="req_001",
        parsed_trade=parsed_trade,
        execution_context=SimpleNamespace(),
        market_snapshot=market_snapshot,
        payoff=object(),
        description="European option",
        selected_model={
            "methodology_summary": {"method_family": "analytical"},
        },
        settlement=date(2026, 4, 4),
    )

    assert compiled.execution_plan.route_method == "analytical"


def test_build_callable_bond_spec_preserves_explicit_zero_call_price():
    from trellis.platform.services.pricing_service import PricingService
    from trellis.platform.services.trade_service import TradeService

    pricing_input = {
        "instrument_type": "callable_bond",
        "notional": 1_000_000.0,
        "coupon": 0.05,
        "start_date": "2025-01-15",
        "end_date": "2035-01-15",
        "call_dates": ("2028-01-15", "2030-01-15"),
        "call_price": 0.0,
    }
    parsed_trade = TradeService().parse_trade(structured_trade=pricing_input)

    spec, assumptions = PricingService._build_callable_bond_spec(
        parsed_trade=parsed_trade,
        pricing_input=pricing_input,
    )

    assert spec.call_price == 0.0
    assert not any("defaulted call_price" in note.lower() for note in assumptions)


def test_build_range_accrual_spec_preserves_explicit_zero_principal_redemption():
    from trellis.platform.services.pricing_service import PricingService
    from trellis.platform.services.trade_service import TradeService

    pricing_input = {
        "instrument_type": "range_accrual",
        "reference_index": "SOFR",
        "coupon_rate": 0.0525,
        "lower_bound": 0.015,
        "upper_bound": 0.0325,
        "observation_schedule": (
            "2026-01-15",
            "2026-04-15",
            "2026-07-15",
            "2026-10-15",
        ),
        "accrual_start_dates": (
            "2025-10-15",
            "2026-01-15",
            "2026-04-15",
            "2026-07-15",
        ),
        "payment_dates": (
            "2026-01-15",
            "2026-04-15",
            "2026-07-15",
            "2026-10-15",
        ),
        "notional": 1_000_000.0,
        "principal_redemption": 0.0,
    }
    parsed_trade = TradeService().parse_trade(structured_trade=pricing_input)

    spec = PricingService._build_range_accrual_spec(
        parsed_trade=parsed_trade,
        pricing_input=pricing_input,
    )

    assert spec.principal_redemption == 0.0
