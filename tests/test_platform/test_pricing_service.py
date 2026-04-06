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
