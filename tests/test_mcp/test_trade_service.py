"""Tests for the transport-neutral governed trade parsing service."""

from __future__ import annotations


def test_trade_parse_normalizes_natural_language_vanilla_option():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        description="European call on AAPL with strike 120 and expiry 2025-11-15",
        instrument_type="european_option",
    )

    assert result.parse_status == "parsed"
    assert result.semantic_id == "vanilla_option"
    assert result.trade_type == "european_option"
    assert result.asset_class == "equity"
    assert result.missing_fields == ()
    assert result.warnings == ()
    assert result.product_ir["payoff_family"] == "vanilla_option"
    assert result.required_market_data == (
        "black_vol_surface",
        "discount_curve",
        "underlier_spot",
    )


def test_trade_parse_supports_structured_swaption_input():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "swaption",
            "observation_schedule": ("2026-01-15",),
            "exercise_style": "european",
            "preferred_method": "analytical",
        }
    )

    assert result.parse_status == "parsed"
    assert result.semantic_id == "rate_style_swaption"
    assert result.asset_class == "rates"
    assert result.trade_type == "swaption"
    assert result.contract_summary["semantic_id"] == "rate_style_swaption"
    assert result.product_ir["exercise_style"] == "european"
    assert "discount_curve" in result.required_market_data


def test_trade_parse_reports_missing_fields_for_incomplete_request():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        description="Price a resettable memory note with a holiday-adjusted schedule and monthly coupons.",
        instrument_type="structured_note",
    )

    assert result.parse_status == "incomplete"
    assert result.semantic_id == ""
    assert "underlier_structure" in result.missing_fields
    assert "observation_schedule" in result.missing_fields
    assert result.warnings
    assert result.contract_summary["gap_types"] == [
        "missing_semantic_contract_field",
        "missing_runtime_primitive",
        "missing_knowledge_lesson",
    ]

