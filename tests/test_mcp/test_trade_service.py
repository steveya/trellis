"""Tests for the transport-neutral governed trade parsing service."""

from __future__ import annotations

import csv
import json


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


def test_trade_parse_supports_structured_option_expiry_date_alias():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "european_option",
            "underliers": ("AAPL",),
            "expiry_date": "2026-12-31",
            "strike": 120.0,
            "option_type": "call",
        }
    )

    assert result.parse_status == "parsed"
    assert result.semantic_id == "vanilla_option"
    assert result.trade_type == "european_option"
    assert result.contract_summary["product"]["observation_schedule"] == ["2026-12-31"]


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


def test_trade_parse_supports_structured_range_accrual_input():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
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
            "call_dates": ("2026-07-15",),
            "payout_currency": "USD",
        }
    )

    assert result.parse_status == "parsed"
    assert result.semantic_id == "range_accrual"
    assert result.asset_class == "rates"
    assert result.trade_type == "range_accrual"
    assert result.contract_summary["product"]["term_fields"]["reference_index"] == "SOFR"
    assert result.contract_summary["product"]["term_fields"]["callability"] == {
        "call_schedule": ["2026-07-15"],
        "call_style": "issuer_callable",
    }
    assert result.required_market_data == (
        "discount_curve",
        "fixing_history",
        "forward_curve",
    )


def test_trade_parse_range_accrual_still_requires_observation_schedule():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "range_accrual",
            "reference_index": "SOFR",
            "coupon_rate": 0.0525,
            "lower_bound": 0.015,
            "upper_bound": 0.0325,
            "call_schedule": ("2026-07-15",),
        }
    )

    assert result.parse_status == "incomplete"
    assert result.missing_fields == ("observation_schedule",)


def test_trade_parse_supports_structured_callable_bond_input_with_term_fields():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "callable_bond",
            "description": "Callable bond with 5% coupons and issuer call dates in 2028 and 2030.",
            "notional": 1_000_000.0,
            "coupon": 0.05,
            "start_date": "2025-01-15",
            "end_date": "2035-01-15",
            "call_dates": ("2028-01-15", "2030-01-15"),
            "call_price": 100.0,
            "frequency": "semi_annual",
            "day_count": "act_365",
            "payout_currency": "USD",
            "reporting_currency": "USD",
            "preferred_method": "rate_tree",
        }
    )

    assert result.parse_status == "parsed"
    assert result.semantic_id == "callable_bond"
    assert result.asset_class == "rates"
    assert result.trade_type == "callable_bond"
    assert result.product_ir["exercise_style"] == "issuer_call"
    assert result.contract_summary["product"]["term_fields"] == {
        "notional": 1_000_000.0,
        "coupon": 0.05,
        "start_date": "2025-01-15",
        "end_date": "2035-01-15",
        "call_price": 100.0,
        "frequency": "SEMI_ANNUAL",
        "day_count": "ACT_365",
        "exercise_schedule": ["2028-01-15", "2030-01-15"],
    }
    assert result.required_market_data == (
        "black_vol_surface",
        "discount_curve",
    )


def test_trade_parse_supports_structured_callable_bond_call_schedule_alias():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "callable_bond",
            "description": "Callable bond with schedule provided via call_schedule alias.",
            "notional": 1_000_000.0,
            "coupon": 0.05,
            "start_date": "2025-01-15",
            "end_date": "2035-01-15",
            "call_schedule": ("2028-01-15", "2030-01-15"),
            "preferred_method": "rate_tree",
        }
    )

    assert result.parse_status == "parsed"
    assert result.semantic_id == "callable_bond"
    assert result.contract_summary["product"]["term_fields"]["exercise_schedule"] == [
        "2028-01-15",
        "2030-01-15",
    ]


def test_trade_parse_supports_structured_callable_bond_day_count_aliases():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "callable_bond",
            "notional": 1_000_000.0,
            "coupon": 0.05,
            "start_date": "2025-01-15",
            "end_date": "2035-01-15",
            "call_dates": ("2028-01-15", "2030-01-15"),
            "day_count": "THIRTY_360_US",
        }
    )

    assert result.parse_status == "parsed"
    assert result.contract_summary["product"]["term_fields"]["day_count"] == "THIRTY_360"


def test_trade_parse_rejects_unsupported_callable_bond_frequency():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "callable_bond",
            "notional": 1_000_000.0,
            "coupon": 0.05,
            "start_date": "2025-01-15",
            "end_date": "2035-01-15",
            "call_dates": ("2028-01-15", "2030-01-15"),
            "frequency": "weekly",
        }
    )

    assert result.parse_status == "invalid"
    assert any("unsupported frequency" in warning.lower() for warning in result.warnings)


def test_trade_parse_supports_structured_bermudan_swaption_input_with_term_fields():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "bermudan_swaption",
            "description": "Bermudan payer swaption with annual exercise dates from 2027 to 2029.",
            "notional": 5_000_000.0,
            "strike": 0.04,
            "exercise_schedule": ("2027-11-15", "2028-11-15", "2029-11-15"),
            "swap_end": "2032-11-15",
            "swap_frequency": "semi_annual",
            "day_count": "act_360",
            "is_payer": True,
            "payout_currency": "USD",
            "reporting_currency": "USD",
            "preferred_method": "rate_tree",
        }
    )

    assert result.parse_status == "parsed"
    assert result.semantic_id == "rate_style_swaption"
    assert result.asset_class == "rates"
    assert result.trade_type == "bermudan_swaption"
    assert result.product_ir["exercise_style"] == "bermudan"
    assert result.contract_summary["product"]["term_fields"] == {
        "notional": 5_000_000.0,
        "strike": 0.04,
        "swap_end": "2032-11-15",
        "swap_frequency": "SEMI_ANNUAL",
        "day_count": "ACT_360",
        "is_payer": True,
        "exercise_schedule": ["2027-11-15", "2028-11-15", "2029-11-15"],
    }
    assert result.required_market_data == (
        "black_vol_surface",
        "discount_curve",
        "forward_curve",
    )


def test_trade_parse_rejects_unsupported_bermudan_swaption_day_count():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "bermudan_swaption",
            "notional": 5_000_000.0,
            "strike": 0.04,
            "exercise_schedule": ("2027-11-15", "2028-11-15", "2029-11-15"),
            "swap_end": "2032-11-15",
            "day_count": "bus_252",
        }
    )

    assert result.parse_status == "invalid"
    assert any("unsupported day_count" in warning.lower() for warning in result.warnings)


def test_position_parse_supports_flat_range_accrual_row_contract():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_position(
        structured_position={
            "position_id": "ra_1",
            "trade_type": "range_accrual",
            "quantity": 2.0,
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
            "payout_currency": "USD",
        }
    )

    assert result.parse_status == "parsed"
    assert result.position_id == "ra_1"
    assert result.instrument_type == "range_accrual"
    assert result.quantity == 2.0
    assert result.field_map["instrument_type"] == "trade_type"
    assert result.field_map["structured_trade"] == "top_level_trade_fields"
    assert result.position_contract["structured_trade"]["instrument_type"] == "range_accrual"
    assert result.trade_summary["semantic_id"] == "range_accrual"


def test_position_parse_supports_nested_callable_bond_trade_payload():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_position(
        structured_position={
            "id": "call_1",
            "qty": 1.5,
            "tags": ("rates", "callable"),
            "trade": {
                "instrument_type": "callable_bond",
                "notional": 1_000_000.0,
                "coupon": 0.05,
                "start_date": "2025-01-15",
                "end_date": "2035-01-15",
                "call_dates": ("2028-01-15", "2030-01-15"),
            },
        }
    )

    assert result.parse_status == "parsed"
    assert result.position_id == "call_1"
    assert result.instrument_type == "callable_bond"
    assert result.quantity == 1.5
    assert result.field_map["position_id"] == "id"
    assert result.field_map["quantity"] == "qty"
    assert result.field_map["structured_trade"] == "trade"
    assert result.position_contract["tags"] == ["rates", "callable"]
    assert result.trade_summary["semantic_id"] == "callable_bond"


def test_position_parse_reports_trade_missing_fields_on_incomplete_contract():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_position(
        structured_position={
            "position_id": "ra_missing",
            "trade_type": "range_accrual",
            "reference_index": "SOFR",
        }
    )

    assert result.parse_status == "incomplete"
    assert result.position_contract["position_id"] == "ra_missing"
    assert result.position_contract["structured_trade"]["instrument_type"] == "range_accrual"
    assert result.missing_fields == (
        "coupon_definition",
        "range_condition",
        "observation_schedule",
    )


def test_position_parse_rejects_instrument_type_mismatch_between_position_and_trade():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_position(
        structured_position={
            "position_id": "bad_1",
            "instrument_type": "callable_bond",
            "trade": {
                "instrument_type": "range_accrual",
                "reference_index": "SOFR",
            },
        }
    )

    assert result.parse_status == "invalid"
    assert result.position_id == "bad_1"
    assert result.instrument_type == "callable_bond"
    assert any("must match" in warning.lower() for warning in result.warnings)


def test_load_positions_supports_mixed_book_rows_with_partial_failures():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().load_positions(
        structured_positions=[
            {
                "position_id": "ra_1",
                "trade_type": "range_accrual",
                "quantity": 2.0,
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
                "payout_currency": "USD",
            },
            {
                "position_id": "call_1",
                "quantity": 1.5,
                "trade": {
                    "instrument_type": "callable_bond",
                    "notional": 1_000_000.0,
                    "coupon": 0.05,
                    "start_date": "2025-01-15",
                    "end_date": "2035-01-15",
                    "call_dates": ("2028-01-15", "2030-01-15"),
                },
            },
            {
                "position_id": "bad_row",
                "trade_type": "range_accrual",
                "reference_index": "SOFR",
            },
        ]
    )

    assert result.load_status == "partial"
    assert result.parsed_count == 2
    assert result.incomplete_count == 1
    assert result.invalid_count == 0
    assert tuple(result.position_book.keys()) == ("ra_1", "call_1")
    assert result.position_book["ra_1"]["instrument_type"] == "range_accrual"
    assert result.position_book["call_1"]["quantity"] == 1.5
    assert result.row_results[0]["row_index"] == 1
    assert result.row_results[0]["parse_status"] == "parsed"
    assert result.row_results[2]["row_index"] == 3
    assert result.row_results[2]["missing_fields"] == [
        "coupon_definition",
        "range_condition",
        "observation_schedule",
    ]


def test_load_positions_csv_reads_mixed_flat_file_book(tmp_path):
    from trellis.platform.services.trade_service import TradeService

    path = tmp_path / "book.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "position_id",
                "instrument_type",
                "quantity",
                "reference_index",
                "coupon_rate",
                "lower_bound",
                "upper_bound",
                "observation_schedule",
                "trade",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "position_id": "ra_csv",
                "instrument_type": "range_accrual",
                "quantity": 2.0,
                "reference_index": "SOFR",
                "coupon_rate": 0.0525,
                "lower_bound": 0.015,
                "upper_bound": 0.0325,
                "observation_schedule": "2026-01-15|2026-04-15|2026-07-15|2026-10-15",
            }
        )
        writer.writerow(
            {
                "position_id": "call_csv",
                "instrument_type": "callable_bond",
                "quantity": 1.0,
                "trade": json.dumps(
                    {
                        "notional": 1_000_000.0,
                        "coupon": 0.05,
                        "start_date": "2025-01-15",
                        "end_date": "2035-01-15",
                        "call_dates": ["2028-01-15", "2030-01-15"],
                    }
                ),
            }
        )

    result = TradeService().load_positions_csv(path)

    assert result.load_status == "parsed"
    assert result.parsed_count == 2
    assert result.position_book["ra_csv"]["structured_trade"]["observation_schedule"] == [
        "2026-01-15",
        "2026-04-15",
        "2026-07-15",
        "2026-10-15",
    ]
    assert result.position_book["call_csv"]["structured_trade"]["instrument_type"] == "callable_bond"
    assert result.row_results[1]["field_map"]["structured_trade"] == "trade"


def test_load_positions_json_reads_row_list_and_reports_invalid_rows(tmp_path):
    from trellis.platform.services.trade_service import TradeService

    path = tmp_path / "book.json"
    path.write_text(
        json.dumps(
            [
                {
                    "position_id": "ra_json",
                    "trade_type": "range_accrual",
                    "quantity": 1.0,
                    "reference_index": "SOFR",
                    "coupon_rate": 0.0525,
                    "lower_bound": 0.015,
                    "upper_bound": 0.0325,
                    "observation_schedule": [
                        "2026-01-15",
                        "2026-04-15",
                        "2026-07-15",
                        "2026-10-15",
                    ],
                },
                {
                    "position_id": "bad_json",
                    "instrument_type": "callable_bond",
                    "quantity": "oops",
                    "trade": {
                        "instrument_type": "callable_bond",
                        "notional": 1_000_000.0,
                        "coupon": 0.05,
                        "start_date": "2025-01-15",
                        "end_date": "2035-01-15",
                        "call_dates": ["2028-01-15", "2030-01-15"],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    result = TradeService().load_positions_json(path)

    assert result.load_status == "partial"
    assert result.parsed_count == 1
    assert result.incomplete_count == 0
    assert result.invalid_count == 1
    assert tuple(result.position_book.keys()) == ("ra_json",)
    assert result.row_results[1]["row_index"] == 2
    assert result.row_results[1]["parse_status"] == "invalid"
    assert any("quantity" in warning.lower() for warning in result.row_results[1]["warnings"])


def test_trade_parse_reports_range_accrual_missing_fields_for_incomplete_request():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        description="Price a range accrual note on SOFR.",
        instrument_type="range_accrual",
    )

    assert result.parse_status == "incomplete"
    assert result.asset_class == "rates"
    assert result.semantic_id == ""
    assert result.missing_fields == (
        "coupon_definition",
        "range_condition",
        "observation_schedule",
    )
    assert any("range accrual" in warning.lower() for warning in result.warnings)


def test_trade_parse_reports_callable_bond_missing_fields_for_incomplete_structured_request():
    from trellis.platform.services.trade_service import TradeService

    result = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "callable_bond",
            "call_dates": ("2028-01-15", "2030-01-15"),
        }
    )

    assert result.parse_status == "incomplete"
    assert result.asset_class == "rates"
    assert result.missing_fields == (
        "notional",
        "coupon",
        "start_date",
        "end_date",
    )


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
