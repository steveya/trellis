"""Tests for deterministic governed model matching."""

from __future__ import annotations


def _store_with_registered_models(tmp_path):
    from trellis.platform.models import (
        ModelLifecycleStatus,
        ModelRecord,
        ModelRegistryStore,
        ModelVersionRecord,
    )

    store = ModelRegistryStore(base_dir=tmp_path)
    store.create_model(
        ModelRecord(
            model_id="vanilla_option_approved",
            semantic_id="vanilla_option",
            semantic_version="c2.1",
            product_family="equity_option",
            instrument_class="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            underlier_structure="single_underlier",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("discount_curve", "underlier_spot", "black_vol_surface"),
            supported_method_families=("analytical",),
        )
    )
    store.create_version(
        ModelVersionRecord(
            model_id="vanilla_option_approved",
            version="v1",
            contract_summary={"semantic_id": "vanilla_option"},
            methodology_summary={"method_family": "analytical"},
            engine_binding={"engine_id": "pricing_engine.local"},
        ),
        actor="builder",
        reason="seed",
    )
    store.transition_version(
        "vanilla_option_approved",
        "v1",
        ModelLifecycleStatus.VALIDATED,
        actor="validator",
        reason="seed_validation",
    )
    store.transition_version(
        "vanilla_option_approved",
        "v1",
        ModelLifecycleStatus.APPROVED,
        actor="reviewer",
        reason="seed_approval",
    )

    store.create_model(
        ModelRecord(
            model_id="vanilla_option_eur",
            semantic_id="vanilla_option",
            semantic_version="c2.1",
            product_family="equity_option",
            instrument_class="european_option",
            payoff_family="vanilla_option",
            exercise_style="european",
            underlier_structure="single_underlier",
            payout_currency="EUR",
            reporting_currency="EUR",
            required_market_data=("discount_curve", "underlier_spot", "black_vol_surface"),
            supported_method_families=("analytical",),
        )
    )
    store.create_version(
        ModelVersionRecord(
            model_id="vanilla_option_eur",
            version="v1",
            status=ModelLifecycleStatus.VALIDATED,
            contract_summary={"semantic_id": "vanilla_option"},
            methodology_summary={"method_family": "analytical"},
            engine_binding={"engine_id": "pricing_engine.local"},
        ),
        actor="builder",
        reason="seed",
    )
    store.transition_version(
        "vanilla_option_eur",
        "v1",
        ModelLifecycleStatus.VALIDATED,
        actor="validator",
        reason="seed_validation",
    )
    return store


def test_model_match_selects_exact_approved_candidate(tmp_path):
    from trellis.platform.services.model_service import ModelService
    from trellis.platform.services.trade_service import TradeService

    parsed = TradeService().parse_trade(
        description="European call on AAPL with strike 120 and expiry 2025-11-15",
        instrument_type="european_option",
    )
    service = ModelService(registry=_store_with_registered_models(tmp_path))

    result = service.match_trade(parsed)

    assert result.match_type == "exact_approved_match"
    assert result.selected_candidate["model_id"] == "vanilla_option_approved"
    assert result.selected_candidate["version"] == "v1"
    assert result.selected_candidate["status"] == "approved"
    assert result.candidates[0]["execution_eligible"] is True


def test_model_match_explains_rejected_candidates(tmp_path):
    from trellis.platform.services.model_service import ModelService
    from trellis.platform.services.trade_service import TradeService

    parsed = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "european_option",
            "underliers": ("AAPL",),
            "observation_schedule": ("2025-11-15",),
            "payout_currency": "USD",
            "reporting_currency": "USD",
            "preferred_method": "analytical",
        }
    )
    service = ModelService(registry=_store_with_registered_models(tmp_path))

    explanation = service.explain_match(parsed)

    assert explanation.match_type == "exact_approved_match"
    rejected = {
        candidate["model_id"]: tuple(candidate["rejections"])
        for candidate in explanation.candidates
        if candidate["rejections"]
    }
    assert "currency_mismatch" in rejected["vanilla_option_eur"]


def test_model_match_returns_no_match_when_structure_does_not_fit(tmp_path):
    from trellis.platform.services.model_service import ModelService
    from trellis.platform.services.trade_service import TradeService

    parsed = TradeService().parse_trade(
        structured_trade={
            "instrument_type": "swaption",
            "observation_schedule": ("2026-01-15",),
            "exercise_style": "european",
            "preferred_method": "analytical",
        }
    )
    service = ModelService(registry=_store_with_registered_models(tmp_path))

    result = service.match_trade(parsed)

    assert result.match_type == "no_match"
    assert result.selected_candidate == {}
    assert any("semantic_id_mismatch" in candidate["rejections"] for candidate in result.candidates)


def test_model_match_selects_exact_approved_range_accrual_candidate(tmp_path):
    from trellis.platform.models import (
        ModelLifecycleStatus,
        ModelRecord,
        ModelRegistryStore,
        ModelVersionRecord,
    )
    from trellis.platform.services.model_service import ModelService
    from trellis.platform.services.trade_service import TradeService

    store = ModelRegistryStore(base_dir=tmp_path)
    store.create_model(
        ModelRecord(
            model_id="range_accrual_checked",
            semantic_id="range_accrual",
            semantic_version="c2.1",
            product_family="rates_exotic",
            instrument_class="range_accrual",
            payoff_family="range_accrual_coupon",
            exercise_style="none",
            underlier_structure="single_curve_rate_style",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("discount_curve", "fixing_history", "forward_curve"),
            supported_method_families=("analytical",),
        )
    )
    store.create_version(
        ModelVersionRecord(
            model_id="range_accrual_checked",
            version="v1",
            contract_summary={"semantic_id": "range_accrual"},
            methodology_summary={"method_family": "analytical"},
            engine_binding={"engine_id": "pricing_engine.local", "adapter_id": "range_accrual_discounted"},
        ),
        actor="builder",
        reason="seed",
    )
    store.transition_version(
        "range_accrual_checked",
        "v1",
        ModelLifecycleStatus.VALIDATED,
        actor="validator",
        reason="seed_validation",
    )
    store.transition_version(
        "range_accrual_checked",
        "v1",
        ModelLifecycleStatus.APPROVED,
        actor="reviewer",
        reason="seed_approval",
    )

    parsed = TradeService().parse_trade(
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
            "payout_currency": "USD",
            "reporting_currency": "USD",
            "preferred_method": "analytical",
        }
    )
    service = ModelService(registry=store)

    result = service.match_trade(parsed)

    assert result.match_type == "exact_approved_match"
    assert result.selected_candidate["model_id"] == "range_accrual_checked"
    assert result.selected_candidate["engine_binding"]["adapter_id"] == "range_accrual_discounted"
