"""Unit tests for governed model-service lifecycle semantics."""

from __future__ import annotations

from dataclasses import replace

import pytest


def _seed_registry(tmp_path):
    from trellis.platform.models import ModelRecord, ModelRegistryStore, ModelVersionRecord

    registry = ModelRegistryStore(base_dir=tmp_path / "models")
    registry.create_model(
        ModelRecord(
            model_id="vanilla_option_candidate",
            semantic_id="vanilla_option",
            semantic_version="1.0.0",
            product_family="equity_option",
            instrument_class="option",
            payoff_family="vanilla_option",
            exercise_style="european",
            underlier_structure="single_name",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("discount_curve",),
            supported_method_families=("analytical",),
        )
    )
    registry.create_version(
        ModelVersionRecord(
            model_id="vanilla_option_candidate",
            version="v1",
            contract_summary={"semantic_id": "vanilla_option"},
            methodology_summary={"method_family": "analytical"},
            engine_binding={"engine_id": "pricing_engine.local", "version": "1"},
            artifacts={
                "contract_uri": "trellis://models/vanilla_option_candidate/versions/v1/contract",
                "validation_plan_uri": "trellis://models/vanilla_option_candidate/versions/v1/validation-plan",
            },
        ),
        actor="builder",
        reason="seed_candidate",
    )
    return registry


def test_promote_requires_latest_validation_to_pass(tmp_path):
    from trellis.mcp.errors import TrellisMcpError
    from trellis.platform.services.model_service import ModelService
    from trellis.platform.services.validation_service import ValidationService
    from trellis.platform.storage import ValidationStore

    registry = _seed_registry(tmp_path)
    validation_store = ValidationStore(tmp_path / "validations")
    validation_service = ValidationService(
        registry=registry,
        validation_store=validation_store,
    )
    model_service = ModelService(registry=registry)

    first = validation_service.validate_model(
        model_id="vanilla_option_candidate",
        version="v1",
        actor="validator",
        reason="initial_validation",
    )
    assert first["validation"]["status"] == "passed"

    registry.save_version(
        replace(
            registry.get_version("vanilla_option_candidate", "v1"),
            engine_binding={},
        )
    )
    second = validation_service.validate_model(
        model_id="vanilla_option_candidate",
        version="v1",
        actor="validator",
        reason="revalidation_after_change",
    )
    assert second["validation"]["status"] == "failed"
    assert second["version"]["validation_summary"]["all_checks_passed"] is False

    with pytest.raises(TrellisMcpError) as excinfo:
        model_service.promote_version(
            model_id="vanilla_option_candidate",
            version="v1",
            to_status="validated",
            actor="reviewer",
            reason="attempt_after_failed_revalidation",
            validation_store=validation_store,
        )
    assert excinfo.value.code == "validation_required"
