"""Unit tests for governed deterministic validation service behavior."""

from __future__ import annotations


def test_validation_service_marks_missing_engine_binding_as_failed(tmp_path):
    from trellis.platform.models import ModelRecord, ModelRegistryStore, ModelVersionRecord
    from trellis.platform.services.validation_service import ValidationService
    from trellis.platform.storage import ValidationStore

    registry = ModelRegistryStore(base_dir=tmp_path / "models")
    registry.create_model(
        ModelRecord(
            model_id="callable_bond_candidate",
            semantic_id="callable_bond",
            semantic_version="1.0.0",
            product_family="callable_bond",
            instrument_class="bond",
            payoff_family="callable_fixed_income",
            exercise_style="bermudan",
            underlier_structure="single_curve",
            payout_currency="USD",
            reporting_currency="USD",
            required_market_data=("discount_curve", "vol_surface"),
            supported_method_families=("rate_tree",),
        )
    )
    registry.create_version(
        ModelVersionRecord(
            model_id="callable_bond_candidate",
            version="v1",
            contract_summary={"semantic_id": "callable_bond"},
            methodology_summary={"method_family": "rate_tree"},
        ),
        actor="builder",
        reason="seed_candidate",
    )

    service = ValidationService(
        registry=registry,
        validation_store=ValidationStore(tmp_path / "validations"),
    )
    payload = service.validate_model(
        model_id="callable_bond_candidate",
        version="v1",
        actor="validator",
        reason="manifest_validation",
    )

    assert payload["validation"]["status"] == "failed"
    assert payload["validation"]["summary"]["checks"]["has_engine_binding"] is False
    assert payload["version"]["status"] == "draft"
