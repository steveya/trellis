"""Deterministic governed validation service for model lifecycle workflows."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from trellis.platform.storage import ValidationRecord


def _new_validation_id() -> str:
    return f"validation_{uuid4().hex[:12]}"


class ValidationService:
    """Persist deterministic validation reports for governed model versions."""

    def __init__(self, *, registry, validation_store):
        self.registry = registry
        self.validation_store = validation_store

    def validate_model(
        self,
        *,
        model_id: str,
        version: str,
        actor: str,
        reason: str,
        notes: str = "",
        refs=(),
        policy_outcome=None,
        metadata=None,
    ) -> dict[str, object]:
        """Run one deterministic validation pass and persist the result."""
        from trellis.mcp.errors import TrellisMcpError

        stored_version = self.registry.get_version(model_id, version)
        if stored_version is None:
            raise TrellisMcpError(
                code="unknown_model_version",
                message=f"Unknown model version: {model_id}:{version}",
                details={"model_id": model_id, "version": version},
            )

        checks = self._deterministic_checks(stored_version)
        passed = all(checks.values())
        report = {
            "validation_mode": "deterministic_manifest_v1",
            "all_checks_passed": passed,
            "checks": checks,
            "reason": reason,
            "notes": notes,
            "actor": actor,
            "metadata": dict(metadata or {}),
        }
        report_uri = self.registry.write_version_artifact(
            model_id,
            version,
            "validation-report",
            report,
        )
        validation_record = self.validation_store.save_validation(
            ValidationRecord(
                validation_id=_new_validation_id(),
                model_id=model_id,
                version=version,
                status="passed" if passed else "failed",
                summary=report,
                refs=tuple(refs or ()) + (report_uri,),
                policy_outcome=policy_outcome or {},
            )
        )
        updated_version = self.registry.save_version(
            replace(
                stored_version,
                validation_summary=report,
                validation_refs=tuple(
                    dict.fromkeys(
                        (*stored_version.validation_refs, validation_record.validation_id, report_uri)
                    )
                ),
                artifacts={
                    **dict(stored_version.artifacts),
                    "validation_report_uri": report_uri,
                },
            )
        )
        model = self.registry.get_model(model_id)
        return {
            "validation": validation_record.to_dict(),
            "model": {} if model is None else model.to_dict(),
            "version": updated_version.to_dict(),
            "artifact_uris": {"validation_report": report_uri},
        }

    @staticmethod
    def _deterministic_checks(version_record) -> dict[str, bool]:
        """Return the deterministic manifest-integrity checks for one version."""
        artifacts = dict(version_record.artifacts)
        return {
            "has_contract_summary": bool(version_record.contract_summary),
            "has_methodology_summary": bool(version_record.methodology_summary),
            "has_engine_binding": bool(version_record.engine_binding),
            "has_contract_artifact": bool(artifacts.get("contract_uri")),
            "has_validation_plan": bool(artifacts.get("validation_plan_uri")),
        }


__all__ = [
    "ValidationService",
]
