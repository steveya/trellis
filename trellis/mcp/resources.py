"""Durable MCP resource surface over governed Trellis artifacts."""

from __future__ import annotations

from trellis.mcp.errors import TrellisMcpError
from trellis.platform.policies import get_policy_bundle


_RESOURCE_TEMPLATES = (
    "trellis://market-snapshots/{snapshot_id}",
    "trellis://models/{model_id}",
    "trellis://models/{model_id}/versions",
    "trellis://models/{model_id}/versions/{version}/contract",
    "trellis://models/{model_id}/versions/{version}/code",
    "trellis://models/{model_id}/versions/{version}/validation-report",
    "trellis://policies/{policy_id}",
    "trellis://providers/{provider_id}",
    "trellis://runs/{run_id}",
    "trellis://runs/{run_id}/audit",
    "trellis://runs/{run_id}/inputs",
    "trellis://runs/{run_id}/logs",
    "trellis://runs/{run_id}/outputs",
)


class ResourceRegistry:
    """Small in-process registry for stable Trellis MCP resource URIs."""

    def __init__(self, services):
        self.services = services

    def list_resources(self) -> tuple[str, ...]:
        return _RESOURCE_TEMPLATES

    def read_resource(self, uri: str):
        normalized_uri = str(uri or "").strip()
        if not normalized_uri.startswith("trellis://"):
            raise TrellisMcpError(
                code="unknown_resource",
                message=f"Unknown Trellis MCP resource: {normalized_uri!r}",
                details={"uri": normalized_uri},
            )
        path = normalized_uri[len("trellis://") :]
        if path.startswith("models/"):
            return self._read_model_resource(normalized_uri, path)
        if path.startswith("runs/"):
            return self._read_run_resource(normalized_uri, path)
        if path.startswith("market-snapshots/"):
            snapshot_id = path[len("market-snapshots/") :]
            return {"snapshot": self._require_snapshot(snapshot_id).to_dict()}
        if path.startswith("providers/"):
            provider_id = path[len("providers/") :]
            return {"provider": self._require_provider(provider_id).to_dict()}
        if path.startswith("policies/"):
            policy_id = path[len("policies/") :]
            return {"policy": get_policy_bundle(policy_id).to_dict()}
        raise TrellisMcpError(
            code="unknown_resource",
            message=f"Unknown Trellis MCP resource: {normalized_uri!r}",
            details={"uri": normalized_uri},
        )

    def _read_model_resource(self, uri: str, path: str):
        parts = path.split("/")
        if len(parts) < 2:
            raise self._unknown_resource(uri)
        model_id = parts[1]
        model = self._require_model(model_id)
        if len(parts) == 2:
            return {"model": model.to_dict()}
        if len(parts) == 3 and parts[2] == "versions":
            return {
                "model": model.to_dict(),
                "versions": [
                    record.to_dict() for record in self.services.model_registry.list_versions(model_id)
                ],
            }
        if len(parts) == 5 and parts[2] == "versions":
            version = self._require_model_version(model_id, parts[3])
            artifact_name = parts[4]
            if artifact_name == "contract":
                return {
                    "uri": uri,
                    "contract": self.services.model_registry.load_version_artifact(
                        model_id, version.version, "contract"
                    )
                    or dict(version.contract_summary),
                }
            if artifact_name == "code":
                return {
                    "uri": uri,
                    "source_code": self.services.model_registry.load_version_artifact(
                        model_id, version.version, "code"
                    )
                    or "",
                    "module_path": str(version.artifacts.get("module_path", "")).strip(),
                }
            if artifact_name == "validation-report":
                return {
                    "uri": uri,
                    "validation_report": self.services.model_registry.load_version_artifact(
                        model_id, version.version, "validation-report"
                    )
                    or dict(version.validation_summary),
                }
        raise self._unknown_resource(uri)

    def _read_run_resource(self, uri: str, path: str):
        parts = path.split("/")
        if len(parts) < 2:
            raise self._unknown_resource(uri)
        run_id = parts[1]
        record = self._require_run(run_id)
        if len(parts) == 2:
            return {"run": record.to_dict()}
        if len(parts) == 3 and parts[2] == "audit":
            return self.services.audit_service.get_audit(run_id=run_id)
        if len(parts) == 3 and parts[2] == "inputs":
            return {
                "run_id": record.run_id,
                "inputs": {
                    "trade_identity": dict(record.trade_identity),
                    "provider_bindings": dict(record.provider_bindings),
                    "market_snapshot_id": record.market_snapshot_id,
                    "valuation_timestamp": record.valuation_timestamp,
                    "policy_id": record.policy_id,
                    "run_mode": record.run_mode,
                    "session_id": record.session_id,
                },
            }
        if len(parts) == 3 and parts[2] == "outputs":
            return {"run_id": record.run_id, "outputs": dict(record.result_summary)}
        if len(parts) == 3 and parts[2] == "logs":
            audit = self.services.run_ledger.build_audit_bundle(run_id).to_dict()
            return {
                "run_id": record.run_id,
                "events": list(audit.get("diagnostics", {}).get("trace_events") or ()),
                "logs": [],
                "artifacts": audit.get("artifacts", {}),
            }
        raise self._unknown_resource(uri)

    def _require_model(self, model_id: str):
        model = self.services.model_registry.get_model(str(model_id or "").strip())
        if model is None:
            raise TrellisMcpError(
                code="unknown_resource",
                message=f"Unknown model resource: {model_id!r}",
                details={"model_id": str(model_id or "").strip()},
            )
        return model

    def _require_model_version(self, model_id: str, version: str):
        record = self.services.model_registry.get_version(
            str(model_id or "").strip(),
            str(version or "").strip(),
        )
        if record is None:
            raise TrellisMcpError(
                code="unknown_resource",
                message=f"Unknown model version resource: {model_id}:{version}",
                details={"model_id": str(model_id or "").strip(), "version": str(version or "").strip()},
            )
        return record

    def _require_run(self, run_id: str):
        record = self.services.run_ledger.get_run(str(run_id or "").strip())
        if record is None:
            raise TrellisMcpError(
                code="unknown_resource",
                message=f"Unknown run resource: {run_id!r}",
                details={"run_id": str(run_id or "").strip()},
            )
        return record

    def _require_snapshot(self, snapshot_id: str):
        record = self.services.snapshot_store.get_snapshot(str(snapshot_id or "").strip())
        if record is None:
            raise TrellisMcpError(
                code="unknown_resource",
                message=f"Unknown snapshot resource: {snapshot_id!r}",
                details={"snapshot_id": str(snapshot_id or "").strip()},
            )
        return record

    def _require_provider(self, provider_id: str):
        try:
            return self.services.provider_registry.get_provider(str(provider_id or "").strip())
        except Exception as exc:
            raise TrellisMcpError(
                code="unknown_resource",
                message=f"Unknown provider resource: {provider_id!r}",
                details={"provider_id": str(provider_id or "").strip()},
            ) from exc

    @staticmethod
    def _unknown_resource(uri: str) -> TrellisMcpError:
        return TrellisMcpError(
            code="unknown_resource",
            message=f"Unknown Trellis MCP resource: {uri!r}",
            details={"uri": uri},
        )


def build_resource_registry(services) -> ResourceRegistry:
    """Build the current MCP resource registry."""
    return ResourceRegistry(services)


__all__ = [
    "ResourceRegistry",
    "build_resource_registry",
]
