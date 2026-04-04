"""Governed provider listing and configuration for transport-neutral MCP workflows."""

from __future__ import annotations

from dataclasses import replace

from trellis.platform.context import ProviderBindings, RunMode


class ProviderService:
    """Expose stable provider listing and session-scoped binding updates."""

    def __init__(
        self,
        *,
        provider_registry,
        session_service,
    ):
        self.provider_registry = provider_registry
        self.session_service = session_service

    def list_providers(
        self,
        *,
        session_id: str | None = None,
        kind: str | None = None,
    ) -> dict[str, object]:
        """List visible providers plus their current binding slots for one session."""
        record = self.session_service.ensure_record(session_id)
        active_bindings = record.provider_bindings.to_dict()
        bound_as = self._bound_slots(record.provider_bindings)
        providers = []
        for provider in self.provider_registry.list_providers(kind=kind):
            payload = provider.to_dict()
            payload["bound_as"] = bound_as.get(provider.provider_id, [])
            providers.append(payload)
        return {
            "session_id": record.session_id,
            "providers": providers,
            "active_bindings": active_bindings,
        }

    def configure(
        self,
        *,
        session_id: str | None = None,
        provider_bindings,
    ) -> dict[str, object]:
        """Persist explicit provider bindings for one governed session."""
        from trellis.mcp.errors import TrellisMcpError

        record = self.session_service.ensure_record(session_id)
        bindings = (
            provider_bindings
            if isinstance(provider_bindings, ProviderBindings)
            else ProviderBindings.from_dict(provider_bindings or {})
        )
        self._validate_bindings(bindings)

        if record.run_mode is RunMode.PRODUCTION and self._uses_mock_market_data(bindings):
            raise TrellisMcpError(
                code="mock_data_not_allowed",
                message="Production sessions cannot bind a mock market-data provider.",
                details={"session_id": record.session_id},
            )

        persisted = self.session_service.save(
            replace(
                record,
                provider_bindings=bindings,
                connected_providers=tuple(
                    provider.provider_id for provider in self.provider_registry.list_providers()
                ),
            )
        )
        return self.session_service.get_context(persisted.session_id)

    def _validate_bindings(self, bindings: ProviderBindings) -> None:
        from trellis.mcp.errors import TrellisMcpError

        family_bindings = {
            "market_data": bindings.market_data,
            "pricing_engine": bindings.pricing_engine,
            "model_store": bindings.model_store,
            "validation_engine": bindings.validation_engine,
        }
        for family, binding_set in family_bindings.items():
            for slot_name in ("primary", "fallback"):
                binding = getattr(binding_set, slot_name)
                if binding is None:
                    continue
                try:
                    provider = self.provider_registry.get_provider(binding.provider_id)
                except Exception as exc:
                    raise TrellisMcpError(
                        code="unknown_provider",
                        message=f"Unknown provider id: {binding.provider_id!r}",
                        details={"family": family, "slot": slot_name},
                    ) from exc
                if family == "market_data" and provider.kind != "market_data":
                    raise TrellisMcpError(
                        code="provider_kind_mismatch",
                        message=(
                            f"Provider {binding.provider_id!r} cannot be bound into the {family!r} family."
                        ),
                        details={"family": family, "provider_kind": provider.kind},
                    )

    @staticmethod
    def _uses_mock_market_data(bindings: ProviderBindings) -> bool:
        binding_set = bindings.market_data
        provider_ids = [
            binding.provider_id
            for binding in (binding_set.primary, binding_set.fallback)
            if binding is not None and binding.provider_id
        ]
        return any(".mock" in provider_id or provider_id.endswith("_mock") for provider_id in provider_ids)

    @staticmethod
    def _bound_slots(bindings: ProviderBindings) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for family_name in ("market_data", "pricing_engine", "model_store", "validation_engine"):
            binding_set = getattr(bindings, family_name)
            for slot_name in ("primary", "fallback"):
                binding = getattr(binding_set, slot_name)
                if binding is None:
                    continue
                result.setdefault(binding.provider_id, []).append(f"{family_name}.{slot_name}")
        return result


__all__ = [
    "ProviderService",
]
