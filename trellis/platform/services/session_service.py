"""Governed session context service for transport-neutral MCP workflows."""

from __future__ import annotations

from dataclasses import replace

from trellis.platform.context import ProviderBinding, ProviderBindingSet, ProviderBindings, RunMode
from trellis.platform.policies import get_policy_bundle
from trellis.platform.storage import SessionContextRecord, SessionContextStore, TrellisServerConfig


class SessionService:
    """Persist and project governed session context for MCP tool consumers."""

    def __init__(
        self,
        *,
        store: SessionContextStore,
        config: TrellisServerConfig,
        provider_registry,
    ):
        self.store = store
        self.config = config
        self.provider_registry = provider_registry

    def get_context(self, session_id: str | None = None) -> dict[str, object]:
        """Load or create one governed session context."""
        record = self._get_or_create(session_id)
        return self._payload(record)

    def set_run_mode(self, *, session_id: str | None = None, run_mode: str) -> dict[str, object]:
        """Persist one explicit run-mode change for a governed session."""
        from trellis.mcp.errors import TrellisMcpError

        record = self._get_or_create(session_id)
        try:
            normalized_mode = RunMode.normalize(run_mode)
        except ValueError as exc:
            raise TrellisMcpError(
                code="invalid_run_mode",
                message=f"Unknown governed run mode: {run_mode!r}",
            ) from exc

        if normalized_mode is not RunMode.SANDBOX and self._session_uses_mock_provider(record):
            raise TrellisMcpError(
                code="mock_data_not_allowed",
                message="Cannot switch this session into a non-sandbox run mode while a mock provider is bound.",
                details={"session_id": record.session_id, "run_mode": normalized_mode.value},
            )

        updated = replace(
            record,
            run_mode=normalized_mode,
            active_policy=f"policy_bundle.{normalized_mode.value}.default",
            allow_mock_data=normalized_mode is RunMode.SANDBOX,
            require_provider_disclosure=normalized_mode is not RunMode.SANDBOX,
        )
        persisted = self.store.save_session(updated)
        return self._payload(persisted)

    def save(self, record: SessionContextRecord) -> SessionContextRecord:
        """Persist one updated governed session record."""
        return self.store.save_session(record)

    def ensure_record(self, session_id: str | None = None) -> SessionContextRecord:
        """Load or create the canonical session record."""
        return self._get_or_create(session_id)

    def _get_or_create(self, session_id: str | None) -> SessionContextRecord:
        resolved_session_id = str(session_id or "default").strip() or "default"
        existing = self.store.get_session(resolved_session_id)
        if existing is not None:
            return existing
        created = self._default_record(resolved_session_id)
        return self.store.save_session(created)

    def _default_record(self, session_id: str) -> SessionContextRecord:
        run_mode = RunMode.normalize(self.config.defaults.run_mode)
        return SessionContextRecord(
            session_id=session_id,
            run_mode=run_mode,
            default_output_mode=self.config.defaults.output_mode,
            default_audit_mode=self.config.defaults.audit_mode,
            connected_providers=tuple(
                provider.provider_id for provider in self.provider_registry.list_providers()
            ),
            provider_bindings=self._default_provider_bindings(),
            active_policy=f"policy_bundle.{run_mode.value}.default",
            allow_mock_data=self.config.defaults.allow_mock_data and run_mode is RunMode.SANDBOX,
            require_provider_disclosure=self.config.defaults.require_explicit_provider_binding,
        )

    def _default_provider_bindings(self) -> ProviderBindings:
        return ProviderBindings(
            market_data=ProviderBindingSet(
                primary=self._binding(self.config.providers.market_data.primary),
                fallback=self._binding(self.config.providers.market_data.fallback),
            ),
            pricing_engine=ProviderBindingSet(
                primary=self._binding(self.config.providers.pricing_engine.primary),
                fallback=self._binding(self.config.providers.pricing_engine.fallback),
            ),
            model_store=ProviderBindingSet(
                primary=self._binding(self.config.providers.model_store.primary),
                fallback=self._binding(self.config.providers.model_store.fallback),
            ),
            validation_engine=ProviderBindingSet(
                primary=self._binding(self.config.providers.validation_engine.primary),
                fallback=self._binding(self.config.providers.validation_engine.fallback),
            ),
        )

    @staticmethod
    def _binding(provider_id: str | None) -> ProviderBinding | None:
        if provider_id in {None, ""}:
            return None
        return ProviderBinding(provider_id)

    def _payload(self, record: SessionContextRecord) -> dict[str, object]:
        bundle = get_policy_bundle(record.active_policy, run_mode=record.run_mode)
        session = record.to_dict()
        session["run_mode"] = record.run_mode.value
        return {
            "session": session,
            "policy_bundle": bundle.to_dict(),
        }

    @staticmethod
    def _session_uses_mock_provider(record: SessionContextRecord) -> bool:
        bindings = record.provider_bindings.market_data
        provider_ids = [
            binding.provider_id
            for binding in (bindings.primary, bindings.fallback)
            if binding is not None and binding.provider_id
        ]
        return any(".mock" in provider_id or provider_id.endswith("_mock") for provider_id in provider_ids)


__all__ = [
    "SessionService",
]
