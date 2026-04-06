"""Transport-neutral platform-service bootstrap for MCP and other hosts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trellis.platform.models import ModelRegistryStore
from trellis.platform.providers import ProviderRegistry
from trellis.platform.runs import RunLedgerStore
from trellis.platform.services.audit_service import AuditService
from trellis.platform.services.model_service import ModelService
from trellis.platform.services.pricing_service import PricingService
from trellis.platform.services.provider_service import ProviderService
from trellis.platform.services.session_service import SessionService
from trellis.platform.services.snapshot_service import SnapshotService
from trellis.platform.services.trade_service import TradeService
from trellis.platform.services.validation_service import ValidationService
from trellis.platform.storage import (
    SessionContextStore,
    SnapshotStore,
    TrellisServerConfig,
    TrellisStatePaths,
    ValidationStore,
    build_state_paths,
    load_trellis_server_config,
)


@dataclass(frozen=True)
class PlatformServiceContainer:
    """Shared service container reused by MCP tools and host adapters."""

    config: TrellisServerConfig
    paths: TrellisStatePaths
    session_store: SessionContextStore
    snapshot_store: SnapshotStore
    validation_store: ValidationStore
    model_registry: ModelRegistryStore
    run_ledger: RunLedgerStore
    provider_registry: ProviderRegistry
    session_service: SessionService
    provider_service: ProviderService
    trade_service: TradeService
    model_service: ModelService
    validation_service: ValidationService
    audit_service: AuditService
    snapshot_service: SnapshotService
    pricing_service: PricingService


def bootstrap_platform_services(
    *,
    state_root: Path | str | None = None,
    config_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> PlatformServiceContainer:
    """Bootstrap one shared service container for transport-neutral server work."""
    config = load_trellis_server_config(
        config_path=config_path,
        state_root=state_root,
        env=env,
    )
    paths = build_state_paths(state_root=config.state_root).ensure()
    registry_store = ModelRegistryStore(paths.models_dir)
    run_ledger = RunLedgerStore(paths.runs_dir)
    snapshot_store = SnapshotStore(paths.snapshots_dir)
    resolved_provider_registry = provider_registry or ProviderRegistry()
    session_service = SessionService(
        store=SessionContextStore(paths.sessions_dir),
        config=config,
        provider_registry=resolved_provider_registry,
        snapshot_store=snapshot_store,
    )
    trade_service = TradeService()
    model_service = ModelService(registry=registry_store)
    validation_service = ValidationService(
        registry=registry_store,
        validation_store=ValidationStore(paths.validations_dir),
    )
    audit_service = AuditService(run_ledger=run_ledger)
    snapshot_service = SnapshotService(
        run_ledger=run_ledger,
        snapshot_store=snapshot_store,
    )
    return PlatformServiceContainer(
        config=config,
        paths=paths,
        session_store=session_service.store,
        snapshot_store=snapshot_store,
        validation_store=validation_service.validation_store,
        model_registry=registry_store,
        run_ledger=run_ledger,
        provider_registry=resolved_provider_registry,
        session_service=session_service,
        provider_service=ProviderService(
            provider_registry=resolved_provider_registry,
            session_service=session_service,
        ),
        trade_service=trade_service,
        model_service=model_service,
        validation_service=validation_service,
        audit_service=audit_service,
        snapshot_service=snapshot_service,
        pricing_service=PricingService(
            session_service=session_service,
            trade_service=trade_service,
            model_service=model_service,
            provider_registry=resolved_provider_registry,
            model_registry=registry_store,
            run_ledger=run_ledger,
            snapshot_store=snapshot_store,
            audit_service=audit_service,
        ),
    )


__all__ = [
    "PlatformServiceContainer",
    "bootstrap_platform_services",
]
