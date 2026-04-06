"""Unit tests for governed session-service market snapshot activation."""

from __future__ import annotations

import pytest


def test_activate_market_snapshot_requires_persisted_snapshot_id(tmp_path):
    from trellis.mcp.errors import TrellisMcpError
    from trellis.platform.services import bootstrap_platform_services

    services = bootstrap_platform_services(state_root=tmp_path / "state")

    with pytest.raises(TrellisMcpError) as excinfo:
        services.session_service.activate_market_snapshot(
            session_id="sess_missing_snapshot",
            snapshot_id="snapshot_missing_001",
        )

    assert excinfo.value.code == "unknown_market_snapshot"


def test_activate_market_snapshot_uses_persisted_snapshot_provider_binding(tmp_path):
    from trellis.platform.services import bootstrap_platform_services
    from trellis.platform.storage import SnapshotRecord

    services = bootstrap_platform_services(state_root=tmp_path / "state")
    services.snapshot_store.save_snapshot(
        SnapshotRecord(
            snapshot_id="snapshot_import_001",
            provider_id="market_data.file_import",
            as_of="2026-04-04",
            source="file_import",
            payload={"bundle_type": "file_import_bundle"},
            provenance={"provider_id": "market_data.file_import"},
        )
    )

    payload = services.session_service.activate_market_snapshot(
        session_id="sess_imported_snapshot",
        snapshot_id="snapshot_import_001",
        provider_id="market_data.not_used",
    )

    assert (
        payload["session"]["provider_bindings"]["market_data"]["primary"]["provider_id"]
        == "market_data.file_import"
    )
    assert payload["session"]["metadata"]["active_market_snapshot_id"] == "snapshot_import_001"
    assert (
        payload["session"]["metadata"]["active_market_snapshot_provider_id"]
        == "market_data.file_import"
    )

    persisted = services.session_service.ensure_record("sess_imported_snapshot")
    assert (
        persisted.provider_bindings.market_data.primary.provider_id
        == "market_data.file_import"
    )
