"""Tests for MCP-oriented platform storage and bootstrap config."""

from __future__ import annotations

from pathlib import Path


def test_load_server_config_prefers_explicit_state_root_and_writes_default_config(tmp_path):
    from trellis.platform.storage import load_trellis_server_config

    state_root = tmp_path / "custom_state"

    config = load_trellis_server_config(state_root=state_root)

    assert config.state_root == state_root
    assert config.config_path == state_root / "config" / "server.yaml"
    assert config.config_path.exists()
    assert config.defaults.run_mode == "research"
    assert config.defaults.require_explicit_provider_binding is True
    assert config.observability.persist_run_ledgers is True


def test_load_server_config_honors_environment_override(tmp_path, monkeypatch):
    from trellis.platform.storage import load_trellis_server_config

    state_root = tmp_path / "env_state"
    monkeypatch.setenv("TRELLIS_STATE_ROOT", str(state_root))

    config = load_trellis_server_config()

    assert config.state_root == state_root
    assert config.config_path.exists()


def test_session_and_snapshot_stores_round_trip_records(tmp_path):
    from trellis.platform.context import ProviderBinding, ProviderBindingSet, ProviderBindings, RunMode
    from trellis.platform.storage import (
        SessionContextRecord,
        SessionContextStore,
        SnapshotRecord,
        SnapshotStore,
    )

    session_store = SessionContextStore(base_dir=tmp_path / "sessions")
    snapshot_store = SnapshotStore(base_dir=tmp_path / "snapshots")

    session = session_store.save_session(
        SessionContextRecord(
            session_id="session_001",
            run_mode=RunMode.RESEARCH,
            provider_bindings=ProviderBindings(
                market_data=ProviderBindingSet(
                    primary=ProviderBinding("market_data.treasury_gov"),
                )
            ),
            active_policy="policy_bundle.research.default",
            require_provider_disclosure=True,
        )
    )
    snapshot = snapshot_store.save_snapshot(
        SnapshotRecord(
            snapshot_id="snapshot_001",
            provider_id="market_data.treasury_gov",
            as_of="2026-04-04",
            source="treasury_gov",
            payload={
                "discount_curves": {"usd_ois": {"type": "yield_curve"}},
                "underlier_spots": {"AAPL": 185.5},
            },
            provenance={"provider_id": "market_data.treasury_gov"},
        )
    )

    assert session_store.get_session("session_001") == session
    assert snapshot_store.get_snapshot("snapshot_001") == snapshot
    assert session.execution_context().run_mode.value == "research"
    assert snapshot.payload["underlier_spots"]["AAPL"] == 185.5


def test_bootstrap_platform_services_uses_canonical_model_and_run_stores(tmp_path):
    from trellis.platform.services import bootstrap_platform_services

    services = bootstrap_platform_services(state_root=tmp_path / "state")

    assert services.config.state_root == tmp_path / "state"
    assert services.model_registry.base_dir == tmp_path / "state" / "models"
    assert services.run_ledger.base_dir == tmp_path / "state" / "runs"
    assert services.session_store.base_dir == tmp_path / "state" / "sessions"
    assert services.snapshot_store.base_dir == tmp_path / "state" / "snapshots"
    assert services.provider_registry.get_provider("market_data.mock").provider_id == "market_data.mock"

