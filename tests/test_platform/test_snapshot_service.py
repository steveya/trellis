"""Unit tests for governed reproducibility-bundle persistence."""

from __future__ import annotations


def test_snapshot_service_persists_bundle_and_attaches_run_artifact(tmp_path):
    from trellis.platform.context import build_execution_context
    from trellis.platform.runs import RunLedgerStore, build_run_record
    from trellis.platform.services.snapshot_service import SnapshotService
    from trellis.platform.storage import SnapshotRecord, SnapshotStore

    snapshot_store = SnapshotStore(tmp_path / "snapshots")
    run_ledger = RunLedgerStore(tmp_path / "runs")
    snapshot_store.save_snapshot(
        SnapshotRecord(
            snapshot_id="snapshot_seed_001",
            provider_id="market_data.test_live",
            as_of="2026-04-04",
            source="test_live",
            payload={"underlier_spots": {"AAPL": 123.0}},
            provenance={"source": "test_live"},
        )
    )
    run_ledger.save_run(
        build_run_record(
            run_id="run_seed_001",
            request_id="request_seed_001",
            status="succeeded",
            action="price_trade",
            execution_context=build_execution_context(
                session_id="sess_snapshot",
                market_source="treasury_gov",
            ),
            trade_identity={"semantic_id": "vanilla_option"},
            selected_model={"model_id": "vanilla_option_candidate", "version": "v1"},
            selected_engine={"engine_id": "pricing_engine.local", "version": "1"},
            market_snapshot_id="snapshot_seed_001",
            valuation_timestamp="2026-04-04",
            result_summary={"price": 12.34},
            provenance={"provider_id": "market_data.test_live"},
        )
    )

    service = SnapshotService(run_ledger=run_ledger, snapshot_store=snapshot_store)
    payload = service.persist_run(
        run_id="run_seed_001",
        tolerances={"price_abs": 1e-6},
        random_seed=11,
        calendars=("NYSE",),
    )

    assert payload["snapshot"]["source"] == "reproducibility_bundle"
    assert payload["snapshot"]["payload"]["run_id"] == "run_seed_001"
    assert payload["snapshot"]["payload"]["random_seed"] == 11
    assert payload["snapshot"]["payload"]["tolerances"]["price_abs"] == 1e-6
    assert any(
        artifact["artifact_kind"] == "reproducibility_bundle"
        for artifact in payload["run"]["artifacts"]
    )
