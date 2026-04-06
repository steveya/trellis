"""Unit tests for governed reproducibility-bundle persistence."""

from __future__ import annotations

from datetime import date
import json
import math

import pytest


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


def test_snapshot_service_imports_file_manifest_warns_and_rehydrates(tmp_path):
    from trellis.platform.runs import RunLedgerStore
    from trellis.platform.services.snapshot_service import SnapshotService
    from trellis.platform.storage import SnapshotStore

    (tmp_path / "usd_ois.yaml").write_text(
        "kind: flat\nrate: 0.045\n",
        encoding="utf-8",
    )
    (tmp_path / "usd_rates_atm.yaml").write_text(
        "kind: flat\nvol: 0.20\nsource_kind: synthetic\n",
        encoding="utf-8",
    )
    (tmp_path / "issuer_credit.yaml").write_text(
        "kind: flat\nhazard_rate: 0.02\n",
        encoding="utf-8",
    )
    (tmp_path / "eurusd.json").write_text(
        json.dumps({"spot": 1.10, "domestic": "USD", "foreign": "EUR"}),
        encoding="utf-8",
    )
    (tmp_path / "spots.yaml").write_text(
        "SPX: 5100.0\n",
        encoding="utf-8",
    )
    (tmp_path / "sofr_fixings.csv").write_text(
        "date,value\n2026-03-30,0.041\n2026-03-31,0.042\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "snapshot.yaml"
    manifest_path.write_text(
        "\n".join(
            (
                "as_of: 2026-04-01",
                "source: trader_upload",
                "stale_after_days: 2",
                "expected_component_families:",
                "  - discount_curves",
                "  - forecast_curves",
                "  - fixing_histories",
                "discount_curves:",
                "  usd_ois: usd_ois.yaml",
                "vol_surfaces:",
                "  usd_rates_atm: usd_rates_atm.yaml",
                "credit_curves:",
                "  issuer: issuer_credit.yaml",
                "fx_rates:",
                "  EURUSD: eurusd.json",
                "underlier_spots:",
                "  file: spots.yaml",
                "fixing_histories:",
                "  SOFR: sofr_fixings.csv",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot_store = SnapshotStore(tmp_path / "snapshots")
    service = SnapshotService(
        run_ledger=RunLedgerStore(tmp_path / "runs"),
        snapshot_store=snapshot_store,
    )

    imported = service.import_files(
        manifest_path=manifest_path,
        reference_date="2026-04-04",
    )

    assert imported["snapshot"]["source"] == "file_import"
    assert imported["snapshot"]["provider_id"] == "market_data.file_import"
    assert any("stale" in warning.lower() for warning in imported["warnings"])
    assert any("forecast_curves" in warning for warning in imported["warnings"])
    assert any("defaulted" in warning.lower() and "discount" in warning.lower() for warning in imported["warnings"])
    assert any("synthetic" in warning.lower() and "usd_rates_atm" in warning for warning in imported["warnings"])

    snapshot = service.load_market_snapshot(imported["snapshot"]["snapshot_id"])
    assert snapshot.discount_curve().discount(1.0) == pytest.approx(math.exp(-0.045))
    assert snapshot.vol_surface().black_vol(1.0, 0.05) == pytest.approx(0.20)
    assert snapshot.credit_curve().hazard_rate(5.0) == pytest.approx(0.02)
    assert snapshot.fx_rates["EURUSD"].spot == pytest.approx(1.10)
    assert snapshot.underlier_spot("SPX") == pytest.approx(5100.0)
    assert snapshot.default_fixing_history == "SOFR"
    assert snapshot.fixing_history("SOFR")[date.fromisoformat("2026-03-30")] == pytest.approx(0.041)
    assert snapshot.fixing_history()[date.fromisoformat("2026-03-31")] == pytest.approx(0.042)

    persisted = snapshot_store.get_snapshot(imported["snapshot"]["snapshot_id"])
    assert persisted is not None
    assert persisted.payload["summary"]["discount_curves"] == ["usd_ois"]
    assert persisted.payload["summary"]["fixing_histories"] == ["SOFR"]


def test_snapshot_service_resolves_named_component_request_with_warnings(tmp_path):
    from trellis.platform.runs import RunLedgerStore
    from trellis.platform.services.snapshot_service import SnapshotService
    from trellis.platform.storage import SnapshotStore

    (tmp_path / "usd_ois.yaml").write_text(
        "kind: flat\nrate: 0.045\n",
        encoding="utf-8",
    )
    (tmp_path / "sofr_fixings.csv").write_text(
        "date,value\n2026-03-30,0.041\n2026-03-31,0.042\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "snapshot.yaml"
    manifest_path.write_text(
        "\n".join(
            (
                "as_of: 2026-04-01",
                "source: trader_upload",
                "stale_after_days: 2",
                "defaults:",
                "  discount_curve: usd_ois",
                "  fixing_history: SOFR",
                "discount_curves:",
                "  usd_ois: usd_ois.yaml",
                "fixing_histories:",
                "  SOFR: sofr_fixings.csv",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot_store = SnapshotStore(tmp_path / "snapshots")
    service = SnapshotService(
        run_ledger=RunLedgerStore(tmp_path / "runs"),
        snapshot_store=snapshot_store,
    )
    imported = service.import_files(
        manifest_path=manifest_path,
        reference_date="2026-04-04",
    )

    result = service.resolve_market_state(
        snapshot_id=imported["snapshot"]["snapshot_id"],
        settlement="2026-04-04",
        selected_components={
            "discount_curve": "usd_ois",
            "fixing_history": "SOFR",
        },
        reference_date="2026-04-04",
    )

    assert result.selection_status == "parsed"
    assert result.selected_components == {
        "discount_curve": "usd_ois",
        "fixing_history": "SOFR",
    }
    assert result.selected_curve_names == {
        "discount_curve": "usd_ois",
        "fixing_history": "SOFR",
    }
    assert any("stale" in warning.lower() for warning in result.warnings)
    assert result.market_state_object.fixing_history("SOFR")[date.fromisoformat("2026-03-31")] == pytest.approx(0.042)
