"""Governed reproducibility-bundle persistence over the canonical run ledger."""

from __future__ import annotations

from trellis.platform.runs import ArtifactReference
from trellis.platform.storage import SnapshotRecord


class SnapshotService:
    """Persist reproducibility bundles derived from canonical governed runs."""

    def __init__(self, *, run_ledger, snapshot_store):
        self.run_ledger = run_ledger
        self.snapshot_store = snapshot_store

    def persist_run(
        self,
        *,
        run_id: str,
        tolerances=None,
        random_seed=None,
        calendars=(),
    ) -> dict[str, object]:
        """Persist one reproducibility bundle derived from an existing run."""
        from trellis.mcp.errors import TrellisMcpError

        record = self.run_ledger.get_run(str(run_id or "").strip())
        if record is None:
            raise TrellisMcpError(
                code="unknown_run",
                message=f"Unknown run id: {run_id!r}",
                details={"run_id": str(run_id or "").strip()},
            )
        source_snapshot = (
            None
            if not record.market_snapshot_id
            else self.snapshot_store.get_snapshot(record.market_snapshot_id)
        )
        bundle_id = f"bundle_{record.run_id}"
        bundle_payload = {
            "bundle_type": "reproducibility_bundle",
            "run_id": record.run_id,
            "request_id": record.request_id,
            "status": record.status,
            "action": record.action,
            "run_mode": record.run_mode,
            "session_id": record.session_id,
            "policy_id": record.policy_id,
            "trade_identity": dict(record.trade_identity),
            "selected_model": dict(record.selected_model),
            "selected_engine": dict(record.selected_engine),
            "market_snapshot_id": record.market_snapshot_id,
            "market_snapshot": None if source_snapshot is None else source_snapshot.to_dict(),
            "valuation_timestamp": record.valuation_timestamp,
            "warnings": list(record.warnings),
            "result_summary": dict(record.result_summary),
            "validation_summary": dict(record.validation_summary),
            "policy_outcome": dict(record.policy_outcome),
            "provider_bindings": dict(record.provider_bindings),
            "provenance": dict(record.provenance),
            "artifacts": [artifact.to_dict() for artifact in record.artifacts],
            "tolerances": dict(tolerances or {}),
            "random_seed": random_seed,
            "calendars": list(calendars or ()),
        }
        snapshot_record = self.snapshot_store.save_snapshot(
            SnapshotRecord(
                snapshot_id=bundle_id,
                provider_id=(
                    ""
                    if source_snapshot is None
                    else source_snapshot.provider_id
                ) or str(record.provenance.get("provider_id", "")).strip(),
                as_of=record.valuation_timestamp,
                source="reproducibility_bundle",
                payload=bundle_payload,
                provenance={
                    "bundle_type": "reproducibility_bundle",
                    "run_id": record.run_id,
                    "source_run_snapshot_id": record.market_snapshot_id,
                },
            )
        )
        updated_run = self.run_ledger.attach_artifacts(
            record.run_id,
            (
                ArtifactReference(
                    artifact_id=bundle_id,
                    artifact_kind="reproducibility_bundle",
                    uri=f"trellis://market-snapshots/{bundle_id}",
                    role="replay",
                ),
            ),
        )
        return {
            "snapshot": snapshot_record.to_dict(),
            "bundle_uri": f"trellis://market-snapshots/{snapshot_record.snapshot_id}",
            "run": updated_run.to_dict(),
        }


__all__ = [
    "SnapshotService",
]
