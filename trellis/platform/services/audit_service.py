"""Governed run-summary and audit retrieval for transport-neutral MCP workflows."""

from __future__ import annotations

from trellis.platform.runs import RunLedgerStore


class AuditService:
    """Expose canonical run records and audit bundles through one read surface."""

    def __init__(self, *, run_ledger: RunLedgerStore):
        self.run_ledger = run_ledger

    def get_run(self, *, run_id: str) -> dict[str, object]:
        """Return the canonical run record for one persisted governed run."""
        record = self._require_run(run_id)
        return {"run": record.to_dict()}

    def get_audit(self, *, run_id: str) -> dict[str, object]:
        """Return the canonical audit bundle for one persisted governed run."""
        record = self._require_run(run_id)
        bundle = self.run_ledger.build_audit_bundle(record.run_id)
        return {
            "run_id": record.run_id,
            "audit": bundle.to_dict(),
        }

    def _require_run(self, run_id: str):
        from trellis.mcp.errors import TrellisMcpError

        normalized_run_id = str(run_id or "").strip()
        if not normalized_run_id:
            raise TrellisMcpError(
                code="unknown_run",
                message="Run id is required.",
            )
        record = self.run_ledger.get_run(normalized_run_id)
        if record is None:
            raise TrellisMcpError(
                code="unknown_run",
                message=f"Unknown run id: {normalized_run_id!r}",
                details={"run_id": normalized_run_id},
            )
        return record


__all__ = [
    "AuditService",
]
