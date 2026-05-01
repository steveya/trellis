"""Cashflow expansion visitors for execution IR artifacts."""

from __future__ import annotations

from trellis.execution.ir import ContractExecutionIR, KnownCashflowObligation


def known_cashflow_obligations(ir: ContractExecutionIR) -> tuple[KnownCashflowObligation, ...]:
    """Return deterministic known-cashflow obligations in stable order."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    return tuple(
        sorted(
            (
                obligation
                for obligation in ir.obligations
                if isinstance(obligation, KnownCashflowObligation)
            ),
            key=lambda item: (item.payment_date, item.obligation_id),
        )
    )


__all__ = ["known_cashflow_obligations"]
