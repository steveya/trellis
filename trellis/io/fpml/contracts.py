"""Immutable contracts for bounded FpML document inspection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.agent.trade_envelope import TradeEnvelope, trade_envelope_summary


@dataclass(frozen=True)
class FpMLProfile:
    """One explicitly admitted FpML version/view/root profile."""

    version: str
    view: str
    namespace: str
    root_names: tuple[str, ...]

    @property
    def id(self) -> str:
        return f"fpml_{self.version.replace('-', '_')}_{self.view}"


FPML_5_13_CONFIRMATION = FpMLProfile(
    version="5-13",
    view="confirmation",
    namespace="http://www.fpml.org/FpML-5/confirmation",
    root_names=("dataDocument",),
)

SUPPORTED_FPML_PROFILES = (FPML_5_13_CONFIRMATION,)


@dataclass(frozen=True)
class FpMLInspectionLimits:
    """Hard resource limits applied before product inspection."""

    max_document_bytes: int = 2_000_000
    max_elements: int = 20_000
    max_depth: int = 128

    def __post_init__(self) -> None:
        for field_name in (
            "max_document_bytes",
            "max_elements",
            "max_depth",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


DEFAULT_FPML_INSPECTION_LIMITS = FpMLInspectionLimits()


@dataclass(frozen=True)
class FpMLImportBlocker:
    """One deterministic reason an FpML document cannot advance."""

    id: str
    category: str
    summary: str
    missing_fields: tuple[str, ...] = ()
    ambiguous_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class FpMLClarification:
    """Caller-facing projection of missing or ambiguous import fields."""

    requires_clarification: bool
    missing_fields: tuple[str, ...] = ()
    ambiguous_fields: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class FpMLDocumentIdentity:
    """Bounded identity extracted from the FpML document root."""

    namespace: str | None
    view: str | None
    version: str | None
    root_name: str
    document_id: str | None
    trade_count: int
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class FpMLTradeIdentity:
    """Non-economic identity and direct product names for one trade."""

    element_id: str | None
    business_id: str | None
    trade_date: date | None
    party_ids: tuple[str, ...]
    product_names: tuple[str, ...]


@dataclass(frozen=True)
class FpMLImportReport:
    """Body-free result of bounded FpML document inspection."""

    status: str
    profile: FpMLProfile | None
    document: FpMLDocumentIdentity | None
    trade: FpMLTradeIdentity | None
    trade_envelope: TradeEnvelope | None
    blockers: tuple[FpMLImportBlocker, ...]
    clarification: FpMLClarification

    def __post_init__(self) -> None:
        if self.status not in {"inspected", "blocked"}:
            raise ValueError("FpML import status must be 'inspected' or 'blocked'")
        if self.status == "inspected" and self.blockers:
            raise ValueError("an inspected FpML report cannot carry blockers")
        if self.status == "blocked" and not self.blockers:
            raise ValueError("a blocked FpML report requires at least one blocker")


def fpml_import_report_summary(report: FpMLImportReport) -> dict[str, object]:
    """Return a stable serializable report without XML content or tree nodes."""

    if not isinstance(report, FpMLImportReport):
        raise TypeError("report must be an FpMLImportReport")
    profile = report.profile
    document = report.document
    trade = report.trade
    return {
        "status": report.status,
        "profile": (
            {
                "id": profile.id,
                "version": profile.version,
                "view": profile.view,
                "namespace": profile.namespace,
                "root_names": list(profile.root_names),
            }
            if profile is not None
            else None
        ),
        "document": (
            {
                "namespace": document.namespace,
                "view": document.view,
                "version": document.version,
                "root_name": document.root_name,
                "document_id": document.document_id,
                "trade_count": document.trade_count,
                "byte_length": document.byte_length,
                "sha256": document.sha256,
            }
            if document is not None
            else None
        ),
        "trade": (
            {
                "element_id": trade.element_id,
                "business_id": trade.business_id,
                "trade_date": trade.trade_date.isoformat() if trade.trade_date else None,
                "party_ids": list(trade.party_ids),
                "product_names": list(trade.product_names),
            }
            if trade is not None
            else None
        ),
        "trade_envelope": trade_envelope_summary(report.trade_envelope),
        "blockers": [
            {
                "id": blocker.id,
                "category": blocker.category,
                "summary": blocker.summary,
                "missing_fields": list(blocker.missing_fields),
                "ambiguous_fields": list(blocker.ambiguous_fields),
            }
            for blocker in report.blockers
        ],
        "clarification": {
            "requires_clarification": report.clarification.requires_clarification,
            "missing_fields": list(report.clarification.missing_fields),
            "ambiguous_fields": list(report.clarification.ambiguous_fields),
            "messages": list(report.clarification.messages),
        },
    }


__all__ = [
    "DEFAULT_FPML_INSPECTION_LIMITS",
    "FPML_5_13_CONFIRMATION",
    "SUPPORTED_FPML_PROFILES",
    "FpMLClarification",
    "FpMLDocumentIdentity",
    "FpMLImportBlocker",
    "FpMLImportReport",
    "FpMLInspectionLimits",
    "FpMLProfile",
    "FpMLTradeIdentity",
    "fpml_import_report_summary",
]
