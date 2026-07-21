"""Typed external-document payloads and pre-import blocker planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from trellis.agent.blocker_planning import BlockerReport, PrimitiveBlocker
from trellis.agent.trade_envelope import TradeEnvelope


def _optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = (value or "").strip()
    return text or None


@dataclass(frozen=True)
class ImportedDocumentPayload:
    """Immutable inline document bytes plus caller-declared import identity."""

    source_format: str
    content: bytes | str | bytearray | memoryview | None = field(
        default=b"",
        repr=False,
    )
    media_type: str = "application/xml"
    declared_view: str | None = None
    declared_version: str | None = None
    source_reference: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_format, str):
            raise TypeError("imported document source_format must be a string")
        source_format = self.source_format.strip().lower()
        if not source_format:
            raise ValueError("imported document source_format is required")
        object.__setattr__(self, "source_format", source_format)

        content = self.content
        if content is None:
            content_bytes = b""
        elif isinstance(content, str):
            content_bytes = content.encode("utf-8")
        elif isinstance(content, (bytes, bytearray, memoryview)):
            content_bytes = bytes(content)
        else:
            raise TypeError("imported document content must be bytes, text, or None")
        object.__setattr__(self, "content", content_bytes)

        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise TypeError("imported document media_type must be a non-empty string")
        object.__setattr__(self, "media_type", self.media_type.strip().lower())
        for field_name in (
            "declared_view",
            "declared_version",
            "source_reference",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_text(
                    getattr(self, field_name),
                    field_name=f"imported document {field_name}",
                ),
            )


def imported_document_summary(
    payload: ImportedDocumentPayload | None,
) -> dict[str, object] | None:
    """Return content identity and declarations without exposing document bytes."""

    if payload is None:
        return None
    if not isinstance(payload, ImportedDocumentPayload):
        raise TypeError("payload must be an ImportedDocumentPayload")
    content = bytes(payload.content)
    return {
        "source_format": payload.source_format,
        "media_type": payload.media_type,
        "declared_view": payload.declared_view,
        "declared_version": payload.declared_version,
        "source_reference": payload.source_reference,
        "has_inline_content": bool(content),
        "byte_length": len(content),
        "sha256": hashlib.sha256(content).hexdigest() if content else None,
    }


def imported_document_blocker_report(
    payload: ImportedDocumentPayload | None,
    *,
    trade_envelope: TradeEnvelope | None,
) -> BlockerReport:
    """Plan deterministic request-boundary blockers before document parsing."""

    blockers: list[PrimitiveBlocker] = []
    if payload is None:
        blockers.append(
            _blocker(
                "missing_contract_field:fpml_payload",
                category="contract_gap",
                summary="The FpML request has no imported document payload.",
            )
        )
        return _report(blockers)

    if payload.source_format != "fpml":
        blockers.append(
            _blocker(
                "external_import:unsupported_source_format",
                category="unsupported_contract",
                summary=(
                    "The FpML request boundary only admits source_format='fpml'."
                ),
            )
        )
    if not payload.content:
        if payload.source_reference:
            blockers.append(
                _blocker(
                    "external_import:source_reference_resolution_unsupported",
                    category="unsupported_operation",
                    summary=(
                        "The request layer does not fetch or resolve external "
                        "document references; provide inline FpML content."
                    ),
                )
            )
        else:
            blockers.append(
                _blocker(
                    "missing_contract_field:fpml_inline_content",
                    category="contract_gap",
                    summary="The FpML request requires inline document content.",
                )
            )
    if not payload.declared_view:
        blockers.append(
            _blocker(
                "missing_contract_field:fpml_source_view",
                category="contract_gap",
                summary="The FpML request requires a declared source view.",
            )
        )
    if not payload.declared_version:
        blockers.append(
            _blocker(
                "missing_contract_field:fpml_source_version",
                category="contract_gap",
                summary="The FpML request requires a declared source version.",
            )
        )

    if trade_envelope is None:
        blockers.append(
            _blocker(
                "missing_contract_field:fpml_trade_envelope",
                category="contract_gap",
                summary="The FpML request requires a trade provenance envelope.",
            )
        )
    else:
        if trade_envelope.source_format != payload.source_format:
            blockers.append(
                _blocker(
                    "contract_conflict:fpml_source_format",
                    category="contract_conflict",
                    summary="The FpML payload and trade envelope source formats differ.",
                )
            )
        if (
            trade_envelope.source_view
            and payload.declared_view
            and trade_envelope.source_view != payload.declared_view
        ):
            blockers.append(
                _blocker(
                    "contract_conflict:fpml_source_view",
                    category="contract_conflict",
                    summary="The FpML payload and trade envelope source views differ.",
                )
            )
        if (
            trade_envelope.source_version
            and payload.declared_version
            and trade_envelope.source_version != payload.declared_version
        ):
            blockers.append(
                _blocker(
                    "contract_conflict:fpml_source_version",
                    category="contract_conflict",
                    summary="The FpML payload and trade envelope source versions differ.",
                )
            )

    if blockers:
        return _report(blockers)
    return BlockerReport(
        blockers=(),
        should_block=False,
        summary="FpML request preflight admitted.",
    )


def fpml_import_blocker_report(report) -> BlockerReport:
    """Project bounded FpML inspection blockers onto the task blocker contract."""

    from trellis.io.fpml import FpMLImportReport

    if not isinstance(report, FpMLImportReport):
        raise TypeError("report must be an FpMLImportReport")
    blockers = [
        _blocker(
            blocker.id,
            category=blocker.category,
            summary=blocker.summary,
            target_package="trellis.io.fpml",
        )
        for blocker in report.blockers
    ]
    if not blockers:
        raise ValueError("an inspected FpML report has no import blockers")
    return _report(blockers)


def fpml_product_normalizer_blocker_report() -> BlockerReport:
    """Report the deliberate boundary after successful document inspection."""

    return _report(
        [
            _blocker(
                "external_import:fpml_product_normalizer_unavailable",
                category="implementation_gap",
                summary=(
                    "The FpML document passed bounded inspection, but its product "
                    "economics cannot yet be normalized into a Trellis contract."
                ),
                target_package="trellis.io.fpml.products",
            )
        ]
    )


def _blocker(
    blocker_id: str,
    *,
    category: str,
    summary: str,
    target_package: str | None = None,
) -> PrimitiveBlocker:
    return PrimitiveBlocker(
        id=blocker_id,
        category=category,
        primitive_kind="external_document_import",
        severity="high",
        summary=summary,
        target_package=target_package,
    )


def _report(blockers: list[PrimitiveBlocker]) -> BlockerReport:
    return BlockerReport(
        blockers=tuple(blockers),
        should_block=True,
        summary="; ".join(blocker.summary for blocker in blockers),
    )


__all__ = [
    "ImportedDocumentPayload",
    "fpml_import_blocker_report",
    "fpml_product_normalizer_blocker_report",
    "imported_document_blocker_report",
    "imported_document_summary",
]
