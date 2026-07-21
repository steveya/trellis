"""Tests for explicit external-document request boundaries."""

from __future__ import annotations

from datetime import date
import hashlib

import pytest


FPML_TEXT = "<dataDocument xmlns='http://www.fpml.org/FpML-5/confirmation'/>"


def _blocker_ids(compiled) -> tuple[str, ...]:
    report = compiled.blocker_report
    return tuple(blocker.id for blocker in report.blockers)


def test_imported_document_payload_canonicalizes_text_and_summarizes_without_content():
    from trellis.agent.imported_documents import (
        ImportedDocumentPayload,
        imported_document_summary,
    )

    payload = ImportedDocumentPayload(
        source_format=" FpML ",
        content=FPML_TEXT,
        media_type="application/xml",
        declared_view="confirmation",
        declared_version="5-13",
        source_reference="s3://opaque-bucket/trade.xml",
    )

    expected_bytes = FPML_TEXT.encode("utf-8")
    summary = imported_document_summary(payload)

    assert payload.source_format == "fpml"
    assert payload.content == expected_bytes
    assert summary == {
        "source_format": "fpml",
        "media_type": "application/xml",
        "declared_view": "confirmation",
        "declared_version": "5-13",
        "source_reference": "s3://opaque-bucket/trade.xml",
        "has_inline_content": True,
        "byte_length": len(expected_bytes),
        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
    }
    assert FPML_TEXT not in repr(summary)
    assert FPML_TEXT not in repr(payload)
    with pytest.raises(TypeError):
        payload.content[0] = 1


def test_make_fpml_request_preserves_request_intent_outside_document_payload():
    from trellis.agent.platform_requests import make_fpml_request

    market_snapshot = object()
    request = make_fpml_request(
        FPML_TEXT,
        source_view="confirmation",
        source_version="5-13",
        source_reference="trade-store:TRADE-001",
        market_snapshot=market_snapshot,
        settlement=date(2026, 7, 1),
        model="hull_white",
        measures=["price", "delta"],
        measure_context={"bump_size": 0.0001},
        request_type="price",
        metadata={"validation": "standard", "expected_outcome": "price"},
    )

    assert request.entry_point == "fpml"
    assert request.request_type == "price"
    assert request.description is None
    assert request.instrument_type is None
    assert request.market_snapshot is market_snapshot
    assert request.settlement == date(2026, 7, 1)
    assert request.model == "hull_white"
    assert request.requested_outputs == ("price", "delta")
    assert request.measure_specs == ("price", "delta")
    assert request.measure_context == {"bump_size": 0.0001}
    assert request.metadata == {
        "expected_outcome": "price",
        "validation": "standard",
    }
    assert request.trade_envelope.source_format == "fpml"
    assert request.trade_envelope.source_view == "confirmation"
    assert request.trade_envelope.source_version == "5-13"
    assert request.imported_document.declared_view == "confirmation"
    assert request.imported_document.declared_version == "5-13"


def test_compile_fpml_request_blocks_at_dedicated_importer_without_semantic_text_parsing(
    monkeypatch,
):
    import trellis.agent.platform_requests as platform_requests

    request = platform_requests.make_fpml_request(
        FPML_TEXT,
        source_view="confirmation",
        source_version="5-13",
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("generic semantic text parsing must not run for FpML")

    monkeypatch.setattr(platform_requests, "_draft_semantic_contract", _unexpected)
    monkeypatch.setattr(platform_requests, "decompose_to_ir", _unexpected)

    compiled = platform_requests.compile_platform_request(request)

    assert compiled.request is request
    assert compiled.imported_document is request.imported_document
    assert compiled.trade_envelope is request.trade_envelope
    assert compiled.execution_plan.action == "block"
    assert compiled.execution_plan.reason == "fpml_import_boundary"
    assert compiled.execution_plan.requires_build is False
    assert compiled.execution_plan.requested_outputs == ()
    assert _blocker_ids(compiled) == ("external_import:fpml_importer_unavailable",)
    assert compiled.product_ir is None
    assert compiled.semantic_contract is None
    assert compiled.generation_plan is None
    assert compiled.validation_contract is None
    assert "imported_document" not in compiled.request.metadata


@pytest.mark.parametrize(
    ("content", "source_reference", "source_view", "source_version", "expected_ids"),
    [
        (
            None,
            None,
            "confirmation",
            "5-13",
            ("missing_contract_field:fpml_inline_content",),
        ),
        (
            None,
            "https://example.invalid/trade.xml",
            "confirmation",
            "5-13",
            ("external_import:source_reference_resolution_unsupported",),
        ),
        (
            FPML_TEXT,
            None,
            None,
            None,
            (
                "missing_contract_field:fpml_source_view",
                "missing_contract_field:fpml_source_version",
            ),
        ),
    ],
)
def test_compile_fpml_request_reports_exact_incomplete_declaration_blockers(
    content,
    source_reference,
    source_view,
    source_version,
    expected_ids,
):
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_fpml_request,
    )

    request = make_fpml_request(
        content,
        source_view=source_view,
        source_version=source_version,
        source_reference=source_reference,
    )

    compiled = compile_platform_request(request)

    assert compiled.execution_plan.action == "block"
    assert _blocker_ids(compiled) == expected_ids


def test_compile_fpml_request_blocks_envelope_declaration_conflicts():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_fpml_request,
    )
    from trellis.agent.trade_envelope import TradeEnvelope

    request = make_fpml_request(
        FPML_TEXT,
        source_view="confirmation",
        source_version="5-13",
        trade_envelope=TradeEnvelope(
            source_format="trellis_contract",
            source_view="recordkeeping",
            source_version="5-12",
        ),
    )

    compiled = compile_platform_request(request)

    assert _blocker_ids(compiled) == (
        "contract_conflict:fpml_source_format",
        "contract_conflict:fpml_source_view",
        "contract_conflict:fpml_source_version",
    )


def test_platform_request_requires_typed_imported_document_payload():
    from trellis.agent.platform_requests import PlatformRequest

    with pytest.raises(TypeError, match="imported_document"):
        PlatformRequest(
            request_id="request-1",
            request_type="price",
            entry_point="fpml",
            imported_document={"source_format": "fpml", "content": FPML_TEXT},
        )


def test_compile_raw_fpml_request_requires_payload_and_trade_envelope():
    from trellis.agent.imported_documents import ImportedDocumentPayload
    from trellis.agent.platform_requests import (
        PlatformRequest,
        compile_platform_request,
    )

    missing_payload = compile_platform_request(
        PlatformRequest(
            request_id="request-missing-payload",
            request_type="price",
            entry_point="fpml",
        )
    )
    missing_envelope = compile_platform_request(
        PlatformRequest(
            request_id="request-missing-envelope",
            request_type="price",
            entry_point="fpml",
            imported_document=ImportedDocumentPayload(
                source_format="fpml",
                content=FPML_TEXT,
                declared_view="confirmation",
                declared_version="5-13",
            ),
        )
    )

    assert _blocker_ids(missing_payload) == (
        "missing_contract_field:fpml_payload",
    )
    assert _blocker_ids(missing_envelope) == (
        "missing_contract_field:fpml_trade_envelope",
    )


def test_platform_request_namespace_exports_fpml_ingress_surface():
    from trellis.agent.imported_documents import ImportedDocumentPayload
    from trellis.agent.platform_requests import make_fpml_request
    from trellis.platform.requests import (
        ImportedDocumentPayload as PlatformImportedDocumentPayload,
    )
    from trellis.platform.requests import make_fpml_request as platform_make_fpml_request

    assert PlatformImportedDocumentPayload is ImportedDocumentPayload
    assert platform_make_fpml_request is make_fpml_request
