"""Tests for explicit external-document request boundaries."""

from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import pytest


FPML_TEXT = "<dataDocument xmlns='http://www.fpml.org/FpML-5/confirmation'/>"
FPML_FIXTURES = Path(__file__).parents[1] / "test_io" / "fixtures" / "fpml"
VALID_FPML = (FPML_FIXTURES / "confirmation_5_13_swap.xml").read_bytes()
NORMALIZABLE_FPML = (
    FPML_FIXTURES / "confirmation_5_13_fixed_float_swap.xml"
).read_bytes()
NORMALIZABLE_SWAPTION_FPML = (
    FPML_FIXTURES / "confirmation_5_13_european_swaption.xml"
).read_bytes()


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


def test_imported_document_preflight_admits_a_coherent_fpml_request():
    from trellis.agent.imported_documents import imported_document_blocker_report
    from trellis.agent.platform_requests import make_fpml_request

    request = make_fpml_request(
        VALID_FPML,
        source_view="confirmation",
        source_version="5-13",
    )

    report = imported_document_blocker_report(
        request.imported_document,
        trade_envelope=request.trade_envelope,
    )

    assert report.blockers == ()
    assert report.should_block is False
    assert report.summary == "FpML request preflight admitted."


def test_compile_incomplete_fpml_swap_blocks_on_missing_product_economics(
    monkeypatch,
):
    import trellis.agent.platform_requests as platform_requests

    request = platform_requests.make_fpml_request(
        VALID_FPML,
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
    assert compiled.execution_plan.reason == "fpml_import_rejected"
    assert compiled.execution_plan.requires_build is False
    assert compiled.execution_plan.requested_outputs == ()
    assert _blocker_ids(compiled) == (
        "missing_contract_field:fpml_swap_stream",
    )
    assert compiled.import_report.status == "blocked"
    assert compiled.import_report.profile.id == "fpml_5_13_confirmation"
    assert compiled.import_report.trade.product_names == ("swap",)
    assert compiled.import_report.trade_envelope.trade_id == "TRADE-001"
    assert compiled.product_ir is None
    assert compiled.semantic_contract is None
    assert compiled.generation_plan is None
    assert compiled.validation_contract is None
    assert "imported_document" not in compiled.request.metadata


def test_compile_fpml_fixed_float_swap_uses_structural_execution_ir(monkeypatch):
    import trellis.agent.platform_requests as platform_requests
    import trellis.io.fpml.normalizer as fpml_normalizer
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty
    from trellis.core.payoff import ExecutionBackedPayoff

    request = platform_requests.make_fpml_request(
        NORMALIZABLE_FPML,
        source_view="confirmation",
        source_version="5-13",
        trade_envelope=TradeEnvelope(
            source_format="fpml",
            source_view="confirmation",
            source_version="5-13",
            parties=(TradeParty("PARTY-A", role="valuation_party"),),
        ),
        settlement=date(2025, 1, 15),
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("generic semantic text parsing must not run for FpML")

    monkeypatch.setattr(platform_requests, "_draft_semantic_contract", _unexpected)
    monkeypatch.setattr(platform_requests, "decompose_to_ir", _unexpected)
    monkeypatch.setattr(fpml_normalizer, "inspect_fpml_document", _unexpected)

    compiled = platform_requests.compile_platform_request(request)

    assert compiled.execution_plan.action == "price_normalized_payoff"
    assert compiled.execution_plan.reason == "normalized_external_contract"
    assert compiled.execution_plan.requires_build is False
    assert compiled.execution_plan.route_method == "structural_execution_ir"
    assert compiled.blocker_report is None
    assert compiled.import_report.status == "normalized"
    assert compiled.import_report.economic_identity.startswith("static_leg:v1:")
    assert isinstance(compiled.request.instrument, ExecutionBackedPayoff)
    execution_ir = compiled.request.instrument.execution_ir
    assert execution_ir.source_track.product_family == "fixed_float_swap"
    assert dict(execution_ir.source_track.source_metadata) == {
        "static_leg_lowering_declaration_id": "static_leg_fixed_float_swap",
        "validation_bundle_id": "static_leg_fixed_float_swap_contract",
        "requested_method": "",
        "callable_ref": "trellis.instruments.swap.SwapPayoff",
    }
    assert compiled.request.metadata["external_contract"]["economic_identity"] == (
        compiled.import_report.economic_identity
    )


def test_compile_fpml_european_swaption_uses_existing_contract_ir_route(monkeypatch):
    import trellis.agent.platform_requests as platform_requests
    from trellis.agent.contract_ir_solver_compiler import ContractIRPricingPayoff
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty

    request = platform_requests.make_fpml_request(
        NORMALIZABLE_SWAPTION_FPML,
        source_view="confirmation",
        source_version="5-13",
        trade_envelope=TradeEnvelope(
            source_format="fpml",
            source_view="confirmation",
            source_version="5-13",
            parties=(TradeParty("PARTY-A", role="valuation_party"),),
        ),
        settlement=date(2025, 1, 15),
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("generic semantic text parsing must not run for FpML")

    monkeypatch.setattr(platform_requests, "_draft_semantic_contract", _unexpected)
    monkeypatch.setattr(platform_requests, "decompose_to_ir", _unexpected)

    compiled = platform_requests.compile_platform_request(request)

    assert compiled.execution_plan.action == "price_normalized_payoff"
    assert compiled.execution_plan.reason == "normalized_external_contract"
    assert compiled.execution_plan.requires_build is False
    assert compiled.execution_plan.route_method == "structural_contract_ir"
    assert compiled.blocker_report is None
    assert compiled.import_report.economic_identity.startswith("contract_ir:v1:")
    assert isinstance(compiled.request.instrument, ContractIRPricingPayoff)
    assert compiled.request.metadata["external_contract"] == {
        "source_format": "fpml",
        "economic_identity": compiled.import_report.economic_identity,
        "structural_declaration_id": "swaption_payer_black76_resolved_kernel",
        "validation_bundle_id": "rate_style_swaption_contract",
        "callable_ref": "trellis.models.rate_style_swaption.price_swaption_black76_raw",
        "mapping_provenance_count": len(compiled.import_report.mapping_provenance),
    }


@pytest.mark.parametrize(
    ("request_type", "expected_outputs"),
    (
        ("analytics", ("price", "dv01", "duration")),
        ("greeks", ("dv01", "duration", "convexity")),
    ),
)
def test_compile_fpml_defaults_non_price_request_outputs(
    request_type,
    expected_outputs,
):
    from trellis.agent.platform_requests import compile_platform_request, make_fpml_request
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty

    compiled = compile_platform_request(
        make_fpml_request(
            NORMALIZABLE_FPML,
            source_view="confirmation",
            source_version="5-13",
            trade_envelope=TradeEnvelope(
                source_format="fpml",
                source_view="confirmation",
                source_version="5-13",
                parties=(TradeParty("PARTY-A", role="valuation_party"),),
            ),
            settlement=date(2025, 1, 15),
            request_type=request_type,
        )
    )

    assert compiled.execution_plan.action == "price_normalized_payoff"
    assert compiled.request.requested_outputs == expected_outputs
    assert compiled.execution_plan.requested_outputs == expected_outputs


def test_compile_complete_fpml_swap_requests_valuation_party_clarification():
    from trellis.agent.platform_requests import compile_platform_request, make_fpml_request

    compiled = compile_platform_request(
        make_fpml_request(
            NORMALIZABLE_FPML,
            source_view="confirmation",
            source_version="5-13",
        )
    )

    assert compiled.execution_plan.action == "block"
    assert _blocker_ids(compiled) == (
        "missing_contract_field:fpml_valuation_party_id",
    )
    assert compiled.import_report.clarification.missing_fields == (
        "valuation_party_id",
    )


def test_compile_fpml_swap_requests_a_deterministic_valuation_date():
    from trellis.agent.platform_requests import compile_platform_request, make_fpml_request
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty

    compiled = compile_platform_request(
        make_fpml_request(
            NORMALIZABLE_FPML,
            source_view="confirmation",
            source_version="5-13",
            trade_envelope=TradeEnvelope(
                source_format="fpml",
                source_view="confirmation",
                source_version="5-13",
                parties=(TradeParty("PARTY-A", role="valuation_party"),),
            ),
        )
    )

    assert compiled.execution_plan.action == "block"
    assert _blocker_ids(compiled) == (
        "missing_contract_field:fpml_valuation_date",
    )


def test_compile_only_fpml_swap_remains_independent_of_valuation_date():
    from trellis.agent.platform_requests import compile_platform_request, make_fpml_request
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty

    compiled = compile_platform_request(
        make_fpml_request(
            NORMALIZABLE_FPML,
            source_view="confirmation",
            source_version="5-13",
            trade_envelope=TradeEnvelope(
                source_format="fpml",
                source_view="confirmation",
                source_version="5-13",
                parties=(TradeParty("PARTY-A", role="valuation_party"),),
            ),
            request_type="build",
        )
    )

    assert compiled.execution_plan.action == "compile_only"
    assert compiled.import_report.status == "normalized"


def test_compile_fpml_swap_blocks_seasoned_coupon_without_runtime_fixing_support():
    from trellis.agent.platform_requests import compile_platform_request, make_fpml_request
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty

    compiled = compile_platform_request(
        make_fpml_request(
            NORMALIZABLE_FPML,
            source_view="confirmation",
            source_version="5-13",
            trade_envelope=TradeEnvelope(
                source_format="fpml",
                source_view="confirmation",
                source_version="5-13",
                parties=(TradeParty("PARTY-A", role="valuation_party"),),
            ),
            settlement=date(2025, 7, 1),
        )
    )

    assert compiled.execution_plan.action == "block"
    assert _blocker_ids(compiled) == (
        "external_import:fpml_historical_fixing_runtime_unsupported",
    )


def test_compile_fpml_swap_rejects_multiple_valuation_parties_at_preflight():
    from trellis.agent.platform_requests import compile_platform_request, make_fpml_request
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty

    compiled = compile_platform_request(
        make_fpml_request(
            NORMALIZABLE_FPML,
            source_view="confirmation",
            source_version="5-13",
            trade_envelope=TradeEnvelope(
                source_format="fpml",
                source_view="confirmation",
                source_version="5-13",
                parties=(
                    TradeParty("PARTY-A", role="valuation_party"),
                    TradeParty("PARTY-B", role="valuation_party"),
                ),
            ),
        )
    )

    assert compiled.execution_plan.reason == "fpml_import_boundary"
    assert _blocker_ids(compiled) == (
        "contract_ambiguity:fpml_valuation_party_id",
    )
    assert compiled.import_report is None


def test_compile_fpml_request_carries_deterministic_import_rejection():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_fpml_request,
    )

    request = make_fpml_request(
        b"<dataDocument>",
        source_view="confirmation",
        source_version="5-13",
    )

    compiled = compile_platform_request(request)

    assert compiled.execution_plan.action == "block"
    assert compiled.execution_plan.reason == "fpml_import_rejected"
    assert _blocker_ids(compiled) == ("external_import:fpml_malformed_xml",)
    assert compiled.import_report.status == "blocked"
    assert compiled.import_report.document is None
    assert compiled.product_ir is None


def test_compile_fpml_request_rejects_parsed_provenance_conflicts():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_fpml_request,
    )
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty

    request = make_fpml_request(
        VALID_FPML,
        source_view="confirmation",
        source_version="5-13",
        trade_envelope=TradeEnvelope(
            source_format="fpml",
            source_view="confirmation",
            source_version="5-13",
            document_id="DOC-OTHER",
            trade_id="TRADE-OTHER",
            trade_date=date(2026, 7, 2),
            parties=(TradeParty("PARTY-C"),),
        ),
    )

    compiled = compile_platform_request(request)

    assert compiled.execution_plan.action == "block"
    assert compiled.execution_plan.reason == "fpml_import_rejected"
    assert _blocker_ids(compiled) == (
        "contract_conflict:fpml_document_id",
        "contract_conflict:fpml_trade_id",
        "contract_conflict:fpml_trade_date",
        "contract_conflict:fpml_parties",
    )
    assert compiled.import_report.status == "inspected"
    assert compiled.import_report.trade_envelope.document_id == "DOC-001"
    assert compiled.trade_envelope.document_id == "DOC-OTHER"


def test_compile_fpml_request_runs_preflight_before_xml_inspection(monkeypatch):
    import trellis.agent.platform_requests as platform_requests

    request = platform_requests.make_fpml_request(
        VALID_FPML,
        source_view=None,
        source_version=None,
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("XML inspection must not run before request preflight passes")

    monkeypatch.setattr(platform_requests, "inspect_fpml_document", _unexpected)
    monkeypatch.setattr(
        platform_requests,
        "_normalize_inspected_fpml_document",
        _unexpected,
    )

    compiled = platform_requests.compile_platform_request(request)

    assert compiled.execution_plan.reason == "fpml_import_boundary"
    assert compiled.import_report is None
    assert _blocker_ids(compiled) == (
        "missing_contract_field:fpml_source_view",
        "missing_contract_field:fpml_source_version",
    )


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
