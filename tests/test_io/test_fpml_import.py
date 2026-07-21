"""Security and contract tests for bounded FpML document inspection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


FIXTURES = Path(__file__).with_name("fixtures") / "fpml"
CONFIRMATION_NAMESPACE = "http://www.fpml.org/FpML-5/confirmation"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _blocker_ids(report) -> tuple[str, ...]:
    return tuple(blocker.id for blocker in report.blockers)


def _document(body: str, *, version: str = "5-13", root: str = "dataDocument") -> bytes:
    return (
        f'<{root} xmlns="{CONFIRMATION_NAMESPACE}" fpmlVersion="{version}">'
        f"{body}"
        f"</{root}>"
    ).encode("utf-8")


def test_inspect_fpml_document_recognizes_profile_and_extracts_provenance():
    from trellis.io.fpml import (
        FPML_5_13_CONFIRMATION,
        fpml_import_report_summary,
        inspect_fpml_document,
    )

    xml = _fixture("confirmation_5_13_swap.xml")
    report = inspect_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
    )

    assert report.status == "inspected"
    assert report.profile == FPML_5_13_CONFIRMATION
    assert report.blockers == ()
    assert report.document.namespace == CONFIRMATION_NAMESPACE
    assert report.document.root_name == "dataDocument"
    assert report.document.document_id == "DOC-001"
    assert report.document.trade_count == 1
    assert report.trade.element_id == "TRADE-ELEMENT-001"
    assert report.trade.business_id == "TRADE-001"
    assert report.trade.trade_date.isoformat() == "2026-07-01"
    assert report.trade.party_ids == ("PARTY-A", "PARTY-B")
    assert report.trade.product_names == ("swap",)
    assert report.trade_envelope.document_id == "DOC-001"
    assert report.trade_envelope.trade_id == "TRADE-001"
    assert tuple(party.party_id for party in report.trade_envelope.parties) == (
        "PARTY-A",
        "PARTY-B",
    )
    assert report.clarification.requires_clarification is False

    summary = fpml_import_report_summary(report)
    assert summary["status"] == "inspected"
    assert summary["document"]["sha256"] == report.document.sha256
    assert xml.decode("utf-8") not in repr(summary)
    assert xml.decode("utf-8") not in repr(report)
    with pytest.raises(FrozenInstanceError):
        report.status = "mutated"


@pytest.mark.parametrize(
    ("xml", "expected_id"),
    [
        (
            _fixture("forbidden_entity.xml"),
            "external_import:fpml_forbidden_xml_declaration",
        ),
        (
            b"<dataDocument>",
            "external_import:fpml_malformed_xml",
        ),
        (
            "<dataDocument/>".encode("utf-16"),
            "external_import:fpml_unsupported_encoding",
        ),
    ],
)
def test_inspect_fpml_document_rejects_unsafe_or_malformed_xml(xml, expected_id):
    from trellis.io.fpml import inspect_fpml_document

    report = inspect_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
    )

    assert report.status == "blocked"
    assert _blocker_ids(report) == (expected_id,)
    assert report.document is None
    assert report.trade is None


@pytest.mark.parametrize(
    ("xml", "limits", "expected_id"),
    [
        (
            _document("<trade><swap /></trade>"),
            {"max_document_bytes": 64, "max_elements": 100, "max_depth": 20},
            "external_import:fpml_document_too_large",
        ),
        (
            _document("<trade><a><b><swap /></b></a></trade>"),
            {"max_document_bytes": 4096, "max_elements": 100, "max_depth": 3},
            "external_import:fpml_nesting_too_deep",
        ),
        (
            _document("<trade><tradeHeader /><swap /></trade><party /><party />"),
            {"max_document_bytes": 4096, "max_elements": 5, "max_depth": 20},
            "external_import:fpml_too_many_elements",
        ),
    ],
)
def test_inspect_fpml_document_enforces_resource_limits(xml, limits, expected_id):
    from trellis.io.fpml import FpMLInspectionLimits, inspect_fpml_document

    report = inspect_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
        limits=FpMLInspectionLimits(**limits),
    )

    assert report.status == "blocked"
    assert _blocker_ids(report) == (expected_id,)


@pytest.mark.parametrize(
    ("xml", "declared_view", "declared_version", "expected_ids"),
    [
        (
            _document("<trade><swap /></trade>"),
            "recordkeeping",
            "5-13",
            ("contract_conflict:fpml_source_view",),
        ),
        (
            _document("<trade><swap /></trade>"),
            "confirmation",
            "5-12",
            ("contract_conflict:fpml_source_version",),
        ),
        (
            _document("<trade><swap /></trade>", version="5-12"),
            "confirmation",
            "5-12",
            ("external_import:fpml_unsupported_version",),
        ),
        (
            b'<dataDocument fpmlVersion="5-13"><trade><swap /></trade></dataDocument>',
            "confirmation",
            "5-13",
            ("missing_contract_field:fpml_document_namespace",),
        ),
    ],
)
def test_inspect_fpml_document_requires_exact_profile_declarations(
    xml,
    declared_view,
    declared_version,
    expected_ids,
):
    from trellis.io.fpml import inspect_fpml_document

    report = inspect_fpml_document(
        xml,
        declared_view=declared_view,
        declared_version=declared_version,
    )

    assert report.status == "blocked"
    assert _blocker_ids(report) == expected_ids


@pytest.mark.parametrize(
    ("xml", "expected_id", "clarification_field"),
    [
        (
            _document(""),
            "missing_contract_field:fpml_trade",
            "trade",
        ),
        (
            _document("<trade><swap /></trade><trade><swap /></trade>"),
            "contract_ambiguity:fpml_multiple_trades",
            "trade",
        ),
        (
            _document("<trade><tradeHeader /></trade>"),
            "missing_contract_field:fpml_product",
            "product",
        ),
        (
            _document("<trade><swap /><swaption /></trade>"),
            "contract_ambiguity:fpml_multiple_products",
            "product",
        ),
        (
            _document("<trade><swap /></trade>", root="requestConfirmation"),
            "external_import:fpml_unsupported_message_root",
            None,
        ),
    ],
)
def test_inspect_fpml_document_blocks_root_trade_and_product_shape_gaps(
    xml,
    expected_id,
    clarification_field,
):
    from trellis.io.fpml import inspect_fpml_document

    report = inspect_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
    )

    assert report.status == "blocked"
    assert _blocker_ids(report) == (expected_id,)
    if clarification_field is None:
        assert report.clarification.requires_clarification is False
    else:
        assert report.clarification.requires_clarification is True
        fields = (
            report.clarification.missing_fields
            + report.clarification.ambiguous_fields
        )
        assert clarification_field in fields
        assert report.clarification.messages


@pytest.mark.parametrize(
    ("xml", "expected_id"),
    [
        (
            _fixture("confirmation_5_13_generic_product.xml"),
            "external_import:fpml_incomplete_generic_product",
        ),
        (
            _document("<trade><nonSchemaProduct /></trade>"),
            "external_import:fpml_incomplete_non_schema_product",
        ),
        (
            _document("<trade><standardProduct /></trade>"),
            "external_import:fpml_incomplete_standard_product",
        ),
    ],
)
def test_inspect_fpml_document_never_treats_incomplete_wrappers_as_economics(
    xml,
    expected_id,
):
    from trellis.io.fpml import inspect_fpml_document

    report = inspect_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
    )

    assert report.status == "blocked"
    assert _blocker_ids(report) == (expected_id,)
    assert report.trade.product_names


def test_inspect_fpml_document_blocks_lifecycle_content_before_product_mapping():
    from trellis.io.fpml import inspect_fpml_document

    report = inspect_fpml_document(
        _fixture("confirmation_5_13_lifecycle.xml"),
        declared_view="confirmation",
        declared_version="5-13",
    )

    assert report.status == "blocked"
    assert _blocker_ids(report) == (
        "external_import:fpml_unsupported_lifecycle_content",
    )
    assert report.trade.product_names == ("swap",)
