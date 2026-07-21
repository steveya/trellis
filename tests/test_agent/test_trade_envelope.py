"""Tests for external trade-envelope identity and provenance."""

from __future__ import annotations

from datetime import date

import pytest


def test_trade_envelope_is_deeply_immutable_and_has_stable_summary():
    from trellis.agent.trade_envelope import (
        TradeEnvelope,
        TradeParty,
        trade_envelope_summary,
    )

    identifiers = {"uti": "UTI-001", "book": "RATES-NY"}
    metadata = {
        "annotations": ["confirmed", {"channel": "electronic"}],
        "source_sequence": 7,
    }
    envelope = TradeEnvelope(
        source_format="fpml",
        source_view="confirmation",
        source_version="5-13",
        document_id="DOC-001",
        trade_id="TRADE-001",
        package_id="PACKAGE-001",
        trade_date=date(2026, 7, 1),
        lifecycle_state="current",
        parties=(
            TradeParty(
                party_id="PARTY-B",
                role="counterparty",
                identifiers={"lei": "LEI-B"},
            ),
            TradeParty(
                party_id="PARTY-A",
                role="reporting_party",
                identifiers={"lei": "LEI-A"},
            ),
        ),
        identifiers=identifiers,
        metadata=metadata,
    )

    identifiers["uti"] = "MUTATED"
    metadata["annotations"][1]["channel"] = "voice"

    assert envelope.identifiers["uti"] == "UTI-001"
    assert envelope.metadata["annotations"][1]["channel"] == "electronic"
    assert tuple(party.party_id for party in envelope.parties) == (
        "PARTY-A",
        "PARTY-B",
    )
    with pytest.raises(TypeError):
        envelope.identifiers["new"] = "value"
    with pytest.raises(TypeError):
        envelope.metadata["annotations"][1]["channel"] = "voice"

    assert trade_envelope_summary(envelope) == {
        "source_format": "fpml",
        "source_view": "confirmation",
        "source_version": "5-13",
        "document_id": "DOC-001",
        "trade_id": "TRADE-001",
        "package_id": "PACKAGE-001",
        "trade_date": "2026-07-01",
        "lifecycle_state": "current",
        "parties": [
            {
                "party_id": "PARTY-A",
                "role": "reporting_party",
                "name": None,
                "identifiers": {"lei": "LEI-A"},
            },
            {
                "party_id": "PARTY-B",
                "role": "counterparty",
                "name": None,
                "identifiers": {"lei": "LEI-B"},
            },
        ],
        "identifiers": {"book": "RATES-NY", "uti": "UTI-001"},
        "metadata": {
            "annotations": ["confirmed", {"channel": "electronic"}],
            "source_sequence": 7,
        },
    }


def test_trade_envelope_rejects_missing_source_identity_and_duplicate_parties():
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty

    with pytest.raises(ValueError, match="source_format"):
        TradeEnvelope(source_format=" ")

    with pytest.raises(ValueError, match="duplicate trade party ids"):
        TradeEnvelope(
            source_format="fpml",
            parties=(
                TradeParty(party_id="PARTY-A"),
                TradeParty(party_id="PARTY-A"),
            ),
        )

    with pytest.raises(ValueError, match="duplicate normalized key"):
        TradeEnvelope(
            source_format="fpml",
            identifiers={"uti": "UTI-1", " uti ": "UTI-2"},
        )

    with pytest.raises(TypeError, match="source_format"):
        TradeEnvelope(source_format=5)


def test_platform_request_requires_typed_trade_envelope():
    from trellis.agent.platform_requests import PlatformRequest

    with pytest.raises(TypeError, match="trade_envelope"):
        PlatformRequest(
            request_id="request-1",
            request_type="build",
            entry_point="executor",
            trade_envelope={"source_format": "fpml"},
        )


def test_platform_request_preserves_legacy_positional_metadata_argument():
    from trellis.agent.platform_requests import PlatformRequest

    request = PlatformRequest(
        "request-legacy",
        "build",
        "executor",
        None,
        None,
        None,
        None,
        (),
        (),
        (),
        {},
        None,
        None,
        None,
        None,
        None,
        None,
        {"task_id": "E23"},
    )

    assert request.metadata == {"task_id": "E23"}
    assert request.trade_envelope is None
