"""Secure, bounded FpML XML inspection without product lowering."""

from __future__ import annotations

from datetime import date
import hashlib
import re
from xml.etree import ElementTree

from trellis.agent.trade_envelope import TradeEnvelope, TradeParty
from trellis.io.fpml.contracts import (
    DEFAULT_FPML_INSPECTION_LIMITS,
    SUPPORTED_FPML_PROFILES,
    FpMLClarification,
    FpMLDocumentIdentity,
    FpMLImportBlocker,
    FpMLImportReport,
    FpMLInspectionLimits,
    FpMLProfile,
    FpMLTradeIdentity,
)


_XML_ENCODING = re.compile(
    br"<\?xml[^>]*\bencoding\s*=\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_FPML_NAMESPACE = re.compile(r"^http://www\.fpml\.org/FpML-5/([^/]+)$")
_FORBIDDEN_DECLARATIONS = ("<!DOCTYPE", "<!ENTITY")
_LIFECYCLE_NAMES = {
    "amendment",
    "increase",
    "isCancellation",
    "isCorrection",
    "novation",
    "originatingEvent",
    "termination",
}
_NON_PRODUCT_TRADE_CHILDREN = {
    "allocations",
    "approvals",
    "barrierDeterminationAgent",
    "brokerPartyReference",
    "calculationAgent",
    "calculationAgentBusinessCenter",
    "clearedDate",
    "collateral",
    "determiningParty",
    "documentation",
    "governingLaw",
    "hedgingParty",
    "originatingPackage",
    "otherPartyPayment",
    "partyTradeIdentifier",
    "partyTradeInformation",
    "productSummary",
    "tradeDate",
    "tradeHeader",
    "tradeSummary",
}
_INCOMPLETE_PRODUCT_BLOCKERS = {
    "genericProduct": (
        "external_import:fpml_incomplete_generic_product",
        "genericProduct does not provide complete schema-defined trade economics.",
    ),
    "nonSchemaProduct": (
        "external_import:fpml_incomplete_non_schema_product",
        "nonSchemaProduct cannot establish complete Trellis trade economics.",
    ),
    "standardProduct": (
        "external_import:fpml_incomplete_standard_product",
        "standardProduct identifiers are not a complete confirmed-trade representation.",
    ),
}


class _ResourceLimitExceeded(Exception):
    def __init__(self, blocker: FpMLImportBlocker):
        self.blocker = blocker
        super().__init__(blocker.summary)


def inspect_fpml_document(
    content: bytes | str | bytearray | memoryview,
    *,
    declared_view: str | None,
    declared_version: str | None,
    limits: FpMLInspectionLimits = DEFAULT_FPML_INSPECTION_LIMITS,
) -> FpMLImportReport:
    """Inspect one inline FpML document under an explicit bounded profile."""

    if not isinstance(limits, FpMLInspectionLimits):
        raise TypeError("limits must be an FpMLInspectionLimits")
    content_bytes = _content_bytes(content)
    if len(content_bytes) > limits.max_document_bytes:
        return _blocked_report(
            _blocker(
                "external_import:fpml_document_too_large",
                "resource_limit",
                (
                    "The FpML document exceeds the configured byte limit "
                    f"({len(content_bytes)} > {limits.max_document_bytes})."
                ),
            )
        )
    digest = hashlib.sha256(content_bytes).hexdigest()

    encoding_blocker = _encoding_blocker(content_bytes)
    if encoding_blocker is not None:
        return _blocked_report(encoding_blocker)
    decoded = content_bytes.decode("utf-8-sig")
    upper = decoded.upper()
    if any(token in upper for token in _FORBIDDEN_DECLARATIONS):
        return _blocked_report(
            _blocker(
                "external_import:fpml_forbidden_xml_declaration",
                "security_policy",
                "FpML input cannot contain DTD or entity declarations.",
            )
        )

    try:
        root = _bounded_parse(content_bytes, limits=limits)
    except _ResourceLimitExceeded as exc:
        return _blocked_report(exc.blocker)
    except ElementTree.ParseError:
        return _blocked_report(
            _blocker(
                "external_import:fpml_malformed_xml",
                "malformed_document",
                "The FpML payload is not well-formed XML.",
            )
        )

    namespace, root_name = _split_tag(root.tag)
    document_version = _optional_text(root.attrib.get("fpmlVersion"))
    actual_view = _view_from_namespace(namespace)
    direct_trades = _direct_children(root, "trade", namespace=namespace)
    document = FpMLDocumentIdentity(
        namespace=namespace,
        view=actual_view,
        version=document_version,
        root_name=root_name,
        document_id=_optional_text(root.attrib.get("id")),
        trade_count=len(direct_trades),
        byte_length=len(content_bytes),
        sha256=digest,
    )

    profile, profile_blockers = _resolve_profile(
        namespace=namespace,
        view=actual_view,
        version=document_version,
        root_name=root_name,
        declared_view=_optional_text(declared_view),
        declared_version=_optional_text(declared_version),
    )
    if profile_blockers:
        return _report(
            profile=profile,
            document=document,
            trade=None,
            trade_envelope=None,
            blockers=profile_blockers,
        )

    root_blocker = _root_blocker(root_name, profile=profile)
    if root_blocker is not None:
        return _report(
            profile=profile,
            document=document,
            trade=None,
            trade_envelope=None,
            blockers=(root_blocker,),
        )

    blockers: list[FpMLImportBlocker] = []
    if _has_lifecycle_content(root, namespace=namespace):
        blockers.append(
            _blocker(
                "external_import:fpml_unsupported_lifecycle_content",
                "unsupported_contract",
                "Lifecycle event content is outside the current-state FpML import cohort.",
            )
        )

    if not direct_trades:
        blockers.append(
            _blocker(
                "missing_contract_field:fpml_trade",
                "contract_gap",
                "The FpML dataDocument does not contain a direct trade.",
                missing_fields=("trade",),
            )
        )
        return _report(
            profile=profile,
            document=document,
            trade=None,
            trade_envelope=None,
            blockers=tuple(blockers),
        )
    if len(direct_trades) > 1:
        blockers.append(
            _blocker(
                "contract_ambiguity:fpml_multiple_trades",
                "contract_ambiguity",
                "The first FpML import cohort accepts exactly one direct trade.",
                ambiguous_fields=("trade",),
            )
        )
        return _report(
            profile=profile,
            document=document,
            trade=None,
            trade_envelope=None,
            blockers=tuple(blockers),
        )

    trade_element = direct_trades[0]
    trade = _trade_identity(root, trade_element, namespace=namespace)
    envelope = _trade_envelope(document, trade, profile=profile)
    if not trade.product_names:
        blockers.append(
            _blocker(
                "missing_contract_field:fpml_product",
                "contract_gap",
                "The FpML trade does not contain a direct product element.",
                missing_fields=("product",),
            )
        )
    elif len(trade.product_names) > 1:
        blockers.append(
            _blocker(
                "contract_ambiguity:fpml_multiple_products",
                "contract_ambiguity",
                "The FpML trade contains multiple direct product candidates.",
                ambiguous_fields=("product",),
            )
        )
    for product_name in trade.product_names:
        incomplete = _INCOMPLETE_PRODUCT_BLOCKERS.get(product_name)
        if incomplete is not None:
            blocker_id, summary = incomplete
            blockers.append(
                _blocker(
                    blocker_id,
                    "incomplete_contract",
                    summary,
                    missing_fields=("schema_defined_product_economics",),
                )
            )

    return _report(
        profile=profile,
        document=document,
        trade=trade,
        trade_envelope=envelope,
        blockers=tuple(blockers),
    )


def _content_bytes(content: bytes | str | bytearray | memoryview) -> bytes:
    if isinstance(content, str):
        return content.encode("utf-8")
    if isinstance(content, (bytes, bytearray, memoryview)):
        return bytes(content)
    raise TypeError("FpML content must be bytes or text")


def _encoding_blocker(content: bytes) -> FpMLImportBlocker | None:
    if content.startswith((b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00")):
        return _unsupported_encoding_blocker()
    if b"\x00" in content:
        return _unsupported_encoding_blocker()
    declaration = content[3:] if content.startswith(b"\xef\xbb\xbf") else content
    declaration_end = declaration.find(b"?>")
    if declaration.startswith(b"<?xml") and declaration_end >= 0:
        declaration = declaration[: declaration_end + 2]
    else:
        declaration = b""
    match = _XML_ENCODING.search(declaration)
    if match is not None:
        encoding = match.group(1).decode("ascii", errors="ignore").lower().replace("_", "-")
        if encoding not in {"utf-8", "utf8"}:
            return _unsupported_encoding_blocker()
    try:
        content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return _unsupported_encoding_blocker()
    return None


def _unsupported_encoding_blocker() -> FpMLImportBlocker:
    return _blocker(
        "external_import:fpml_unsupported_encoding",
        "unsupported_contract",
        "The bounded FpML importer accepts UTF-8 XML only.",
    )


def _bounded_parse(content: bytes, *, limits: FpMLInspectionLimits):
    parser = ElementTree.XMLPullParser(events=("start", "end"))
    root = None
    depth = 0
    element_count = 0
    for offset in range(0, len(content), 16_384):
        parser.feed(content[offset : offset + 16_384])
        root, depth, element_count = _consume_events(
            parser,
            root=root,
            depth=depth,
            element_count=element_count,
            limits=limits,
        )
    parser.close()
    root, depth, element_count = _consume_events(
        parser,
        root=root,
        depth=depth,
        element_count=element_count,
        limits=limits,
    )
    if root is None:
        raise ElementTree.ParseError("empty XML document")
    return root


def _consume_events(
    parser,
    *,
    root,
    depth: int,
    element_count: int,
    limits: FpMLInspectionLimits,
):
    for event, element in parser.read_events():
        if event == "start":
            if root is None:
                root = element
            depth += 1
            element_count += 1
            if depth > limits.max_depth:
                raise _ResourceLimitExceeded(
                    _blocker(
                        "external_import:fpml_nesting_too_deep",
                        "resource_limit",
                        "The FpML document exceeds the configured nesting-depth limit.",
                    )
                )
            if element_count > limits.max_elements:
                raise _ResourceLimitExceeded(
                    _blocker(
                        "external_import:fpml_too_many_elements",
                        "resource_limit",
                        "The FpML document exceeds the configured element-count limit.",
                    )
                )
        else:
            depth -= 1
    return root, depth, element_count


def _resolve_profile(
    *,
    namespace: str | None,
    view: str | None,
    version: str | None,
    root_name: str,
    declared_view: str | None,
    declared_version: str | None,
) -> tuple[FpMLProfile | None, tuple[FpMLImportBlocker, ...]]:
    blockers: list[FpMLImportBlocker] = []
    if namespace is None:
        blockers.append(
            _blocker(
                "missing_contract_field:fpml_document_namespace",
                "contract_gap",
                "The FpML document root requires an explicit FpML namespace.",
                missing_fields=("source_view",),
            )
        )
        return None, tuple(blockers)

    if view is None:
        blockers.append(
            _blocker(
                "external_import:fpml_unsupported_namespace",
                "unsupported_contract",
                f"The XML namespace {namespace!r} is not an admitted FpML namespace.",
            )
        )
    elif declared_view and declared_view != view:
        blockers.append(
            _blocker(
                "contract_conflict:fpml_source_view",
                "contract_conflict",
                "The declared FpML view conflicts with the document namespace.",
            )
        )
    elif not any(profile.view == view for profile in SUPPORTED_FPML_PROFILES):
        blockers.append(
            _blocker(
                "external_import:fpml_unsupported_view",
                "unsupported_contract",
                f"FpML view {view!r} is not admitted by the current import profile.",
            )
        )

    if version is None:
        blockers.append(
            _blocker(
                "missing_contract_field:fpml_document_version",
                "contract_gap",
                "The FpML document root requires an fpmlVersion attribute.",
                missing_fields=("source_version",),
            )
        )
    elif declared_version and declared_version != version:
        blockers.append(
            _blocker(
                "contract_conflict:fpml_source_version",
                "contract_conflict",
                "The declared FpML version conflicts with the document fpmlVersion.",
            )
        )
    elif not any(profile.version == version for profile in SUPPORTED_FPML_PROFILES):
        blockers.append(
            _blocker(
                "external_import:fpml_unsupported_version",
                "unsupported_contract",
                f"FpML version {version!r} is not admitted by the current import profile.",
            )
        )

    profile = next(
        (
            candidate
            for candidate in SUPPORTED_FPML_PROFILES
            if candidate.namespace == namespace and candidate.version == version
        ),
        None,
    )
    return profile, tuple(blockers)


def _root_blocker(root_name: str, *, profile: FpMLProfile | None):
    admitted_roots = profile.root_names if profile is not None else ()
    if root_name in admitted_roots:
        return None
    return _blocker(
        "external_import:fpml_unsupported_message_root",
        "unsupported_contract",
        f"FpML root {root_name!r} is outside the admitted dataDocument cohort.",
    )


def _trade_identity(root, trade, *, namespace: str | None) -> FpMLTradeIdentity:
    trade_header = _first_direct_child(trade, "tradeHeader", namespace=namespace)
    business_id = _first_descendant_text(trade_header, "tradeId", namespace=namespace)
    trade_date_text = _first_descendant_text(trade_header, "tradeDate", namespace=namespace)
    parsed_trade_date = None
    if trade_date_text:
        try:
            parsed_trade_date = date.fromisoformat(trade_date_text)
        except ValueError:
            parsed_trade_date = None
    party_ids = tuple(
        sorted(
            {
                party_id
                for party in _direct_children(root, "party", namespace=namespace)
                if (party_id := _optional_text(party.attrib.get("id"))) is not None
            }
        )
    )
    product_names = tuple(
        sorted(
            local_name
            for child in tuple(trade)
            for child_namespace, local_name in (_split_tag(child.tag),)
            if child_namespace == namespace
            and local_name not in _NON_PRODUCT_TRADE_CHILDREN
        )
    )
    return FpMLTradeIdentity(
        element_id=_optional_text(trade.attrib.get("id")),
        business_id=business_id,
        trade_date=parsed_trade_date,
        party_ids=party_ids,
        product_names=product_names,
    )


def _trade_envelope(
    document: FpMLDocumentIdentity,
    trade: FpMLTradeIdentity,
    *,
    profile: FpMLProfile | None,
) -> TradeEnvelope:
    identifiers = {}
    if trade.element_id:
        identifiers["fpml_trade_element_id"] = trade.element_id
    return TradeEnvelope(
        source_format="fpml",
        source_view=profile.view if profile else document.view,
        source_version=profile.version if profile else document.version,
        document_id=document.document_id,
        trade_id=trade.business_id or trade.element_id,
        trade_date=trade.trade_date,
        parties=tuple(TradeParty(party_id=party_id) for party_id in trade.party_ids),
        identifiers=identifiers,
    )


def _has_lifecycle_content(root, *, namespace: str | None) -> bool:
    return any(
        child_namespace == namespace and local_name in _LIFECYCLE_NAMES
        for child in root.iter()
        for child_namespace, local_name in (_split_tag(child.tag),)
    )


def _first_descendant_text(element, local_name: str, *, namespace: str | None):
    if element is None:
        return None
    for child in element.iter():
        child_namespace, child_name = _split_tag(child.tag)
        if child_namespace == namespace and child_name == local_name:
            return _optional_text(child.text)
    return None


def _direct_children(element, local_name: str, *, namespace: str | None):
    return tuple(
        child
        for child in tuple(element)
        if _split_tag(child.tag) == (namespace, local_name)
    )


def _first_direct_child(element, local_name: str, *, namespace: str | None):
    matches = _direct_children(element, local_name, namespace=namespace)
    return matches[0] if matches else None


def _split_tag(tag: str) -> tuple[str | None, str]:
    if tag.startswith("{") and "}" in tag:
        namespace, local_name = tag[1:].split("}", 1)
        return namespace or None, local_name
    return None, tag


def _view_from_namespace(namespace: str | None) -> str | None:
    if not namespace:
        return None
    match = _FPML_NAMESPACE.match(namespace)
    return match.group(1) if match is not None else None


def _optional_text(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _blocker(
    blocker_id: str,
    category: str,
    summary: str,
    *,
    missing_fields: tuple[str, ...] = (),
    ambiguous_fields: tuple[str, ...] = (),
) -> FpMLImportBlocker:
    return FpMLImportBlocker(
        id=blocker_id,
        category=category,
        summary=summary,
        missing_fields=missing_fields,
        ambiguous_fields=ambiguous_fields,
    )


def _clarification(blockers: tuple[FpMLImportBlocker, ...]) -> FpMLClarification:
    missing = tuple(
        dict.fromkeys(
            field_name
            for blocker in blockers
            for field_name in blocker.missing_fields
        )
    )
    ambiguous = tuple(
        dict.fromkeys(
            field_name
            for blocker in blockers
            for field_name in blocker.ambiguous_fields
        )
    )
    messages = tuple(
        [f"Provide unambiguous FpML values for: {', '.join(missing)}."]
        if missing
        else []
    ) + tuple(
        [f"Disambiguate the FpML fields: {', '.join(ambiguous)}."]
        if ambiguous
        else []
    )
    return FpMLClarification(
        requires_clarification=bool(missing or ambiguous),
        missing_fields=missing,
        ambiguous_fields=ambiguous,
        messages=messages,
    )


def _report(
    *,
    profile: FpMLProfile | None,
    document: FpMLDocumentIdentity | None,
    trade: FpMLTradeIdentity | None,
    trade_envelope: TradeEnvelope | None,
    blockers: tuple[FpMLImportBlocker, ...],
) -> FpMLImportReport:
    return FpMLImportReport(
        status="blocked" if blockers else "inspected",
        profile=profile,
        document=document,
        trade=trade,
        trade_envelope=trade_envelope,
        blockers=blockers,
        clarification=_clarification(blockers),
    )


def _blocked_report(blocker: FpMLImportBlocker) -> FpMLImportReport:
    return _report(
        profile=None,
        document=None,
        trade=None,
        trade_envelope=None,
        blockers=(blocker,),
    )


__all__ = ["inspect_fpml_document"]
