"""Bounded FpML product normalization into Trellis semantic IR."""

from __future__ import annotations

from datetime import date
import hashlib

from trellis.agent.contract_ir import contract_ir_economic_identity
from trellis.agent.static_leg_contract import static_leg_economic_identity
from trellis.io.fpml.contracts import (
    DEFAULT_FPML_INSPECTION_LIMITS,
    FpMLClarification,
    FpMLImportReport,
    FpMLInspectionLimits,
    FpMLPremiumMetadata,
)
from trellis.io.fpml.importer import (
    _bounded_parse,
    _content_bytes,
    _direct_children,
    _first_direct_child,
    _optional_text,
    inspect_fpml_document,
)
from trellis.io.fpml._normalization_common import (
    _NormalizationBlocked,
    _blocked_from,
    _blocker,
    _fail,
    _has_unresolved_historical_fixing,
    _reject_unadmitted_direct_children,
    _validate_document_metadata,
)
from trellis.io.fpml._normalization_cap_floor import _normalize_cap_floor
from trellis.io.fpml._normalization_swap import (
    _normalize_fixed_float_swap,
    _reject_unresolved_swap_historical_fixings,
)
from trellis.io.fpml._normalization_swaption import _normalize_european_swaption


_ALLOWED_DOCUMENT_CHILDREN = {"party", "trade"}
_ALLOWED_TRADE_CHILDREN = {"capFloor", "swap", "swaption", "tradeHeader"}


def normalize_fpml_document(
    content: bytes | str | bytearray | memoryview,
    *,
    declared_view: str | None,
    declared_version: str | None,
    valuation_party_id: str | None,
    valuation_date: date | None = None,
    require_valuation_date: bool = False,
    limits: FpMLInspectionLimits = DEFAULT_FPML_INSPECTION_LIMITS,
) -> FpMLImportReport:
    """Normalize one admitted FpML product without using XML names as route authority."""

    inspected = inspect_fpml_document(
        content,
        declared_view=declared_view,
        declared_version=declared_version,
        limits=limits,
    )
    if inspected.blockers:
        return inspected

    return _normalize_inspected_fpml_document(
        content,
        inspected=inspected,
        valuation_party_id=valuation_party_id,
        valuation_date=valuation_date,
        require_valuation_date=require_valuation_date,
        limits=limits,
    )


def _normalize_inspected_fpml_document(
    content: bytes | str | bytearray | memoryview,
    *,
    inspected: FpMLImportReport,
    valuation_party_id: str | None,
    valuation_date: date | None = None,
    require_valuation_date: bool = False,
    limits: FpMLInspectionLimits = DEFAULT_FPML_INSPECTION_LIMITS,
) -> FpMLImportReport:
    """Normalize content that has already passed the bounded FpML inspection."""

    if inspected.blockers:
        return inspected
    if inspected.status != "inspected" or inspected.document is None:
        raise ValueError("inspected must be an admitted FpML inspection report")

    content_bytes = _content_bytes(content)
    if hashlib.sha256(content_bytes).hexdigest() != inspected.document.sha256:
        raise ValueError("inspected report does not describe the supplied FpML content")

    root = _bounded_parse(content_bytes, limits=limits)
    namespace = inspected.document.namespace if inspected.document else None
    trade = _direct_children(root, "trade", namespace=namespace)[0]
    product_names = inspected.trade.product_names
    try:
        _reject_unadmitted_direct_children(
            root,
            allowed=_ALLOWED_DOCUMENT_CHILDREN,
            scope="document",
            namespace=namespace,
        )
        _reject_unadmitted_direct_children(
            trade,
            allowed=_ALLOWED_TRADE_CHILDREN,
            scope="trade",
            namespace=namespace,
        )
        _validate_document_metadata(
            root,
            trade,
            namespace=namespace,
        )
    except _NormalizationBlocked as exc:
        return _blocked_from(inspected, exc.blocker)

    if product_names not in {("capFloor",), ("swap",), ("swaption",)}:
        return _blocked_from(
            inspected,
            _blocker(
                "external_import:fpml_product_normalizer_unavailable",
                "implementation_gap",
                "The inspected FpML product is outside the admitted normalization cohort.",
            ),
        )
    if product_names == ("swap",):
        swap = _first_direct_child(trade, "swap", namespace=namespace)
        if swap is None:
            raise AssertionError("inspected swap product is missing its element")
        if not _direct_children(swap, "swapStream", namespace=namespace):
            return _blocked_from(
                inspected,
                _blocker(
                    "missing_contract_field:fpml_swap_stream",
                    "contract_gap",
                    "A fixed-float swap requires swapStream economics.",
                    missing_fields=("swap_stream",),
                ),
            )

    valuation_party = _optional_text(valuation_party_id)
    if valuation_party is None:
        return _blocked_from(
            inspected,
            _blocker(
                "missing_contract_field:fpml_valuation_party_id",
                "contract_gap",
                "FpML pricing requires an explicit valuation party for signed cashflows.",
                missing_fields=("valuation_party_id",),
            ),
        )
    if inspected.trade is None or valuation_party not in inspected.trade.party_ids:
        return _blocked_from(
            inspected,
            _blocker(
                "contract_conflict:fpml_valuation_party_id",
                "contract_conflict",
                "The valuation party is not a party identified by the FpML trade.",
            ),
        )

    if valuation_date is not None and not isinstance(valuation_date, date):
        raise TypeError("valuation_date must be a date")

    premium_metadata: tuple[FpMLPremiumMetadata, ...] = ()
    try:
        if product_names == ("swap",):
            swap = _first_direct_child(trade, "swap", namespace=namespace)
            if swap is None:
                raise AssertionError("inspected swap product is missing its element")
            contract, provenance = _normalize_fixed_float_swap(
                swap,
                namespace=namespace,
                valuation_party_id=valuation_party,
                known_party_ids=inspected.trade.party_ids,
            )
            if valuation_date is not None:
                _reject_unresolved_swap_historical_fixings(contract, valuation_date)
            economic_identity = static_leg_economic_identity(contract)
        elif product_names == ("swaption",):
            swaption = _first_direct_child(trade, "swaption", namespace=namespace)
            if swaption is None:
                raise AssertionError(
                    "inspected swaption product is missing its element"
                )
            contract, provenance, premium_metadata = _normalize_european_swaption(
                swaption,
                namespace=namespace,
                valuation_party_id=valuation_party,
                valuation_date=valuation_date,
                known_party_ids=inspected.trade.party_ids,
            )
            economic_identity = contract_ir_economic_identity(contract)
        elif product_names == ("capFloor",):
            cap_floor = _first_direct_child(trade, "capFloor", namespace=namespace)
            if cap_floor is None:
                raise AssertionError(
                    "inspected capFloor product is missing its element"
                )
            contract, provenance, premium_metadata = _normalize_cap_floor(
                cap_floor,
                namespace=namespace,
                valuation_party_id=valuation_party,
                valuation_date=valuation_date,
                known_party_ids=inspected.trade.party_ids,
            )
            if valuation_date is not None and _has_unresolved_historical_fixing(
                contract,
                valuation_date,
            ):
                _fail(
                    "external_import:fpml_historical_fixing_runtime_unsupported",
                    "implementation_gap",
                    "The static-leg runtime does not yet consume historical fixings "
                    "for unpaid seasoned cap/floor periods.",
                )
            economic_identity = static_leg_economic_identity(contract)
        else:  # pragma: no cover - guarded by the admitted product set above
            raise AssertionError("unreachable FpML product dispatch")
    except _NormalizationBlocked as exc:
        return _blocked_from(inspected, exc.blocker)

    if require_valuation_date and valuation_date is None:
        return _blocked_from(
            inspected,
            _blocker(
                "missing_contract_field:fpml_valuation_date",
                "contract_gap",
                "FpML pricing requires an explicit deterministic valuation date.",
                missing_fields=("valuation_date",),
            ),
        )

    return FpMLImportReport(
        status="normalized",
        profile=inspected.profile,
        document=inspected.document,
        trade=inspected.trade,
        trade_envelope=inspected.trade_envelope,
        blockers=(),
        clarification=FpMLClarification(requires_clarification=False),
        normalized_contract=contract,
        economic_identity=economic_identity,
        mapping_provenance=provenance,
        premium_metadata=premium_metadata,
    )
