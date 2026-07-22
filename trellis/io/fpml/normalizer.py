"""Bounded FpML product normalization into Trellis semantic IR."""

from __future__ import annotations

from datetime import date
import hashlib

from trellis.agent.contract_ir import (
    Annuity,
    Constant,
    ContractIR,
    Exercise,
    FiniteSchedule,
    ForwardRate,
    Max,
    Observation,
    Scaled,
    Singleton,
    Strike,
    Sub,
    SwapRate,
    Underlying,
    contract_ir_economic_identity,
)
from trellis.agent.static_leg_contract import (
    CouponLeg,
    FixedCouponFormula,
    SettlementRule,
    static_leg_economic_identity,
)
from trellis.io.fpml.contracts import (
    DEFAULT_FPML_INSPECTION_LIMITS,
    FpMLClarification,
    FpMLFieldProvenance,
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
    _split_tag,
    inspect_fpml_document,
)
from trellis.io.fpml._normalization_common import (
    _ALLOWED_LEAF_CHILDREN,
    _NormalizationBlocked,
    _adjustable_date,
    _blocked_from,
    _blocker,
    _fail,
    _has_unresolved_historical_fixing,
    _normalize_option_premiums,
    _provenance,
    _reject_nested_metadata_children,
    _reject_unadmitted_direct_children,
    _required_child,
    _required_href,
    _validate_document_metadata,
)
from trellis.io.fpml._normalization_cap_floor import _normalize_cap_floor
from trellis.io.fpml._normalization_swap import (
    _normalize_fixed_float_swap,
    _reject_unresolved_swap_historical_fixings,
)


_ALLOWED_DOCUMENT_CHILDREN = {"party", "trade"}
_ALLOWED_TRADE_CHILDREN = {"capFloor", "swap", "swaption", "tradeHeader"}
_ALLOWED_SWAPTION_CHILDREN = {
    "assetClass",
    "buyerPartyReference",
    "cashSettlement",
    "europeanExercise",
    "physicalSettlement",
    "premium",
    "primaryAssetClass",
    "productId",
    "productType",
    "secondaryAssetClass",
    "sellerPartyReference",
    "swap",
    "swaptionStraddle",
}
_SWAPTION_METADATA_CHILDREN = {
    "assetClass",
    "primaryAssetClass",
    "productId",
    "productType",
    "secondaryAssetClass",
}
_ALLOWED_EUROPEAN_EXERCISE_CHILDREN = {"expirationDate"}


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


def _normalize_european_swaption(
    swaption,
    *,
    namespace: str | None,
    valuation_party_id: str,
    valuation_date: date | None,
    known_party_ids: tuple[str, ...],
) -> tuple[
    ContractIR,
    tuple[FpMLFieldProvenance, ...],
    tuple[FpMLPremiumMetadata, ...],
]:
    """Normalize the bounded physical European fixed-float swaption cohort."""

    exercise_styles = tuple(
        local_name
        for child in tuple(swaption)
        for child_namespace, local_name in (_split_tag(child.tag),)
        if child_namespace == namespace
        and local_name in {"americanExercise", "bermudaExercise", "europeanExercise"}
    )
    if not exercise_styles:
        _fail(
            "missing_contract_field:fpml_swaption_exercise",
            "contract_gap",
            "A swaption requires an explicit exercise declaration.",
            missing_fields=("exercise",),
        )
    if exercise_styles != ("europeanExercise",):
        _fail(
            "external_import:fpml_swaption_exercise_style_unsupported",
            "unsupported_contract",
            "The admitted swaption cohort requires exactly one European exercise.",
        )
    if _direct_children(swaption, "cashSettlement", namespace=namespace):
        _fail(
            "external_import:fpml_swaption_cash_settlement_unsupported",
            "unsupported_contract",
            "Cash-settled swaptions require settlement-method semantics outside "
            "the admitted physical cohort.",
        )
    _reject_unadmitted_direct_children(
        swaption,
        allowed=_ALLOWED_SWAPTION_CHILDREN,
        scope="swaption",
        namespace=namespace,
    )
    _reject_nested_metadata_children(
        swaption,
        names=_SWAPTION_METADATA_CHILDREN,
        namespace=namespace,
    )

    buyer = _required_href(
        swaption,
        "buyerPartyReference",
        namespace=namespace,
        missing_field="buyer_party_reference",
    )
    seller = _required_href(
        swaption,
        "sellerPartyReference",
        namespace=namespace,
        missing_field="seller_party_reference",
    )
    if buyer == seller:
        _fail(
            "contract_conflict:fpml_swaption_parties",
            "contract_conflict",
            "Swaption buyer and seller must differ.",
        )
    if not {buyer, seller}.issubset(set(known_party_ids)):
        _fail(
            "contract_conflict:fpml_swaption_party_reference",
            "contract_conflict",
            "The swaption references a party not identified by the FpML document.",
        )
    if valuation_party_id not in {buyer, seller}:
        _fail(
            "contract_conflict:fpml_valuation_party_id",
            "contract_conflict",
            "The valuation party is neither buyer nor seller of the swaption.",
        )

    physical = _required_child(
        swaption,
        "physicalSettlement",
        namespace=namespace,
        missing_field="physical_settlement",
    )
    _reject_unadmitted_direct_children(
        physical,
        allowed=_ALLOWED_LEAF_CHILDREN,
        scope="physical_settlement",
        namespace=namespace,
    )
    straddle_elements = _direct_children(
        swaption,
        "swaptionStraddle",
        namespace=namespace,
    )
    if len(straddle_elements) > 1:
        _fail(
            "contract_ambiguity:fpml_swaption_straddle",
            "contract_ambiguity",
            "FpML normalization found multiple swaptionStraddle declarations.",
            ambiguous_fields=("swaption_straddle",),
        )
    straddle = (
        _optional_text(straddle_elements[0].text).lower()
        if straddle_elements and _optional_text(straddle_elements[0].text)
        else "false"
    )
    if straddle not in {"true", "false", "1", "0"}:
        _fail(
            "external_import:fpml_malformed_swaption_straddle",
            "malformed_document",
            "FpML swaptionStraddle must use XML boolean syntax.",
        )
    if straddle in {"true", "1"}:
        _fail(
            "external_import:fpml_swaption_straddle_unsupported",
            "unsupported_contract",
            "Swaption straddles are outside the admitted payer/receiver cohort.",
        )

    exercise = _required_child(
        swaption,
        "europeanExercise",
        namespace=namespace,
        missing_field="european_exercise",
    )
    _reject_unadmitted_direct_children(
        exercise,
        allowed=_ALLOWED_EUROPEAN_EXERCISE_CHILDREN,
        scope="european_exercise",
        namespace=namespace,
    )
    _, expiry = _adjustable_date(
        exercise,
        "expirationDate",
        namespace=namespace,
    )
    if valuation_date is not None and valuation_date >= expiry:
        _fail(
            "external_import:fpml_expired_swaption_unsupported",
            "unsupported_contract",
            "The admitted swaption pricing cohort requires valuation before expiry.",
        )

    swaps = _direct_children(swaption, "swap", namespace=namespace)
    if len(swaps) != 1:
        _fail(
            "missing_contract_field:fpml_swaption_underlying_swap"
            if not swaps
            else "contract_ambiguity:fpml_swaption_underlying_swap",
            "contract_gap" if not swaps else "contract_ambiguity",
            "A European swaption requires exactly one complete underlying swap.",
            missing_fields=("underlying_swap",) if not swaps else (),
            ambiguous_fields=("underlying_swap",) if len(swaps) > 1 else (),
        )
    underlying_contract, raw_underlying_provenance = _normalize_fixed_float_swap(
        swaps[0],
        namespace=namespace,
        valuation_party_id=buyer,
        known_party_ids=known_party_ids,
        required_party_pair=frozenset((buyer, seller)),
        xml_base_path="/dataDocument/trade/swaption/swap",
    )
    underlying_provenance = tuple(
        FpMLFieldProvenance(
            xml_path=item.xml_path,
            semantic_field=f"underlying_contract.{item.semantic_field}",
            normalized_value=item.normalized_value,
        )
        for item in raw_underlying_provenance
        if item.semantic_field != "valuation_party_id"
    )
    fixed_signed_leg = next(
        (
            item
            for item in underlying_contract.legs
            if isinstance(item.leg, CouponLeg)
            and isinstance(item.leg.coupon_formula, FixedCouponFormula)
        ),
        None,
    )
    if fixed_signed_leg is None:
        raise AssertionError("normalized fixed-float swap has no fixed leg")
    fixed_leg = fixed_signed_leg.leg
    notional_step = fixed_leg.notional_schedule.steps[0]
    if expiry >= notional_step.start_date:
        _fail(
            "contract_conflict:fpml_swaption_expiry_underlying_start",
            "contract_conflict",
            "Swaption expiry must precede the underlying swap effective date.",
        )
    payment_schedule = FiniteSchedule(
        tuple(period.payment_date for period in fixed_leg.coupon_periods)
    )
    underlier_id = (
        f"{fixed_leg.currency}-IRS-"
        f"{notional_step.start_date:%Y%m%d}-{notional_step.end_date:%Y%m%d}"
    )
    rate = fixed_leg.coupon_formula.rate
    intrinsic = (
        Sub(SwapRate(underlier_id, payment_schedule), Strike(rate))
        if fixed_signed_leg.direction == "pay"
        else Sub(Strike(rate), SwapRate(underlier_id, payment_schedule))
    )
    expiry_schedule = Singleton(expiry)
    contract = ContractIR(
        payoff=Scaled(
            Annuity(underlier_id, payment_schedule),
            Max((intrinsic, Constant(0.0))),
        ),
        exercise=Exercise("european", expiry_schedule),
        observation=Observation("terminal", expiry_schedule),
        underlying=Underlying(ForwardRate(underlier_id, "lognormal_forward")),
        position="long" if valuation_party_id == buyer else "short",
        settlement=SettlementRule(
            settlement_kind="physical",
            payout_currency=fixed_leg.currency,
        ),
        underlying_contract=underlying_contract,
    )
    premium_metadata = _normalize_option_premiums(
        swaption,
        namespace=namespace,
        valuation_date=valuation_date,
        known_party_ids=known_party_ids,
        option_party_ids=(buyer, seller),
        product_scope="swaption",
        product_label="Swaption",
    )
    provenance = (
        _provenance("", "valuation_party_id", valuation_party_id),
        _provenance(
            "/dataDocument/trade/swaption/buyerPartyReference/@href",
            "position",
            contract.position,
        ),
        _provenance(
            "/dataDocument/trade/swaption/europeanExercise/expirationDate/"
            "unadjustedDate",
            "exercise.schedule.t",
            expiry,
        ),
        _provenance(
            "/dataDocument/trade/swaption/physicalSettlement",
            "settlement.settlement_kind",
            "physical",
        ),
        _provenance(
            "/dataDocument/trade/swaption/swap",
            "underlying_contract",
            static_leg_economic_identity(underlying_contract),
        ),
        *underlying_provenance,
    )
    return contract, provenance, premium_metadata
