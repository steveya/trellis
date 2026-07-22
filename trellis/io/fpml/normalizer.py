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
    NotionalSchedule,
    NotionalStep,
    PeriodRateOptionPeriod,
    PeriodRateOptionStripLeg,
    SettlementRule,
    SignedLeg,
    StaticLegContractIR,
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
    _ALLOWED_STREAM_CHILDREN,
    _NormalizationBlocked,
    _STREAM_METADATA_REFERENCES,
    _STUB_FIELDS,
    _adjustable_date,
    _blocked_from,
    _blocker,
    _contains_any,
    _descendant,
    _descendants,
    _fail,
    _finite_float,
    _has_unresolved_historical_fixing,
    _normalize_stream,
    _provenance,
    _reject_nested_metadata_children,
    _reject_unadmitted_direct_children,
    _required_child,
    _required_href,
    _required_text,
    _validate_document_metadata,
)
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
_ALLOWED_CAP_FLOOR_CHILDREN = {
    "additionalPayment",
    "assetClass",
    "capFloorStream",
    "earlyTerminationProvision",
    "premium",
    "primaryAssetClass",
    "productId",
    "productType",
    "secondaryAssetClass",
}
_CAP_FLOOR_METADATA_CHILDREN = {
    "assetClass",
    "primaryAssetClass",
    "productId",
    "productType",
    "secondaryAssetClass",
}
_ALLOWED_EUROPEAN_EXERCISE_CHILDREN = {"expirationDate"}
_ALLOWED_PREMIUM_CHILDREN = {
    "payerPartyReference",
    "paymentAmount",
    "paymentDate",
    "receiverPartyReference",
}
_ALLOWED_PAYMENT_AMOUNT_CHILDREN = {"amount", "currency"}
_ALLOWED_STRIKE_SCHEDULE_CHILDREN = {"buyer", "initialValue", "seller", "step"}


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


def _normalize_option_premiums(
    product,
    *,
    namespace: str | None,
    valuation_date: date | None,
    known_party_ids: tuple[str, ...],
    option_party_ids: tuple[str, str],
    product_scope: str,
    product_label: str,
) -> tuple[FpMLPremiumMetadata, ...]:
    result = []
    for premium in _direct_children(product, "premium", namespace=namespace):
        _reject_unadmitted_direct_children(
            premium,
            allowed=_ALLOWED_PREMIUM_CHILDREN,
            scope=f"{product_scope}_premium",
            namespace=namespace,
        )
        payer = _required_href(
            premium,
            "payerPartyReference",
            namespace=namespace,
            missing_field="premium_payer_party_reference",
        )
        receiver = _required_href(
            premium,
            "receiverPartyReference",
            namespace=namespace,
            missing_field="premium_receiver_party_reference",
        )
        if (
            payer == receiver
            or not {payer, receiver}.issubset(set(known_party_ids))
            or {payer, receiver} != set(option_party_ids)
        ):
            _fail(
                f"contract_conflict:fpml_{product_scope}_premium_parties",
                "contract_conflict",
                f"{product_label} premium parties must be distinct identified trade parties.",
            )
        amount_element = _required_child(
            premium,
            "paymentAmount",
            namespace=namespace,
            missing_field="premium_payment_amount",
        )
        _reject_unadmitted_direct_children(
            amount_element,
            allowed=_ALLOWED_PAYMENT_AMOUNT_CHILDREN,
            scope="premium_payment_amount",
            namespace=namespace,
        )
        currency = _required_text(
            amount_element,
            "currency",
            namespace=namespace,
            missing_field="premium_currency",
        ).upper()
        amount = _finite_float(
            _required_text(
                amount_element,
                "amount",
                namespace=namespace,
                missing_field="premium_amount",
            ),
            field_name="premium_amount",
            positive=True,
        )
        _, payment_date = _adjustable_date(
            premium,
            "paymentDate",
            namespace=namespace,
        )
        if valuation_date is not None and payment_date >= valuation_date:
            _fail(
                f"external_import:fpml_{product_scope}_unsettled_premium_unsupported",
                "unsupported_contract",
                "Only premiums settled before the valuation date are admitted as "
                "separate source metadata.",
            )
        result.append(
            FpMLPremiumMetadata(
                payer_party_id=payer,
                receiver_party_id=receiver,
                payment_date=payment_date,
                currency=currency,
                amount=amount,
            )
        )
    return tuple(result)


def _normalize_cap_floor(
    cap_floor,
    *,
    namespace: str | None,
    valuation_party_id: str,
    valuation_date: date | None,
    known_party_ids: tuple[str, ...],
) -> tuple[
    StaticLegContractIR,
    tuple[FpMLFieldProvenance, ...],
    tuple[FpMLPremiumMetadata, ...],
]:
    """Normalize one bounded cap or floor into scheduled strip semantics."""

    _reject_unadmitted_direct_children(
        cap_floor,
        allowed=_ALLOWED_CAP_FLOOR_CHILDREN,
        scope="cap_floor",
        namespace=namespace,
    )
    _reject_nested_metadata_children(
        cap_floor,
        names=_CAP_FLOOR_METADATA_CHILDREN,
        namespace=namespace,
    )
    if _direct_children(cap_floor, "additionalPayment", namespace=namespace):
        _fail(
            "external_import:fpml_cap_floor_additional_payment_unsupported",
            "unsupported_contract",
            "Additional cap/floor payments are outside the admitted cohort.",
        )
    if _direct_children(
        cap_floor,
        "earlyTerminationProvision",
        namespace=namespace,
    ):
        _fail(
            "external_import:fpml_cap_floor_early_termination_unsupported",
            "unsupported_contract",
            "Early-termination provisions are outside the admitted cap/floor cohort.",
        )
    streams = _direct_children(cap_floor, "capFloorStream", namespace=namespace)
    if len(streams) != 1:
        _fail(
            "missing_contract_field:fpml_cap_floor_stream"
            if not streams
            else "contract_ambiguity:fpml_cap_floor_stream",
            "contract_gap" if not streams else "contract_ambiguity",
            "An admitted cap or floor requires exactly one capFloorStream.",
            missing_fields=("cap_floor_stream",) if not streams else (),
            ambiguous_fields=("cap_floor_stream",) if len(streams) > 1 else (),
        )
    stream = streams[0]
    _reject_unadmitted_direct_children(
        stream,
        allowed=_ALLOWED_STREAM_CHILDREN,
        scope="cap_floor_stream",
        namespace=namespace,
    )
    _reject_nested_metadata_children(
        stream,
        names=_STREAM_METADATA_REFERENCES,
        namespace=namespace,
    )
    if _contains_any(stream, _STUB_FIELDS, namespace=namespace):
        _fail(
            "external_import:fpml_stub_period_unsupported",
            "unsupported_contract",
            "Explicit first or last stub periods are outside the admitted cap/floor cohort.",
        )
    if _descendants(stream, "compoundingMethod", namespace=namespace):
        _fail(
            "external_import:fpml_compounding_unsupported",
            "unsupported_contract",
            "Compounded cap/floor periods are outside the admitted cohort.",
        )
    if _descendants(stream, "fixedRateSchedule", namespace=namespace):
        _fail(
            "external_import:fpml_cap_floor_fixed_coupon_unsupported",
            "unsupported_contract",
            "A fixed coupon cannot be combined with the admitted cap/floor strip.",
        )
    notional_schedule = _descendant(
        stream,
        "notionalStepSchedule",
        namespace=namespace,
    )
    if notional_schedule is not None and _direct_children(
        notional_schedule,
        "step",
        namespace=namespace,
    ):
        _fail(
            "external_import:fpml_amortizing_notional_unsupported",
            "unsupported_contract",
            "Amortizing or accreting notionals are outside the admitted cap/floor cohort.",
        )

    floating = _required_child(
        _required_child(
            _required_child(
                stream,
                "calculationPeriodAmount",
                namespace=namespace,
                missing_field="calculation_period_amount",
            ),
            "calculation",
            namespace=namespace,
            missing_field="calculation",
        ),
        "floatingRateCalculation",
        namespace=namespace,
        missing_field="floating_rate_calculation",
    )
    cap_schedules = _direct_children(
        floating,
        "capRateSchedule",
        namespace=namespace,
    )
    floor_schedules = _direct_children(
        floating,
        "floorRateSchedule",
        namespace=namespace,
    )
    if _direct_children(floating, "averagingMethod", namespace=namespace):
        _fail(
            "external_import:fpml_cap_floor_averaging_unsupported",
            "unsupported_contract",
            "Averaged rate observations are outside the admitted cap/floor cohort.",
        )
    if cap_schedules and floor_schedules:
        _fail(
            "external_import:fpml_cap_floor_collar_unsupported",
            "unsupported_contract",
            "A combined cap and floor schedule is a collar outside the admitted cohort.",
        )
    schedules = cap_schedules or floor_schedules
    if not schedules:
        _fail(
            "missing_contract_field:fpml_cap_floor_strike_schedule",
            "contract_gap",
            "An admitted cap or floor requires one strike schedule.",
            missing_fields=("strike_schedule",),
        )
    if len(schedules) != 1:
        _fail(
            "contract_ambiguity:fpml_cap_floor_strike_schedule",
            "contract_ambiguity",
            "An admitted cap or floor requires exactly one strike schedule.",
            ambiguous_fields=("strike_schedule",),
        )
    strike_schedule = schedules[0]
    option_side = "call" if cap_schedules else "put"
    strike, buyer_role, seller_role = _normalize_cap_floor_strike_schedule(
        strike_schedule,
        namespace=namespace,
    )

    terms = _normalize_stream(
        stream,
        kind="floating",
        namespace=namespace,
        valuation_party_id=valuation_party_id,
        allow_rate_option_strikes=True,
        stream_scope="cap_floor_stream",
    )
    if not {terms.payer, terms.receiver}.issubset(set(known_party_ids)):
        _fail(
            "contract_conflict:fpml_cap_floor_party_reference",
            "contract_conflict",
            "The cap/floor stream references a party not identified by the document.",
        )
    if terms.rate_index is None:
        raise AssertionError("normalized cap/floor stream has no rate index")
    if terms.spread != 0.0 or terms.gearing != 1.0:
        _fail(
            "external_import:fpml_cap_floor_coupon_transform_unsupported",
            "unsupported_contract",
            "The admitted pure cap/floor strip requires zero spread and unit gearing.",
        )

    role_parties = {"Payer": terms.payer, "Receiver": terms.receiver}
    option_buyer = role_parties[buyer_role]
    option_seller = role_parties[seller_role]
    if valuation_party_id == option_buyer:
        direction = "receive"
    elif valuation_party_id == option_seller:
        direction = "pay"
    else:  # pragma: no cover - _normalize_stream already proves membership
        raise AssertionError("valuation party is outside cap/floor counterparties")

    notional = NotionalSchedule((NotionalStep(terms.start, terms.end, terms.notional),))
    option_periods = tuple(
        PeriodRateOptionPeriod(
            accrual_start=period.accrual_start,
            accrual_end=period.accrual_end,
            fixing_date=period.fixing_date,
            payment_date=period.payment_date,
        )
        for period in terms.periods
        if period.fixing_date is not None
    )
    if len(option_periods) != len(terms.periods):
        raise AssertionError("normalized cap/floor period is missing a fixing date")
    contract = StaticLegContractIR(
        legs=(
            SignedLeg(
                direction,
                PeriodRateOptionStripLeg(
                    currency=terms.currency,
                    notional_schedule=notional,
                    option_periods=option_periods,
                    rate_index=terms.rate_index,
                    strike=strike,
                    option_side=option_side,
                    day_count=terms.day_count,
                    payment_frequency=terms.frequency_name,
                    label="cap_strip" if option_side == "call" else "floor_strip",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency=terms.currency),
        metadata={"semantic_family": "period_rate_option_strip"},
    )
    strike_name = "capRateSchedule" if option_side == "call" else "floorRateSchedule"
    base = "/dataDocument/trade/capFloor/capFloorStream"
    floating_base = (
        f"{base}/calculationPeriodAmount/calculation/floatingRateCalculation"
    )
    provenance = (
        _provenance("", "valuation_party_id", valuation_party_id),
        _provenance(
            f"{floating_base}/{strike_name}/buyer",
            "legs[0].direction",
            direction,
        ),
        _provenance(
            f"{base}/calculationPeriodDates",
            "legs[0].option_periods",
            len(option_periods),
        ),
        _provenance(
            f"{base}/calculationPeriodDates/effectiveDate/unadjustedDate",
            "legs[0].notional_schedule.steps[0].start_date",
            terms.start,
        ),
        _provenance(
            f"{base}/calculationPeriodDates/terminationDate/unadjustedDate",
            "legs[0].notional_schedule.steps[0].end_date",
            terms.end,
        ),
        _provenance(
            f"{base}/calculationPeriodAmount/calculation/notionalSchedule/"
            "notionalStepSchedule/initialValue",
            "legs[0].notional_schedule.steps[0].amount",
            terms.notional,
        ),
        _provenance(
            f"{base}/calculationPeriodAmount/calculation/notionalSchedule/"
            "notionalStepSchedule/currency",
            "legs[0].currency",
            terms.currency,
        ),
        _provenance(
            f"{base}/calculationPeriodAmount/calculation/dayCountFraction",
            "legs[0].day_count",
            terms.day_count,
        ),
        _provenance(
            f"{base}/paymentDates/paymentFrequency",
            "legs[0].payment_frequency",
            terms.frequency_name,
        ),
        _provenance(
            f"{floating_base}/floatingRateIndex",
            "legs[0].rate_index",
            f"{terms.rate_index.name}:{terms.rate_index.tenor}",
        ),
        _provenance(
            f"{floating_base}/{strike_name}/initialValue",
            "legs[0].strike",
            strike,
        ),
        _provenance(
            f"{floating_base}/{strike_name}",
            "legs[0].option_side",
            option_side,
        ),
    )
    premium_metadata = _normalize_option_premiums(
        cap_floor,
        namespace=namespace,
        valuation_date=valuation_date,
        known_party_ids=known_party_ids,
        option_party_ids=(option_buyer, option_seller),
        product_scope="cap_floor",
        product_label="Cap/floor",
    )
    return contract, provenance, premium_metadata


def _normalize_cap_floor_strike_schedule(
    schedule,
    *,
    namespace: str | None,
) -> tuple[float, str, str]:
    if _direct_children(schedule, "step", namespace=namespace):
        _fail(
            "external_import:fpml_step_strike_schedule_unsupported",
            "unsupported_contract",
            "Stepped cap/floor strikes are outside the admitted cohort.",
        )
    _reject_unadmitted_direct_children(
        schedule,
        allowed=_ALLOWED_STRIKE_SCHEDULE_CHILDREN,
        scope="cap_floor_strike_schedule",
        namespace=namespace,
    )
    strike = _finite_float(
        _required_text(
            schedule,
            "initialValue",
            namespace=namespace,
            missing_field="strike",
        ),
        field_name="strike",
    )
    buyer = _required_text(
        schedule,
        "buyer",
        namespace=namespace,
        missing_field="strike_buyer",
    )
    seller = _required_text(
        schedule,
        "seller",
        namespace=namespace,
        missing_field="strike_seller",
    )
    if {buyer, seller} != {"Payer", "Receiver"}:
        _fail(
            "contract_conflict:fpml_cap_floor_option_parties",
            "contract_conflict",
            "Cap/floor strike buyer and seller must be opposite Payer/Receiver roles.",
        )
    return strike, buyer, seller
