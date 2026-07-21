FpML Import And Normalization
=============================

Trellis treats FpML as an explicit external contract source, not as its
internal derivative representation. The first executable cohort is deliberately
small: one FpML 5.13 confirmation-view ``dataDocument`` containing one regular,
single-currency, constant-notional fixed-float interest-rate swap.

Lifecycle
---------

The governed request path is deterministic:

1. ``make_fpml_request(...)`` creates an ``ImportedDocumentPayload`` and a
   separate ``TradeEnvelope``.
2. request preflight requires inline bytes plus declared view and version; it
   never resolves an external URI.
3. ``inspect_fpml_document(...)`` applies the secure XML limits and extracts
   body-free document/trade identity.
4. caller and document provenance are reconciled before economics are used.
5. ``normalize_fpml_document(...)`` maps the admitted economics into
   ``StaticLegContractIR`` and records XML-path mapping provenance.
6. the ordinary static-leg structural selector lowers that contract to
   ``ContractExecutionIR``.
7. ``ExecutionBackedPayoff`` prices the execution artifact through the generic
   ``price_normalized_payoff`` platform action.

The FpML ``swap`` wrapper is only a normalizer dispatch fact. It does not select
``static_leg_fixed_float_swap``, ``SwapPayoff``, a validation bundle, a model,
or a solver. Those artifacts are selected from the normalized leg structure.
The admitted path performs no natural-language parsing, LLM call, code
generation, cookbook update, or model-validator review.

Valuation Perspective
---------------------

An FpML swap stream identifies its payer and receiver, but the document does
not choose whose NPV a caller wants. A pricing request must therefore identify
exactly one ``TradeParty`` with ``role="valuation_party"``. The normalizer maps
each stream to ``pay`` or ``receive`` relative to that party.

A missing valuation party produces
``missing_contract_field:fpml_valuation_party_id``. A party that is not in the
document produces ``contract_conflict:fpml_valuation_party_id``. Trellis never
chooses the first party or assumes the reporting party owns the requested NPV.

Admitted Economics
------------------

The first fixed-float cohort requires:

- exactly two ``swapStream`` elements, one fixed and one floating
- one currency and matching positive constant notionals
- matching effective and termination dates
- regular annual, semiannual, quarterly, or monthly calculation and payment
  schedules with one calculation period per payment
- matching payer/receiver counterparty pairs that reference parties declared
  in the document
- fixed rate, supported day counts, and a term floating-rate index whose one
  index tenor matches the calculation and reset frequency
- reset dates relative to calculation-period starts
- fixed and floating rate schedules without steps; a constant spread or
  multiplier may be represented
- internal FpML schedule references that identify the schedule they claim to
  reference

Pricing requests also require a deterministic valuation date. Until the
static-leg runtime consumes historical fixing histories, Trellis rejects a
swap with an unpaid floating coupon whose fixing date is on or before that
valuation date. Build-only normalization remains independent of valuation
date.

Supported business-day conventions are ``NONE``, ``FOLLOWING``,
``MODFOLLOWING``, ``PRECEDING``, and ``MODPRECEDING``. Adjusted dates require
an admitted explicit business center. The current center map is bounded to the
corresponding Trellis calendars for ``AUSY``, ``BRSP``, ``CATO``, ``CHZU``,
``EUTA``, ``GBLO``, ``JPTO``, and ``USNY``.

When FpML supplies an ``adjustedDate`` for an effective or termination date,
the normalizer requires it to equal the date recomputed from the unadjusted
date and declared adjustments. Payment dates relative to
``CalculationPeriodEndDate`` are anchored to adjusted calculation-period ends
before their own payment-date adjustment is applied.

Fail-Closed Boundary
--------------------

Unsupported amortization, compounding, stubs, cross-currency legs, stepped
rates or spreads, mismatched schedules or notionals, non-term floating-rate
forms, duplicate optional rate schedules, unpaid seasoned floating coupons,
duplicate roll declarations, fixed-leg reset schedules, initial-fixing
overrides, mismatched counterparty pairs, end-of-month or clamped high-day
rolls, foreign-namespaced extension children, unsupported
frequency, tenor, date-adjustment, or business-center children,
calendars/conventions, lifecycle content, and missing schedule or valuation-date
terms produce exact ``external_import:*``,
``missing_contract_field:*``, ``contract_ambiguity:*``, or
``contract_conflict:*`` blockers. The report carries clarification fields when
the caller can supply missing or disambiguating information.

The importer is not a complete FpML schema validator. It supports one view,
version, root, trade count, and product cohort; it does not claim package,
lifecycle, recordkeeping, basis, cross-currency, OIS compounding, inflation,
swaption, or cap/floor coverage.

Identity And Provenance
-----------------------

``static_leg_economic_identity(...)`` hashes a versioned canonical projection
of the normalized static-leg economics. Labels, metadata, XML paths, document
ids, and source format are excluded, so an equivalent native contract and FpML
contract have the same identity. Signed leg direction, schedules, notionals,
coupon formulas, indices, and settlement remain part of the identity.

``FpMLImportReport.mapping_provenance`` separately records XML paths and their
normalized semantic fields. ``fpml_import_report_summary(...)`` exposes the
identity, economic projection, and mapping evidence without retaining raw XML
or parser nodes.
