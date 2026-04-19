# Leg-Based Contract IR — Future Track Foundation

## Status

Draft. Parking-lot design document. Not yet an execution mirror and not
yet tied to a filed Linear child issue.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-904 — Phase 2 umbrella for payoff-expression Contract IR
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- `doc/plan/draft__contract-ir-phase-2-ast-foundation.md`
- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`
- `doc/plan/draft__contract-ir-phase-4-route-retirement.md`
- `docs/unified_pricing_engine_model_grammar.md`
- Existing implementation surfaces:
  - `trellis/models/cashflow_engine/*`
  - `trellis/models/contingent_cashflows.py`
  - `docs/mathematical/cashflow_engine.rst`

## Purpose

Capture the future track for products whose semantics are leg and
cashflow based rather than a single algebraic payoff expression. The
goal is to prevent Phase 2's payoff AST from becoming a dumping ground
for coupon schedules, accrual rules, or payment conventions, while still
preserving the route-retirement objective: a fresh build should
eventually compile these products from semantic IR rather than from
hard-coded per-instrument routes.

## Why This Is A Separate Track

The current Phase 2 `ContractIR` is a payoff-expression tree. That is
the right fit for products whose economic meaning is "evaluate this
observable expression at observation time(s)." It is not the right fit
for products whose economic meaning is "assemble and price these dated
cashflow legs."

Two future tracks need to stay distinct:

- **Quoted-observable products.** These settle on one or more explicit
  quoted market observables at an observation surface. Examples:
  vol-skew products paying a function of two implied-vol surface points,
  or a terminal 10Y-2Y curve-spread product paying
  `S_{10Y}(T) - S_{2Y}(T)`.
- **Leg-based cashflow products.** These are defined by accrual,
  fixing, payment, notional, and settlement rules across one or more
  legs. Examples: vanilla interest-rate swaps, SOFR-FF basis swaps,
  coupon bonds, CMS spread notes, callable coupon products.

Boundary rule: classify by contract semantics, not by desk nickname. A
trade described as a "basis" trade is leg-based only if the contract is
actually a schedule of exchanged coupon legs. If it is a one-shot
terminal spread payoff, it belongs to the quoted-observable track, not
this one.

## Design Objective

The later-track IR should let Trellis compile leg-based products into a
structural contract representation that is:

- additive alongside the current Phase 2 payoff-expression `ContractIR`
- explicit about accrual, fixing, payment, and settlement conventions
- independent of route ids, instrument strings, and backend-binding ids
- suitable for multiple lowerings: discounted cashflow engines, trees,
  Monte Carlo, or future pathwise evaluators
- honest about what is contractual semantics versus what is pricing
  convention or market-quote transform

## Non-Goals For The First Slice

- Do not solve callable / putable / cancelable structures in the first
  slice.
- Do not fold credit-default or structured waterfall products into the
  initial leg schema.
- Do not encode discounting directly in the contract nodes. The IR
  should represent undiscounted contractual obligations; lowerings
  decide how present value is computed.
- Do not collapse quoted-observable products into the leg track just
  because they mention rates or spreads.

## Semantic Requirements

Any serious leg-based IR must make the following contractual surfaces
explicit:

1. **Leg polarity.** Receive / pay cannot be implicit in a product name.
2. **Notional schedule.** Constant, step-up/down, amortizing, accreting.
3. **Coupon accrual periods.** Start date, end date, day count, stubs.
4. **Fixing rules.** Index, fixing date or lag, observation method,
   lookback / lockout / payment delay when relevant.
5. **Rate construction.** Fixed rate, simple floating rate, compounded
   overnight rate, averaged rate, CMS-style quoted rate, or future
   quote-observable coupon reference.
6. **Payment rules.** Payment date, payment lag, currency, settlement
   adjustments.
7. **Exchange rules.** Initial / final notional exchanges and fees.
8. **Event coupling.** Optional future support for callability, barriers,
   knockouts, accrual interruption, or default.

If any of those are hidden in one opaque leaf node, the IR will not be
good enough to support route-free fresh builds.

## Candidate Surface

This document does not lock the final ADT, but the minimal useful shape
looks closer to a leg/coupon algebra than to a single `PayoffExpr`
leaf.

Pseudo-ADT sketch:

```text
LegContractIR =
    { legs: tuple[SignedLeg, ...]
    ; settlement: SettlementRule
    ; exercise: Exercise | None
    ; observation: ObservationContext | None
    ; underlying: UnderlyingUniverse
    }

SignedLeg =
    { direction: Direction  # receive | pay
    ; leg: Leg
    }

Leg =
    | FixedCouponLeg(currency, notional_schedule, coupon_periods, fixed_rate)
    | FloatingCouponLeg(currency, notional_schedule, coupon_periods, rate_index, spread, gearing)
    | KnownCashflowLeg(currency, cashflows)

CouponPeriod =
    { accrual_start: Date
    ; accrual_end: Date
    ; payment_date: Date
    ; fixing_rule: FixingRule | None
    ; day_count: DayCount
    ; compounding: CompoundingRule
    }

FixingRule =
    | SpotFixing(observation_date: Date)
    | LaggedFixing(lag: int, calendar: str)
    | AveragedFixing(schedule: FiniteSchedule)
    | CompoundedOvernight(schedule: FiniteSchedule, lockout_days: int | None)

RateIndex =
    | FixedRate(value: float)
    | TermRateIndex(name: str, tenor: str)
    | OvernightIndex(name: str)
    | CmsQuote(name: str, tenor: str)
```

The important architectural point is not the exact field list. It is
that coupon assembly, fixing conventions, and signed legs are explicit
IR structure, not buried in route-local adapter logic.

## Examples

### Example 1 — Vanilla fixed-float IRS

Receive fixed 3.75% annually, pay 3M SOFR quarterly, notional 10mm,
five-year maturity.

This is leg-based because the contract is the difference between two
coupon schedules. The fixed leg and floating leg each require accrual
periods, payment dates, and notional handling.

### Example 2 — SOFR-FF basis swap

Receive compounded SOFR plus spread on one quarterly schedule, pay Fed
Funds plus spread on another quarterly schedule.

This is also leg-based. Even if the desk thinks of it as "SOFR minus
FF," the product is not one terminal spread observation. It is a pair of
floating coupon legs with potentially different fixing and compounding
rules.

### Example 3 — Terminal 10Y-2Y spread product

Pay `max(S_{10Y}(T) - S_{2Y}(T) - K, 0)` at one expiry.

This is **not** leg-based. It belongs to the future quoted-observable
extension of the payoff-expression `ContractIR`, because the contract is
still a one-shot function of observed quote points.

## Relationship To The Current Phase 2 Contract IR

The likely long-run shape is a broader semantic contract surface that
admits at least two sibling representations:

- **Payoff-expression Contract IR** for terminal / schedule / path
  observable payoffs
- **Leg-based Contract IR** for coupon schedules and dated obligations

They should coexist rather than force one representation to impersonate
the other. A later unifying root may wrap both, but the first step is to
let each domain have honest semantics.

## Dependency On Phases 3 And 4

This later track depends on the authority model proven by the
payoff-expression Contract IR program.

Concretely:

- Phase 3 proves that fresh-build solver selection can be driven from
  structural IR plus market capabilities rather than route ids.
- Phase 4 proves that migrated fresh builds can actually delete
  route-local authority while preserving provenance and replay.

The leg-based track should reuse that authority model. It should not
invent a second dispatch regime just because the semantic nodes are
coupon / leg shaped instead of payoff-expression shaped.

## First Implementable Slice

When this track is eventually promoted from parking lot to active work,
the first useful scope should stay narrow:

1. Vanilla fixed-float interest-rate swaps
2. Float-float basis swaps, starting with SOFR-FF or a closely related
   desk-supported pair
3. Static coupon bonds only if they naturally reuse the same dated
   cashflow representation without adding optionality

Deferred from that first slice:

- callable / putable structures
- amortizing or accreting schedules if they materially complicate the
  first schema
- CMS coupons and CMS spread notes
- range accruals, barriers, and event-coupled coupon interruption
- credit default swaps and contingent default legs

## Pricing Boundary

This track should follow the same discipline adopted in the payoff IR:

- contract nodes represent contractual semantics
- lowerings represent pricing method and quote convention

Examples:

- discounting belongs in the lowering, not the leg node
- projected floating coupons belong in the lowering, not as cached
  numbers embedded in the IR
- par-rate or implied-vol transforms belong in quoted-observable leaves
  or lowering helpers, not in the contract root

## Risks To Avoid

- **Recreating route ids in disguise.** A `FloatingLeg(kind="sofr_ff_basis")`
  node would just smuggle route logic back into the IR.
- **Baking pricing engines into the contract.** A leg node should not
  say "price with discounted cashflow engine X."
- **Mixing quote transforms with contractual semantics.** CMS and
  vol-quoted coupons need explicit quote references, not magical
  "already transformed" numbers.
- **Skipping coupon-level tests.** This track will need fixture-driven
  tests at the coupon-period level, not just whole-product regression
  tests.

## Open Questions

1. Should the long-run semantic root be one tagged union with
   `payoff_contract | leg_contract`, or should the two IRs stay separate
   and only meet at the blueprint level?
2. How much schedule normalization belongs in the IR constructor versus
   in upstream decomposition helpers?
3. Should coupon sign live on the leg (`receive` / `pay`) or as an outer
   scalar multiplier?
4. When CMS-style coupons arrive, do they reference the quoted-
   observable track directly, or do they define a coupon-local quoted
   rate abstraction?
5. Which live desk product is the best first proving ground:
   vanilla IRS or SOFR-FF basis swap?

## Next Steps

1. Keep this document as the parking-lot spec for the leg-based track
   while Phase 2 and Phase 3 land.
2. When the quoted-observable follow-on is scoped, explicitly confirm
   the boundary between that track and this one with real fixtures.
3. File a future Linear child issue under QUA-887 for the first active
   leg-based slice.
4. Before implementation, audit reuse opportunities in the existing
   cashflow engine and contingent-cashflow modules so the new IR does
   not duplicate already-correct schedule machinery.
