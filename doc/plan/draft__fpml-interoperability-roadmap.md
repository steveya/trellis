# FpML Interoperability Roadmap

## Status

Draft. Cross-cutting roadmap for external interoperability. Not yet an
execution mirror.

No linked Linear umbrella exists yet. The `FPI.*` queue ids below are
repo-local placeholders until the implementation program is filed.

Status snapshot as of `2026-04-23`.

## Linked Context

- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/draft__post-phase-4-semantic-closure-execution-plan.md`
- `doc/plan/draft__semantic-contract-target-and-trade-envelope.md`
- `doc/plan/draft__valuation-session-and-request-surface.md`
- `doc/plan/draft__portfolio-path-and-result-set-surface.md`
- `doc/plan/draft__valuation-result-identity-and-provenance.md`
- `doc/plan/draft__external-prior-art-adoption-map.md`
- existing repo surfaces:
  - `trellis.agent.platform_requests.PlatformRequest`
  - `trellis.agent.valuation_context.ValuationContext`
  - `trellis.agent.market_binding.MarketBindingSpec`
  - `trellis.agent.semantic_contracts.SemanticContract`
  - `trellis.agent.contract_ir.ContractIR`
  - `trellis.agent.static_leg_contract.StaticLegContractIR`
  - `trellis.agent.dynamic_contract_ir.DynamicContractIR`

## Purpose

Define a backward roadmap from the long-run FpML interoperability target
to the smallest honest first implementation slice.

This note is intentionally not a claim that "Trellis should price
anything FpML can represent" in the literal sense. FpML also represents:

- parties and identifiers
- confirmations and reporting payloads
- lifecycle events such as novations, terminations, and amendments
- workflow messages that are not themselves price targets

The useful ambition is narrower:

- Trellis should eventually ingest any bounded FpML product or lifecycle
  state that contains enough economic detail
- normalize that payload into Trellis semantic representations plus a
  separate trade envelope
- price the economically well-defined subset whose representation,
  decomposition, lowering, parity, and provenance are closed inside
  Trellis

## End State

The target architecture should satisfy all of the following.

1. **External-document import is explicit.**
   FpML view, version, document identity, and extraction mode are
   carried on a request-side import surface, not hidden in text parsing.
2. **Trade-envelope separation is real.**
   Parties, identifiers, package metadata, booking metadata, and
   lifecycle payloads live outside the semantic contract itself.
3. **Economics normalize onto Trellis semantic tracks.**
   Imported economics land on:
   - `SemanticContract`
   - `ContractIR`
   - `StaticLegContractIR`
   - `DynamicContractIR`
4. **Valuation policy remains independent.**
   Imported documents do not become a substitute for `ValuationContext`
   or `MarketBindingSpec`.
5. **Support claims remain closure-based.**
   FpML import breadth never outruns Trellis' internal semantic closure.
6. **Result navigation preserves imported identity.**
   Imported document, trade, and package provenance remain visible in
   path-aware result surfaces.

## Non-Goals

- Do not use the raw FpML schema as Trellis' internal semantic IR.
- Do not let FpML message type or product wrapper names become solver
  selectors.
- Do not claim that every FpML-representable payload is priceable.
- Do not mix lifecycle transport semantics with semantic product meaning.
- Do not broaden into recordkeeping, reporting, or lifecycle mutation
  flows before the confirmation/current-state slice is stable.

## Reverse Dependency Ladder

Work backward from the end state in this order.

### Layer 6 — Industrial interoperability

- versioned FpML support contract
- bulk import and reconciliation
- package/book integration
- provenance and audit closeout

### Layer 5 — Dynamic and lifecycle-rich products

- imported dynamic wrappers become executable
- imported lifecycle state can be projected onto current economic state
- recordkeeping and confirmation sources can reconcile

### Layer 4 — Cross-asset current-state product breadth

- multiple asset classes
- bounded exotics where Trellis already has real pricing lanes
- no lifecycle mutation dependence yet

### Layer 3 — Current-state external import

- bounded confirmation import
- economics separated from trade envelope
- one trade at a time

### Layer 2 — Request and provenance seam

- explicit `fpml` request entry point
- explicit imported-document metadata
- stable fail-closed diagnostics for unsupported constructs

### Layer 1 — Target-side boundary

- first-class trade envelope
- result-path vocabulary that can later carry document/trade/package axes

### Layer 0 — Existing semantic closure

- Trellis' own representation / decomposition / lowering closure remains
  the authority boundary

## Delivery Rule

Every cohort must satisfy the same honesty rule:

- importing a payload is not the same as supporting its pricing
- representation is not the same as executable lowering
- executable lowering is not the same as parity and provenance closeout

If an imported FpML construct exceeds the currently closed Trellis
family, the importer should fail closed with an explicit blocker instead
of silently coercing it into a nearby product family.

## Cohort Table

| Cohort | Status | Objective | Hard prerequisites |
| --- | --- | --- | --- |
| `FPI.0` | Proposed | foundation seam for imported documents and trade envelopes | none |
| `FPI.1` | Proposed | confirmation-view vanilla-rates current-state pricing | `FPI.0` |
| `FPI.2` | Proposed | rates breadth and package hygiene | `FPI.1` |
| `FPI.3` | Proposed | cross-asset current-state single-trade coverage | `FPI.1` |
| `FPI.4` | Proposed | lifecycle-state and alternate-view normalization | `FPI.2`, `FPI.3` |
| `FPI.5` | Proposed | dynamic and stateful imported exotics | `FPI.4` plus executable dynamic lowering inside Trellis |
| `FPI.6` | Proposed | industrial interoperability closeout | `FPI.5` |

## Cohort Details

### FPI.0 — Foundation seam

Objective:

Reserve the right architecture before any real product-support claim is
made.

Queue:

- `FPI.0a` add a first-class `TradeEnvelope` surface for imported trade
  metadata and source-document provenance
- `FPI.0b` add a distinct `fpml` entry point to
  `trellis.agent.platform_requests`
- `FPI.0c` add `trellis/io/fpml/` with fixture-backed XML loading and a
  stable unsupported-construct error vocabulary
- `FPI.0d` add invariance tests proving envelope-only metadata cannot
  change semantic selection
- `FPI.0e` document the import boundary and failure policy

Acceptance:

- FpML import has its own request seam
- trade metadata no longer needs to be smuggled through semantic
  contract fields
- unsupported constructs fail closed at a named import boundary

### FPI.1 — Confirmation-view vanilla rates

Objective:

Ship the first honest end-to-end import slice.

Scope:

- confirmation view only
- one trade per request
- fixed-float IRS
- European payer / receiver swaption
- scheduled cap / floor strips on the canonical
  `period_rate_option_strip` track
- optional stretch: constant-notional basis swap

Queue:

- `FPI.1a` implement confirmation parser adapter for fixed-float IRS
  onto `StaticLegContractIR`
- `FPI.1b` implement confirmation parser adapter for European swaption
  onto the existing semantic + `ContractIR` path
- `FPI.1c` implement confirmation parser adapter for scheduled cap/floor
  strips onto canonical static-leg semantics
- `FPI.1d` add structural-selection and parity tests against existing
  checked lanes
- `FPI.1e` add explicit blockers for amortization, cross-currency,
  Bermudan exercise, and lifecycle event payloads
- `FPI.1f` publish bounded user/developer docs

Acceptance:

- a bounded confirmation payload can be parsed, normalized, and priced
  through checked lanes
- envelope-only metadata changes do not affect structural selection
- unsupported rates payloads block honestly

### FPI.2 — Rates breadth and package hygiene

Objective:

Widen the rates slice without yet depending on lifecycle mutation or
dynamic execution.

Scope:

- richer static-leg conventions
- first package/document identity surfaces
- still current-state only

Queue:

- `FPI.2a` broaden static-leg import for richer day-count, stub, and
  compounding conventions that remain inside admitted static-leg closure
- `FPI.2b` make basis-swap import first-class rather than stretch-only
- `FPI.2c` add bounded package/document import for multiple
  independently priceable rates trades
- `FPI.2d` add result-path support for imported document, package, and
  trade axes
- `FPI.2e` add fail-closed handling for package semantics that imply
  unsupported netting or path-dependent package logic

Acceptance:

- the rates importer is no longer limited to toy single-trade fixtures
- package identity is visible without becoming a semantic classifier

### FPI.3 — Cross-asset current-state single-trade coverage

Objective:

Extend the current-state import architecture across product families that
already have honest Trellis pricing lanes.

Candidate proving families:

- FX vanilla
- equity vanilla
- CDS
- variance swaps
- bounded basket/spread products
- quanto where the existing lane remains support-contract-correct

Queue:

- `FPI.3a` add one adapter map per imported family onto existing Trellis
  semantic families
- `FPI.3b` add a per-family closure gate so import support cannot outrun
  representation and lowering closure
- `FPI.3c` add cross-asset golden fixtures plus parity and blocker tests
- `FPI.3d` publish explicit unsupported-family diagnostics for payloads
  that exceed current Trellis closure

Acceptance:

- Trellis can import and price a bounded single-trade, current-state
  subset across multiple asset classes
- support claims remain family-by-family and closure-based

### FPI.4 — Lifecycle state and alternate-view normalization

Objective:

Move from pristine current-state confirmations to bounded projections of
current economic state.

Scope:

- amendments
- partial terminations
- novation-style envelope changes
- recordkeeping as an alternate current-state source

Queue:

- `FPI.4a` introduce a canonical imported lifecycle-state surface on the
  target/envelope side
- `FPI.4b` define a bounded "current economic state" projection from
  confirmation plus lifecycle payloads
- `FPI.4c` add recordkeeping-view import for the overlapping priced
  subset
- `FPI.4d` add reconciliation tests between confirmation-derived and
  recordkeeping-derived current state
- `FPI.4e` fail closed on lifecycle flows that require unsupported
  dynamic execution semantics

Acceptance:

- Trellis can price the current state of a bounded imported contract,
  not just the original pristine trade description

### FPI.5 — Dynamic and stateful imported exotics

Objective:

Make imported dynamic families executable only where Trellis' own
dynamic-lowering lanes are executable.

Queue:

- `FPI.5a` wire imported automatic event/state families into checked
  executable dynamic lanes
- `FPI.5b` wire imported discrete-control families into checked
  executable dynamic lanes
- `FPI.5c` wire imported continuous/singular-control families into
  checked executable dynamic lanes
- `FPI.5d` preserve insurance overlays as a separate wrapper boundary
  until the executable support contract is real
- `FPI.5e` add parity, provenance, and masked-authority gates before any
  imported dynamic cutover claim

Acceptance:

- imported autocallable, callable, swing, or GMxB-style products are
  priceable only where Trellis can already lower those dynamic semantics
  honestly

### FPI.6 — Industrial interoperability closeout

Objective:

Turn bounded import slices into a durable external interoperability
surface.

Queue:

- `FPI.6a` publish a versioned FpML support matrix by view, family, and
  lifecycle state
- `FPI.6b` add schema/version/document provenance reporting on imported
  requests
- `FPI.6c` add bulk import, audit, and reconciliation tooling for books
  and packages
- `FPI.6d` align imported results with the path-aware result-set design
- `FPI.6e` publish the supported FpML-to-Trellis benchmark and evidence
  ledger

Acceptance:

- Trellis exposes a versioned, benchmarked, support-contract-correct
  interoperability surface rather than a collection of ad hoc adapters

## Pickup Rule

Use this rule when selecting the next cohort.

- do not import a family before Trellis has an honest semantic home for
  it
- do not claim support when import works but executable lowering does
  not
- do not add lifecycle mutation imports before the target/envelope split
  is landed
- do not widen into dynamic imported exotics before current-state
  single-trade import is stable

## Immediate Next Slice

If implementation began now, the smallest coherent queue would be:

1. `FPI.0a` trade envelope
2. `FPI.0b` `fpml` request entry point
3. `FPI.0c` XML fixture/parser harness
4. `FPI.1a` IRS adapter
5. `FPI.1b` swaption adapter
6. `FPI.1c` cap/floor adapter
7. `FPI.1d` invariance and parity tests
8. `FPI.1f` docs

That slice proves the architecture without over-claiming support beyond
the currently realistic Trellis closure surface.
