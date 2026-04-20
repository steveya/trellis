# Contract IR — Phase 4: Route Retirement And Semantic Lowering Pipeline Cutover

## Status

Draft. Pre-queue design document. Not yet the live execution mirror for
Phase 4.

## Linked Linear

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-905 — Phase 3 umbrella (structural solver compiler)
- QUA-906 — Phase 4 umbrella (this plan expands)
- QUA-904 — Phase 2 umbrella (semantic IR substrate already additive)

## Companion Docs

- `doc/plan/draft__external-prior-art-adoption-map.md`
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/draft__contract-ir-phase-2-ast-foundation.md`
- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`
- `doc/plan/draft__market-coordinate-overlay-and-shock-model.md`
- `doc/plan/draft__semantic-contract-target-and-trade-envelope.md`
- `doc/plan/draft__valuation-result-identity-and-provenance.md`
- `doc/plan/draft__quoted-observable-contract-ir-foundation.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `doc/plan/draft__post-phase-4-semantic-closure-execution-plan.md`
- `doc/plan/draft__contract-ir-compiler-retiring-route-registry.md`
- `docs/quant/contract_ir.rst`
- `docs/quant/contract_algebra.rst`

## Purpose

Specify the cutover phase that turns the structural compiler from
shadow-mode evidence into the only fresh-build authority for supported
surface.

This document names the route-free fresh-build mechanism the
**Semantic Lowering Pipeline**:

1. semantic contract validation
2. admitted semantic-IR emission
3. structural solver selection
4. lowering / adapter assembly
5. exact helper or kernel binding
6. provenance and validation attachment

The core requirement is simple and strict:

- for supported fresh-build families, a rebuild must no longer depend on
  direct hard-coded route ids, route-family tables,
  `ProductIR.instrument`, or opaque trade-wrapper labels
- the build must go through the Semantic Lowering Pipeline, which is
  Contract-IR-backed for the current admitted cohort
- Phase 4 is not complete while any supported fresh-build family still
  depends on route-local selector authority

Phase 4 is therefore not new pricing math. It is governed removal of
redundant authority after Phase 3 has already proved parity.

## Framing

### What "route-free fresh build" means

For a migrated contract, the solver-selection result must be determined
by:

- `contract_ir`
- requested method / valuation policy
- market capability surface
- valuation market identity, including overlay / scenario identity when
  applicable
- explicit structural declaration precedence

and NOT by:

- `ProductIR.instrument`
- route ids
- route families
- compatibility aliases
- hand-written per-instrument branch ladders

By the same logic, it also must not be determined by opaque trade-wrapper
labels once Trellis introduces a thinner contract-target envelope.
Trade date, desk tags, booking metadata, or package wrappers may exist,
but they are not allowed to become replacement route ids.

This is the success criterion that motivated the whole Contract IR
program.

### What route-free does NOT mean

Phase 4 does not require deleting every legacy artifact from the repo on
day one, and it does not require future not-yet-admitted semantic
families to exist before the current supported surface can cut over.

It is valid to retain:

- replay-time route aliases
- provenance ids for historical comparison
- summary projections from `ContractIR` back onto `ProductIR`
- transitional fallback for families not yet admitted into the migration
  set during rollout

as long as those artifacts are no longer selection authority for
migrated fresh-build surface.

The Phase 4 exit condition is stricter than the rollout condition:

- by Phase 4 close, supported fresh-build surface must no longer rely on
  route fallback at all
- any residual route ids must be replay-only, audit-only, or
  compatibility-only
- future families admitted after Phase 4 must plug directly into the
  Semantic Lowering Pipeline instead of reintroducing temporary route
  selectors

### Closure role of Phase 4

The semantic-contract closure model is defined in
`doc/plan/draft__semantic-contract-closure-program.md`.

Phase 4 does not create representation closure, decomposition closure,
or lowering closure. It consumes them.

Its job is narrower and stricter:

- once a family is already semantically closed
- and parity / provenance evidence exists
- remove route-local fresh-build authority for that family

If Phase 4 discovers that a family still depends on a hidden
product-specific payload, missing semantic domain, ad hoc route-local
normalization, or trade-wrapper discriminators, that family is not ready
for migration into `\mathcal{D}_{\text{mig}}`.

That rollout rule does not weaken the phase-exit rule. Phase 4 is only
done when the supported fresh-build universe has been fully moved onto
the Semantic Lowering Pipeline and route authority has become non-
authoritative everywhere on that surface.

Phase 4 also must preserve route-free valuation identity. Removing route
authority must not collapse operator meaning onto a thinner,
less-auditable surface. The deletion phase should preserve stable
identity for:

- the structural declaration
- the valuation snapshot / market identity
- any overlay / scenario identity
- requested output identity
- exact helper / kernel provenance

### Replay and provenance are separate from dispatch

Fresh build and replay are different modes.

- **Fresh build** asks: what should Trellis select now for this semantic
  contract?
- **Replay / audit** asks: what did we previously bind and validate?

Replay may still carry legacy route ids during transition. Fresh build
must not read them once a family is migrated.

## Mathematical Invariants

### Legacy projection is observational only

Let

$$\pi : ContractIR \to ProductIR$$

be the lossy summary projection already discussed in the Phase 2 plan.

For the current migrated slice, `ContractIR` is the correct domain on
the left-hand side. The broader semantic-contract closure program will
later need the same invariant generalized from this bounded
`ContractIR` cohort to a wider semantic root spanning quoted-
observable, static leg, and event/state/control domains.

Phase 4's contract is that, for migrated surface, dispatch is no longer
a function of `\pi(c)`:

$$\operatorname{Select}_{\text{fresh}}(c, v, m) \neq f(\pi(c))$$

in the sense of authority. `\pi(c)` may still be emitted for telemetry,
searchability, or compatibility, but selection correctness for migrated
surface must be invariant under changes to legacy metadata.

### Metadata-masking invariance

For any migrated contract `c` and any two metadata packets
`h_1, h_2` that differ only in non-semantic selector-forbidden fields
such as route-local metadata or trade-envelope wrappers,

$$\operatorname{Select}_{\text{fresh}}(c, v, m, h_1)
=
\operatorname{Select}_{\text{fresh}}(c, v, m, h_2).$$

This invariant is the formal phase-exit condition for "route-free fresh
build."

This includes the non-semantic trade-envelope fields described in
`doc/plan/draft__semantic-contract-target-and-trade-envelope.md`.
Changing external ids, booking tags, or similar wrapper metadata must
not change migrated fresh-build selection unless that data has been
explicitly promoted into semantic contract or valuation-policy
authority.

The corresponding non-invariant is also required:

- changing the actual valuation market or scenario overlay may change
  the result
- changing route aliases or instrument strings must not

So route retirement removes route-local authority while retaining
correct sensitivity to valuation identity.

### Migration domain

Let `\mathcal{D}_{\text{mig}}` be the subset of Contract IR space whose
families have:

1. documented representation closure on an admitted semantic IR domain
2. documented decomposition closure from supported request surfaces into
   that semantic domain
3. a shipped lowering / declaration set for the relevant method cohort
4. shadow-mode parity evidence
5. migrated observability and provenance surfaces

Phase 4 only deletes authority for `c \in \mathcal{D}_{\text{mig}}`.
Everything else keeps the old fallback until it joins that set.

Let `\mathcal{U}_{\text{sup}}` be the supported fresh-build universe in
scope for this Phase 4 program. Then the rollout and exit rules are:

- during rollout, route authority may be deleted family-by-family for
  `c \in \mathcal{D}_{\text{mig}}`
- Phase 4 exits only when
  `\mathcal{D}_{\text{mig}} = \mathcal{U}_{\text{sup}}`
- after that equality holds, route ids may remain only on replay,
  provenance, and compatibility surfaces, not on fresh-build selection

For the current payoff-expression migration tranche, that bounded
definition is enough. Future post-Phase-4 closure tracks should inherit
the same rule by replacing the left-hand domain with the broader
semantic root once those sibling representations are admitted.

### Parity contract

For each validation case `x = (c, v, m)` in the migrated cohort,

$$
\left|
PV_{\text{legacy}}(x) - PV_{\text{fresh-ir}}(x)
\right|
\le
\varepsilon_{\text{abs}} + \varepsilon_{\text{rel}} \cdot S(x)
$$

where `S(x)` is the scale policy for that family.

Phase 4 may only delete a legacy path after the relevant family's
parity policy is satisfied on the agreed benchmark and fixture set.

### Provenance contract

Once fresh-build authority shifts, the emitted provenance packet must
still identify:

- the selected structural declaration id
- the exact helper / kernel refs used
- the valuation identity for the market snapshot
- overlay / scenario identity when present
- requested output identity
- resolved market-coordinate references when the lowering can express
  them faithfully
- the validation bundle covering that declaration
- compatibility alias policy for historical lookups

Phase 4 changes *who selects* the path, not whether the selected path is
auditable.

## What Gets Deleted, In What Order

### 4A. Primary selector authority flips first

The first deletion is conceptual:

- `rank_primitive_routes(...)` and equivalent fresh-build selection
  paths stop using legacy route registry results as primary authority
  for migrated families
- structural compiler output becomes primary for migrated families
- legacy route matching becomes fallback only for unmigrated surface

This is the smallest possible change that proves the program goal.

### 4B. Route-card clauses retire family by family

After the selector flip is validated, remove redundant route authority
for migrated families:

- `routes.yaml` conditional clauses that are no longer consulted for
  migrated fresh builds
- corresponding backend-binding authority that duplicates the structural
  declaration set

Deletion must be per family / per slice, not one giant PR.

### 4C. `ProductIR.instrument` dispatch reads retire next

The field should be audited by consumer class:

1. dispatch / compiler reads
2. observability / trace reads
3. benchmark / learning / reporting reads
4. compatibility-only reads

Delete in exactly that order.

The field may temporarily remain as a derived summary while observability
and reporting consumers migrate. It must no longer be read on the
fresh-build selector path once Phase 4A lands for a migrated family.

### 4D. Observability migrates last

Operator tooling still needs stable identities.

So traces, scorecards, and replay summaries should move from
"route id as primary meaning" to:

- structural declaration id
- Contract IR family / pattern identity
- valuation identity key
- market / overlay / scenario identity
- exact helper / kernel provenance

Only after that migration is stable should route ids be removed from the
operator-facing primary identity.

## Required Architecture Outcome

### Fresh-build path

For migrated families the intended pipeline is:

1. semantic contract validation
2. admitted semantic-IR emission for the family in scope
3. structural solver selection from that admitted semantic IR
4. adapter / lowering assembly
5. exact helper / kernel binding
6. provenance + validation metadata attachment

For the current Phase 4 tranche, the admitted semantic IR is
`ContractIR`. Future sibling semantic homes should plug into the same
Semantic Lowering Pipeline without restoring route authority.

`ProductIR` may still be emitted as a lossy compatibility or observability
projection, but it is not a required authority-bearing stage on the
fresh-build selector path.

There is no direct branch of the form:

- `if product_ir.instrument == ...`
- `if route_family == ...`
- `if route_id == ...`
- `if trade_envelope.trade_type == ...`
- `if desk_tag == ...`

on this fresh-build path.

### Replay path

Replay and audit may still carry:

- route aliases
- historical binding ids
- legacy identifiers needed to compare against archived canaries

but that logic must stay outside the selector for migrated fresh-build
surface.

## Ordered Sub-Ticket Queue

### P4.1 — Primary structural selector switch

**Objective.** Make the structural compiler the primary selector for
migrated families in fresh-build code paths.

**Artifacts.**

- selector integration in `trellis/agent/`
- route-masked and envelope-masked regression tests
- feature flag / rollout guard if needed

### P4.2 — Family-by-family route-card retirement

**Objective.** Delete redundant `routes.yaml` and binding-catalog
authority for migrated families, and close the residual inventory until
supported fresh-build surface has no selector dependence on route.

**Artifacts.**

- per-family deletion PRs
- shrinking residual inventory during rollout
- no fresh-build route fallback remaining at phase exit

### P4.3 — `ProductIR.instrument` dispatch-read audit

**Objective.** Enumerate and remove every production selector read of
`ProductIR.instrument`.

**Artifacts.**

- read-site inventory
- dispatch-read removals
- explicit residual list for observability-only reads

### P4.4 — Provenance and trace migration

**Objective.** Move operator-facing identity away from route ids and
toward structural declaration + binding provenance.

**Artifacts.**

- trace payload updates
- scorecard / replay identity updates
- stable valuation-identity / provenance-key surface
- documentation of the new primary identity

### P4.5 — Fresh-build hard guardrails

**Objective.** Prevent regressions back to direct route-local selection.
This includes regressions that try to move the same authority onto trade
or booking wrapper metadata.

**Artifacts.**

- regression tests that mask route metadata and non-semantic trade
  envelope metadata and expect the same outcome
- guard tests that fail if migrated families reintroduce
  `ProductIR.instrument`-keyed branching
- guard tests that fail if selector logic keys off non-semantic trade
  envelope labels

### P4.6 — Knowledge, docs, and closeout

**Objective.** Update the official docs and any knowledge assets that
still describe route ids as the primary fresh-build authority.

**Artifacts.**

- `docs/quant/`
- `docs/developer/`
- relevant knowledge / prompt text describing migrated family selection

## Acceptance Criteria

- Every family admitted into `\mathcal{D}_{\text{mig}}` has an explicit
  closure record: representation closure, decomposition closure,
  lowering closure, parity evidence, and provenance-readiness.
- Phase 4 does not close until
  `\mathcal{D}_{\text{mig}} = \mathcal{U}_{\text{sup}}` for the
  supported fresh-build universe in scope.
- For every migrated family, masking `ProductIR.instrument`,
  `route_id`, `route_family`, and non-semantic trade-envelope or
  position-wrapper metadata does not change fresh-build solver
  selection.
- The primary fresh-build path for migrated families goes through the
  Semantic Lowering Pipeline, with `ContractIR` as the admitted
  semantic IR for the current tranche.
- Redundant route authority is deleted for migrated families during
  rollout and fully absent from supported fresh-build surface at phase
  exit.
- Operator-facing traces still identify the selected solver, valuation
  snapshot, and any overlay/scenario identity without depending on
  deleted route ids.
- After Phase 4 close, remaining legacy route artifacts are clearly
  limited to replay-only, audit-only, or compatibility-only surface.

## Validation

- route-masked and trade-envelope-masked fresh-build integration tests
- parity harness re-run after the primary selector flip
- observability regression tests for traces / scorecards / replays
- audit test that enumerates forbidden `ProductIR.instrument` selector
  reads on migrated fresh-build paths
- audit test that enumerates forbidden selector reads of non-semantic
  trade-envelope or position-wrapper metadata
- residual-inventory audit proving no supported fresh-build family
  remains behind route fallback at phase exit

## Failure Modes To Watch

- **Shadow-mode complacency.** If the team treats shadow-mode success as
  equivalent to deletion readiness, Phase 4 will remove authority before
  the observability and provenance surfaces are ready.
- **Hidden read sites.** `ProductIR.instrument` is used outside the
  obvious selector modules. Benchmarks, telemetry, and review surfaces
  need their own audit.
- **Replay breakage.** Removing route ids from fresh build must not
  break historical replay packages that still reference them.
- **Partial deletions without guardrails.** If route cards are deleted
  before route-masked tests exist, regressions will slip in silently.
- **Family over-claim.** Only families with actual Phase 3 parity
  evidence belong in `\mathcal{D}_{\text{mig}}`.
- **False completion.** If route fallback survives anywhere on supported
  fresh-build surface, the Semantic Lowering Pipeline is not yet the
  authoritative mechanism and Phase 4 must stay open.

## Relationship To Future Tracks

Phase 4 should not block on future semantic domains that are not yet part
of the supported fresh-build universe.

Specifically, Phase 4 for the current migrated payoff-expression
families does NOT require:

- quoted-observable follow-ons such as future `CurveQuote` /
  `SurfaceQuote` nodes
- leg-based cashflow IR
- callable / event-coupled leg products
- the future event/state/control semantic track for autocallables,
  target-redemption structures, swing-style control, or GMWB/GMxB

Those tracks must fit the same eventual authority model, but they are
not prerequisites for deleting fresh-build route authority from the
already-migrated payoff-expression families.

The stronger rule after Phase 4 is:

- no newly admitted family gets a temporary route-based fresh-build path
- later quoted-observable, leg-based, or event/state/control families
  must enter through the Semantic Lowering Pipeline from their first
  supported admission
- post-Phase-4 closure work is therefore an admission program into the
  route-free mechanism, not a justification for keeping route alive

## Next Steps

1. Land this document as the dedicated Phase 4 draft.
2. Keep the deletion sequence tied to the Phase 3 parity ledger; no
   family enters Phase 4 deletion without evidence.
3. Audit `ProductIR.instrument` and non-semantic trade-envelope
   consumers before Phase 4 coding starts.
4. Keep a per-family closure ledger so route retirement is gated on
   semantic closure, not on intuition or branch-local confidence.
5. Treat the Phase 4 exit criterion as a hard product rule:
   by phase close, every supported fresh rebuild must go through the
   Semantic Lowering Pipeline, and route must be replay-only.
