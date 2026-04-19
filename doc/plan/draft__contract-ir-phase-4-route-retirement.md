# Contract IR — Phase 4: Route Retirement And ProductIR Dispatch Phaseout

## Status

Draft. Pre-queue design document. Not yet the live execution mirror for
Phase 4.

## Linked Linear

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-905 — Phase 3 umbrella (structural solver compiler)
- QUA-906 — Phase 4 umbrella (this plan expands)
- QUA-904 — Phase 2 umbrella (semantic IR substrate already additive)

## Companion Docs

- `doc/plan/draft__contract-ir-phase-2-ast-foundation.md`
- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__contract-ir-compiler-retiring-route-registry.md`
- `docs/quant/contract_ir.rst`
- `docs/quant/contract_algebra.rst`

## Purpose

Specify the deletion phase that turns the structural compiler from
shadow-mode evidence into primary fresh-build authority.

The core requirement is simple and strict:

- for migrated payoff families, a fresh rebuild must no longer depend on
  direct hard-coded route ids, route-family tables, or
  `ProductIR.instrument`
- the build must go through the Contract IR -> structural selection ->
  lowering / helper-binding path

Phase 4 is therefore not new pricing math. It is governed removal of
redundant authority after Phase 3 has already proved parity.

## Framing

### What "route-free fresh build" means

For a migrated contract, the solver-selection result must be determined
by:

- `contract_ir`
- requested method / valuation policy
- market capability surface
- explicit structural declaration precedence

and NOT by:

- `ProductIR.instrument`
- route ids
- route families
- compatibility aliases
- hand-written per-instrument branch ladders

This is the success criterion that motivated the whole Contract IR
program.

### What route-free does NOT mean

Phase 4 does not require deleting every legacy artifact from the repo on
day one.

It is valid to retain:

- replay-time route aliases
- provenance ids for historical comparison
- summary projections from `ContractIR` back onto `ProductIR`
- fallback for unmigrated families

as long as those artifacts are no longer selection authority for
migrated fresh-build surface.

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

Phase 4's contract is that, for migrated surface, dispatch is no longer
a function of `\pi(c)`:

$$\operatorname{Select}_{\text{fresh}}(c, v, m) \neq f(\pi(c))$$

in the sense of authority. `\pi(c)` may still be emitted for telemetry,
searchability, or compatibility, but selection correctness for migrated
surface must be invariant under changes to legacy metadata.

### Metadata-masking invariance

For any migrated contract `c` and any two legacy-metadata packets
`h_1, h_2` that differ only in route-local fields,

$$\operatorname{Select}_{\text{fresh}}(c, v, m, h_1)
=
\operatorname{Select}_{\text{fresh}}(c, v, m, h_2).$$

This invariant is the formal phase-exit condition for "route-free fresh
build."

### Migration domain

Let `\mathcal{D}_{\text{mig}}` be the subset of Contract IR space whose
families have:

1. a shipped solver declaration set
2. shadow-mode parity evidence
3. migrated observability surfaces

Phase 4 only deletes authority for `c \in \mathcal{D}_{\text{mig}}`.
Everything else keeps the old fallback until it joins that set.

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
- exact helper / kernel provenance

Only after that migration is stable should route ids be removed from the
operator-facing primary identity.

## Required Architecture Outcome

### Fresh-build path

For migrated families the intended pipeline is:

1. semantic contract validation
2. bounded decomposition to `ProductIR` and `ContractIR`
3. structural solver selection from `ContractIR`
4. adapter / lowering assembly
5. exact helper / kernel binding
6. provenance + validation metadata attachment

There is no direct branch of the form:

- `if product_ir.instrument == ...`
- `if route_family == ...`
- `if route_id == ...`

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
- route-masked regression tests
- feature flag / rollout guard if needed

### P4.2 — Family-by-family route-card retirement

**Objective.** Delete redundant `routes.yaml` and binding-catalog
authority for migrated families.

**Artifacts.**

- per-family deletion PRs
- fallback retained only for unmigrated surface

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
- documentation of the new primary identity

### P4.5 — Fresh-build hard guardrails

**Objective.** Prevent regressions back to direct route-local selection.

**Artifacts.**

- regression tests that mask route metadata and expect the same outcome
- guard tests that fail if migrated families reintroduce
  `ProductIR.instrument`-keyed branching

### P4.6 — Knowledge, docs, and closeout

**Objective.** Update the official docs and any knowledge assets that
still describe route ids as the primary fresh-build authority.

**Artifacts.**

- `docs/quant/`
- `docs/developer/`
- relevant knowledge / prompt text describing migrated family selection

## Acceptance Criteria

- For every migrated family, masking `ProductIR.instrument`,
  `route_id`, and `route_family` does not change fresh-build solver
  selection.
- The primary fresh-build path for migrated families goes through the
  structural compiler.
- Redundant route authority is deleted for migrated families.
- Operator-facing traces still identify the selected solver and its
  provenance without depending on deleted route ids.
- Remaining legacy route authority is clearly limited to unmigrated or
  replay-only surface.

## Validation

- route-masked fresh-build integration tests
- parity harness re-run after the primary selector flip
- observability regression tests for traces / scorecards / replays
- audit test that enumerates forbidden `ProductIR.instrument` selector
  reads on migrated fresh-build paths

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

## Relationship To Future Tracks

Phase 4 should not block on future semantic domains.

Specifically, Phase 4 for the current migrated payoff-expression
families does NOT require:

- quoted-observable follow-ons such as future `CurveQuote` /
  `SurfaceQuote` nodes
- leg-based cashflow IR
- callable / event-coupled leg products

Those tracks must fit the same eventual authority model, but they are
not prerequisites for deleting fresh-build route authority from the
already-migrated payoff-expression families.

## Next Steps

1. Land this document as the dedicated Phase 4 draft.
2. Keep the deletion sequence tied to the Phase 3 parity ledger; no
   family enters Phase 4 deletion without evidence.
3. Audit `ProductIR.instrument` consumers before Phase 4 coding starts.
4. Treat the Phase 4 exit criterion as a hard product rule:
   for migrated families, even the simplest fresh rebuild must go
   through the IR -> structural selection -> lowering path.
