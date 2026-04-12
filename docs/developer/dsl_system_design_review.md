# Trellis Semantic DSL — System Design Review

**Date:** 2026-04-01  
**Status:** current shipped boundary  
**Supersedes:** earlier design-review notes that described the pre-boundary in-flight state

---

## 1. Architecture Summary (as-built)

```text
Request / term sheet
  -> SemanticContract
  -> semantic validation
  -> ValuationContext
  -> RequiredDataSpec / MarketBindingSpec
  -> ProductIR
  -> EventProgramIR / ControlProgramIR
  -> PricingPlan + RouteSpec admissibility
  -> family lowering IR
  -> existing helper-backed numerical route
```

The important point is that Trellis now has an explicit boundary between:

1. semantic contract meaning
2. valuation and market binding
3. route admissibility and lowering
4. numerical execution

This boundary is shipped code, not just prompt guidance.

---

## 2. Current Authority Model

### 2.1 Semantic authority

`SemanticContract` is still evolved in place. The authoritative typed fields are:

- `ConventionEnv`
- `SemanticTimeline`
- `ObservableSpec`
- `StateField`
- `ObligationSpec`
- `ControllerProtocol`
- `EventMachine`

Automatic triggers live in event/state machinery. Strategic rights live in
`ControllerProtocol`.

### 2.2 Valuation authority

`ValuationContext` is the authoritative valuation-policy object. It owns:

- market snapshot/source handle
- model spec
- measure spec
- discounting policy
- optional collateral policy
- reporting policy
- canonical `requested_outputs`

`requested_measures` is retained only as a compatibility shim.

### 2.3 Routing authority

`ProductIR` is the shared checked summary used for route selection. It is not
the full semantic contract and not a universal solver IR.

Typed route admissibility is owned by:

- `RouteSpec.admissibility`
- `BuildGateDecision`

Routes are no longer the primary synthesis plan. The compiler now emits lane
obligations first and then a structured route-binding authority packet that
answers a narrower question:

- which exact checked backend fit was selected
- which modules, primitives, and helper refs are in authority for that fit
- which validation bundle and canary tasks cover the fit
- which typed admissibility contract and failures apply to the request

That packet is the backend-binding and provenance surface shared by prompt
rendering, trace replay, validation events, checkpoints, and diagnostics.
Route-card prose is secondary evidence, not the planning authority.

### 2.4 Lowering authority

The current lowering boundary is family-specific, but it now sits underneath a
universal semantic event/control compiler program:

- `EventProgramIR`
- `ControlProgramIR`

That shared program is emitted once from product semantics and then projected
into the bounded numerical families. The current family-specific layer is:

- `AnalyticalBlack76IR`
- `TransformPricingIR`
- `EventAwarePDEIR`
- `VanillaEquityPDEIR` as the current compatibility wrapper for the vanilla
  theta-method PDE route
- `ExerciseLatticeIR`
- `CorrelatedBasketMonteCarloIR`

These lower onto existing checked-in helpers and kernels. The pricing math
remains in `trellis/models/`.

For transform routes, the dedicated lowered contract now carries:

- one terminal-state transform state spec
- one characteristic-function family and model family
- explicit quote semantics and strike semantics
- a transform-lane control contract (`identity`) plus the upstream semantic
  control provenance
- backend capability split (`helper_backed` versus `raw_kernel_only`)

That removes the earlier ambiguity where transform admissibility could fall
back to raw option-family tags and mistakenly inherit exercise/state facts that
are irrelevant to terminal-only transform pricing.

For PDE routes, the typed lowering surface now includes explicit contracts for:

- PDE state (`PDEStateSpec`)
- operator/stepping family (`PDEOperatorSpec`)
- event-time buckets (`PDEEventTimeSpec`)
- event transforms (`PDEEventTransformSpec`)
- control style (`PDEControlSpec`)
- boundary/terminal semantics (`PDEBoundarySpec`)

Those PDE contracts are now projections of the shared event/control program,
not a separate PDE-local event vocabulary.

The runtime now also ships a bounded generic rollback substrate in
`trellis.models.pde.event_aware`:

- `EventAwarePDEGridSpec`
- `EventAwarePDEOperatorSpec`
- `EventAwarePDEBoundarySpec`
- `EventAwarePDEEventBucket`
- `EventAwarePDETransform`
- `EventAwarePDEProblemSpec`
- `EventAwarePDEProblem`

That substrate is intentionally 1D and deterministic-schedule-only, but it
means the compiler no longer lowers into a family IR without a matching
product-agnostic rollback assembly layer.

The checked vanilla-equity PDE helper now uses that same runtime substrate. The
route still exposes `price_vanilla_equity_option_pde`, but the helper assembles
an `EventAwarePDEProblem` with no event buckets instead of carrying a separate
vanilla-only rollback loop. `VanillaEquityPDEIR` is now explicitly
transitional-only: the intended end state is to retire the wrapper once trace
and review consumers stop keying on the legacy family-IR type.

For supported schedule-driven PDE requests, the compiler can now preserve a
typed event timeline on `EventAwarePDEIR` even when DSL lowering cannot yet
bind a helper target. That changes the failure mode from “raw schedule-state
leaked through routing” to “explicit event-aware PDE contract exists, but the
selected route still lacks the required operator/control backend or route
migration.”

The platform trace boundary now mirrors that contract through
`generation_boundary.lowering.family_ir_summary`, which condenses the family IR
into the fields operators actually review:

- state variable and tags
- operator family and solver family
- control style
- event-transform kinds and event dates
- helper symbol
- compatibility status for transitional wrappers

The same boundary now also emits `generation_boundary.construction_identity`.
That summary is the operator-facing answer to “what construction contract am I
actually looking at?” It prefers:

- the exact backend binding id when the compiler found a native checked fit
- otherwise the family IR type
- otherwise the lane family

The compatibility route alias is still recorded, but only as secondary
provenance context instead of the primary explanation.

That secondary context is now policy-driven. Migrated exact-helper families can
mark their route alias as internal-only, in which case:

- the route id remains in raw authority metadata for replay and canary history
- operator-facing trace and prompt surfaces suppress the alias
- the backend binding id or family IR remains the only surfaced explanation

The same rule now applies inside migrated exact-helper route cards for the FX
and quanto cohort. When the backend binding is already a stable semantic-facing
helper, the route card no longer surfaces raw-kernel reconstruction steps or
route-local input mappers as live build instructions. Review and validation are
expected to fail if generated code bypasses that helper surface or calls it on
the wrong signature.

The credit and copula helper cohort now follows the same rule. The
single-name CDS, nth-to-default, and tranche-style basket-credit route cards
keep only backend binding, admissibility, and validation ownership. Schedule
loops, survival/default plumbing, copula initialization, and tranche-loss
projection are no longer carried as route-card instructions once the checked
helpers already own that surface.

The rate-tree cohort now follows the same discipline. ``exercise_lattice``,
``rate_tree_backward_induction``, ``zcb_option_rate_tree``, and
``zcb_option_analytical`` still expose the backend binding and the validation
surface, but they no longer supply prompt-level lattice or short-rate
construction guidance when the checked helper already owns that assembly. The
semantic drafting boundary is also stricter here: an explicit ``instrument_type``
such as ``zcb_option`` now blocks fallback drafting into a generic
``vanilla_option`` contract when the prose only happens to contain generic
``option`` language.

The Monte Carlo compiler now has the matching bounded family surface for the
next migration tranche:

- `EventAwareMonteCarloIR`
- `MCStateSpec`
- `MCProcessSpec`
- `MCEventTimeSpec`
- `MCEventSpec`
- `MCPathRequirementSpec`
- `MCPayoffReducerSpec`
- `MCControlSpec`
- `MCMeasureSpec`
- `MCCalibrationBindingSpec`

Those Monte Carlo contracts are likewise projections of the shared
`EventProgramIR` / `ControlProgramIR` boundary. The compiler now lowers
deterministic schedule/event semantics into that family for bounded products
without inventing a separate Monte Carlo-only event language. The first
concrete proof slice is the European rate-style swaption path: the semantic
compiler can emit `EventAwareMonteCarloIR` with explicit event buckets, replay
requirements, and typed payoff-reducer semantics on the Monte Carlo route
instead of dropping the product straight into an untyped fallback.

The runtime now also has the matching bounded assembly layer in
`trellis.models.monte_carlo.event_aware`:

- `EventAwareMonteCarloProcessSpec`
- `EventAwareMonteCarloEvent`
- `EventAwareMonteCarloProblemSpec`
- `EventAwareMonteCarloProblem`

That layer resolves bounded process families (`gbm_1d`, `local_vol_1d`,
`hull_white_1f`), compiles deterministic event buckets onto the existing
`PathEventTimeline` substrate, and assembles `StateAwarePayoff` contracts over
reduced Monte Carlo state instead of requiring product-local path replay code.

That still does not mean the full Monte Carlo migration is complete. The
current state is:

- bounded compiler/lowering support exists
- bounded runtime problem assembly now exists
- vanilla European Monte Carlo now normalizes onto a terminal-only
  `EventAwareMonteCarloIR` contract instead of carrying a synthetic event
  replay timeline
- the local-vol vanilla helper is now a compatibility wrapper over the generic
  event-aware Monte Carlo runtime
- generic vanilla migration and proof-route recovery remain separate follow-on
  slices
- comparison-request compilation no longer bypasses semantic compilation for
  rate-style swaptions; each method plan now carries its own semantic blueprint
  so fixed/float leg conventions, rate-index roles, and the bounded
  Hull-White calibration contract survive into comparison assembly

---

## 3. Shipped Route Families

The typed semantic boundary is proven end-to-end for these route families:

| Route ID | Family IR | Current contract families |
|---|---|---|
| `analytical_black76` | `AnalyticalBlack76IR` | vanilla European options |
| `vanilla_equity_theta_pde` | `VanillaEquityPDEIR` on `EventAwarePDEIR` | vanilla European options |
| `pde_theta_1d` | `EventAwarePDEIR` | bounded event-aware 1D rollback for holder-max equity exercise and issuer-min Hull-White callable bonds |
| `exercise_lattice` | `ExerciseLatticeIR` | callable bonds, Bermudan swaptions |
| `correlated_basket_monte_carlo` | `CorrelatedBasketMonteCarloIR` | ranked-observation baskets |

This boundary preserves current route IDs and helper entry surfaces.

The semantic-contract layer feeding that boundary is now also registry-backed:
ordered draft rules select the family, registered family definitions own the
admissible method matrix, and a single shared specialization helper rebuilds
method-sensitive contracts for the request layer and semantic compiler. That
eliminates one of the recurring lower-layer bug classes where method support
drifted between ``semantic_contracts.py``, ``platform_requests.py``, and
``semantic_contract_compiler.py``.

For Monte Carlo, the next bounded compiler target is `EventAwareMonteCarloIR`:
single-state / one-factor, deterministic event timelines, reduced-state path
requirements, and explicit model/quote bindings. The route table above remains
accurate because the generic vanilla MC routes have not yet all switched over
to that family, even though the compiler now emits it for the bounded
schedule-driven swaption slice.

Callable-bond wrappers now follow the same “thin public shell over reusable
family helpers” rule as the newer event-aware routes. The public PDE/tree
helpers still exist, but coupon timeline compilation, embedded exercise
projection, and straight-bond reference assembly have moved into
``trellis.models.short_rate_fixed_income`` so later short-rate claim families
can reuse that substrate instead of copying callable-bond-local logic.

---

## 4. Post-Boundary Runtime Leverage

The semantic boundary now also has concrete runtime leverage layers on top of
it. These do not reopen the contract/compiler split; they exploit it:

- deterministic checked-in route reuse is now route-metadata-driven rather than
  executor-hardcoded
- route matching now treats explicit semantic ``route_families`` as a stronger
  authority than broad engine-family preferences, so lower layers stop
  widening candidate sets after the semantic compiler has already chosen a
  family boundary
- replay, checkpoints, and diagnosis packets consume the same
  route-binding-authority packet rather than inferring route meaning from
  scattered helper text
- low-confidence and cold-start builds can surface similar-product retrieval
  through gap-check, knowledge retrieval, and prompt formatting
- deterministic validation is compiled through `CompiledValidationContract`
  and can attach route-specific financial checks before reviewer escalation
- eligible single-method builds can run a post-bundle reference oracle against
  exact helper-backed or bound-style checked-in references

This matters operationally because the shipped boundary is no longer only a
compiler refactor. It is the authority surface that retrieval, validation,
reuse, and runtime diagnostics now share.

---

## 5. Warning And Error Policy

The current system distinguishes:

- hard semantic validation errors
- typed admissibility failures
- successful compilation with warnings

Warnings are used when:

- legacy mirrors are normalized into typed fields
- a migrated route ignores a stale legacy mirror
- output requests fall back to bump-and-reprice support

Errors are used when:

- typed phase order is invalid
- observables or state fields future-peek
- an automatic trigger is represented as strategic control
- settlement-bearing contracts omit typed obligations
- route capabilities do not support the requested control, outputs, or state tags
- route capabilities do not support the requested PDE operator family or
  event-transform set

---

## 6. Migrated-Family Authority Rules

For the migrated families above:

- typed obligations and timeline are authoritative for settlement semantics
- typed `EventMachine` is authoritative for automatic event semantics
- `settlement_rule` and `event_transitions` are mirrors, not truth sources

This is intentionally narrow. Non-migrated code paths may still consume those
legacy fields.

The practical consequence is that a stale legacy mirror can now warn without
blocking the migrated family path, as long as the typed surface is valid.

---

## 7. What Changed Relative To The Earlier In-Flight Review

The earlier review was directionally useful but stale on several points. The
following are now shipped, not hypothetical:

- typed semantic fields on `SemanticContract`
- structured semantic validation findings
- explicit `ValuationContext`
- compiled `RequiredDataSpec` and `MarketBindingSpec`
- typed route admissibility
- `ProductIR` as the shared checked summary
- family-specific lowering IRs for proven route families
- migrated-family authority flip away from legacy settlement/event mirrors

The following remain intentionally deferred:

- ordered sequential multi-controller protocols
- nonlinear funding/XVA semantics inside `ValuationContext`
- a universal numerical IR
- a full desk-task DSL

---

## 8. Engineering Rules For The Current Slice

These rules are now part of the shipped design:

- do not change pricing math in semantic-boundary PRs unless a test proves an existing bug
- do not re-create schedules, timing order, or settlement semantics inside route-local glue
- do not model automatic triggers as control nodes
- do not treat free-form legacy strings as authoritative on migrated paths
- do not introduce a parallel event DSL; reuse `EventMachine`

---

## 9. Remaining Gaps Worth Tracking

These are real follow-ons, but they are outside the current shipped boundary:

1. broader route-family migration beyond the current proven four
2. richer typed observable algebra beyond the tranche-1 descriptors
3. typed desk-task objects for exposure, explain, and calibration workflows
4. proof-carrying analytical rewrite lemmas beyond current helper-backed kernels
5. portfolio-level nonlinear valuation layers such as funding and XVA

Those should build on the current semantic/valuation/lowering split rather than
reopening it.
