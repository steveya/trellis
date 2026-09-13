# Implementation Journey: From Prompt to Price

This note records how Trellis moved from a loosely connected "prompt plus code
generation" workflow toward a more unified path from request to priced result.

## Starting Point

The early system already had useful pieces:

- `trellis.ask(...)`
- `Session.price(...)`
- `Pipeline.run()`
- a build loop that could generate unsupported payoffs

But those surfaces did not all pass through the same semantic backbone. The
center of the platform was stronger than the entry surfaces around it.

Typical problems in the earlier shape were:

- natural-language entry and structured entry took different conceptual paths
- the build loop understood more semantics than some direct execution flows
- comparison tasks did not fit the same path as single-method pricing requests
- market context was often implicit rather than first-class

## The First Real Shift: Strengthen the Build Core

The turning point was when `trellis/agent/executor.py` stopped acting like a
thin wrapper around a builder prompt and started enforcing structure:

- decompose to `ProductIR`
- build a `PricingPlan`
- build a `GenerationPlan`
- classify blockers before code generation
- validate imports before file write
- validate semantics before accepting the artifact

That changed the question from:

> "Can the model write something plausible?"

to:

> "Can the platform compile this request into a defensible route and reject bad
> artifacts before they become part of the system?"

## Unifying the Front Doors

Once the center of the system was stronger, the next move was to unify entry
surfaces through `trellis.agent.platform_requests`.

That layer gave Trellis a canonical internal path:

1. create a `PlatformRequest`
2. compile it into a `CompiledPlatformRequest`
3. attach shared knowledge, route choice, blocker reports, and execution intent
4. execute deterministic pricing or agent-backed build/validation
5. persist traces and task-run records

The important result was not the dataclasses themselves. It was that the same
internal machinery could now serve:

- `ask(...)`
- direct `Session` requests
- `Pipeline` book/scenario workflows
- structured user-defined products
- comparison tasks

The newer semantic-compiler path now adds one more important property: route
selection and DSL lowering use the same ranked route chooser as the live build
path. That keeps the semantic blueprint aligned with runtime execution instead
of exposing one route in the compiler and a different one in generation. In
practice this is what lets one semantic contract lower cleanly to an
analytical kernel, a PDE helper, or a lattice helper based on the selected
method without changing the contract shape.

The practical follow-on is that comparison routes can expose the smallest
stable construction surfaces instead of forcing the builder to rediscover
market binding or hiding the whole target behind a product wrapper. Bermudan
swaptions now illustrate both cases:

- the tree lane lowers to ``price_bermudan_swaption_tree(...)``
- the analytical comparison lane normalizes the exercise dates, selects the
  final date strictly after settlement and before swap end, resolves that
  European swaption with ``resolve_swaption_black76_inputs(...)``, and passes
  the typed result to ``price_swaption_black76_raw(...)``

That keeps the comparison target explicit and bounded. The analytical lane is
not "write some Black76-like code for a Bermudan swaption" or "call a product
wrapper". The adapter owns the small derivative-specific rule: return zero if
no valid date remains, otherwise price the European swaption exercisable only
on the final Bermudan date. It must not aggregate or maximize European values
over the schedule. The retained
``price_bermudan_swaption_black76_lower_bound(...)`` function is comparison and
compatibility evidence, not generated construction authority.

T73 applies the same boundary to a European swaption Monte Carlo target. The
route and exact binding no longer point at ``price_swaption_monte_carlo(...)``
or ``resolve_swaption_monte_carlo_problem(...)``. They expose the reusable
expiry resolver, explicit-start fixed and floating payment timelines on one
authored model-time clock, Hull-White process binding, discounted swap-PV
payload, short-rate discount reducer, event/problem
contracts, problem compiler, and generic event-aware estimator. DSL lowering
records those stages as an ordered ``ThenExpr`` and deterministic offline
generation materializes the same sequence. This makes source identity and
validation evidence describe the estimator that ran rather than a product
wrapper that hid the assembly.

T73's European rate-tree target is primitive-composed too. The route and exact
binding identify generic ``price_on_lattice(...)`` as the estimator. Lowering
records one-exercise contract construction, curve-basis adjustment, resolved
tree inputs, topology, mesh, calibration target, lattice construction,
swaption-contract compilation, and rollback as an ordered ``ThenExpr``.
Deterministic generation preserves the Hull-White/BDT model, explicit
comparison parameters, conventions, and tree-step controls. The retained
``price_swaption_tree(...)`` wrapper remains an independent reference rather
than the source identity of the generated artifact.

The repository row for T73 now authors the economics, named market scenario,
exercise-value/output contract, model parameters, numerical controls, reference
target, and per-target tolerances directly. Runtime synthesis must therefore
remain title-independent. The proof does not establish external-library
parity, independent Black-vs-Hull-White model agreement, Bermudan exercise,
contractual cash/physical settlement, delivery lifecycle, stochastic basis, or
production calibration coverage. Its semantic product and obligation record
only the positive payer underlying-swap NPV at exercise, discounted to valuation;
they do not assert a cash-settlement method.
The semantic timeline explicitly has no settlement dates. Family lowering
retains the exercise-value obligation as a valuation event on the exercise
decision date; neither family nor DSL signatures advertise a settlement role.
This distinction survives method specialization and the Monte Carlo event
projection. Ordinary contractual settlement obligations retain their existing
settlement timeline semantics.
Dictionary/YAML parsing hydrates explicit event machines, including their
nested guards, actions, and ordered parameters, instead of leaving raw mappings
in the typed contract. Malformed explicit machines fail rather than silently
regenerating a different lifecycle; serialized exercise-value contracts retain
the same strict validation as in-memory objects.
T73's replay seed is taken from
`cross_validate.target_contracts.hw_mc.spec_overrides.seed`, the same seed
executed by its sole Monte Carlo target; generic task or market defaults cannot
replace it. Exact admission rejects top-level `seed` and `simulation_seed`
aliases. Other task seed-resolution rules remain unchanged.

Generated schedule defaults resolve frequency and day-count conventions to
verified enum members (including supported aliases). Unknown conventions fail
closed instead of being copied into generated Python source.

Per-target-only tolerance maps must cover every non-reference price target.
Missing entries fail before comparison execution; a tolerance for one target
cannot become an implicit global allowance for the others. Result evidence
reports no global tolerance for these maps and retains each target's authored
tolerance. The legacy 5% fallback for requests with no authored tolerance remains
tracked separately in QUA-1252.
Promotion reviews use the candidate's own authored allowance and reject missing
or malformed allowances in per-target-only evidence.

Compiled request metadata now also carries a compact semantic-blueprint summary
for downstream tooling. That summary records the canonical lowered route, the
helper and primitive references selected by DSL lowering, and any explicit
control styles so trace, review, and UI code do not need to reconstruct those
details from the full compiler objects.

## Comparison Tasks Changed the Architecture

Comparison tasks exposed a deeper truth: not every pricing request is asking
for one method and one artifact.

Tasks like "tree vs PDE vs MC vs FFT vs COS" forced the platform to stop
treating every request as a single-route build. That led to:

- comparison-aware request compilation
- method-specific plans
- runtime cross-validation
- per-method traces and task results

This was important because it turned the platform from "build one pricer" into
"assemble and evaluate a pricing workflow."

## Market Data Had To Enter The Same Path

Another major shift was the move from ad hoc runtime market inputs to explicit
market-data compilation:

- `MarketSnapshot`
- named discount/forecast/vol/credit/FX components
- task-level market selection
- persisted `market_context` in task-run records

Without that, the answer to "why did this price differ?" or "why did this task
fail?" was never stable enough.

## The Current Operational Flow

Today, the intended flow is:

1. a request enters from `ask`, `Session`, `Pipeline`, structured spec, or task
2. the request compiles into product semantics, market requirements, and route intent
3. the platform resolves or verifies market context
4. the platform either:
   - prices deterministically from supported substrate, or
   - enters the guarded build loop
5. validation and review gates decide whether the result is acceptable
6. traces, issues, and task-run records preserve the full audit trail

That means "prompt to price" is no longer really about prompts. It is about a
compiler/runtime with prompts in the places where ambiguity still matters.

## The Most Important Design Lesson

The most important lesson from this journey is that a pricing platform gets
better when prompt use becomes narrower and more explicit.

The winning architecture was not:

- make the builder prompt smarter

It was:

- make routing, market context, comparison intent, and validation more
  structured
- leave the LLM for the residual ambiguity

## What Is Still Transitional

The system is much more coherent than it was, but a few things remain
transitional:

- direct existing-instrument flows still benefit less from the full semantic
  shared-knowledge path than build-oriented flows
- some route families still rely on more prompt-time interpretation than they
  should
- the missing-primitive workflow is good at diagnosis and escalation, but not
  yet fully autonomous implementation

## Why This Matters For The Next Milestone

The next milestone is no longer "support prompt pricing." It is:

- make repeated reruns trustworthy
- make missing substrate visible and bounded
- add missing pricing infrastructure under guardrails

That only works because the path from prompt to price has become structured
enough to tell the difference between:

- a weak prompt
- a provider problem
- a market-data gap
- a knowledge gap
- a real missing primitive
