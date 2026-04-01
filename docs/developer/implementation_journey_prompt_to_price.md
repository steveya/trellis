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

The practical follow-on is that comparison routes can now target checked-in
helper surfaces instead of forcing the builder to reassemble the same
mathematical glue repeatedly. A recent example is Bermudan swaptions:

- the tree lane lowers to ``price_bermudan_swaption_tree(...)``
- the analytical comparison lane lowers to
  ``price_bermudan_swaption_black76_lower_bound(...)``

That keeps the comparison target explicit and bounded. The analytical lane is
no longer "write some Black76-like code for a Bermudan swaption"; it is "call
the checked-in lower-bound helper that represents the intended comparison
contract". In this case the intended comparison contract is the European
swaption exercisable only on the final Bermudan date, not an aggregate or
max-over-schedule analytical surrogate.

Compiled request metadata now also carries a compact semantic-blueprint summary
for downstream tooling. That summary records the canonical lowered route, the
helper references selected by DSL lowering, and any explicit control styles so
trace, review, and UI code do not need to reconstruct those details from the
full compiler objects.

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
