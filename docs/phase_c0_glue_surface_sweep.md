# Phase C0 Glue Surface Sweep

## Goal

Systematically inventory the "glue code" the embedded agent still has to
invent during fresh-build generation, using the current `quanto` proving path
as the concrete evidence base.

This note is meant to answer one architectural question:

- should we keep extracting small deterministic helper surfaces into Trellis,
  or do we now need a larger generated-payoff scaffold?

## Evidence Base

Primary files inspected:

- `trellis/instruments/_agent/_fresh/quantooptionanalytical.py`
- `trellis/instruments/_agent/_fresh/quantooptionmontecarlo.py`
- `trellis/instruments/_agent/quantooptionanalytical.py`
- `trellis/instruments/_agent/quantooptionmontecarlo.py`
- `trellis/models/resolution/quanto.py`
- `trellis/core/payoff.py`
- `trellis/agent/executor.py`
- `trellis/agent/prompts.py`
- `trellis/agent/assembly_tools.py`
- `trellis/agent/knowledge/traces/promotion_candidates/20260326_195934_t105_quanto_bs.yaml`
- `trellis/agent/knowledge/traces/promotion_candidates/20260326_195934_t105_mc_quanto.yaml`
- `trellis/agent/knowledge/traces/promotion_reviews/20260326_200517_t105_quanto_bs_approved.yaml`
- `trellis/agent/knowledge/traces/promotion_reviews/20260326_200517_t105_mc_quanto_approved.yaml`

## Executive Summary

The agent can now generate working `quanto` code on the fresh-build `T105`
path, but it still succeeds by inventing more route glue than we should want.

What is already in good shape:

- family market binding is mostly extracted into `trellis.models.resolution.quanto`
- generic resolved-input and Monte Carlo adapter glue is partly extracted into
  `trellis.core.payoff`
- build-loop recovery, validation diagnostics, and promotion gates are now
  explicit and auditable

What is still too implicit:

- the generator still writes the full module shell, spec dataclass, imports, and
  `requirements` property from scratch
- the generator still reconstructs analytical and Monte Carlo route bodies
  instead of calling a stable route helper
- the Monte Carlo path remains especially glue-heavy: drift mapping, process
  assembly, engine setup, payoff-shape handling, and discount semantics are
  still open-coded in the generated module

Recommendation:

- continue extracting glue into Trellis
- do not jump yet to a giant universal abstraction
- the next warranted expansion is a **medium** surface:
  - family-specific route kernels for `quanto`
  - route-specific skeletons/scaffolds for resolved analytical and resolved MC
  - stronger prompt obligations to use those checked-in surfaces

## Sweep

### 1. Module Shell Glue

Current generated examples:

- fresh analytical candidate defines its own docstring, imports, `QuantoOptionSpec`,
  `QuantoOptionAnalyticalPayoff`, `spec` property, and `requirements` property
- fresh MC candidate does the same for the Monte Carlo route

Files:

- `trellis/instruments/_agent/_fresh/quantooptionanalytical.py`
- `trellis/instruments/_agent/_fresh/quantooptionmontecarlo.py`

Current Trellis support:

- `trellis/agent/executor.py` provides a generic module skeleton
- `trellis/agent/prompts.py` asks the model to fill in `evaluate()`

Assessment:

- this is still agent-written boilerplate
- it is not the highest-risk math surface, but it adds noise and increases the
  chance of import/name drift

Recommendation:

- extract family-aware skeleton variants for known family/method pairs
- keep the class/spec names deterministic instead of letting the model restate
  them each time

Priority:

- medium

### 2. Import and Symbol Glue

Current generated examples:

- analytical candidate imports `generate_schedule` and `Frequency` even though
  they are unused
- analytical candidate uses `__import__("math").exp` instead of a stable local
  import
- MC candidate imports `black76_call` and `black76_put` even though they are
  unused

Files:

- `trellis/instruments/_agent/_fresh/quantooptionanalytical.py`
- `trellis/instruments/_agent/_fresh/quantooptionmontecarlo.py`

Current Trellis support:

- import registry and resolution tools exist
- the builder prompt includes inspected/reference modules

Assessment:

- not the main correctness blocker, but it is still evidence that the scaffold
  is too loose

Recommendation:

- strengthen route-specific skeletons so required imports are mostly fixed
- reduce the generator's responsibility to choosing among a very small allowed
  import set

Priority:

- medium-low

### 3. Family Market-Resolution Glue

Current generated examples:

- analytical candidate correctly calls `resolve_quanto_inputs(...)`, but then
  still re-checks and re-derives:
  - valuation date
  - underlier spot fallback
  - domestic discount factor fallback
  - foreign discount factor fallback
  - underlier vol fallback
  - FX vol fallback
  - correlation fallback

Files:

- `trellis/instruments/_agent/_fresh/quantooptionanalytical.py`
- `trellis/models/resolution/quanto.py`

Current Trellis support:

- `ResolvedQuantoInputs`
- `resolve_quanto_inputs`
- dedicated helpers for underlier spot, foreign curve, and correlation

Assessment:

- this was the critical extraction that made fresh-build `quanto` viable
- but the generator still does not trust the resolved contract fully and keeps
  open-coding fallback probes

Recommendation:

- push more of the contract into the resolver return type
- make route skeletons assume the resolver contract is authoritative
- discourage fallback probing in generated code for known families

Priority:

- high

### 4. Analytical Route Glue

Current generated examples:

- analytical candidate still open-codes:
  - time-to-expiry computation
  - quanto forward construction
  - option-type branching
  - kernel invocation
  - final notional/discount scaling

Files:

- `trellis/instruments/_agent/_fresh/quantooptionanalytical.py`
- `trellis/instruments/_agent/quantooptionanalytical.py`

Current Trellis support:

- `ResolvedInputPayoff`
- `ResolvedQuantoInputs`
- `black76_call` / `black76_put`

Assessment:

- the deterministic checked-in route is already a thin adapter, but the fresh
  candidate still does not reuse that pattern strongly enough
- the analytical route body is small enough that a dedicated checked-in helper
  is probably worth it

Recommendation:

- add a shared `price_quanto_option_analytical(...)` helper or equivalent route
  kernel
- keep the payoff class as a thin adapter over that helper plus the resolver

Priority:

- high

### 5. Monte Carlo Route Glue

Current generated examples:

- MC candidate still open-codes:
  - `CorrelatedGBM` drift vector construction
  - domestic/foreign rate extraction
  - engine selection and engine kwargs
  - initial state assembly
  - payoff function shape assumptions
  - discounting policy through `engine.price(..., discount_rate=...)`

Files:

- `trellis/instruments/_agent/_fresh/quantooptionmontecarlo.py`
- `trellis/instruments/_agent/quantooptionmontecarlo.py`
- `trellis/core/payoff.py`

Current Trellis support:

- `MonteCarloPathPayoff`
- shared resolver
- `CorrelatedGBM`
- `MonteCarloEngine`

Assessment:

- this is still the largest remaining glue surface
- the agent can now make it work, but it still reconstructs too much route
  policy in-line
- this is also where prior failures were most likely to happen

Recommendation:

- extract a shared `build_quanto_mc_process(...)` or full
  `price_quanto_option_mc(...)` helper
- consider a family-specific MC adapter scaffold instead of only the generic
  `MonteCarloPathPayoff`
- this is the strongest argument for a larger family-specific surface

Priority:

- very high

### 6. Validation and Repair Glue

Current generated examples:

- not written by the payoff generator directly, but still part of the autonomy
  loop

Current Trellis support:

- structured invariant diagnostics in `trellis/agent/invariants.py`
- bundle-level failure detail propagation in
  `trellis/agent/validation_bundles.py`
- fragment/syntax recovery in `trellis/agent/executor.py`

Assessment:

- this surface is in reasonably good shape
- it was necessary to make fresh-build runs informative, but it is not the main
  remaining route-math bottleneck

Recommendation:

- keep iterating, but do not prioritize this over route-kernel extraction

Priority:

- medium-low

### 7. Promotion and Adoption Glue

Current Trellis support:

- candidate capture
- candidate review
- dry-run adoption

Files:

- `trellis/agent/task_runtime.py`
- `trellis/agent/knowledge/promotion.py`
- `scripts/review_promotion_candidates.py`
- `scripts/adopt_promotion_candidates.py`

Assessment:

- this is now explicit and auditable
- it is not the bottleneck for fresh-build success anymore

Recommendation:

- stop expanding this until route reliability improves

Priority:

- low

## What The Agent Still Has To Invent Today

For a fresh-build `quanto` route, the generator still has to invent:

- module-level imports
- the spec dataclass body
- the payoff class shell
- the `requirements` property
- the exact analytical route body
- the exact MC route body
- some family-specific fallback behavior around resolved inputs

The agent no longer has to invent:

- raw underlier/FX correlation lookup rules
- foreign-curve lookup conventions
- the generic resolved-input evaluation pattern
- the generic MC path normalization and aggregation pattern
- promotion/review/adoption bookkeeping

## Small vs Larger Surface

### Small surfaces are still warranted for

- route kernels:
  - `price_quanto_option_analytical(...)`
  - `price_quanto_option_mc(...)`
- process builders:
  - `build_quanto_mc_process(...)`
- fixed family skeletons:
  - known `QuantoOptionSpec`
  - known payoff class shells

These reduce the exact code the agent must synthesize while keeping the
architecture modular.

### A larger family-specific scaffold is warranted for

- Monte Carlo family-method pairs where the route policy itself is repetitive
  and fragile
- future products like Himalaya where the agent would otherwise have to invent:
  - multi-asset state assembly
  - correlation/covariance policy
  - path tensor semantics
  - pathwise observation rules
  - discounting/aggregation contracts

In other words:

- `quanto analytical`: small-surface extraction is probably enough
- `quanto MC`: medium-surface extraction is justified
- `Himalaya MC`: likely needs a larger family-specific scaffold before we trust
  autonomous synthesis

## Recommended Next Extraction Order

1. Extract a shared analytical quanto route helper.
2. Extract a shared Monte Carlo quanto route helper or process-builder helper.
3. Add fixed route skeletons so fresh-build generation stops rewriting the spec
   dataclass and payoff shell.
4. Tighten prompt/guardrail obligations so known family routes must use:
   - the family resolver
   - the route helper
   - the adapter scaffold
5. Re-run fresh-build `T105` repeatedly across model settings and measure
   whether success becomes materially less model-sensitive.

## Decision

The current evidence does **not** support a giant generic abstraction sweep
across all products.

It **does** support a systematic medium-sized extraction for known family/method
pairs, starting with `quanto`:

- family resolver
- family route kernel
- family-aware skeleton
- generic adapter base only where it genuinely removes boilerplate

That is the right balance before asking the embedded agent to synthesize
runtime-authored products such as Himalaya.
