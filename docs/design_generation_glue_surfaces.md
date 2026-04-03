# Generation Glue Surfaces Design

This note keeps the useful design heuristics from the earlier glue-surface
sweep. The concrete evidence came from the quanto proving path, but the design
rules are broader.

## Purpose

Decide when Trellis should extract more deterministic helper surface and when
it should stop short of giant generic abstractions.

## Core Design Rule

Prefer medium-sized helper-backed surfaces for known route or family pairs.

That means:

- resolvers should own market binding
- route helpers should own repeated pricing policy
- fixed skeletons should own repetitive module and adapter boilerplate
- prompts and guardrails should require use of those surfaces

It does not mean introducing a giant universal scaffold before the shared
semantics and helper boundaries are stable.

## Retained Lessons

### Module and import glue

- deterministic route skeletons reduce import drift and boilerplate noise
- required imports should be mostly fixed for known route shapes

### Market-resolution glue

- resolver return types should be treated as authoritative
- generated code should not re-probe market state when a shared resolver has
  already normalized the inputs

### Analytical route glue

- small shared route kernels are worthwhile when the math is narrow and stable
- the adapter should be thinner than the kernel

### Monte Carlo route glue

- MC routes often justify a larger helper surface than analytical routes
- process construction, initial state, engine defaults, discounting, and payoff
  shape should not be regenerated ad hoc when the route is known

### Validation and promotion glue

- these are necessary for autonomy, but they are not the main remaining route-
  math bottleneck once structured diagnostics and review gates exist

## Current Implications

- `quanto` analytical remains a small-surface case
- `quanto` Monte Carlo remains a medium-surface case
- ranked-observation basket Monte Carlo should stay helper-heavy rather than
  leaving state assembly and path semantics to generated code

## Current Implementation Anchors

- `trellis.models.resolution.quanto`
- `trellis.models.analytical.quanto`
- `trellis.models.monte_carlo.quanto`
- `trellis.core.payoff`
- `trellis.agent.prompts`
- `trellis.agent.executor`

## Decision

The retained decision is still correct:

- do not jump to a giant generic abstraction sweep
- keep extracting stable resolver, route-helper, and skeleton surfaces where
  they materially reduce generated glue
- use semantic and helper-backed boundaries to keep generated code thin

## Related Linear Tickets

Status snapshot as of 2026-04-02:

- No direct Linear issue was identified from the original glue-surface note.
- The clearest downstream follow-on issues are:
  - `QUA-289` Analytical support: `Done`
  - `QUA-291` Reusable analytical kernels: `Done`
  - `QUA-292` Refactor analytical route kernels: `Done`
  - `QUA-293` Builder guidance for analytical assembly: `Done`
