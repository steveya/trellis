# Analytical Support Substrate Design

This document keeps the useful follow-on design from the earlier C1 plan while
renaming it as a standing design reference.

## Purpose

Reduce Trellis-specific analytical glue by moving repeated discounting,
forward-building, payoff-transform, cross-asset logic, and event-family payoff
assembly into pure helper surfaces.

## Scope

In scope:

- reusable analytical support functions that are `MarketState`-free
- support kernels usable both by top-level analytical routes and subproblems
  inside larger engines
- shared event-family kernels that are reused by more than one checked route
  helper
- prompt and scaffold guidance that forces fresh-build analytical routes to use
  the checked-in support layer

Out of scope:

- claiming broad closed-form support that Trellis does not have
- a giant symbolic formula system
- replacing every product-specific kernel with one generic formula API

## Design Principles

- analytical kernels do not read `MarketState`
- family or semantic resolvers own market binding
- support helpers stay pure and typed
- route kernels compose support helpers
- generated code should prefer resolver, support helper, route kernel, then
  thin adapter
- support helpers should be reusable, but not so fine-grained that they become
  formula-confetti

## Current Support Taxonomy

Foundational helpers now live under `trellis/models/analytical/support/`:

- discounting and rate transforms
- forward builders
- payoff transforms
- cross-asset transforms

Useful next extension areas remain:

- distribution helpers
- approximation helpers
- inversion and quoting helpers

## Current Implementation Anchors

- `trellis/models/analytical/support/discounting.py`
- `trellis/models/analytical/support/forwards.py`
- `trellis/models/analytical/support/payoffs.py`
- `trellis/models/analytical/support/cross_asset.py`
- `trellis/models/contingent_cashflows.py`
- `trellis/core/runtime_contract.py`
- `trellis/models/monte_carlo/event_state.py`
- `trellis/models/credit_default_swap.py`
- `trellis/instruments/nth_to_default.py`
- `trellis/instruments/mortgage_pass_through.py`
- `trellis/models/analytical/quanto.py`
- `trellis/agent/prompts.py`
- `trellis/agent/executor.py`

## Retained Guidance

- keep `quanto` as the first proving route for support-layer reuse
- add helpers only when they represent a stable reusable subproblem
- keep prompt regressions next to support regressions
- prefer a measured expansion of reusable kernels over abstract generality

## Current Design Implications

- the support layer should keep route kernels thin
- a second checked-in consumer is the best reason to promote a helper
- the event-family substrate is now a checked boundary: CDS and
  nth-to-default reuse shared contingent-cashflow support, and mortgage
  pass-throughs reuse the same promotion rule for extracted kernels
- future analytical refactors should look first for repeated discounting,
  normalization, distribution, carry logic, and event-state assembly

## Related Linear Tickets

Status snapshot as of 2026-04-02:

- `QUA-289` Analytical support: `Done`
- `QUA-291` Reusable analytical kernels: `Done`
- `QUA-292` Refactor analytical route kernels: `Done`
- `QUA-293` Builder guidance for analytical assembly: `Done`
- Parent umbrella `QUA-277` Autonomous library development: `In Progress`
