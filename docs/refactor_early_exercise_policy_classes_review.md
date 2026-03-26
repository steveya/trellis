# Early-Exercise Monte Carlo Policy Classes Review

## Summary

This tranche relaxed the semantic and prompt-level contract for
`exercise_monte_carlo`.

Previously, Trellis effectively treated Monte Carlo early exercise as
synonymous with `longstaff_schwartz`. That was too strict. The new contract is:

- Monte Carlo early exercise must use an approved control-policy class
- approved classes currently are:
  - `longstaff_schwartz` [implemented]
  - `tsitsiklis_van_roy` [planned]
  - `primal_dual_mc` [planned]
  - `stochastic_mesh` [planned]

We stayed honest about implementation state:

- semantic validation now recognizes the broader policy family
- prompt and route-card guidance mention the broader family
- import validation remains strict, so planned classes cannot be silently
  treated as already available imports

## Main Code Changes

- added `trellis.agent.early_exercise_policy`
- updated `trellis.agent.semantic_validation`
- updated `trellis.agent.prompts`
- updated `trellis.agent.codegen_guardrails`

## Why This Is Better

- avoids baking a single algorithm into the route contract
- keeps room for future `primal_dual_mc` and `stochastic_mesh` support
- avoids overclaiming implementation state
- aligns the roadmap with the deterministic validation layer

## Remaining Work

- add shared early-exercise Monte Carlo contracts in library code
- refactor `longstaff_schwartz` onto that common contract
- implement the other approved classes in bounded phases
- extend route/task comparison support to compare policy classes explicitly
