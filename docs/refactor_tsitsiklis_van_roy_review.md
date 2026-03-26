# Tsitsiklis-Van Roy Early-Exercise Policy Review

Date: 2026-03-26

This tranche completed `EEMC.3`.

## What changed

- Added `trellis.models.monte_carlo.tv_regression` with:
  - `tsitsiklis_van_roy_result(...)`
  - `tsitsiklis_van_roy(...)`
- Exported the new primitive from `trellis.models.monte_carlo`
- Marked `tsitsiklis_van_roy` as implemented in
  `trellis.agent.early_exercise_policy`
- Added the primitive to deterministic route planning in
  `trellis.agent.codegen_guardrails`
- Extended deterministic semantic validation tests so the new primitive is
  recognized as an approved early-exercise control

## Implementation shape

The implementation is intentionally conservative:

- same shared early-exercise result contract as the other policy classes
- same path-array input surface as `longstaff_schwartz`
- fitted continuation-value recursion over exercise dates
- no claim of dual bounds or sensitivity-native behavior

So this is a sibling primitive in the policy family, not a new architecture.

## Validation

- shared-policy-result tests
- side-by-side comparison against `longstaff_schwartz`
- planning and semantic-validation tests
- nearby prompt / knowledge-store / import-registry regression slices

## Remaining work

The early-exercise family is now complete at the policy-class level, but still
subject to the broader sensitivity-support and route-selection priorities
already captured elsewhere in the roadmap.
