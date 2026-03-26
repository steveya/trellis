# Early-Exercise Monte Carlo Policy Family Completion Review

## Summary

This tranche completed the remaining implemented members of the
`exercise_monte_carlo` policy family and aligned the agent stack around the
policy-class contract instead of a single required symbol.

Implemented:

- `trellis.models.monte_carlo.primal_dual.primal_dual_mc`
- `trellis.models.monte_carlo.stochastic_mesh.stochastic_mesh`

Integrated:

- route planning now advertises multiple implemented exercise-control
  primitives for `exercise_monte_carlo`
- semantic validation and prompt guidance now report the implemented set from
  the shared policy registry
- task comparison target mapping now treats `primal_dual_mc` and
  `stochastic_mesh` as Monte Carlo family targets

## Important Precision Note

`primal_dual_mc` is implemented as:

- a primal lower bound from the admissible Longstaff-Schwartz stopping policy
- an optimistic pathwise upper-bound diagnostic from discounted intrinsic
  exercise payoffs

That keeps the implementation auditable and useful for route-level validation,
but it is not a full Andersen-Broadie nested dual construction. The docs and
policy summaries were kept explicit about that.

## Main Code Changes

- added `trellis.models.monte_carlo.primal_dual`
- added `trellis.models.monte_carlo.stochastic_mesh`
- updated `trellis.models.monte_carlo.__init__`
- updated `trellis.agent.early_exercise_policy`
- updated `trellis.agent.codegen_guardrails`
- updated `trellis.agent.semantic_validation`
- updated `trellis.agent.prompts`
- updated `trellis.agent.task_runtime`
- updated `trellis.agent.assembly_tools`
- updated `trellis.agent.knowledge.import_registry`

## Validation

- focused agent/policy slice:
  - `113 passed, 1 deselected`
- numerical/task slice:
  - `47 passed`

## Remaining Work

- `tsitsiklis_van_roy` is still planned, not implemented
- if we later need a tighter upper bound, we should add a true fitted or nested
  dual construction instead of broadening the claims around the current
  `primal_dual_mc`
