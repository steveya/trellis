# Local-Vol Monte Carlo Slice Review

Date: 2026-03-26

This note records the implementation of `M4.2`, the bounded local-vol Monte
Carlo slice defined in
[`refactor_local_vol_preflight_review.md`](./refactor_local_vol_preflight_review.md).

## Scope implemented

Implemented:

- reusable vanilla local-vol Monte Carlo helper in
  `trellis.models.monte_carlo.local_vol`
- local-vol context override in `trellis.agent.quant`
- deterministic `local_vol_monte_carlo` primitive route in
  `trellis.agent.codegen_guardrails`
- lite-review support for `local_vol_surface` and the new route
- validation-fixture support for `spot + local_vol_surface` market states

Still not implemented in this slice:

- a dedicated `local_vol_pde` primitive/operator contract
- canonical knowledge-layer registration under `trellis.agent.knowledge`
- a live `E23` rerun

## Resulting behavior

For bounded vanilla-equity local-vol prompts such as:

- `European equity call under local vol: PDE vs MC`
- `Local volatility: Dupire PDE forward vs MC with local vol`

the planner now treats the request as surface-driven rather than implied-vol
analytical. In the Monte Carlo branch:

- `spot`
- `discount_curve`
- `local_vol_surface`

become the deterministic market-data contract, and the route planner now
selects `local_vol_monte_carlo` instead of generic `monte_carlo_paths`.

## Reusable primitive

The new reusable helper is:

- `trellis.models.monte_carlo.local_vol.local_vol_european_vanilla_price(...)`

It wraps:

- `trellis.models.processes.local_vol.LocalVol`
- `trellis.models.monte_carlo.engine.MonteCarloEngine`

for European vanilla call/put pricing under a supplied local-vol surface.

## Validation

Focused validation passed:

- `tests/test_models/test_monte_carlo/test_mc.py`
- `tests/test_agent/test_quant.py`
- `tests/test_agent/test_primitive_planning.py`
- `tests/test_agent/test_lite_review.py`
- `tests/test_agent/test_task_runtime.py`

Result:

- `84 passed, 1 deselected`

## Important nuance

A broader package-level regression attempt surfaced a separate worktree issue:

- `trellis/models/monte_carlo/discretization.py` is currently deleted in the
  local worktree
- several existing modules still import it

That inconsistency is outside the bounded local-vol slice and blocks a larger
`trellis.models.monte_carlo` package-import regression sweep until it is
resolved separately.

## Recommended next step

Use this narrower MC slice as the new baseline, then decide `M4.3` from an
actual `E23` rerun:

1. rerun `E23`
2. if the MC branch now succeeds and the PDE branch blocks narrowly, implement
   the local-vol PDE adapter/operator tranche next
3. if both branches succeed, move local-vol work to promotion and knowledge
   registration
