# Local-Vol Preflight Review

Date: 2026-03-26

This note closes `M4.1` by bounding the local-vol proving ground around `E23`
 before any broader implementation work.

## Target task

- `E23`: European equity call under local vol: PDE vs MC

Current task contract:

- construct families: `pde`, `monte_carlo`
- comparison targets: `local_vol_pde`, `local_vol_mc`
- required market context:
  - `discount_curve`
  - `spot`
  - `local_vol_surface`

## What already exists

The blocker is not missing market data.

Available today:

- mock local-vol surfaces in `trellis.data.mock`
  - `spx_local_vol`
  - `aapl_local_vol`
- task/runtime market-state wiring for `local_vol_surface`
- reusable local-vol process in `trellis.models.processes.local_vol.LocalVol`
- reusable Dupire-style calibration helper in
  `trellis.models.calibration.local_vol.dupire_local_vol`
- generic 1D theta-method PDE solver in `trellis.models.pde.theta_method`

So the core substrate pieces are present.

## What is still missing

The gap is route integration and thin execution wiring, not raw capability
absence.

Missing today:

- no deterministic local-vol-specific primitive route in
  `trellis.agent.codegen_guardrails`
- no local-vol-specific route selection in `trellis.agent.quant`
- no deterministic MC adapter from:
  - `market_state.local_vol_surface`
  - `market_state.spot`
  - `market_state.discount`
  into `LocalVol` + Monte Carlo execution
- no deterministic PDE adapter from the same market inputs into a reusable
  local-vol PDE operator/solver contract
- no canonical latest `E23` run in the task-run store, so the proving ground
  still lacks a fresh post-FX baseline trace

## Scope control

This proving ground should not expand into:

- stochastic-vol / Heston support
- path-dependent local-vol exotics
- early exercise under local vol
- multi-asset local-vol coupling
- calibration UX or surface-building workflows beyond the already existing
  callable local-vol surface input

The bounded goal remains:

- European vanilla equity option
- local vol supplied directly by `market_state.local_vol_surface`
- one MC path
- one PDE path

## Recommended M4.2 slice

Start with the narrower Monte Carlo slice first.

Why:

- `LocalVol` already exists as a reusable process
- the Monte Carlo path only needs a thin adapter and validation slice
- it gives us an honest narrowed blocker if the PDE path still lacks the right
  operator contract

Recommended order:

1. `local_vol_mc_equity`
   - build the deterministic adapter from `MarketState` to `LocalVol`
   - validate on a European vanilla call
2. rerun `E23`
   - expect either one method to execute and the PDE side to block narrowly, or
     both to execute if the PDE path is already close enough
3. only then decide whether `local_vol_pde` needs a separate primitive/operator
   tranche

## Exit from preflight

The local-vol proving ground is now bounded as:

- first implementation target: local-vol MC for European vanilla equity
- second possible target: local-vol PDE operator/adapter only if the rerun
  shows that as the remaining blocker
