# Phase 4 Review: American / LSM Proving Ground

Phase 4 used the generated American put payoff as the first end-to-end proving
ground for the new semantic agent architecture.

## Accepted Contract

For an equity American put routed to Monte Carlo exercise control, the accepted
Trellis contract is:

- simulate paths with `MonteCarloEngine`
- use a real discretization (`exact`, `euler`, or `milstein`)
- call `longstaff_schwartz(...)` explicitly for exercise control
- import `LaguerreBasis` from `trellis.models.monte_carlo.schemes`
- pass a spot-value payoff callback of shape `(n_paths,) -> (n_paths,)`
- do not invent `method="lsm"`
- do not treat `engine.price(...)` as a substitute for regression-based control

## What Changed

- The canonical Monte Carlo cookbook now includes an explicit early-exercise
  branch using `longstaff_schwartz` and `LaguerreBasis`.
- Monte Carlo method requirements now state the same exercise-control rules.
- The API map exposes the real exercise-control imports.
- The cached generated American artifact was brought into line with the route:
  it now uses `get_numpy()`, `year_fraction(...)`, `engine.simulate(...)`, and
  `longstaff_schwartz(...)`.
- The build loop now has a direct proving-ground regression that rebuilds a
  valid American payoff through the normal generation path and prices it.

## What This Phase Proved

- `ProductIR` + `PrimitivePlan` can drive a real exercise/control product.
- Semantic validation rejects the old fake-MC contract and accepts the real one.
- The generated American adapter prices plausibly against a lattice reference.
- The main build path, cookbook path, and cached artifact now agree on the same
  LSM contract.

## What Is Still Deferred

- Bermudan swaption and callable products still need their own proving-ground
  passes.
- Route selection is still heuristic, not learned.
- Continuous-American fixed-income exercise remains an honest blocker unless the
  lattice/PDE substrate is explicitly widened.
