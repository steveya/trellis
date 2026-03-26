# Phase 5 Review: Bermudan / Callable Proving Ground

Phase 5 extended the proving-ground workflow from the American/LSM route to the
rate-tree exercise family.

## Accepted Contracts

### Callable bond

- use a calibrated short-rate lattice via `build_rate_lattice(...)`
- price by backward induction via `lattice_backward_induction(...)`
- map call dates to Bermudan exercise steps
- use `exercise_fn=min` because the issuer minimizes liability
- ensure the callable price does not exceed the corresponding straight bond

### Bermudan swaption

- use a calibrated short-rate lattice via `build_rate_lattice(...)`
- compute node-level swap intrinsic values against a fixed underlying swap
- price the option by Bermudan backward induction via `lattice_backward_induction(...)`
- use `exercise_fn=max` because the holder exercises to maximize value

## What Changed

- Added proving-ground tests for cached generated callable and Bermudan artifacts.
- Tightened the rate-tree cookbook so it explicitly states:
  - `exercise_fn=min` for issuer-callable structures
  - `exercise_fn=max` for holder-exercised structures
- Rebuilt the cached Bermudan swaption artifact as a thin lattice adapter around
  the real tree primitives instead of a heuristic fixed-leg approximation.
- Added a build-loop regression proving that a rebuilt Bermudan rate-tree module
  prices plausibly through the normal generation path.

## What This Phase Proved

- The rate-tree exercise route now has the same proving-ground coverage that
  the American/LSM route gained in Phase 4.
- The cached callable artifact is route-compliant and numerically plausible.
- The cached Bermudan artifact is now route-compliant and numerically plausible
  against the task reference tree.
- Cookbook guidance, route planning, semantic validation, and cached artifacts
  agree on the lattice exercise family at a high level.

## What Is Still Deferred

- Semantic validation does not yet enforce callable vs puttable vs Bermudan
  exercise semantics explicitly; that is the next phase.
- Route selection between lattice, Monte Carlo control, and other exercise
  families is still heuristic.
