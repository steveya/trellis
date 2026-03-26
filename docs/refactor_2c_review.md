# Tranche 2C Review

This note records the review pass for Tranche 2C before any export-normalization
tests or code changes.

## Scope

Tranche 2C is the public-surface normalization tranche.

The goal is not a sweeping rename. The goal is to make the existing package
surface more coherent by:

- keeping the curated top-level `trellis` API stable
- giving `trellis.core` a real package surface
- giving `trellis.models` a real package surface
- moving docs toward canonical package-level entry points
- preserving compatibility where practical

## Current State

### `trellis`

- [KEEP] `trellis.__init__` is already a curated user-facing surface
- it exposes the v2 API (`Session`, `Book`, `price`, `ask`, samples, key
  instruments, key curves, and common conventions)
- this should remain the primary top-level user entry point

### `trellis.core`

- [REFACTOR] `trellis/core/__init__.py` is empty
- the package exists conceptually, and docs already treat it as a public area,
  but users must import from deep modules such as `trellis.core.market_state`
  and `trellis.core.payoff`
- this is a documentation/API mismatch

### `trellis.models`

- [REFACTOR] `trellis/models/__init__.py` is empty
- the family packages under `trellis.models.*` are mostly well-shaped already:
  - analytical
  - trees
  - monte_carlo
  - pde
  - transforms
  - processes
  - copulas
  - calibration
  - cashflow_engine
- those subpackages already provide useful curated `__init__` exports
- the missing piece is the package hub at `trellis.models`

### Docs

- [REFACTOR] API docs still point to stale or low-level module paths
- examples:
  - `docs/api/models.rst` still documents
    `trellis.models.pde.crank_nicolson.crank_nicolson_1d`
    and `trellis.models.pde.implicit_fd.implicit_fd_1d`
    even though the canonical PDE surface is now `theta_method_1d` with
    backward-compat wrappers exposed from `trellis.models.pde`
- `docs/api/core.rst` documents deep module paths rather than a package-level
  surface

### Metadata

- [REFACTOR] `pyproject.toml` is the real package metadata
- [DEPRECATE / RECONCILE] `setup.py` is stale:
  - package name is still `rate-model`
  - dependency list is incomplete
  - classifiers stop at a different surface than the current package reality
- [REFACTOR] `README.md` is also stale and still says `# rate-model`

## Canonical Target for 2C

### Top-level package

Keep `trellis` as the stable, curated, user-facing API.

### `trellis.core`

Promote a canonical package surface for:

- `MarketState`
- `MissingCapabilityError`
- `Payoff`
- `DeterministicCashflowPayoff`
- `Cashflows`
- `PresentValue`
- `StateSpace`
- `Frequency`
- `DayCountConvention`
- `PricingResult`
- capability helpers such as:
  - `analyze_gap`
  - `check_market_data`
  - `discover_capabilities`
  - `capability_summary`

### `trellis.models`

Promote a canonical package surface that:

- re-exports the most common direct helpers:
  - `black76_call`
  - `black76_put`
  - `FlatVol`
  - `VolSurface`
- re-exports the package-level family surfaces that already exist via their
  subpackage `__init__` files
- does not try to flatten every symbol from every method family into one giant
  namespace

The intended mental model is:

- `trellis.models` is the family hub
- `trellis.models.pde`, `trellis.models.trees`, `trellis.models.monte_carlo`,
  etc. are the canonical family packages
- deep-module imports remain valid where already used, but are not the primary
  documented surface

## Planned 2C.2 Tests

The first export/doc tests should lock down:

1. `trellis.core` package imports for the core types listed above
2. `trellis.models` package imports for:
   - `black76_call`, `black76_put`
   - `FlatVol`, `VolSurface`
   - family subpackages such as `pde`, `trees`, `monte_carlo`, `transforms`,
     `processes`, `copulas`, `calibration`, `cashflow_engine`, `analytical`
3. package-level family exports that are already intended to be public, e.g.:
   - `trellis.models.pde.theta_method_1d`
   - `trellis.models.trees.BinomialTree`
   - `trellis.models.monte_carlo.MonteCarloEngine`
   - `trellis.models.transforms.fft_price`
4. docs reference the canonical package-level entry points instead of stale
   low-level paths
5. top-level `trellis` exports remain stable

## Non-goals for 2C

- no broad package rename
- no forced migration to a new `pricing/` root
- no removal of deep-module imports that are already part of real usage
- no method-family expansion
- no behavior changes in the pricing engines themselves
