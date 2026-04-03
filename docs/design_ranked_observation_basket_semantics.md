# Ranked-Observation Basket Semantics Design

This document keeps the useful canonical-slice ideas from the earlier
Himalaya-focused plan, but names the runtime boundary by semantics rather than
by product-family label.

## Naming Boundary

- "Himalaya" is acceptable as request language or explanatory prose.
- The runtime boundary should be `ranked_observation_basket` semantics, not a
  `himalaya_option` branch.

## Canonical Slice

The retained canonical proving slice is:

- a basket of risky assets
- a fixed ordered observation schedule
- at each observation, select the best performer among remaining assets
- remove the selected asset from the remaining set
- lock the selected return into contract memory
- settle a single maturity payoff from the locked observations

The initial route family is Monte Carlo only.

## Required Semantics

- `constituents`
- ordered `observation_schedule`
- observation basis
- selection operator, scope, and count
- lock or remove rule
- aggregation rule
- maturity settlement rule
- explicit state variables and event transitions

## Required Market Inputs

- per-name spots
- per-name vols
- per-name carry or dividend inputs
- discount curve
- correlation matrix

Later-extension inputs may include FX and local-vol or stochastic-vol variants,
but they are not part of the canonical proving slice.

## Provenance And Binding Rules

- the runtime must keep constituent identifiers stable enough for connector
  lookup
- the runtime must distinguish observed, derived, estimated, and user-supplied
  inputs
- the runtime must not silently fabricate a correlation matrix
- any future estimation policy must make its data window, PSD repair, and
  provenance labels explicit

## Validation And Reduction Checks

Structural requirements:

- at least two constituents
- non-empty ordered observation schedule
- selection is over the remaining basket, not the original full basket after
  removals

Coherence requirements:

- `path_dependence` and `schedule_dependence` are required
- the route is MC-only for the canonical slice
- lock semantics and payoff aggregation must agree

Reduction checks:

- a one-observation case should reduce toward best-of or rainbow semantics
- identical-name fixtures should not create contradictory state-machine results
- schedule monotonicity and locked-state progression should remain invariant

## Current Implementation Anchors

- `trellis.agent.semantic_contracts.make_ranked_observation_basket_contract`
- `trellis.agent.platform_requests._draft_semantic_contract`
- `trellis.models.resolution.basket_semantics`
- `trellis.models.monte_carlo.basket_state`
- `trellis.models.monte_carlo.ranked_observation_payoffs`
- `trellis.models.monte_carlo.semantic_basket`

## Non-Goals

- autocallable wrappers
- coupon-bearing note wrappers unless reduced to the core payoff
- principal guarantees
- callable redemption state machines
- claiming full commercial structured-note coverage

## Related Linear Tickets

Status snapshot as of 2026-04-02:

- `QUA-284` Runtime-request contract: `Done`
- `QUA-329` Derivative-agnostic synthesis roadmap: `Done`
- `QUA-333` Phase 4 representative derivative regression matrix: `Done`
- `QUA-334` Phase 5 documentation, knowledge, and roadmap hardening: `Done`
- `QUA-376` Semantic DSL gap classification for novel requests: `Done`
- `QUA-397` Semantic control plane and provenance: `Done`
