# Quanto Runtime Contract Design

This document keeps the useful narrow-slice ideas from the earlier quanto plan
without treating the old family-contract writeup as the primary architecture.

## Purpose

Preserve the design for the checked-in single-underlier European quanto slice
used by the `T105` proving path and its migrated semantic equivalent.

## Scope

In scope:

- single-underlier European quanto option semantics
- explicit domestic payout currency and foreign underlier currency
- analytical quanto adjustment and correlated Monte Carlo as the two initial
  route families
- honest, provenance-aware market binding

Out of scope:

- American or Bermudan quanto structures
- path-dependent quanto exotics
- multi-asset quanto baskets
- silent widening from the proving slice to generic cross-currency exotics

## Retained Contract Requirements

Minimum semantic and runtime fields:

- underlier identity
- option type
- strike and expiry
- notional
- domestic payout currency
- underlier currency
- FX linkage between those currencies
- explicit settlement behavior

Required market inputs:

- domestic discount curve
- foreign discount or carry curve
- underlier spot
- FX spot
- underlier volatility
- FX volatility
- underlier/FX correlation

Optional or later-extension inputs:

- dividend or convenience yield
- local-vol inputs

## Provenance And Binding Rules

- the runtime must distinguish observed, derived, estimated, and user-supplied
  inputs
- foreign discounting may be bridged only when the connector policy makes that
  mapping explicit
- the runtime must not fabricate `fx_vol` or `underlier_fx_correlation`
- missing required quanto inputs should fail with family-specific wording rather
  than degrade into generic vanilla semantics

Accepted narrow-slice bridge forms:

- canonical foreign carry keys such as `EUR-DISC`, `EUR_DISC`, or `EUR`
- explicit `quanto_foreign_curve_policy` using the selected forecast curve
- explicit `quanto_foreign_curve_policy` reusing the domestic discount curve
  when that bridge is intentionally declared

## Route Expectations

The retained route hierarchy is:

- primary route: analytical quanto adjustment
- independent cross-check: correlated underlier/FX Monte Carlo

The contract and compiled blueprint should preserve enough detail to:

- keep domestic-vs-foreign payout semantics explicit
- keep FX vol and correlation requirements explicit
- choose analytical only for the narrow supported slice
- fall back honestly when analytical assumptions or required inputs are not met

## Current Implementation Anchors

The useful checked-in surfaces are:

- `trellis.agent.semantic_contracts.make_quanto_option_contract`
- `trellis.agent.family_contract_templates`
- `trellis.models.resolution.quanto`
- `trellis.models.analytical.quanto`
- `trellis.models.monte_carlo.quanto`
- `trellis.core.payoff`

## Current Design Implications

- use resolver-backed route helpers rather than open-coded market binding
- keep the analytical and Monte Carlo helpers as stable route surfaces
- preserve the narrow proving slice explicitly rather than implying broad quanto
  support
- treat richer connector-backed runtime binding as follow-on work, not as
  already solved

## Related Linear Tickets

Status snapshot as of 2026-04-02:

- No dedicated quanto-only issue was identified from the source plan set.
- Nearest related follow-on issues found in Linear:
  - `QUA-289` Analytical support: `Done`
  - `QUA-291` Reusable analytical kernels: `Done`
  - `QUA-292` Refactor analytical route kernels: `Done`
  - `QUA-293` Builder guidance for analytical assembly: `Done`
- Legacy family-contract tickets still referenced by the old plan:
  - `QUA-286` Validator rules and draft fixtures: `Backlog`
  - `QUA-287` Compiler and request routing: `Backlog`
