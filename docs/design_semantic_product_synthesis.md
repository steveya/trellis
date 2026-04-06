# Semantic Product Synthesis Design

This document is the current design reference for Trellis' semantic-contract
and DSL-based product synthesis path. It supersedes the older C0 family-
contract planning sequence as the primary architecture note.

Read this together with:

- `docs/quant/contract_algebra.rst`
- `docs/quant/dsl_algebra.rst`
- `docs/quant/pricing_stack.rst`
- `docs/qua-284-arbitrary-derivative-proving-run.md`

## Current Boundary

The shipped boundary is:

- `SemanticContract` is the authoritative product-meaning object
- `ValuationContext`, `RequiredDataSpec`, and `MarketBindingSpec` separate
  valuation policy and market binding from contract meaning
- `ProductIR` remains the shared checked summary used for route selection
- lowering narrows from `ProductIR` into family-specific lowering IRs and then
  onto helper-backed numerical routes
- for migrated routes, typed timeline, obligation, and event-state surfaces are
  authoritative; legacy mirror strings remain only for compatibility

This is intentionally not a universal solver IR. The current design goal is a
narrow, typed, auditable route boundary.

## Retained Design Ideas

These ideas are still useful from the earlier plan set:

- keep `ProductIR` coarse; rich product meaning lives above it
- preserve required market inputs, provenance, admissibility, and route
  obligations in the compiled blueprint instead of flattening them away
- prefer medium helper-backed surfaces over giant fake-generic abstractions
- treat product names as request-language hints, not as the runtime
  architecture boundary
- use checked-in templates only as compatibility bridges or regression fixtures,
  not as the long-term truth source for novel products

## Canonical Proving Slices

### `quanto_option`

The narrow checked-in quanto slice still matters, but it now sits inside the
semantic path in two ways:

- a compatibility bridge from the legacy family template
- a generic semantic contract constructor for the migrated route

Current anchors:

- `trellis.agent.semantic_contracts.make_quanto_option_contract`
- `trellis.agent.family_contract_templates.family_template_as_semantic_contract`
- `trellis.models.resolution.quanto`
- `trellis.models.analytical.quanto`
- `trellis.models.monte_carlo.quanto`

### `ranked_observation_basket`

The former Himalaya-style proving case is now expressed as ranked-observation
basket semantics. The runtime should not depend on a `himalaya_option` branch.

Current anchors:

- `trellis.agent.semantic_contracts.make_ranked_observation_basket_contract`
- `trellis.agent.platform_requests._draft_semantic_contract`
- `trellis.models.resolution.basket_semantics`
- `trellis.models.monte_carlo.basket_state`
- `trellis.models.monte_carlo.ranked_observation_payoffs`
- `trellis.models.monte_carlo.semantic_basket`

## Legacy And Migration Notes

- `trellis.agent.family_contract_compiler` has been removed. Checked-in family
  templates must route through the semantic bridge instead of reviving the old
  family-blueprint compiler.
- `trellis.agent.family_contract_templates` still matters for the checked-in
  quanto bridge, but new work should compile through
  `compile_semantic_contract(...)`.
- The earlier family-contract implementation docs were useful sketches for the
  migration, but they are no longer the authoritative architecture notes.
- `QUA-286` and `QUA-287` remain in `Backlog`. Treat them as legacy planning
  references rather than the source of truth for shipped behavior.

## Design Rules

- do not add new runtime branches keyed on product names like `himalaya_option`
- preserve typed semantic meaning through validation, admissibility, and
  lowering
- prefer helper-backed synthesis where generated code is only the thin product-
  specific layer
- keep unsupported routes explicit through semantic validation errors or
  admissibility failures rather than degrading silently to generic products

## Related Linear Tickets

Status snapshot as of 2026-04-02:

- `QUA-286` Validator rules and draft fixtures: `Backlog`
- `QUA-287` Compiler and request routing: `Backlog`
- `QUA-284` Runtime-request contract: `Done`
- `QUA-329` Derivative-agnostic synthesis roadmap: `Done`
- `QUA-333` Phase 4 representative derivative regression matrix: `Done`
- `QUA-334` Phase 5 documentation, knowledge, and roadmap hardening: `Done`
- `QUA-397` Semantic control plane and provenance: `Done`
- `QUA-409` DSL enforcement and expressiveness: `Done`

## What This Document Replaces

This design note absorbs the useful architectural material from the older:

- C0 contract-synthesis implementation sketch
- C0 quanto and Himalaya planning notes
- C0 handoff note
- C2 family-name-free semantic synthesis plan

The orthogonal follow-on ideas now live in separate design notes:

- `docs/design_quanto_runtime_contract.md`
- `docs/design_ranked_observation_basket_semantics.md`
- `docs/design_generation_glue_surfaces.md`
- `docs/design_analytical_support_substrate.md`
- `docs/design_connector_stress_regression_gate.md`
