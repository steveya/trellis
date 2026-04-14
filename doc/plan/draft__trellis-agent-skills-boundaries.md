# Trellis Agent Skills Boundary Draft

## Purpose

This document drafts the first Trellis-specific skill set for coding agents.

It is intentionally a planning artifact only. It does not create a committed
skill implementation queue yet.

The goal is to define which parts of the Trellis workflow should become agent
skills and which parts should remain runtime/library helpers.

## Boundary Rule

A Trellis runtime helper and a Trellis coding skill should not be the same
thing.

- runtime helpers belong in the repo and are imported by pricing/runtime code
- skills belong to the agent control plane and teach the agent how to work with
  the repo correctly

So a skill should tell the agent:

- which abstractions are authoritative
- which helper surfaces are approved
- which validation workflow to run
- which docs and Linear closeout steps are required

A skill should not:

- duplicate repo pricing math
- replace checked repo helpers with prompt-only logic
- carry giant static copies of codebase reference material

## Proposed First Skill Set

### 1. `trellis-semantic-assembly`

Use when the work concerns:

- semantic contract interpretation
- family lowering
- event/control compilation
- family-first assembly instead of route-first synthesis

Owns:

- semantic-to-family workflow
- event/control compiler guidance
- approved lowering and assembly surfaces
- anti-patterns around route-local meaning

Does not own:

- concrete pricing formulas
- runtime helper implementation

### 2. `trellis-market-regime-normalization`

Use when the work concerns:

- calibration planning
- quote semantics
- comparison tasks
- multi-curve or model-binding normalization

Owns:

- `EngineModelSpec` use
- quote semantics and quote authority workflow
- calibration/market-regime normalization workflow
- anti-patterns around comparing outputs before regime alignment

Does not own:

- raw solver implementations

### 3. `trellis-adapter-hardening`

Use when the work concerns:

- generated adapters
- checked-in adapters
- stale/fresh adapter handling
- adapter architecture minimization

Owns:

- thin-adapter rules
- allowed helper surfaces
- fresh-build versus checked-in adapter workflow
- anti-patterns around adapters owning business logic

Does not own:

- primary product semantics
- primary kernel math

### 4. `trellis-family-helper-design`

Use when building or extending reusable helper surfaces for PDE, MC, lattice,
or other bounded family kits.

Owns:

- end-to-end helper-kit design
- helper layering
- public helper surface rules
- anti-patterns around kernel-only helpers

Does not own:

- product-specific one-off shortcuts unless they are explicitly intended as
  compatibility wrappers

### 5. `trellis-canary-recovery`

Use when working from canary failures and recovery tickets.

Owns:

- failure classification by layer
- trace-reading workflow
- task rerun workflow
- deciding whether the gap is semantic, helper, adapter, validation, or
  calibration

Does not own:

- long-lived runtime abstractions directly; it points to the right workstream

## Why These Are The Right Granularity

These skills are workflow-sized, not product-sized.

They are broad enough to stay reusable, but narrow enough that an agent can be
taught:

- how to approach the task
- which surfaces are authoritative
- what to avoid
- how to validate and close out the work

## What Should Not Become Skills

Avoid:

- one skill per instrument
- one skill per pricing method
- one skill per helper
- one skill per Linear epic

That would mostly duplicate the repo structure and become hard to maintain.

## Future Implementation Notes

If these skills are implemented later, each one should contain:

- a short `SKILL.md`
- explicit trigger conditions
- approved repo surfaces
- anti-patterns
- required validation and doc/Linear closeout rules
- minimal references, loaded only as needed

No Linear workstream is created for this draft yet. It is intentionally held
for future discussion and iteration after the semantic-platform hardening
program has started.
