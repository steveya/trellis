# Phase C0: AI Contract Synthesis for Quanto and Himalaya

This note is the checked-in handoff for the next top-priority tranche in
Trellis.

The goal is not to manually implement another isolated pricer. The goal is to
make AI capable of defining the guardrailed contract layers for known product
families so the library can evolve in a bounded, auditable way.

## Why This Is Now the Priority

Recent FX work clarified the distinction between:

- a family that is already formalized enough to support deterministic adapters
  and reuse
- a family that still falls back to generic `european_option` reasoning and
  free-form code generation

Vanilla FX is now in the first category:

- `E25` passes
- `T108` passes

Quanto is still in the second category:

- `T105` fails
- the failure is no longer market-data plumbing or missing FX cookbook support
- it is a real missing-family problem

That same gap will matter for other known exotics such as:

- Himalaya / mountain options
- rainbow / best-of / worst-of basket options
- autocallables with multi-asset state

If Trellis is going to become a self-evolving library, the system needs to be
able to formalize these families into structured contracts before it tries to
write implementations.

## Tranche Goal

Implement the first executable contract-synthesis layer for known product
families, starting with:

- `quanto`
- `Himalaya`

The output of this layer must be machine-readable and suitable for:

- product semantics
- market-data requirements
- method-family expectations
- sensitivity support
- validation bundle selection
- implementation blueprint generation

## What "AI Defines the Contract Layer" Means

For a known product family, AI should be able to draft a structured contract
covering:

1. Product semantics
- `family_id`
- `payoff_family`
- `exercise_style`
- `path_dependence`
- observation schedule semantics
- payoff state variables
- event/state-machine transitions

2. Market-data contract
- required inputs
- optional inputs
- input aliases or bridges
- mapping notes into pricing inputs

3. Method contract
- candidate method families
- reference vs production methods
- honest limitations

4. Sensitivity contract
- support level:
  - `native`
  - `bump_only`
  - `experimental`
  - `unsupported`
- supported measures
- stability notes

5. Validation bundle hints
- universal checks
- no-arbitrage checks
- family-specific sanity checks
- comparison expectations

6. Implementation blueprint hints
- primitive families required
- adapter expectations
- likely target modules
- likely proving-ground tasks

The system may use AI to draft these contracts, but deterministic validation
must decide whether they are coherent enough to proceed.

## Initial Families

### Quanto

The initial contract draft should capture:

- domestic payout currency
- foreign underlying currency
- FX linkage between underlying and payout currency
- domestic and foreign discounting / carry inputs
- underlier volatility
- FX volatility
- underlier/FX correlation
- likely methods:
  - analytical quanto adjustment
  - correlated Monte Carlo
- honest initial sensitivity support:
  - likely `bump_only`

### Himalaya

The initial contract draft should capture:

- basket constituents
- observation schedule
- selection rule at each observation
- lock-in / removal / freeze semantics
- coupon or redemption semantics if present
- path-state updates across observations
- likely methods:
  - multi-asset Monte Carlo first
- required market inputs:
  - spots
  - vols
  - dividends/carry
  - correlation matrix
  - schedule
- honest initial sensitivity support:
  - likely `bump_only`

## Existing Surfaces to Reuse

Useful current components:

- multi-asset process:
  - `/Users/steveyang/Projects/steveya/trellis/trellis/models/processes/correlated_gbm.py`
- planner-side basket recognition:
  - `/Users/steveyang/Projects/steveya/trellis/trellis/agent/planner.py`
- route / primitive planning:
  - `/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py`
- validation bundles:
  - `/Users/steveyang/Projects/steveya/trellis/trellis/agent/validation_bundles.py`
- framework task runner:
  - `/Users/steveyang/Projects/steveya/trellis/trellis/agent/framework_runtime.py`
- roadmap and task inventories:
  - `/Users/steveyang/Projects/steveya/trellis/docs/autonomous_library_development_workstream.md`
  - `/Users/steveyang/Projects/steveya/trellis/TASKS.yaml`
  - `/Users/steveyang/Projects/steveya/trellis/FRAMEWORK_TASKS.yaml`

Relevant framework and pricing tasks already exist:

- multi-asset MC extraction
- exotic payoff protocol extraction
- `T102` rainbow
- `T105` quanto

## Deliverables

This tranche should deliver:

1. A concrete contract schema module
- checked-in, typed, deterministic

2. Deterministic contract validation
- structure checks
- coherence checks
- unsupported-claim checks

3. Initial family contract drafts or templates
- `quanto`
- `Himalaya`

4. Contract-to-blueprint compilation
- validated contract -> implementation plan shape

5. Documentation and roadmap updates

## Non-Goals

This tranche should not try to do all of the following at once:

- fully implement quanto pricing
- fully implement Himalaya pricing
- solve full term-sheet extraction
- allow AI to self-approve unsupported claims

The objective is to formalize the contract layer that future implementation
tranches will consume.

## Proposed Implementation Workflow

1. Review current surfaces
- inspect planner, quant, route planning, validation bundles, framework runner,
  and multi-asset process support

2. Write red deterministic tests first
- contract schema validation
- quanto contract validation
- Himalaya contract validation
- blueprint generation

3. Implement the contract layer
- schema
- validators
- family templates
- blueprint compiler

4. Run targeted validation
- use `/Users/steveyang/miniforge3/bin/python3`

5. Update docs and roadmap

## Acceptance Criteria

This tranche is successful when:

- Trellis has a checked-in machine-readable contract schema for product-family
  definitions
- Trellis can validate whether a drafted `quanto` contract is coherent
- Trellis can validate whether a drafted `Himalaya` contract is coherent
- Trellis can compile a validated contract into a deterministic blueprint /
  implementation-plan shape
- the resulting artifacts are explicit enough that future AI implementation
  work can target bounded primitives and adapters instead of falling back to
  generic prompt-only code generation

## How This Moves Us Toward the Final Goal

The final goal is not just "agents write more code." The final goal is:

- new product-family requests become formalizable
- validated contracts become reusable library substrate
- future implementations become bounded, testable, and promotable
- failures become specific family-work items instead of generic LLM timeouts

This tranche is the first direct step toward that self-evolving-library model.
