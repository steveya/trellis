# Shared Memory Substrate Integration Review

Date: 2026-03-25

## What changed

The shared knowledge substrate is no longer limited to the builder loop.

- Request compilation now materializes a single shared-knowledge bundle with:
  - builder prompt text
  - reviewer/model-validator prompt text
  - decomposition/routing prompt text
  - a compact trace/task summary
- The bundle is now carried through:
  - free-form build compilation
  - term-sheet compilation, including the existing-payoff path
  - structured user-defined product compilation
  - comparison-task compilation
- Platform traces persist the compact shared-knowledge summary so request traces
  now show which lessons/principles/cookbook route were active.
- Knowledge-aware build results preserve the compact shared-knowledge summary.
- Task runtime payloads now expose:
  - structured agent observations
  - shared-knowledge summary
  - aggregated knowledge summaries across comparison builds
- The remaining legacy fallback in `trellis.agent.prompts.evaluate_prompt()` now
  uses unified retrieval/formatting instead of assembling cookbook, contracts,
  requirements, and experience separately.

## Why this matters

The request/compiler layer is now the default producer of shared knowledge,
rather than the executor reconstructing separate prompt contexts ad hoc. This
keeps builder, critic, arbiter, model-validator, traces, and task results on
the same semantic substrate.

## Remaining isolated mechanisms

These remain intentionally unintegrated or only partially integrated:

1. `trellis.agent.analytics_cookbook`
   - Still a standalone analytics guidance source outside the knowledge store.
   - It is useful, but it is not yet part of the unified shared-memory payload.

2. `trellis.agent.experience`
   - Now mainly a compatibility wrapper over knowledge retrieval.
   - Legacy flat-file fallback remains for safety, but it is no longer the
     primary substrate for builder prompts.

3. `trellis.agent.test_resolution`
   - Still references the older experience workflow directly.
   - This is a remediation/debugging side path, not the main shared-memory loop.

4. Direct `Session` / `Pipeline` requests for existing instruments
   - These now get canonical platform traces, but they do not synthesize a
     `ProductIR` or shared knowledge bundle unless the request flows through a
     semantic compile/build path.
   - This is acceptable for direct-existing execution, but it remains a gap if
     we want every request to carry semantic memory uniformly.

5. UI consumption
   - The trace/task surfaces in Trellis now emit richer shared-knowledge
     metadata, but any UI work to render those new fields still lives outside
     this repo.

## Suggested next follow-up

If we want to push this further, the next clean steps are:

1. fold `analytics_cookbook` into the knowledge store or a parallel analytics
   retrieval API using the same payload pattern
2. move `test_resolution` off the old experience file path onto the same shared
   lesson/promoted-trace substrate
3. decide whether direct `Session` / `Pipeline` requests should infer a minimal
   semantic request summary even when no build is required
