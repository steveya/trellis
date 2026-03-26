# Agent Evals

This note documents the first deterministic eval layer added in Tranche 2D.

## Purpose

These eval helpers are not a separate framework. They are a thin, deterministic
layer over the Tranche 2B guardrails.

They exist to answer narrow questions such as:

- did the generated code use only real, approved Trellis imports?
- did the generated code satisfy Trellis engine/payoff semantic contracts?
- did the generation plan inspect the repo before codegen?
- did the generation plan include the expected validation targets?
- did an eval artifact claim unsupported method families?

## Current Graders

### `import_correctness`

Uses:

- generated source code
- generation plan
- import registry

Passes only if:

- every `trellis.*` import resolves
- imported symbols are exported by the referenced module
- imported modules are approved by the generation plan
- wildcard imports are absent

### `inspection_evidence_present`

Uses:

- generation plan

Passes only if:

- `inspected_modules` is non-empty
- `approved_modules` is non-empty

### `semantic_validity`

Uses:

- generated source code
- optional `ProductIR`
- generation plan

Passes only if:

- Monte Carlo code does not invent unsupported engine modes
- early-exercise Monte Carlo code uses a real control primitive such as
  `longstaff_schwartz`
- regression bases are imported from the correct module
- payoff callbacks passed to `MonteCarloEngine.price(...)` do not appear to
  return path-matrix-shaped values
- transform characteristic functions avoid scalar `math`/`cmath` usage
- extracted engine families are compatible with the product IR when available

### `test_selection`

Uses:

- generation plan

Passes only if:

- the expected family/instrument validation targets are present in
  `proposed_tests`

### `unsupported_claims`

Uses:

- canonical method-family names

Fails if an eval artifact claims unsupported method families or capability
labels outside canonical knowledge.

## Test-Suite Cleanup

As part of Tranche 2D, `tests/test_v2_api.py` was removed.

It had become a catch-all file whose contents already belonged in more
descriptive suites:

- `tests/test_book.py`
- `tests/test_session.py`
- `tests/test_pipeline.py`
- `tests/test_data/test_resolver.py`
- `tests/test_samples.py`

Only necessary coverage was kept. Pure duplicates were not copied forward.
