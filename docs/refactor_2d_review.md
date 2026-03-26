# Tranche 2D Review

This note records the review pass for Tranche 2D before adding grader tests or
reorganizing the remaining evaluation surface.

## Scope

Tranche 2D is the eval/grader tranche.

The goals are:

- define deterministic grading contracts around the Tranche 2B harness
- wire those graders into the existing test/task infrastructure
- clean up ambiguous test naming where the current suite hides intent

## Existing Evaluation Surfaces

### Agent/harness validation

Current coverage already exists for:

- import registry lookups
- codegen guardrails
- build-loop retries
- planner/quant behavior
- knowledge retrieval and formatting

This makes Tranche 2D an incremental wiring task, not a greenfield framework.

### Task and verification tests

The repo already contains:

- `tests/test_agent/*`
- `tests/test_tasks/*`
- `tests/test_verification/*`
- `tests/test_crossval/*`

These should remain the primary evidence sources. 2D should not create a second
parallel validation world.

## `test_v2_api.py` Review

`tests/test_v2_api.py` is not a coherent concept any more.

It currently bundles:

- `Book`
- `BookResult`
- `Session`
- `Pipeline`
- `Resolver`
- `Samples`

Most of these already have better homes:

- `tests/test_book.py`
- `tests/test_session.py`
- `tests/test_pipeline.py`
- `tests/test_data/test_resolver.py`
- `tests/test_samples.py`

So the right 2D action is:

- decompose `test_v2_api.py` into those descriptive suites
- preserve any unique assertions by moving them into the corresponding file
- remove `test_v2_api.py` only after the moved tests are covered and green

## Planned `test_v2_api.py` Mapping

### Move into `tests/test_book.py`

- Book construction from dict/list
- `getitem`, iteration, empty book
- `from_dataframe`
- `BookResult` totals and serialization

### Move into `tests/test_session.py`

- session construction
- pricing with/without greeks
- `with_curve_shift`, `with_tenor_bumps`, `with_curve`
- `spread_to_curve`
- `risk_report`
- `agent_enabled`
- mock data source

### Move into `tests/test_pipeline.py`

- basic run
- scenarios
- compute subsets
- missing instruments
- CSV output

### Move into `tests/test_data/test_resolver.py`

- mock/latest/string-date/unknown-source coverage if any unique assertions remain

### Move into `tests/test_samples.py`

- `sample_session`
- `sample_book`
- `quickstart`

## First Deterministic Grader Set

The first 2D graders should be small and fully deterministic.

### G1. import_correctness

Evidence:

- generated code
- approved generation plan
- import registry

Passes only if:

- all `trellis.*` imports are real
- imported symbols are exported by the referenced module
- imported modules are approved by the generation plan
- wildcard imports are absent

### G2. inspection_evidence_present

Evidence:

- generation plan

Passes only if:

- inspected modules are present
- approved modules are not empty

### G3. reuse_existing_symbols

Evidence:

- generation plan
- import registry
- generated code

Initial deterministic version should stay narrow:

- fail if code uses non-registry `trellis.*` symbols when equivalent approved
  registry-backed symbols were already available

### G4. test_selection

Evidence:

- generation plan

Passes only if:

- family-level expected test targets are present
- instrument-specific targets are present when applicable

### G5. unsupported_claims

Evidence:

- canonical method names
- canonical cookbooks/requirements
- generated summaries or eval metadata

The first version should be narrow and static:

- fail when an eval artifact claims a method family or capability that Trellis
  does not list in canonical knowledge

## Planned 2D.2 Tests

1. grader unit tests for the five graders above
2. characterization tests for the `test_v2_api.py` split:
   - prove the moved assertions live in descriptive files
   - then delete the old catch-all file

## Non-goals for 2D

- no probabilistic or LLM-based grading
- no new standalone eval framework
- no pricing-engine behavior changes
