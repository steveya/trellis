# Fresh-Generated FinancePy Pilot

This plan corrects the benchmark methodology for FinancePy parity work.

The benchmark goal is not "make the checked-in `_agent` adapter correct." The
benchmark goal is "prove that the current agent can assemble a fresh pricing
engine from Trellis primitives and match FinancePy."

## Objective

For a representative pilot subset of benchmark tasks:

- run the parity benchmark from fresh generated adapters only
- keep checked-in `trellis/instruments/_agent/` code out of the benchmark critical path
- separate benchmark success from adapter admission
- keep timestamped run history so repeated reruns remain comparable across code and knowledge revisions

## Why This Exists

The previous FinancePy parity wave surfaced real shared gaps in:

- CDS timing/convention handling
- benchmark market-state alignment
- output extraction and parity reporting

Those fixes were valuable, but the benchmark path still allowed methodology drift:
the checked-in `_agent` surface could influence or absorb benchmark fixes.

That is the wrong evaluation boundary. `_agent` is an admitted/generated artifact
surface, not the thing we should hand-tune and then claim the agent assembled.

## Pilot Scope

Use a bounded but representative subset of FinancePy parity tasks:

- `F001` equity vanilla Black-Scholes
- `F002` FX vanilla Garman-Kohlhagen
- `F003` USD cap strip Black
- `F007` single-name CDS analytical
- `F009` equity barrier Black-Scholes
- `F012` equity chooser Black-Scholes

These cover:

- plain analytical equity
- FX analytical
- schedule/rates
- credit conventions
- barrier/path-style contract structure
- analytical exotic assembly

## Required Behavior

### Benchmark execution

- The benchmark runner must materialize fresh generated adapters in an ephemeral workspace.
- The parity run must execute those fresh generated adapters, not the checked-in `_agent` module.
- The runner must persist enough metadata to prove which generated artifact was executed.

### Admission separation

- Passing the benchmark must not overwrite the checked-in `_agent` adapter.
- Promotion into `trellis/instruments/_agent/` must be a separate explicit admission step.
- Admission metadata should record which benchmark run and which generated artifact justified promotion.

### Guardrails

- Pilot benchmark tests should fail if the parity path imports the checked-in `_agent` adapter for the selected tasks.
- Promotion tooling should fail closed if the generated artifact does not match the validated benchmark record.

## Ticket Map

1. runner isolation
   Build an ephemeral generated-adapter execution path for the pilot subset.

2. benchmark-path decoupling
   Ensure the pilot parity path no longer relies on checked-in `_agent` adapters.

3. promotion workflow
   Create an explicit admission/promotion step from validated fresh artifacts into `_agent`.

4. pilot guardrails and validation
   Add contract tests and rerun the pilot subset against FinancePy with timestamped history.

## Success Criteria

- The selected pilot tasks price from fresh generated adapters only.
- The pilot subset passes FinancePy parity on the current benchmark tolerances.
- Each run record includes timestamps, code revision, knowledge revision, and generated-artifact provenance.
- Checked-in `_agent` updates happen only through the explicit promotion workflow, not during benchmark execution.
