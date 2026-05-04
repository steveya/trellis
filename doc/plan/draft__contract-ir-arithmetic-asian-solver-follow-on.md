# Contract IR Arithmetic Asian Solver Follow-on

## Status

Completed for the bounded Phase 3 closure target. Bounded structural
analytical and Monte Carlo lanes are admitted; broader family generalization
remains a separate follow-on.

## Linked Linear

- QUA-905 — Phase 3 umbrella (structural solver compiler)
- QUA-906 — Phase 4 umbrella (route retirement / selector flip)
- QUA-996 — bounded arithmetic-Asian structural Monte Carlo lane
- QUA-997 — Asian compatibility shim cleanup
- QUA-998 — bounded analytical arithmetic-Asian helper

## Purpose

Record the bounded arithmetic-Asian closeout explicitly instead of letting the
family disappear into a generic statement that "ContractIR can already
represent it."

Phase 2 established representation and bounded decomposition for arithmetic
Asian payoffs. The checked repository now admits a bounded analytical
approximation for call/put European schedule-based equity-diffusion requests
plus the earlier bounded Monte Carlo proving lane. Broad family retirement is
still governed and does not imply universal arithmetic-Asian closure.

## Current Evidence

The checked parity ledger now records arithmetic Asians as:

- `source = request_decomposition`
- `preferred_method = analytical` -> `shadow_status = bound`
- `preferred_method = monte_carlo` -> `shadow_status = bound`

See:

- `docs/benchmarks/contract_ir_solver_parity.json`
- `docs/benchmarks/contract_ir_solver_parity.md`

That is the correct current behavior. It proves the family is representable,
admits bounded structural analytical and Monte Carlo lanes, and is no longer a
Phase 3 blocker for the admitted cohort.

## What Landed

1. A checked arithmetic-Asian solver surface now exists under
   `trellis/models/asian_option.py`.
   - bounded analytical approximation via discrete moment matching
   - bounded Monte Carlo declaration with explicit seed discipline
2. The declaration domain is structural, not product-keyed.
   It binds from:
   - `ArithmeticMean(Spot(u), schedule)`
   - the generic term environment
   - valuation context and market state
3. The parity policy is family-appropriate.
   - analytical approximation -> deterministic parity against the same checked
     helper call surface
   - Monte Carlo -> explicit seed discipline
4. Request-level observability is preserved.
   The family now reports `shadow_status = bound` in the request-level
   `contract_ir_compiler` summary for the admitted analytical and Monte Carlo
   slices.

## Remaining Follow-ons

- widen the admitted family beyond the current European schedule-based
  equity-diffusion cohort
- decide whether Monte Carlo put parity should be admitted structurally in the
  same bounded family
- only promote broader family retirement claims when the widened cohort is
  checked and benchmarked

## Non-goals

- Pretending a route-local Asian implementation counts as structural lowering
- Treating the bounded analytical approximation as universal arithmetic-Asian
  support
- Promoting broader family retirement without widening the admitted cohort
