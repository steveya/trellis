# Contract IR Arithmetic Asian Solver Follow-on

## Status

Draft. Explicit Phase 3 blocker note for the arithmetic-Asian family.

## Linked Linear

- QUA-905 — Phase 3 umbrella (structural solver compiler)
- QUA-906 — Phase 4 umbrella (route retirement / selector flip)

## Purpose

Keep arithmetic Asians visible as a governed blocker instead of letting them
disappear behind the generic statement that "ContractIR can already represent
them."

Phase 2 established representation and bounded decomposition for arithmetic
Asian payoffs. Phase 3 intentionally does **not** migrate them because the
current checked repository still lacks an admitted solver declaration surface
that satisfies the structural compiler contract.

## Current Evidence

The checked parity ledger records arithmetic Asians as:

- `source = request_decomposition`
- `shadow_status = no_match`
- `shadow_error.error_type = ContractIRSolverNoMatchError`

See:

- `docs/benchmarks/contract_ir_solver_parity.json`
- `docs/benchmarks/contract_ir_solver_parity.md`

That is the correct current behavior. It proves the family is representable
without over-claiming route-free lowering support.

## Why Phase 3 Blocks Here

Arithmetic Asians fail the current lowering-closure gate:

- there is no admitted analytical arithmetic-Asian helper comparable to the
  current vanilla / digital / swaption / basket / variance first wave
- there is therefore no bounded structural declaration with checked parity
  evidence to bind
- promoting the family anyway would force Phase 3 to hide a solver gap behind
  route-local product authority, which is exactly what this program is trying
  to remove

So the correct Phase 3 contract is:

- representation closed
- bounded decomposition closed
- lowering blocked

## Minimum Follow-on Surface

The next solver slice should land only when all of the following are explicit:

1. A checked arithmetic-Asian solver surface exists.
   Acceptable first target:
   - an admitted analytical approximation helper, or
   - a deliberately scoped Monte Carlo declaration with its own parity and RNG
     policy
2. The declaration domain is structural, not product-keyed.
   It must bind from:
   - `ArithmeticMean(Spot(u), schedule)`
   - the generic term environment
   - valuation context and market state
3. The parity policy is family-appropriate.
   - deterministic helper -> tight floating-point tolerance
   - Monte Carlo -> explicit confidence-interval / seed discipline
4. Request-level observability is preserved.
   The family must move from `shadow_status = no_match` to `shadow_status = bound`
   in the request-level `contract_ir_compiler` summary before any Phase 4
   selector flip is allowed.

## Non-goals

- Pretending a route-local Asian implementation counts as structural lowering
- Widening the current first-wave declarations to include arithmetic means
- Promoting Phase 4 retirement for Asians before the parity ledger turns green
