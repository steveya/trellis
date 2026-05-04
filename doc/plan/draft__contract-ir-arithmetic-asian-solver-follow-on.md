# Contract IR Arithmetic Asian Solver Follow-on

## Status

Active. Bounded structural Monte Carlo lane admitted; analytical closure still blocked.

## Linked Linear

- QUA-905 — Phase 3 umbrella (structural solver compiler)
- QUA-906 — Phase 4 umbrella (route retirement / selector flip)
- QUA-996 — bounded arithmetic-Asian structural Monte Carlo lane

## Purpose

Keep arithmetic Asians visible as a governed blocker instead of letting them
disappear behind the generic statement that "ContractIR can already represent
them."

Phase 2 established representation and bounded decomposition for arithmetic
Asian payoffs. The current checked repository now admits one bounded structural
Monte Carlo call lane, but Phase 3 still does **not** retire the family
broadly because the analytical surface and broader family coverage remain open.

## Current Evidence

The checked parity ledger now records arithmetic Asians as:

- `source = request_decomposition`
- `preferred_method = analytical` -> `shadow_status = no_match`
- `preferred_method = monte_carlo` -> `shadow_status = bound`

See:

- `docs/benchmarks/contract_ir_solver_parity.json`
- `docs/benchmarks/contract_ir_solver_parity.md`

That is the correct current behavior. It proves the family is representable and
admits one governed structural lane without over-claiming analytical or
phase-4-ready coverage.

## Why Phase 3 Blocks Here

Arithmetic Asians still fail the current full lowering-closure gate:

- there is no admitted analytical arithmetic-Asian helper comparable to the
  current vanilla / digital / swaption / basket / variance first wave
- the admitted Monte Carlo lane is intentionally narrower than broad family
  retirement and does not erase the analytical blocker
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
   The first bounded target is now landed:
   - an admitted Monte Carlo declaration with explicit seed discipline and
     parity coverage
   Still open:
   - an admitted analytical approximation helper, or
   - a broader Monte Carlo family contract if retirement scope expands
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
