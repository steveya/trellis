# Basis-Claim Patterns

Recurring patterns for constructing analytical pricing formulae from a
terminal basis evaluation, a claim adjustment, and risk-neutral discounting.

## Terminal Basis-Claim (Proven)

The canonical pattern for vanilla analytical pricing:

1. **Forward construction** -- compute the forward price from the spot and
   cost-of-carry over the life of the instrument.
2. **Basis evaluation** -- evaluate the terminal payoff distribution under
   the risk-neutral measure (e.g. Black-76 d1/d2 integrals for a call/put).
3. **Discounting** -- multiply the undiscounted expected payoff by the
   discount factor to the payment date.

This pattern is implemented in `trellis.models.black` and the analytical
support modules under `trellis.models.analytical.support`.

## Barrier Monitoring (Proven, Route-Local)

Proven in task T09 (barrier option analytical pricing).  The barrier
variant extends the terminal basis-claim pattern with a survival
probability and rebate term:

    PV = terminal_basis x survival_probability + rebate_value

- `terminal_basis` is the vanilla Black-76 value.
- `survival_probability` is the probability that the barrier is never
  breached under continuous or discrete monitoring.
- `rebate_value` is the discounted rebate paid on breach (zero for
  many contracts).

Currently route-local in `trellis.models.analytical.barrier`.  See
`trellis/models/analytical/support/barriers.py` for the placeholder
awaiting a second consumer before promotion to shared support.

## Path-Dependent (Future)

A prospective extension for path-dependent analytical approximations
(e.g. Asian options with moment-matching):

    PV = terminal_basis(path_statistic, adjusted_vol)

- `path_statistic` summarises the path (e.g. arithmetic average, geometric
  average, running max).
- `adjusted_vol` is a volatility that accounts for the averaging effect.

Not yet implemented.  Will require a second proving task before extraction.

## Extraction Policy

> Extract to shared support only after a second consumer proves reuse.

A pattern that appears in only one route stays route-local.  Promotion to
`trellis.models.analytical.support` happens when a second, independent
pricing route needs the same logic and the shared interface is validated
by both consumers.
