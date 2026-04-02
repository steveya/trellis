Lattice Algebra Boundary ADR
============================

Status
------

Accepted on April 2, 2026.

Context
-------

Trellis now has a unified lattice substrate with explicit topology, mesh,
model, calibration, contract, and rollback layers. That substrate is broad
enough to support one-factor rate trees, equity trees, event overlays,
local-volatility lattices, and low-dimensional two-factor products.

Without a hard boundary, the platform will overpromise. Some families can be
represented cleanly as positive recombining pricing operators. Others cannot,
or can only do so after state growth that defeats the point of the algebra.

Decision
--------

The general lattice algebra is the default route only when all of the
following hold:

- finite horizon
- Markov after finite augmentation
- recombining state graph with polynomial node growth
- factor count less than or equal to 2
- single numeraire
- single-controller obstacle structure only

The platform should route away from the general lattice algebra when any of
the following are true:

- non-recombining or effectively non-recombining state growth
- rough-vol or other non-Markov state requirements
- HJM or LMM families without a checked Markov projection
- high-dimensional baskets and hybrids beyond two state factors
- multi-controller game structures such as callable-puttable-convertible logic

Implementation Notes
--------------------

The checked routing gate is ``lattice_algebra_eligible(...)`` in
``trellis.models.trees.algebra`` and is wired into route admissibility.

The current fast-path contract is intentionally narrower than the algebra
boundary:

- 1D linear claims and single-controller obstacles use the lattice fast path
- finite-state overlays and edge-aware contracts fall back to Python with a warning
- two-factor product lattices currently have a dedicated terminal-claim fast path

Consequences
------------

- The unified lattice surface is the canonical target for helper-backed rate
  and equity routes.
- Specialized trees remain legitimate first-class implementations when they do
  not satisfy the algebra boundary.
- Route selection can now reject ineligible products explicitly instead of
  silently generating a partial lattice implementation.
