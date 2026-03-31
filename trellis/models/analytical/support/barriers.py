"""Barrier analytical support primitives.

STATUS: Route-local — extraction intentionally deferred.

The barrier monitoring pattern (survival_probability, rebate_value) is
currently implemented inline in ``trellis.models.analytical.barrier``.
This module is a placeholder for shared barrier support functions that
will be extracted once a **second consumer** proves stable reuse of the
same primitives (e.g. a discrete monitoring approximation or a PDE
boundary kernel that references the same survival probability formula).

Design rule (from ``docs/quant/basis_claim_patterns.md``):
  Barrier primitives are *route-local* until two independent routes share
  the same formula verbatim.  Extraction before that point creates
  premature abstraction that couples routes unnecessarily.

When extraction is ready:
  1. Move ``rebate_raw`` and ``barrier_image_raw`` into this module.
  2. Add a ``ResolvedBarrierSupport`` dataclass here alongside the
     existing ``ResolvedBarrierInputs`` in ``barrier.py``.
  3. Update both consumers to import from this module.
  4. Add a test in ``tests/test_models/test_analytical_support.py`` that
     verifies gradient flow through the extracted helpers.

See also: ``trellis/models/analytical/barrier.py`` for the current
implementation and ``docs/quant/basis_claim_patterns.md`` §Barrier
Monitoring for the extraction policy.
"""
