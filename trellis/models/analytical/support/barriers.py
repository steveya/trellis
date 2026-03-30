"""Barrier analytical support primitives.

STATUS: Route-local.

The barrier monitoring pattern (survival_probability, rebate_value) is
currently implemented inline in ``trellis.models.analytical.barrier``.
This module is a placeholder for shared barrier support functions that
will be extracted once a second consumer proves reuse of the same
primitives.

See QUA-289 for the follow-on work to extract and validate shared
barrier helpers.
"""
