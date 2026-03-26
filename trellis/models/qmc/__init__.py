"""Quasi-Monte Carlo helpers layered on top of Monte Carlo internals.

This package is the canonical home for low-discrepancy sampling helpers and
path constructions that accelerate Monte Carlo estimators.

What belongs here:
- low-discrepancy drivers such as Sobol-based normal variates
- path constructions such as Brownian bridge

What does not belong here:
- generic Monte Carlo estimators
- process discretization schemes
- exercise logic such as Longstaff-Schwartz

Those components remain in :mod:`trellis.models.monte_carlo`.
"""

from trellis.models.monte_carlo.brownian_bridge import brownian_bridge
from trellis.models.monte_carlo.variance_reduction import sobol_normals

__all__ = [
    "sobol_normals",
    "brownian_bridge",
]
