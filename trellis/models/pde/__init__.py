"""PDE solvers for option pricing."""

from trellis.models.pde.event_aware import (
    EventAwarePDEBoundarySpec,
    EventAwarePDEEventBucket,
    EventAwarePDEGridSpec,
    EventAwarePDEOperatorSpec,
    EventAwarePDEProblem,
    EventAwarePDEProblemSpec,
    EventAwarePDETransform,
    apply_event_bucket,
    apply_event_transform,
    build_event_aware_pde_operator,
    build_event_aware_pde_problem,
    interpolate_pde_values,
    solve_event_aware_pde,
)
from trellis.models.pde.theta_method import theta_method_1d
from trellis.models.pde.psor import psor_1d
from trellis.models.pde.grid import Grid
from trellis.models.pde.rate_operator import HullWhitePDEOperator
from trellis.models.pde.thomas import thomas_solve


def crank_nicolson_1d(*args, **kwargs):
    """Backward compat — delegates to theta_method_1d with theta=0.5."""
    kwargs.setdefault('theta', 0.5)
    return theta_method_1d(*args, **kwargs)


def implicit_fd_1d(*args, **kwargs):
    """Backward compat — delegates to theta_method_1d with theta=1.0."""
    kwargs.setdefault('theta', 1.0)
    return theta_method_1d(*args, **kwargs)
