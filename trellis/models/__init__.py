"""Numerical method families and reusable pricing-model components.

`trellis.models` is the hub for method-family packages such as PDE, trees,
Monte Carlo, transforms, processes, copulas, calibration, and cashflow
engines. It also re-exports a few high-signal helpers that are widely used.
"""

from trellis.models.black import (
    black76_call,
    black76_put,
    garman_kohlhagen_call,
    garman_kohlhagen_put,
)
from trellis.models.vol_surface import FlatVol, GridVolSurface, VolSurface
from trellis.models.vol_surface_shocks import (
    VolSurfaceShockBucket,
    VolSurfaceShockSurface,
    VolSurfaceShockWarning,
    build_vol_surface_shock_surface,
)

from . import (
    analytical,
    bermudan_swaption_tree,
    calibration,
    cashflow_engine,
    callable_bond_tree,
    credit_default_swap,
    equity_option_pde,
    copulas,
    equity_option_tree,
    monte_carlo,
    pde,
    processes,
    qmc,
    rate_style_swaption,
    transforms,
    trees,
    vol_surface_shocks,
    zcb_option,
    zcb_option_tree,
)

__all__ = [
    "black76_call",
    "black76_put",
    "garman_kohlhagen_call",
    "garman_kohlhagen_put",
    "FlatVol",
    "GridVolSurface",
    "VolSurface",
    "VolSurfaceShockBucket",
    "VolSurfaceShockSurface",
    "VolSurfaceShockWarning",
    "build_vol_surface_shock_surface",
    "analytical",
    "bermudan_swaption_tree",
    "trees",
    "monte_carlo",
    "qmc",
    "pde",
    "transforms",
    "processes",
    "copulas",
    "calibration",
    "cashflow_engine",
    "rate_style_swaption",
    "callable_bond_tree",
    "credit_default_swap",
    "equity_option_pde",
    "equity_option_tree",
    "vol_surface_shocks",
    "zcb_option",
    "zcb_option_tree",
]
