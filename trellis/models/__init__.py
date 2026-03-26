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

from . import (
    analytical,
    calibration,
    cashflow_engine,
    copulas,
    monte_carlo,
    pde,
    processes,
    qmc,
    transforms,
    trees,
)

__all__ = [
    "black76_call",
    "black76_put",
    "garman_kohlhagen_call",
    "garman_kohlhagen_put",
    "FlatVol",
    "GridVolSurface",
    "VolSurface",
    "analytical",
    "trees",
    "monte_carlo",
    "qmc",
    "pde",
    "transforms",
    "processes",
    "copulas",
    "calibration",
    "cashflow_engine",
]
