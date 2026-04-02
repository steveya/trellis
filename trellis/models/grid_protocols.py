"""Shared grid/exercise protocols for lattice and PDE backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as raw_np

from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import BlackScholesOperator
from trellis.models.pde.theta_method import theta_method_1d
from trellis.models.trees.algebra import (
    LatticeContractSpec,
    LatticeControlSpec,
    LatticeLinearClaimSpec,
    price_on_lattice,
)


class SpatialGrid(Protocol):
    """Protocol shared by discrete lattice and PDE grids."""

    def time(self, step: int) -> float:
        """Return the model time at ``step``."""

    def node_count(self, step: int) -> int:
        """Return the number of active nodes at ``step``."""


@dataclass(frozen=True)
class AmericanPutExerciseBoundary:
    """Shared obstacle contract for an American put exercise region."""

    strike: float
    objective: str = "holder_max"

    def lattice_value(self, step: int, node: int, lattice, obs) -> float:
        del step, node, lattice
        if "spot" in obs:
            spot = float(obs["spot"])
        else:
            spot = float(obs["state"])
        return max(float(self.strike) - spot, 0.0)

    def pde_values(self, grid: Grid) -> raw_np.ndarray:
        return raw_np.maximum(float(self.strike) - raw_np.asarray(grid.x, dtype=float), 0.0)


@dataclass(frozen=True)
class LatticeSpatialGrid:
    """Adapter exposing a built lattice through the shared grid protocol."""

    lattice: object

    @classmethod
    def from_lattice(cls, lattice) -> "LatticeSpatialGrid":
        return cls(lattice=lattice)

    def time(self, step: int) -> float:
        return float(step) * float(self.lattice.dt)

    def node_count(self, step: int) -> int:
        return int(self.lattice.n_nodes(step))


@dataclass(frozen=True)
class PDEUniformGrid:
    """Adapter for 1D finite-difference grids."""

    spot: float
    maturity: float
    n_x: int
    n_t: int
    s_max_multiplier: float = 4.0

    def build(self) -> Grid:
        s_max = max(self.s_max_multiplier * self.spot, 2.0 * self.spot)
        return Grid(x_min=0.0, x_max=s_max, n_x=self.n_x, T=self.maturity, n_t=self.n_t)

    def time(self, step: int) -> float:
        return float(step) * (float(self.maturity) / max(int(self.n_t), 1))

    def node_count(self, step: int) -> int:
        del step
        return int(self.n_x)


class LatticeBackwardInductionEngine:
    """Shared lattice-side engine built from the generalized contract surface."""

    def price_vanilla_put(
        self,
        grid: LatticeSpatialGrid,
        *,
        strike: float,
        boundary: AmericanPutExerciseBoundary,
    ) -> float:
        claim = LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice, obs: max(float(strike) - float(obs["spot"]), 0.0)
        )
        control = LatticeControlSpec(
            objective=boundary.objective,
            exercise_value_fn=boundary.lattice_value,
        )
        return float(price_on_lattice(grid.lattice, LatticeContractSpec(claim=claim, control=control)))


class PDEThetaEngine:
    """Shared PDE-side engine using the theta-method obstacle solver."""

    def price_vanilla_put(
        self,
        grid_spec: PDEUniformGrid,
        *,
        strike: float,
        rate: float,
        sigma_fn,
        boundary: AmericanPutExerciseBoundary,
    ) -> float:
        grid = grid_spec.build()
        operator = BlackScholesOperator(sigma_fn=sigma_fn, r_fn=lambda t: float(rate))
        terminal = raw_np.maximum(float(strike) - grid.x, 0.0)
        values = theta_method_1d(
            grid,
            operator,
            terminal,
            theta=1.0,
            lower_bc_fn=lambda t: float(strike) * raw_np.exp(-float(rate) * (float(grid.T) - t)),
            upper_bc_fn=lambda t: 0.0,
            exercise_values=boundary.pde_values(grid),
        )
        idx = raw_np.searchsorted(grid.x, float(grid_spec.spot))
        idx = max(1, min(int(idx), len(grid.x) - 1))
        weight = (float(grid_spec.spot) - grid.x[idx - 1]) / (grid.x[idx] - grid.x[idx - 1])
        return float(values[idx - 1] * (1.0 - weight) + values[idx] * weight)


__all__ = [
    "AmericanPutExerciseBoundary",
    "LatticeBackwardInductionEngine",
    "LatticeSpatialGrid",
    "PDEThetaEngine",
    "PDEUniformGrid",
    "SpatialGrid",
]
