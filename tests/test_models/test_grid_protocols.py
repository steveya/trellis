from __future__ import annotations

import pytest

from trellis.models.equity_option_pde import price_vanilla_equity_option_pde
from trellis.models.trees.algebra import (
    LATTICE_MODEL_REGISTRY,
    LOG_SPOT_MESH,
    BINOMIAL_1F_TOPOLOGY,
    build_lattice,
)


def test_shared_grid_protocols_price_american_put_consistently():
    from datetime import date, timedelta

    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.grid_protocols import (
        AmericanPutExerciseBoundary,
        LatticeBackwardInductionEngine,
        LatticeSpatialGrid,
        PDEThetaEngine,
        PDEUniformGrid,
    )

    settlement = date(2026, 4, 2)
    expiry = settlement + timedelta(days=365)
    strike = 100.0
    spot = 100.0
    curve = YieldCurve.flat(0.03)

    lattice = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        LOG_SPOT_MESH,
        LATTICE_MODEL_REGISTRY["crr"],
        calibration_target=None,
        spot=spot,
        rate=0.03,
        sigma=0.20,
        maturity=1.0,
        n_steps=300,
    )
    boundary = AmericanPutExerciseBoundary(strike=strike)

    lattice_grid = LatticeSpatialGrid.from_lattice(lattice)
    lattice_price = LatticeBackwardInductionEngine().price_vanilla_put(
        lattice_grid,
        strike=strike,
        boundary=boundary,
    )

    class _FlatVol:
        def black_vol(self, t: float, strike_: float) -> float:
            del t, strike_
            return 0.20

    class _MarketState:
        def __init__(self):
            self.as_of = settlement
            self.settlement = settlement
            self.discount = curve
            self.vol_surface = _FlatVol()

    class _Spec:
        def __init__(self):
            self.notional = 1.0
            self.spot = spot
            self.strike = strike
            self.expiry_date = expiry
            self.option_type = "put"

    pde_grid = PDEUniformGrid(spot=spot, maturity=1.0, n_x=401, n_t=401)
    pde_price = PDEThetaEngine().price_vanilla_put(
        pde_grid,
        strike=strike,
        rate=0.03,
        sigma_fn=lambda s, t: 0.20,
        boundary=boundary,
    )

    # The shipped helper prices the European PDE, so the American obstacle should dominate it.
    helper_price = price_vanilla_equity_option_pde(_MarketState(), _Spec(), theta=1.0, n_x=401, n_t=401)

    assert pde_price >= helper_price
    assert lattice_price == pytest.approx(pde_price, rel=0.05)
