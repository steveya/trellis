"""Shared test helpers for the unified lattice-construction surface."""

from __future__ import annotations

from trellis.curves.yield_curve import YieldCurve
from trellis.models.trees.algebra import (
    BINOMIAL_1F_TOPOLOGY,
    LOG_SPOT_MESH,
    NO_CALIBRATION_TARGET,
    TERM_STRUCTURE_TARGET,
    UNIFORM_ADDITIVE_MESH,
    LATTICE_MODEL_REGISTRY,
)
from trellis.models.trees.lattice import build_lattice
from trellis.models.trees.models import MODEL_REGISTRY


def build_equity_lattice(
    spot: float,
    rate: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    *,
    model: str = "crr",
):
    """Build a one-factor equity lattice through the current unified API."""
    model_name = str(model).strip().lower()
    return build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        LOG_SPOT_MESH,
        LATTICE_MODEL_REGISTRY[model_name],
        calibration_target=NO_CALIBRATION_TARGET(),
        spot=spot,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        n_steps=n_steps,
    )


def build_short_rate_lattice(
    r0: float,
    sigma: float,
    a: float,
    maturity: float,
    n_steps: int,
    *,
    discount_curve=None,
    model: str = "hull_white",
):
    """Build a calibrated one-factor short-rate lattice through the unified API."""
    curve = discount_curve if discount_curve is not None else YieldCurve.flat(r0)
    model_name = str(model).strip().lower()
    return build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        UNIFORM_ADDITIVE_MESH,
        MODEL_REGISTRY[model_name].as_lattice_model_spec(),
        calibration_target=TERM_STRUCTURE_TARGET(curve),
        r0=r0,
        sigma=sigma,
        a=a,
        T=maturity,
        n_steps=n_steps,
    )
