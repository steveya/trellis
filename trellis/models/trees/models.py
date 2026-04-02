"""One-factor lattice model specifications.

A model spec defines:
1. **displacement_fn**: How latent node coordinates spread at each node
2. **probability_fn**: Transition probabilities (or a compatibility fallback)
3. **rate_fn**: How latent level + displacement map into node state
4. **state_metric_fn**: Which latent metric mean reversion acts on
5. **vol_type**: Normal (additive) vs. lognormal (multiplicative)

Different one-factor models become different parameterizations of the same
recombining lattice:
- Hull-White (normal):        x(m,j) = (2j - m) * σ√Δt
- Black-Derman-Toy (lognormal latent state): x(m,j) = (2j - m) * σ√Δt
- Black-Karasinski (lognormal): same as BDT with mean reversion in log-space
- Ho-Lee (normal, no MR):     x(m,j) = (2j - m) * σ√Δt, p = 0.5

These specs are the transitional bridge toward the broader lattice algebra:
topology, model kernel, contract overlay, and rollback logic should remain
separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as raw_np


@dataclass(frozen=True)
class TreeModel:
    """Specification for a one-factor lattice model.

    Parameters
    ----------
    name : str
        Model identifier (e.g., "hull_white", "bdt", "black_karasinski").
    displacement_fn : callable(step, node, dr) -> float
        Returns the displacement x(m,j) at node (step=m, node=j).
        The short rate is r(m,j) = phi(m) + x(m,j) for normal models,
        or r(m,j) = phi(m) * x(m,j) for lognormal models.
    probability_fn : callable(lattice, step, node, phis, a, dr) -> list[float]
        Returns transition probabilities [p_down, p_up] at (step, node).
        `phis` is the calibrated drift array, `a` is mean reversion.
    rate_fn : callable(phi, displacement) -> float
        Combines the calibrated drift and displacement into the short rate.
        Normal: r = phi + x.  Lognormal: r = phi * exp(x) or r = exp(phi + x).
    discount_fn : callable(rate, dt) -> float
        One-step discount factor from the rate. Usually exp(-r * dt).
    state_metric_fn : callable(state) -> float
        Latent state metric used for probability drift / mean reversion.
        Normal short-rate models use the observed state directly.
        Lognormal short-rate models use ``log(rate)``.
    vol_type : str
        "normal" or "lognormal" — affects how vol is interpreted.
    description : str
        Human-readable description for agent guidance.
    """
    name: str
    displacement_fn: Callable
    probability_fn: Callable
    rate_fn: Callable
    discount_fn: Callable
    vol_type: str
    description: str
    state_metric_fn: Callable[[float], float]
    supported_branchings: tuple[int, ...] = (2, 3)
    factor_family: str = "short_rate"

    def as_lattice_model_spec(self):
        """Return the generalized lattice-model specification for this tree model."""
        from trellis.models.trees.algebra import _lattice_model_from_tree_model

        return _lattice_model_from_tree_model(self)


# ---------------------------------------------------------------------------
# Displacement functions
# ---------------------------------------------------------------------------

def hw_displacement(step: int, node: int, dr: float) -> float:
    """Hull-White normal displacement: x(m,j) = (2j - m) * dr."""
    return (2 * node - step) * dr


def bdt_displacement(step: int, node: int, dr: float) -> float:
    """Black-Derman-Toy lognormal displacement: x(m,j) = (2j - m) * dr.

    Same additive structure in LOG-space. The rate_fn exponentiates.
    """
    return (2 * node - step) * dr


def ho_lee_displacement(step: int, node: int, dr: float) -> float:
    """Ho-Lee normal displacement (same as HW but no mean reversion)."""
    return (2 * node - step) * dr


# ---------------------------------------------------------------------------
# Probability functions
# ---------------------------------------------------------------------------

def identity_state_metric(state: float) -> float:
    """Latent metric for additive models."""
    return float(state)


def log_rate_state_metric(state: float) -> float:
    """Latent metric for lognormal short-rate models."""
    return float(raw_np.log(max(float(state), 1e-10)))


def binomial_mean_reversion_probabilities_from_metric(
    current_metric: float,
    target_metric: float,
    *,
    dt: float,
    a: float,
    dr: float,
) -> list[float]:
    """Return binomial transition probabilities from a latent drift metric."""
    drift = a * (target_metric - current_metric) * dt
    p_up = 0.5 + drift / (2 * dr) if dr > 0 else 0.5
    p_up = max(0.01, min(0.99, p_up))
    return [1 - p_up, p_up]


def trinomial_mean_reversion_probabilities_from_metric(
    current_metric: float,
    target_metric: float,
    *,
    dt: float,
    a: float,
    dr: float,
) -> list[float]:
    """Return trinomial transition probabilities from a latent drift metric."""
    drift = a * (target_metric - current_metric) * dt
    half_drift = drift / (2 * dr) if dr > 0 else 0.0
    p_up = max(0.01, min(0.98, 1.0 / 6 + half_drift))
    p_down = max(0.01, min(0.98, 1.0 / 6 - half_drift))
    p_mid = max(0.01, 1.0 - p_up - p_down)
    total = p_up + p_mid + p_down
    return [p_down / total, p_mid / total, p_up / total]


def equal_probabilities(lattice, step, node, phis, a, dr):
    """Equal probabilities: p_up = p_down = 0.5. No mean reversion."""
    return [0.5, 0.5]


def hw_mean_reversion_probabilities(lattice, step, node, phis, a, dr):
    """Hull-White mean-reversion adjusted probabilities.

    p_up = 0.5 + a*(phi(m+1) - r(m,j))*dt / (2*dr)
    Clamped to [0.01, 0.99] for stability.
    """
    n_steps = lattice.n_steps
    target = phis[min(step + 1, n_steps)]
    dt = lattice.dt
    current_metric = identity_state_metric(lattice.get_state(step, node))
    return binomial_mean_reversion_probabilities_from_metric(
        current_metric,
        float(target),
        dt=dt,
        a=a,
        dr=dr,
    )


def bdt_mean_reversion_probabilities(lattice, step, node, phis, a, dr):
    """BDT mean-reversion in log-space.

    Mean reversion acts on log(r), not r directly.
    p_up = 0.5 + a*(log_target - log_r)*dt / (2*dr)
    """
    n_steps = lattice.n_steps
    target = phis[min(step + 1, n_steps)]
    dt = lattice.dt
    current_metric = log_rate_state_metric(lattice.get_state(step, node))
    return binomial_mean_reversion_probabilities_from_metric(
        current_metric,
        float(target),
        dt=dt,
        a=a,
        dr=dr,
    )


# ---------------------------------------------------------------------------
# Rate functions (phi + displacement → rate)
# ---------------------------------------------------------------------------

def normal_rate(phi: float, displacement: float) -> float:
    """Normal model: r = phi + x. Can go negative."""
    return phi + displacement


def lognormal_rate(phi: float, displacement: float) -> float:
    """Lognormal model: r = exp(phi + x). Always positive."""
    return raw_np.exp(phi + displacement)


def shifted_lognormal_rate(phi: float, displacement: float, shift: float = 0.0) -> float:
    """Shifted lognormal: r = exp(phi + x) + shift. Allows mild negativity."""
    return raw_np.exp(phi + displacement) + shift


# ---------------------------------------------------------------------------
# Standard discount function
# ---------------------------------------------------------------------------

def standard_discount(rate: float, dt: float) -> float:
    """Return the one-step discount factor ``exp(-rate * dt)``."""
    return raw_np.exp(-rate * dt)


# ---------------------------------------------------------------------------
# Pre-built model specifications
# ---------------------------------------------------------------------------

HULL_WHITE = TreeModel(
    name="hull_white",
    displacement_fn=hw_displacement,
    probability_fn=hw_mean_reversion_probabilities,
    rate_fn=normal_rate,
    discount_fn=standard_discount,
    vol_type="normal",
    description=(
        "Hull-White one-factor normal model. Rates can go negative. "
        "Best for: rate derivatives where negative rates are plausible. "
        "Vol input: absolute rate vol (sigma_HW), NOT Black vol. "
        "Convert: sigma_HW = sigma_Black * forward_rate."
    ),
    state_metric_fn=identity_state_metric,
)

BLACK_DERMAN_TOY = TreeModel(
    name="bdt",
    displacement_fn=bdt_displacement,
    probability_fn=bdt_mean_reversion_probabilities,
    rate_fn=lognormal_rate,
    discount_fn=standard_discount,
    vol_type="lognormal",
    description=(
        "Black-Derman-Toy lognormal model. Rates always positive. "
        "Best for: markets where negative rates are impossible. "
        "Vol input: yield volatility (proportional), NOT basis-point vol. "
        "The model assumes rates are lognormally distributed."
    ),
    state_metric_fn=log_rate_state_metric,
)

BLACK_KARASINSKI = TreeModel(
    name="black_karasinski",
    displacement_fn=bdt_displacement,  # same displacement as BDT
    probability_fn=bdt_mean_reversion_probabilities,
    rate_fn=lognormal_rate,
    discount_fn=standard_discount,
    vol_type="lognormal",
    description=(
        "Black-Karasinski model. Lognormal with explicit mean reversion. "
        "Differs from BDT in that mean reversion speed is a free parameter "
        "(BDT calibrates it implicitly from the vol term structure)."
    ),
    state_metric_fn=log_rate_state_metric,
)

HO_LEE = TreeModel(
    name="ho_lee",
    displacement_fn=ho_lee_displacement,
    probability_fn=equal_probabilities,
    rate_fn=normal_rate,
    discount_fn=standard_discount,
    vol_type="normal",
    description=(
        "Ho-Lee model. Normal, no mean reversion (a=0). "
        "Simplest arbitrage-free model. Equal probabilities at all nodes. "
        "Best for: short-dated instruments where mean reversion doesn't matter."
    ),
    state_metric_fn=identity_state_metric,
)

# Model registry — the agent chooses from these
MODEL_REGISTRY: dict[str, TreeModel] = {
    "hull_white": HULL_WHITE,
    "bdt": BLACK_DERMAN_TOY,
    "black_karasinski": BLACK_KARASINSKI,
    "ho_lee": HO_LEE,
}
