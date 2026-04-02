"""Generalized lattice algebra surface built on the shipped lattice numerics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import exp, log, sqrt
from typing import Any, Callable, Mapping, Protocol
import warnings

import numpy as raw_np

from trellis.models._numba import NUMBA_AVAILABLE
from trellis.models.trees.models import MODEL_REGISTRY as LEGACY_MODEL_REGISTRY


def _as_dict(mapping: Mapping[str, object] | None) -> dict[str, object]:
    """Return a shallow mutable dict for recipe parameters."""
    return dict(mapping or {})


def _node_values(count: int, generator) -> raw_np.ndarray:
    """Collect scalar node values with a compact NumPy conversion."""
    return raw_np.fromiter(generator, dtype=float, count=int(count))


def _branching_value(branching: int | tuple[int, ...]) -> int:
    """Return the scalar branching factor for 1D topologies."""
    if isinstance(branching, tuple):
        if len(branching) != 1:
            raise ValueError("Only one-factor branching is supported by the shipped lattice builder")
        return int(branching[0])
    return int(branching)


def _lookup_spec(spec_or_name, registry, kind: str):
    if isinstance(spec_or_name, str):
        try:
            return registry[spec_or_name]
        except KeyError as exc:
            raise KeyError(f"Unknown lattice {kind} {spec_or_name!r}") from exc
    return spec_or_name


def _binomial_node_count(step: int) -> int:
    return int(step) + 1


def _binomial_child_indices(step: int, node: int) -> tuple[int, int]:
    del step
    return int(node), int(node) + 1


def _binomial_parent_indices(step: int, node: int) -> tuple[int, ...]:
    if step <= 0:
        return ()
    if node <= 0:
        return (0,)
    if node >= step:
        return (step - 1,)
    return (node - 1, node)


def _trinomial_node_count(step: int) -> int:
    return 2 * int(step) + 1


def _trinomial_child_indices(step: int, node: int) -> tuple[int, int, int]:
    del step
    return int(node), int(node) + 1, int(node) + 2


def _trinomial_parent_indices(step: int, node: int) -> tuple[int, ...]:
    if step <= 0:
        return ()
    max_parent = 2 * (step - 1)
    parents = []
    for candidate in (node - 2, node - 1, node):
        if 0 <= candidate <= max_parent:
            parents.append(candidate)
    return tuple(parents)


def _product_binomial_2f_node_count(step: int) -> int:
    width = int(step) + 1
    return width * width


def _product_binomial_2f_child_indices(step: int, node: int) -> tuple[int, int, int, int]:
    width = int(step) + 1
    i, j = divmod(int(node), width)
    next_width = width + 1
    base = i * next_width + j
    return base, base + 1, base + next_width, base + next_width + 1


def _product_binomial_2f_parent_indices(step: int, node: int) -> tuple[int, ...]:
    if step <= 0:
        return ()
    width = int(step) + 1
    i, j = divmod(int(node), width)
    prev_width = width - 1
    parents: list[int] = []
    for di in (0, 1):
        for dj in (0, 1):
            parent_i = i - di
            parent_j = j - dj
            if 0 <= parent_i < prev_width and 0 <= parent_j < prev_width:
                parents.append(parent_i * prev_width + parent_j)
    return tuple(sorted(set(parents)))


def _uniform_additive_coordinate(step: int, node: int, params: Mapping[str, object]) -> float:
    dx = float(params.get("dx", params.get("dr", 0.0)))
    branching = int(params.get("branching", 2))
    if branching == 2:
        return (2.0 * float(node) - float(step)) * dx
    return (float(node) - float(step)) * dx


def _uniform_additive_step_size(step: int, params: Mapping[str, object]) -> float:
    del step
    return float(params.get("dx", params.get("dr", 0.0)))


def _log_spot_coordinate(step: int, node: int, params: Mapping[str, object]) -> float:
    spot = float(params.get("spot", params.get("S0", 1.0)))
    rate = float(params.get("rate", params.get("r", 0.0)))
    sigma = float(params.get("sigma", 0.0))
    maturity = float(params.get("maturity", params.get("T", 0.0)))
    n_steps = max(int(params.get("n_steps", 1)), 1)
    dt = maturity / n_steps
    model = str(params.get("model", "crr")).strip().lower()
    if model in {"jarrow_rudd", "jr"}:
        u = exp((rate - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt))
        d = exp((rate - 0.5 * sigma * sigma) * dt - sigma * sqrt(dt))
    else:
        u = exp(sigma * sqrt(dt))
        d = 1.0 / max(u, 1e-12)
    return log(max(spot, 1e-12)) + float(node) * log(max(u, 1e-12)) + float(step - node) * log(max(d, 1e-12))


def _log_spot_step_size(step: int, params: Mapping[str, object]) -> float:
    del step
    sigma = float(params.get("sigma", 0.0))
    maturity = float(params.get("maturity", params.get("T", 0.0)))
    n_steps = max(int(params.get("n_steps", 1)), 1)
    return sigma * sqrt(max(maturity / n_steps, 0.0))


def _product_log_spot_coordinate(step: int, node: int, params: Mapping[str, object]) -> tuple[float, float]:
    width = int(step) + 1
    i, j = divmod(int(node), width)
    spots = tuple(float(value) for value in params.get("spots", params.get("spot_pair", (1.0, 1.0))))
    sigmas = tuple(float(value) for value in params.get("sigmas", (0.0, 0.0)))
    maturity = float(params.get("maturity", params.get("T", 0.0)))
    n_steps = max(int(params.get("n_steps", 1)), 1)
    dt = maturity / n_steps
    log_s1 = log(max(spots[0], 1e-12)) + (2 * i - step) * sigmas[0] * sqrt(dt)
    log_s2 = log(max(spots[1], 1e-12)) + (2 * j - step) * sigmas[1] * sqrt(dt)
    return log_s1, log_s2


def _identity_metric(value):
    return value


def _default_short_rate_observable(state) -> dict[str, object]:
    rate = float(state)
    return {
        "rate": rate,
        "short_rate": rate,
        "value": rate,
        "state": rate,
        "latent_state": rate,
    }


def _default_equity_observable(state) -> dict[str, object]:
    spot = float(state)
    latent = log(max(spot, 1e-12))
    return {
        "spot": spot,
        "value": spot,
        "state": spot,
        "latent_state": latent,
    }


def _default_two_factor_equity_observable(state) -> dict[str, object]:
    spot_1 = float(state[0])
    spot_2 = float(state[1])
    return {
        "spot_1": spot_1,
        "spot_2": spot_2,
        "spots": (spot_1, spot_2),
        "basket": spot_1 + spot_2,
        "state": (spot_1, spot_2),
        "value": (spot_1, spot_2),
        "latent_state": (log(max(spot_1, 1e-12)), log(max(spot_2, 1e-12))),
    }


@dataclass(frozen=True)
class LatticeTopologySpec:
    """Pure lattice graph topology with no coordinate semantics."""

    name: str
    branching: int | tuple[int, ...]
    node_count_fn: Callable[[int], int]
    child_indices_fn: Callable[[int, int], tuple[int, ...]]
    factor_count: int = 1
    parent_indices_fn: Callable[[int, int], tuple[int, ...]] | None = None
    max_nodes_fn: Callable[[int], int] | None = None
    product_topology: bool = False

    def node_count(self, step: int) -> int:
        return int(self.node_count_fn(int(step)))

    def child_indices(self, step: int, node: int) -> tuple[int, ...]:
        return tuple(int(child) for child in self.child_indices_fn(int(step), int(node)))

    def parent_indices(self, step: int, node: int) -> tuple[int, ...]:
        if self.parent_indices_fn is None:
            return ()
        return tuple(int(parent) for parent in self.parent_indices_fn(int(step), int(node)))

    def max_nodes(self, n_steps: int) -> int:
        if self.max_nodes_fn is not None:
            return int(self.max_nodes_fn(int(n_steps)))
        return self.node_count(int(n_steps))


@dataclass(frozen=True)
class LatticeMeshSpec:
    """Numerical coordinate layer separated from topology."""

    name: str
    coordinate_fn: Callable[[int, int, Mapping[str, object]], object]
    step_size_fn: Callable[[int, Mapping[str, object]], object] | None = None
    metric_fn: Callable[[object], object] | None = None
    truncation_rule: Callable[..., object] | None = None
    transform_name: str | None = None


@dataclass(frozen=True)
class LatticeModelSpec:
    """Generalized model specification consumed by the lattice algebra."""

    name: str
    factor_family: str
    factor_count: int = 1
    state_space_type: str = "additive_normal"
    numeraire: str = "money_market"
    calibration_strategy: str = "none"
    supported_topologies: tuple[str, ...] = ()
    supported_branchings: tuple[int, ...] = (2, 3)
    observable_fn: Callable[[object], Mapping[str, object]] | None = None
    state_metric_fn: Callable[[object], object] | None = None
    pricing_operator_fn: Callable[..., object] | None = None
    probability_fn: Callable[..., object] | None = None
    discount_fn: Callable[..., object] | None = None
    supported_calibration_targets: tuple[str, ...] = ()
    admissibility_fn: Callable[..., object] | None = None
    legacy_tree_model: object | None = None


class CalibrationStrategy(Protocol):
    """Protocol for calibration/build strategies."""

    def calibrate(
        self,
        topology: LatticeTopologySpec,
        mesh: LatticeMeshSpec,
        model: LatticeModelSpec,
        target: object | None,
        **params,
    ) -> "CalibratedLatticeData":
        """Build and calibrate a lattice using one strategy."""


@dataclass(frozen=True)
class TermStructureTarget:
    discount_curve: object
    market_times: tuple[float, ...] | None = None


@dataclass(frozen=True)
class VolSurfaceTarget:
    vol_surface: object
    discount_curve: object | None = None
    smoothing_policy: str | None = None
    arbitrage_checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class NoCalibrationTarget:
    reason: str = "analytical"


@dataclass(frozen=True)
class CalibrationDiagnostics:
    residuals: Mapping[str, object] = field(default_factory=dict)
    iterations: Mapping[str, object] = field(default_factory=dict)
    positivity_violations: int = 0
    fallback_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibratedLatticeData:
    lattice: object
    diagnostics: CalibrationDiagnostics = field(default_factory=CalibrationDiagnostics)


@dataclass(frozen=True)
class LatticeLinearClaimSpec:
    """Linear payoff/cashflow layer on a lattice."""

    terminal_payoff: Callable[[int, int, object, Mapping[str, object]], float]
    node_cashflow_fn: Callable[[int, int, object, Mapping[str, object]], float] | None = None
    edge_cashflow_fn: Callable[[int, int, int, object, Mapping[str, object], Mapping[str, object]], float] | None = None
    observable_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class LatticeControlSpec:
    """Normalized single-controller exercise layer."""

    objective: str
    exercise_steps: tuple[int, ...] = ()
    exercise_value_fn: Callable[[int, int, object, Mapping[str, object]], float] | None = None


@dataclass(frozen=True)
class EventOverlaySpec:
    """Finite-state edge-aware overlay on a base lattice."""

    states: tuple[str, ...]
    initial_state: str
    transition_fn: Callable[[int, int, int, str, object, Mapping[str, object], Mapping[str, object]], Mapping[str, float]]
    observable_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class LatticeContractSpec:
    """Composite contract surface for claim, control, and overlay layers."""

    claim: LatticeLinearClaimSpec
    control: LatticeControlSpec | None = None
    overlay: EventOverlaySpec | None = None
    timeline: object | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class LatticeRecipe:
    """Declarative recipe compiled into topology/mesh/model/contract specs."""

    topology_family: str
    mesh_family: str
    model_family: str
    calibration_target: object | None
    claim_kind: str
    claim_params: Mapping[str, object] = field(default_factory=dict)
    control_kind: str | None = None
    control_params: Mapping[str, object] | None = None
    overlay_kind: str | None = None
    overlay_params: Mapping[str, object] | None = None
    build_params: Mapping[str, object] | None = None


BINOMIAL_1F_TOPOLOGY = LatticeTopologySpec(
    name="binomial_1f",
    branching=2,
    node_count_fn=_binomial_node_count,
    child_indices_fn=_binomial_child_indices,
    parent_indices_fn=_binomial_parent_indices,
)

TRINOMIAL_1F_TOPOLOGY = LatticeTopologySpec(
    name="trinomial_1f",
    branching=3,
    node_count_fn=_trinomial_node_count,
    child_indices_fn=_trinomial_child_indices,
    parent_indices_fn=_trinomial_parent_indices,
)

PRODUCT_BINOMIAL_2F_TOPOLOGY = LatticeTopologySpec(
    name="product_binomial_2f",
    branching=(2, 2),
    node_count_fn=_product_binomial_2f_node_count,
    child_indices_fn=_product_binomial_2f_child_indices,
    parent_indices_fn=_product_binomial_2f_parent_indices,
    factor_count=2,
    product_topology=True,
)

UNIFORM_ADDITIVE_MESH = LatticeMeshSpec(
    name="uniform_additive_1f",
    coordinate_fn=_uniform_additive_coordinate,
    step_size_fn=_uniform_additive_step_size,
    metric_fn=_identity_metric,
    transform_name="identity",
)

LOG_SPOT_MESH = LatticeMeshSpec(
    name="log_spot_1f",
    coordinate_fn=_log_spot_coordinate,
    step_size_fn=_log_spot_step_size,
    metric_fn=_identity_metric,
    transform_name="log",
)

PRODUCT_LOG_SPOT_2F_MESH = LatticeMeshSpec(
    name="product_log_spot_2f",
    coordinate_fn=_product_log_spot_coordinate,
    metric_fn=_identity_metric,
    transform_name="log",
)

TOPOLOGY_REGISTRY: dict[str, LatticeTopologySpec] = {
    BINOMIAL_1F_TOPOLOGY.name: BINOMIAL_1F_TOPOLOGY,
    TRINOMIAL_1F_TOPOLOGY.name: TRINOMIAL_1F_TOPOLOGY,
    PRODUCT_BINOMIAL_2F_TOPOLOGY.name: PRODUCT_BINOMIAL_2F_TOPOLOGY,
}

MESH_REGISTRY: dict[str, LatticeMeshSpec] = {
    UNIFORM_ADDITIVE_MESH.name: UNIFORM_ADDITIVE_MESH,
    LOG_SPOT_MESH.name: LOG_SPOT_MESH,
    PRODUCT_LOG_SPOT_2F_MESH.name: PRODUCT_LOG_SPOT_2F_MESH,
}


def _lattice_model_from_tree_model(model) -> LatticeModelSpec:
    state_space_type = "additive_normal" if getattr(model, "vol_type", "normal") == "normal" else "multiplicative_lognormal"
    observable_fn = _default_short_rate_observable
    return LatticeModelSpec(
        name=str(model.name),
        factor_family=str(getattr(model, "factor_family", "short_rate")),
        factor_count=1,
        state_space_type=state_space_type,
        numeraire="money_market",
        calibration_strategy="term_structure",
        supported_topologies=("binomial_1f", "trinomial_1f"),
        supported_branchings=tuple(getattr(model, "supported_branchings", (2, 3))),
        observable_fn=observable_fn,
        state_metric_fn=getattr(model, "state_metric_fn", None),
        probability_fn=getattr(model, "probability_fn", None),
        discount_fn=getattr(model, "discount_fn", None),
        supported_calibration_targets=("term_structure",),
        legacy_tree_model=model,
    )


def _equity_model_spec(name: str) -> LatticeModelSpec:
    return LatticeModelSpec(
        name=name,
        factor_family="equity",
        factor_count=1,
        state_space_type="spot",
        numeraire="money_market",
        calibration_strategy="none",
        supported_topologies=("binomial_1f",),
        supported_branchings=(2,),
        observable_fn=_default_equity_observable,
        state_metric_fn=_identity_metric,
        supported_calibration_targets=("none",),
    )


def _local_vol_model_spec() -> LatticeModelSpec:
    return LatticeModelSpec(
        name="local_vol",
        factor_family="equity",
        factor_count=1,
        state_space_type="spot",
        numeraire="money_market",
        calibration_strategy="vol_surface",
        supported_topologies=("trinomial_1f",),
        supported_branchings=(3,),
        observable_fn=_default_equity_observable,
        state_metric_fn=_identity_metric,
        supported_calibration_targets=("vol_surface",),
    )


def _two_factor_equity_model_spec() -> LatticeModelSpec:
    return LatticeModelSpec(
        name="correlated_gbm_2f",
        factor_family="equity_hybrid",
        factor_count=2,
        state_space_type="spot_pair",
        numeraire="money_market",
        calibration_strategy="joint_analytical_2f",
        supported_topologies=("product_binomial_2f",),
        observable_fn=_default_two_factor_equity_observable,
        state_metric_fn=_identity_metric,
        supported_calibration_targets=("none",),
    )


LATTICE_MODEL_REGISTRY: dict[str, LatticeModelSpec] = {
    **{name: _lattice_model_from_tree_model(model) for name, model in LEGACY_MODEL_REGISTRY.items()},
    "crr": _equity_model_spec("crr"),
    "jarrow_rudd": _equity_model_spec("jarrow_rudd"),
    "jr": _equity_model_spec("jarrow_rudd"),
    "local_vol": _local_vol_model_spec(),
    "correlated_gbm_2f": _two_factor_equity_model_spec(),
}


TERM_STRUCTURE_TARGET = TermStructureTarget
NO_CALIBRATION_TARGET = NoCalibrationTarget


def _target_name(target: object | None) -> str:
    if isinstance(target, TermStructureTarget):
        return "term_structure"
    if isinstance(target, VolSurfaceTarget):
        return "vol_surface"
    if isinstance(target, NoCalibrationTarget):
        return "none"
    return "custom" if target is not None else "none"


def _term_structure_residuals(lattice, discount_curve) -> dict[str, float]:
    from trellis.models.trees.lattice import _propagate_arrow_debreu

    residuals: dict[str, float] = {}
    q_current = raw_np.array([1.0], dtype=float)
    for step in range(lattice.n_steps):
        n_nodes = lattice.n_nodes(step)
        discounts = lattice._discounts[step, :n_nodes]
        probs = lattice._probs[step, :n_nodes, :lattice.branching]
        q_current = _propagate_arrow_debreu(q_current, discounts, probs, lattice.branching)
        market_df = float(discount_curve.discount((step + 1) * lattice.dt))
        residuals[f"t_{step + 1}"] = float(abs(raw_np.sum(q_current) - market_df))
    return residuals


class TermStructureCalibration:
    """Calibration/build strategy for short-rate lattices."""

    def calibrate(
        self,
        topology: LatticeTopologySpec,
        mesh: LatticeMeshSpec,
        model: LatticeModelSpec,
        target: object | None,
        **params,
    ) -> CalibratedLatticeData:
        del mesh
        from trellis.models.trees.lattice import build_generic_lattice

        branching = _branching_value(topology.branching)
        if branching not in model.supported_branchings:
            raise ValueError(f"Model {model.name!r} does not support branching={branching}")

        target_curve = getattr(target, "discount_curve", target)
        if target_curve is None:
            target_curve = params.get("discount_curve")
        if target_curve is None:
            raise ValueError("Term-structure lattice building requires a discount_curve")
        legacy_model = model.legacy_tree_model or LEGACY_MODEL_REGISTRY.get(model.name)
        if legacy_model is None:
            raise ValueError(f"Model {model.name!r} is missing a legacy tree bridge")

        maturity = float(params.get("T", params.get("maturity", 0.0)))
        lattice = build_generic_lattice(
            legacy_model,
            r0=float(params["r0"]),
            sigma=float(params["sigma"]),
            a=float(params.get("a", 0.0)),
            T=maturity,
            n_steps=int(params["n_steps"]),
            discount_curve=target_curve,
            branching=branching,
        )
        diagnostics = CalibrationDiagnostics(
            residuals=_term_structure_residuals(lattice, target_curve),
            iterations={"passes": 3, "strategy": "term_structure"},
        )
        return CalibratedLatticeData(lattice=lattice, diagnostics=diagnostics)


class AnalyticalCalibration:
    """Analytical builder for CRR/Jarrow-Rudd equity trees."""

    def calibrate(
        self,
        topology: LatticeTopologySpec,
        mesh: LatticeMeshSpec,
        model: LatticeModelSpec,
        target: object | None,
        **params,
    ) -> CalibratedLatticeData:
        del mesh, target
        from trellis.models.trees.lattice import _build_spot_lattice_impl

        branching = _branching_value(topology.branching)
        if branching != 2:
            raise ValueError("The shipped analytical equity lattice only supports binomial branching")
        if branching not in model.supported_branchings:
            raise ValueError(f"Model {model.name!r} does not support branching={branching}")

        maturity = float(params.get("maturity", params.get("T", 0.0)))
        lattice = _build_spot_lattice_impl(
            float(params.get("spot", params.get("S0"))),
            float(params.get("rate", params.get("r"))),
            float(params["sigma"]),
            maturity,
            int(params["n_steps"]),
            model=model.name,
        )
        diagnostics = CalibrationDiagnostics(
            residuals={},
            iterations={"passes": 1, "strategy": "analytical"},
        )
        return CalibratedLatticeData(lattice=lattice, diagnostics=diagnostics)


def _resolve_discount_step(discount_curve, *, step: int, dt: float) -> float:
    t0 = float(step) * float(dt)
    t1 = float(step + 1) * float(dt)
    df0 = float(discount_curve.discount(t0))
    df1 = float(discount_curve.discount(t1))
    if df0 <= 0.0 or df1 <= 0.0:
        raise ValueError("Discount curve must return positive discount factors")
    return df1 / df0


def _resolve_local_vol_surface(surface):
    if callable(surface):
        return surface
    if hasattr(surface, "black_vol"):
        return lambda spot, time: float(surface.black_vol(max(float(time), 1e-6), float(spot)))
    raise TypeError("VolSurfaceTarget.vol_surface must be callable or expose black_vol(t, strike)")


class LocalVolCalibration:
    """Local-volatility calibration on a recombining trinomial log-spot mesh."""

    def calibrate(
        self,
        topology: LatticeTopologySpec,
        mesh: LatticeMeshSpec,
        model: LatticeModelSpec,
        target: object | None,
        **params,
    ) -> CalibratedLatticeData:
        del mesh, model
        if topology.name != "trinomial_1f":
            raise ValueError("The shipped local-vol lattice currently requires the trinomial 1F topology")

        from trellis.models.trees.lattice import RecombiningLattice

        if not isinstance(target, VolSurfaceTarget):
            raise TypeError("Local-vol calibration requires a VolSurfaceTarget")
        local_vol_surface = _resolve_local_vol_surface(target.vol_surface)
        spot = float(params.get("spot", params.get("S0")))
        maturity = float(params.get("maturity", params.get("T", 0.0)))
        n_steps = int(params["n_steps"])
        dt = maturity / max(n_steps, 1)
        lattice = RecombiningLattice(n_steps, dt, branching=3, state_dim=1)

        discount_curve = target.discount_curve
        rate = float(params.get("rate", params.get("r", 0.0)))
        sample_spots = raw_np.geomspace(max(spot * 0.35, 1e-6), max(spot * 3.0, 1e-6), num=25)
        sample_times = raw_np.linspace(max(dt, 1e-6), max(maturity, dt), num=min(max(n_steps, 2), 12))
        reference_sigma = max(
            float(params.get("sigma", 0.0)),
            max(float(local_vol_surface(sample_spot, sample_time)) for sample_spot in sample_spots for sample_time in sample_times),
            1e-6,
        )
        dx = reference_sigma * sqrt(3.0 * dt)
        min_probability = 1.0
        max_probability = 0.0
        positivity_violations = 0
        fallback_flags: list[str] = []

        for step in range(n_steps + 1):
            n_nodes = lattice.n_nodes(step)
            for node in range(n_nodes):
                log_spot = log(max(spot, 1e-12)) + (node - step) * dx
                lattice.set_state(step, node, exp(log_spot))
                if step < n_steps:
                    if discount_curve is not None:
                        lattice.set_discount(step, node, _resolve_discount_step(discount_curve, step=step, dt=dt))
                    else:
                        lattice.set_discount(step, node, exp(-rate * dt))

        for step in range(n_steps):
            t = float(step) * dt
            n_nodes = lattice.n_nodes(step)
            for node in range(n_nodes):
                state = float(lattice.get_state(step, node))
                sigma = max(float(local_vol_surface(state, max(t, 1e-6))), 1e-8)
                drift = rate - 0.5 * sigma * sigma
                scale = (sigma * sigma * dt + drift * drift * dt * dt) / max(dx * dx, 1e-16)
                p_up = 0.5 * (scale + drift * dt / max(dx, 1e-16))
                p_down = 0.5 * (scale - drift * dt / max(dx, 1e-16))
                p_mid = 1.0 - scale
                raw_probs = [p_down, p_mid, p_up]
                if any(probability < -1e-12 or probability > 1.0 + 1e-12 for probability in raw_probs):
                    positivity_violations += 1
                    fallback_flags.append(f"clip@{step}:{node}")
                clipped = [min(max(float(probability), 0.0), 1.0) for probability in raw_probs]
                total = sum(clipped)
                if total <= 0.0:
                    positivity_violations += 1
                    clipped = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
                else:
                    clipped = [probability / total for probability in clipped]
                min_probability = min(min_probability, min(clipped))
                max_probability = max(max_probability, max(clipped))
                lattice.set_probabilities(step, node, clipped)

        diagnostics = CalibrationDiagnostics(
            residuals={
                "dx": float(dx),
                "min_probability": float(min_probability),
                "max_probability": float(max_probability),
            },
            iterations={
                "passes": 1,
                "strategy": "vol_surface",
                "reference_sigma": float(reference_sigma),
            },
            positivity_violations=int(positivity_violations),
            fallback_flags=tuple(fallback_flags),
        )
        return CalibratedLatticeData(lattice=lattice, diagnostics=diagnostics)


class TwoFactorAnalyticalCalibration:
    """Analytical builder for two-factor product spot lattices."""

    def calibrate(
        self,
        topology: LatticeTopologySpec,
        mesh: LatticeMeshSpec,
        model: LatticeModelSpec,
        target: object | None,
        **params,
    ) -> CalibratedLatticeData:
        del mesh, model, target
        if topology.name != "product_binomial_2f":
            raise ValueError("The shipped two-factor lattice requires the product_binomial_2f topology")

        from trellis.models.trees.product_lattice import build_product_spot_lattice_2d

        lattice, summary = build_product_spot_lattice_2d(
            spots=tuple(float(value) for value in params["spots"]),
            rate=float(params.get("rate", params.get("r", 0.0))),
            sigmas=tuple(float(value) for value in params["sigmas"]),
            maturity=float(params.get("maturity", params.get("T", 0.0))),
            n_steps=int(params["n_steps"]),
            correlation=float(params.get("correlation", 0.0)),
        )
        diagnostics = CalibrationDiagnostics(
            residuals={
                "min_probability": float(summary["min_probability"]),
                "max_probability": float(summary["max_probability"]),
            },
            iterations={"passes": 1, "strategy": "joint_analytical_2f"},
            positivity_violations=int(summary["positivity_violations"]),
            fallback_flags=tuple(
                flag
                for flag, count in (
                    ("correlation_clip", int(summary["correlation_clips"])),
                )
                if count > 0
            ),
        )
        return CalibratedLatticeData(lattice=lattice, diagnostics=diagnostics)


CALIBRATION_STRATEGIES: dict[str, CalibrationStrategy] = {
    "term_structure": TermStructureCalibration(),
    "none": AnalyticalCalibration(),
    "vol_surface": LocalVolCalibration(),
    "joint_analytical_2f": TwoFactorAnalyticalCalibration(),
}


def _attach_lattice_metadata(lattice, *, topology: LatticeTopologySpec, mesh: LatticeMeshSpec, model: LatticeModelSpec, calibration_target: object | None, diagnostics: CalibrationDiagnostics) -> None:
    lattice._lattice_topology_spec = topology
    lattice._lattice_mesh_spec = mesh
    lattice._lattice_model_spec = model
    lattice._lattice_calibration_target = calibration_target
    lattice._lattice_calibration_diagnostics = diagnostics


def build_lattice(
    topology: LatticeTopologySpec,
    mesh: LatticeMeshSpec,
    model: LatticeModelSpec | object,
    calibration_target: object | None = None,
    **params,
):
    """Build and calibrate a lattice through the generalized API."""
    topology = _lookup_spec(topology, TOPOLOGY_REGISTRY, "topology")
    mesh = _lookup_spec(mesh, MESH_REGISTRY, "mesh")
    if isinstance(model, str):
        model = _lookup_spec(model, LATTICE_MODEL_REGISTRY, "model")
    if not isinstance(model, LatticeModelSpec):
        model = _lattice_model_from_tree_model(model)
    if topology.name not in model.supported_topologies and model.supported_topologies:
        raise ValueError(
            f"Model {model.name!r} does not support topology {topology.name!r}. "
            f"Supported: {model.supported_topologies}"
        )

    target_name = _target_name(calibration_target)
    if calibration_target is not None and model.supported_calibration_targets and target_name not in model.supported_calibration_targets:
        raise ValueError(
            f"Model {model.name!r} does not support calibration target {target_name!r}. "
            f"Supported: {model.supported_calibration_targets}"
        )

    strategy = CALIBRATION_STRATEGIES.get(model.calibration_strategy)
    if strategy is None:
        raise ValueError(f"Unsupported lattice calibration strategy {model.calibration_strategy!r}")

    result = strategy.calibrate(topology, mesh, model, calibration_target, **params)
    _attach_lattice_metadata(
        result.lattice,
        topology=topology,
        mesh=mesh,
        model=model,
        calibration_target=calibration_target,
        diagnostics=result.diagnostics,
    )
    return result.lattice


def _base_observable(lattice, step: int, node: int) -> dict[str, object]:
    state = lattice.get_state(step, node)
    model = getattr(lattice, "_lattice_model_spec", None)
    if isinstance(model, LatticeModelSpec) and callable(model.observable_fn):
        observable = model.observable_fn(state)
        data = dict(observable) if isinstance(observable, Mapping) else {"observable": observable}
    else:
        data = {}
    if isinstance(state, tuple):
        data.setdefault("state", state)
        data.setdefault("value", state)
        data.setdefault("latent_state", state)
    else:
        value = float(state)
        data.setdefault("state", value)
        data.setdefault("value", value)
        data.setdefault("spot", value)
        data.setdefault("rate", value)
        data.setdefault("short_rate", value)
        data.setdefault("latent_state", value)
    return data


def _observable_with_event(base_observable: Mapping[str, object], event_state: str | None) -> dict[str, object]:
    observable = dict(base_observable)
    if event_state is not None:
        observable["event_state"] = event_state
    return observable


def _control_policy(control: LatticeControlSpec | None):
    if control is None or control.objective == "identity":
        return None
    objective = str(control.objective).strip().lower()
    if objective not in {"holder_max", "issuer_min"}:
        raise ValueError(f"Unsupported lattice control objective {control.objective!r}")
    if control.exercise_value_fn is None:
        raise ValueError("Non-identity lattice controls require an exercise_value_fn")
    return {
        "exercise_type": "bermudan" if control.exercise_steps else "american",
        "exercise_steps": tuple(int(step) for step in control.exercise_steps),
        "exercise_fn": max if objective == "holder_max" else min,
    }


def _price_general_contract(lattice, contract: LatticeContractSpec) -> float:
    overlay = contract.overlay
    control = contract.control
    claim = contract.claim

    event_states = overlay.states if overlay is not None else ("base",)
    state_index = {name: idx for idx, name in enumerate(event_states)}
    initial_state = overlay.initial_state if overlay is not None else "base"
    n_steps = lattice.n_steps
    terminal_nodes = lattice.n_nodes(n_steps)
    values = raw_np.zeros((len(event_states), terminal_nodes), dtype=float)

    for event_state in event_states:
        idx = state_index[event_state]
        for node in range(terminal_nodes):
            obs = _observable_with_event(_base_observable(lattice, n_steps, node), None if overlay is None else event_state)
            values[idx, node] = float(claim.terminal_payoff(n_steps, node, lattice, obs))

    for step in range(n_steps - 1, -1, -1):
        n_nodes = lattice.n_nodes(step)
        new_values = raw_np.zeros((len(event_states), n_nodes), dtype=float)
        for event_state in event_states:
            idx = state_index[event_state]
            for node in range(n_nodes):
                base_parent = _base_observable(lattice, step, node)
                parent_obs = _observable_with_event(base_parent, None if overlay is None else event_state)
                discount = float(lattice.get_discount(step, node))
                continuation = 0.0
                for probability, child in zip(lattice.get_probabilities(step, node), lattice.child_indices(step, node)):
                    base_child = _base_observable(lattice, step + 1, child)
                    edge_cashflow = 0.0
                    if claim.edge_cashflow_fn is not None:
                        edge_cashflow = float(
                            claim.edge_cashflow_fn(
                                step,
                                node,
                                child,
                                lattice,
                                parent_obs,
                                _observable_with_event(base_child, None if overlay is None else event_state),
                            )
                        )
                    if overlay is None:
                        continuation += discount * float(probability) * (edge_cashflow + float(values[idx, child]))
                        continue

                    transitions = overlay.transition_fn(
                        step,
                        node,
                        child,
                        event_state,
                        lattice,
                        parent_obs,
                        _observable_with_event(base_child, event_state),
                    )
                    for next_state, overlay_probability in transitions.items():
                        child_idx = state_index[next_state]
                        continuation += (
                            discount
                            * float(probability)
                            * float(overlay_probability)
                            * (edge_cashflow + float(values[child_idx, child]))
                        )

                node_cashflow = 0.0
                if claim.node_cashflow_fn is not None:
                    node_cashflow = float(claim.node_cashflow_fn(step, node, lattice, parent_obs))
                node_value = node_cashflow + continuation
                if control is not None and control.objective != "identity":
                    exercise_open = step in control.exercise_steps if control.exercise_steps else True
                    if exercise_open and control.exercise_value_fn is not None:
                        exercise_value = float(control.exercise_value_fn(step, node, lattice, parent_obs))
                        if str(control.objective).strip().lower() == "holder_max":
                            node_value = max(node_value, exercise_value)
                        else:
                            node_value = min(node_value, exercise_value)
                new_values[idx, node] = node_value
        values = new_values

    return float(values[state_index[initial_state], 0])


def price_on_lattice(lattice, contract: LatticeContractSpec) -> float:
    """Price a compiled lattice contract on a built lattice."""
    claim = contract.claim
    control = contract.control

    def _set_path(path: str) -> None:
        lattice._lattice_last_pricing_path = path

    if (
        contract.overlay is None
        and claim.node_cashflow_fn is None
        and claim.edge_cashflow_fn is None
        and control is None
        and hasattr(lattice, "fast_terminal_rollback")
    ):
        n_steps = lattice.n_steps
        terminal_nodes = lattice.n_nodes(n_steps)
        values = _node_values(
            terminal_nodes,
            (
                claim.terminal_payoff(n_steps, node, lattice, _base_observable(lattice, n_steps, node))
                for node in range(terminal_nodes)
            ),
        )
        _set_path(f"fast_product_2d_{'numba' if NUMBA_AVAILABLE else 'numpy'}")
        return float(lattice.fast_terminal_rollback(values))

    if contract.overlay is not None or claim.edge_cashflow_fn is not None or getattr(lattice, "state_dim", 1) != 1:
        path = "python_overlay_fallback" if contract.overlay is not None or claim.edge_cashflow_fn is not None else "python_multid_fallback"
        warnings.warn(
            "price_on_lattice() used a Python fallback because the contract is outside the 1D fast-path contract",
            RuntimeWarning,
            stacklevel=2,
        )
        _set_path(path)
        return _price_general_contract(lattice, contract)

    policy = _control_policy(control)

    def terminal_payoff(step: int, node: int, lattice_):
        return float(claim.terminal_payoff(step, node, lattice_, _base_observable(lattice_, step, node)))

    cashflow_at_node = None
    if claim.node_cashflow_fn is not None:
        def cashflow_at_node(step: int, node: int, lattice_):
            return float(claim.node_cashflow_fn(step, node, lattice_, _base_observable(lattice_, step, node)))

    exercise_value = None
    if control is not None and control.objective != "identity" and control.exercise_value_fn is not None:
        def exercise_value(step: int, node: int, lattice_):
            return float(control.exercise_value_fn(step, node, lattice_, _base_observable(lattice_, step, node)))

    from trellis.models.trees.lattice import lattice_backward_induction

    _set_path(
        f"fast_{'obstacle' if control is not None and control.objective != 'identity' else 'linear'}_{'numba' if NUMBA_AVAILABLE else 'numpy'}"
    )
    return float(
        lattice_backward_induction(
            lattice,
            terminal_payoff,
            exercise_value=exercise_value,
            exercise_type="european" if policy is None else policy["exercise_type"],
            exercise_steps=None if policy is None else list(policy["exercise_steps"]),
            cashflow_at_node=cashflow_at_node,
            exercise_fn=None if policy is None else policy["exercise_fn"],
        )
    )


def short_rate_tree(
    *,
    model_family: str = "hull_white",
    branching: int = 2,
    claim_kind: str = "short_rate_claim",
    calibration_target: object | None = None,
    claim_params: Mapping[str, object] | None = None,
    **build_params,
) -> LatticeRecipe:
    """Return a declarative short-rate lattice recipe."""
    topology_family = "binomial_1f" if int(branching) == 2 else "trinomial_1f"
    return LatticeRecipe(
        topology_family=topology_family,
        mesh_family="uniform_additive_1f",
        model_family=model_family,
        calibration_target=calibration_target if calibration_target is not None else NoCalibrationTarget(),
        claim_kind=claim_kind,
        claim_params=_as_dict(claim_params),
        build_params=dict(build_params),
    )


def equity_tree(
    *,
    model_family: str = "crr",
    branching: int = 2,
    claim_kind: str = "vanilla_option",
    calibration_target: object | None = None,
    claim_params: Mapping[str, object] | None = None,
    **claim_kwargs,
) -> LatticeRecipe:
    """Return a declarative equity lattice recipe."""
    topology_family = "binomial_1f" if int(branching) == 2 else "trinomial_1f"
    payload = _as_dict(claim_params)
    payload.update(claim_kwargs)
    return LatticeRecipe(
        topology_family=topology_family,
        mesh_family="log_spot_1f",
        model_family=model_family,
        calibration_target=calibration_target if calibration_target is not None else NoCalibrationTarget(),
        claim_kind=claim_kind,
        claim_params=payload,
        build_params={},
    )


def with_control(recipe: LatticeRecipe, control_kind: str, **control_params) -> LatticeRecipe:
    """Attach a control layer to a lattice recipe."""
    return replace(
        recipe,
        control_kind=str(control_kind),
        control_params=dict(control_params),
    )


def with_overlay(recipe: LatticeRecipe, overlay_kind: str, **overlay_params) -> LatticeRecipe:
    """Attach an overlay layer to a lattice recipe."""
    return replace(
        recipe,
        overlay_kind=str(overlay_kind),
        overlay_params=dict(overlay_params),
    )


def _compile_vanilla_option_claim(claim_params: Mapping[str, object]) -> LatticeLinearClaimSpec:
    strike = float(claim_params["strike"])
    option_type = str(claim_params.get("option_type", "call")).strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError(f"Unsupported vanilla option type {option_type!r}")

    def terminal_payoff(step: int, node: int, lattice, obs: Mapping[str, object]) -> float:
        del step, node, lattice
        spot = float(obs["spot"])
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    return LatticeLinearClaimSpec(
        terminal_payoff=terminal_payoff,
        observable_requirements=("spot",),
    )


def _compile_control(control_kind: str | None, control_params: Mapping[str, object] | None) -> LatticeControlSpec | None:
    if control_kind is None:
        return None
    params = _as_dict(control_params)
    normalized = str(control_kind).strip().lower()
    if normalized in {"american", "holder_max"}:
        return LatticeControlSpec(
            objective="holder_max",
            exercise_steps=tuple(int(step) for step in params.get("exercise_steps", ())),
            exercise_value_fn=params.get("exercise_value_fn"),
        )
    if normalized in {"bermudan", "holder_put"}:
        return LatticeControlSpec(
            objective="holder_max",
            exercise_steps=tuple(int(step) for step in params.get("exercise_steps", ())),
            exercise_value_fn=params.get("exercise_value_fn"),
        )
    if normalized in {"issuer_call", "issuer_min"}:
        return LatticeControlSpec(
            objective="issuer_min",
            exercise_steps=tuple(int(step) for step in params.get("exercise_steps", ())),
            exercise_value_fn=params.get("exercise_value_fn"),
        )
    if normalized in {"identity", "european"}:
        return None
    raise ValueError(f"Unsupported lattice control kind {control_kind!r}")


def _compile_overlay(overlay_kind: str | None, overlay_params: Mapping[str, object] | None) -> EventOverlaySpec | None:
    if overlay_kind is None:
        return None
    params = _as_dict(overlay_params)
    normalized = str(overlay_kind).strip().lower()
    if normalized == "knock_out_barrier":
        barrier = float(params["barrier"])
        direction = str(params.get("direction", "up")).strip().lower()

        def transition_fn(step, parent, child, event_state, lattice, obs_parent, obs_child):
            del step, parent, child, lattice, obs_parent
            if event_state == "dead":
                return {"dead": 1.0}
            spot = float(obs_child["spot"])
            crossed = spot >= barrier if direction == "up" else spot <= barrier
            return {"dead": 1.0} if crossed else {"alive": 1.0}

        return EventOverlaySpec(
            states=("alive", "dead"),
            initial_state="alive",
            transition_fn=transition_fn,
            observable_requirements=("spot",),
        )
    raise ValueError(f"Unsupported lattice overlay kind {overlay_kind!r}")


def compile_lattice_recipe(recipe: LatticeRecipe) -> tuple[LatticeTopologySpec, LatticeMeshSpec, LatticeModelSpec, LatticeContractSpec]:
    """Compile a declarative lattice recipe into concrete specs."""
    topology = TOPOLOGY_REGISTRY[recipe.topology_family]
    mesh = MESH_REGISTRY[recipe.mesh_family]
    model = LATTICE_MODEL_REGISTRY[recipe.model_family]

    if recipe.claim_kind == "vanilla_option":
        claim = _compile_vanilla_option_claim(recipe.claim_params)
    else:
        raise ValueError(f"Unsupported lattice claim kind {recipe.claim_kind!r}")

    default_exercise_value = None
    if recipe.control_kind in {"american", "bermudan", "holder_max"}:
        default_exercise_value = claim.terminal_payoff
    control_params = _as_dict(recipe.control_params)
    if default_exercise_value is not None and "exercise_value_fn" not in control_params:
        control_params["exercise_value_fn"] = claim.terminal_payoff
    control = _compile_control(recipe.control_kind, control_params)
    overlay = _compile_overlay(recipe.overlay_kind, recipe.overlay_params)
    return topology, mesh, model, LatticeContractSpec(
        claim=claim,
        control=control,
        overlay=overlay,
        metadata=recipe.build_params,
    )


@dataclass(frozen=True)
class LatticeAlgebraEligibilityDecision:
    ok: bool
    reasons: tuple[str, ...] = ()


def lattice_algebra_eligible(*, product=None, product_ir=None) -> LatticeAlgebraEligibilityDecision:
    """Return whether a semantic product fits the generalized lattice boundary."""
    reasons: list[str] = []

    semantic_product = product
    model_family = str(
        getattr(semantic_product, "model_family", "")
        or getattr(product_ir, "model_family", "")
    ).strip().lower()
    underlier_structure = str(getattr(semantic_product, "underlier_structure", "")).strip().lower()
    multi_asset = bool(getattr(semantic_product, "multi_asset", False))
    constituents = tuple(getattr(semantic_product, "constituents", ()) or ())
    if multi_asset or len(constituents) >= 5 or underlier_structure == "multi_asset_basket" or model_family == "equity_multi_asset":
        reasons.append("multi_asset")

    state_dependence = str(
        getattr(semantic_product, "state_dependence", "")
        or getattr(product_ir, "state_dependence", "")
    ).strip().lower()
    path_dependence = str(getattr(semantic_product, "path_dependence", "")).strip().lower()
    if state_dependence == "path_dependent" or path_dependence == "path_dependent":
        reasons.append("non_markov")

    if model_family in {"rough_vol", "hjm", "lmm"}:
        reasons.append("unsupported_model_family")

    controller_protocol = getattr(semantic_product, "controller_protocol", None)
    actions = tuple(getattr(controller_protocol, "admissible_actions", ()) or ())
    non_continue_actions = tuple(action for action in actions if str(action).strip().lower() not in {"", "continue"})
    if len(non_continue_actions) > 1 and {str(action).strip().lower() for action in non_continue_actions} not in ({"exercise"}, {"call"}):
        reasons.append("multi_controller")

    decision = tuple(dict.fromkeys(reasons))
    return LatticeAlgebraEligibilityDecision(ok=not decision, reasons=decision)


__all__ = [
    "AnalyticalCalibration",
    "BINOMIAL_1F_TOPOLOGY",
    "PRODUCT_BINOMIAL_2F_TOPOLOGY",
    "CALIBRATION_STRATEGIES",
    "CalibrationDiagnostics",
    "CalibrationStrategy",
    "CalibratedLatticeData",
    "EventOverlaySpec",
    "LATTICE_MODEL_REGISTRY",
    "LOG_SPOT_MESH",
    "PRODUCT_LOG_SPOT_2F_MESH",
    "LatticeAlgebraEligibilityDecision",
    "LatticeContractSpec",
    "LatticeControlSpec",
    "LatticeLinearClaimSpec",
    "LatticeMeshSpec",
    "LatticeModelSpec",
    "LatticeRecipe",
    "LatticeTopologySpec",
    "MESH_REGISTRY",
    "NO_CALIBRATION_TARGET",
    "NoCalibrationTarget",
    "TERM_STRUCTURE_TARGET",
    "TOPOLOGY_REGISTRY",
    "TRINOMIAL_1F_TOPOLOGY",
    "TermStructureCalibration",
    "TermStructureTarget",
    "TwoFactorAnalyticalCalibration",
    "UNIFORM_ADDITIVE_MESH",
    "LocalVolCalibration",
    "VolSurfaceTarget",
    "build_lattice",
    "compile_lattice_recipe",
    "equity_tree",
    "lattice_algebra_eligible",
    "price_on_lattice",
    "short_rate_tree",
    "with_control",
    "with_overlay",
]
