"""Stable Bermudan swaption tree helpers on the generalized lattice substrate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Protocol

from trellis.core.date_utils import (
    build_payment_timeline,
    normalize_explicit_dates,
    year_fraction,
)
from trellis.core.differentiable import get_numpy
from trellis.core.types import ContractTimeline, DayCountConvention, Frequency
import trellis.models.trees.algebra as lattice_algebra
import trellis.models.trees.models as tree_models
from trellis.models.trees.control import (
    lattice_step_from_time,
    lattice_steps_from_timeline,
    resolve_lattice_exercise_policy_from_control_style,
)
from trellis.models.trees.lattice import (
    RecombiningLattice,
    build_lattice,
    price_on_lattice,
)
from trellis.models.hull_white_parameters import resolve_hull_white_parameters


class DiscountCurveLike(Protocol):
    """Discount interface required by the Bermudan swaption tree helpers."""

    def discount(self, t: float) -> float:
        """Return a discount factor to time ``t``."""
        ...

    def zero_rate(self, t: float) -> float:
        """Return a zero rate to time ``t``."""
        ...


class VolSurfaceLike(Protocol):
    """Volatility interface required by the Bermudan swaption tree helpers."""

    def black_vol(self, t: float, strike: float) -> float:
        """Return a Black-style volatility quote."""
        ...


class BermudanSwaptionTreeMarketStateLike(Protocol):
    """Market-state interface required by the Bermudan swaption tree helpers."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None
    vol_surface: VolSurfaceLike | None


class BermudanSwaptionSpecLike(Protocol):
    """Spec fields consumed by the Bermudan swaption tree helpers."""

    notional: float
    strike: float
    exercise_dates: ContractTimeline | Iterable[date | str]
    swap_end: date
    swap_frequency: object
    day_count: DayCountConvention
    is_payer: bool


@dataclass(frozen=True)
class BermudanSwaptionTreeSpec:
    """Stable typed spec for the supported Bermudan swaption tree route."""

    notional: float
    strike: float
    exercise_dates: tuple[date, ...]
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True


@dataclass(frozen=True)
class ResolvedBermudanSwaptionTreeInputs:
    """Resolved market and schedule inputs for a Bermudan swaption tree."""

    settlement: date
    swap_start: date
    exercise_dates: tuple[date, ...]
    exercise_frequency: int
    tree_horizon: float
    option_horizon: float
    r0: float
    sigma: float
    n_steps: int


def resolve_bermudan_swaption_tree_inputs(
    market_state: BermudanSwaptionTreeMarketStateLike,
    spec: BermudanSwaptionSpecLike,
    *,
    model: str = "hull_white",
    mean_reversion: float | None = None,
    sigma: float | None = None,
    n_steps: int | None = None,
) -> ResolvedBermudanSwaptionTreeInputs:
    """Resolve settlement, exercise schedule, and lattice calibration inputs."""
    settlement = _settlement_date(market_state, spec)
    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("Bermudan swaption tree pricing requires market_state.discount")

    exercise_dates = _effective_exercise_dates(spec, settlement)
    if not exercise_dates:
        raise ValueError("Bermudan swaption tree pricing requires exercise dates before swap_end")

    swap_start = min(exercise_dates)
    day_count = getattr(spec, "day_count", DayCountConvention.ACT_360)
    exercise_frequency = _infer_schedule_frequency(exercise_dates, day_count)
    option_horizon = _quantize_time(
        float(year_fraction(settlement, max(exercise_dates), day_count)),
        frequency=exercise_frequency,
    )
    tenor = _quantize_time(
        float(year_fraction(swap_start, spec.swap_end, day_count)),
        frequency=_frequency_per_year(spec.swap_frequency),
    )
    tree_horizon = option_horizon + tenor
    if tree_horizon <= 0.0:
        raise ValueError("Bermudan swaption tree horizon must be positive")
    if tenor <= 0.0:
        raise ValueError("swap_end must be after the first Bermudan exercise date")

    r0 = float(discount_curve.zero_rate(max(tree_horizon / 2.0, 1e-6)))
    tree_model = tree_models.MODEL_REGISTRY[str(model).strip().lower()]
    default_sigma = None
    if market_state.vol_surface is not None:
        black_vol = float(
            market_state.vol_surface.black_vol(max(option_horizon, 1e-6), max(abs(float(spec.strike)), 1e-6))
        )
        default_sigma = black_vol if tree_model.vol_type == "lognormal" else black_vol * max(abs(r0), 1e-6)
    resolved_mean_reversion, resolved_sigma = resolve_hull_white_parameters(
        market_state,
        mean_reversion=mean_reversion,
        sigma=sigma,
        default_mean_reversion=0.1,
        default_sigma=default_sigma,
    )
    step_count = int(n_steps or min(400, max(100, int(tree_horizon * 20.0))))
    return ResolvedBermudanSwaptionTreeInputs(
        settlement=settlement,
        swap_start=swap_start,
        exercise_dates=exercise_dates,
        exercise_frequency=exercise_frequency,
        tree_horizon=tree_horizon,
        option_horizon=option_horizon,
        r0=r0,
        sigma=float(resolved_sigma),
        n_steps=step_count,
    )


def build_bermudan_swaption_lattice(
    market_state: BermudanSwaptionTreeMarketStateLike,
    spec: BermudanSwaptionSpecLike,
    *,
    model: str = "hull_white",
    mean_reversion: float | None = None,
    sigma: float | None = None,
    n_steps: int | None = None,
) -> RecombiningLattice:
    """Build the calibrated tree used by a Bermudan swaption."""
    resolved = resolve_bermudan_swaption_tree_inputs(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        sigma=sigma,
        n_steps=n_steps,
    )
    tree_model = tree_models.MODEL_REGISTRY[str(model).strip().lower()]
    return build_lattice(
        lattice_algebra.BINOMIAL_1F_TOPOLOGY,
        lattice_algebra.UNIFORM_ADDITIVE_MESH,
        tree_model.as_lattice_model_spec(),
        calibration_target=lattice_algebra.TERM_STRUCTURE_TARGET(market_state.discount),
        r0=resolved.r0,
        sigma=resolved.sigma,
        a=float(resolve_hull_white_parameters(
            market_state,
            mean_reversion=mean_reversion,
            sigma=resolved.sigma,
            default_mean_reversion=0.1,
            default_sigma=resolved.sigma,
        )[0]),
        T=resolved.tree_horizon,
        n_steps=resolved.n_steps,
    )


def build_bermudan_swaption_coupon_map(
    spec: BermudanSwaptionSpecLike,
    *,
    settlement: date,
    swap_start: date,
    dt: float,
    n_steps: int,
) -> dict[int, float]:
    """Map the fixed-leg coupon schedule onto lattice steps."""
    payment_timeline = build_payment_timeline(
        swap_start,
        spec.swap_end,
        spec.swap_frequency,
        day_count=spec.day_count,
        time_origin=settlement,
        label="bermudan_swaption_fixed_leg_timeline",
    )
    coupon_by_step: dict[int, float] = {}
    exercise_frequency = _infer_schedule_frequency(_normalized_exercise_dates(spec.exercise_dates), spec.day_count)
    first_exercise_step = lattice_step_from_time(
        _quantize_time(
            float(year_fraction(settlement, swap_start, spec.day_count)),
            frequency=exercise_frequency,
        ),
        dt=dt,
        n_steps=n_steps,
        allow_terminal_step=True,
    )
    if first_exercise_step is None:
        return coupon_by_step
    frequency_per_year = _frequency_per_year(spec.swap_frequency)
    steps_per_coupon = max(1, int(round((1.0 / frequency_per_year) / dt)))
    coupon = float(spec.notional) * float(spec.strike) / frequency_per_year
    tenor = float(year_fraction(swap_start, spec.swap_end, spec.day_count))
    quantized_tenor = round(tenor * frequency_per_year) / frequency_per_year
    swap_end_step = min(
        int(round((first_exercise_step * dt + quantized_tenor) / dt)),
        n_steps,
    )
    step = first_exercise_step + steps_per_coupon
    while step <= swap_end_step:
        coupon_by_step[step] = coupon
        step += steps_per_coupon
    return coupon_by_step


def build_bermudan_swaption_exercise_policy(
    spec: BermudanSwaptionSpecLike,
    *,
    settlement: date,
    dt: float,
    n_steps: int,
):
    """Resolve Bermudan exercise dates into a checked-in lattice policy."""
    exercise_dates = list(_effective_exercise_dates(spec, settlement))
    if not exercise_dates:
        return resolve_lattice_exercise_policy_from_control_style(
            "holder_max",
            exercise_steps=(),
            exercise_style="bermudan",
        )
    frequency = _infer_schedule_frequency(exercise_dates, spec.day_count)
    exercise_steps: list[int] = []
    for exercise_date in exercise_dates:
        quantized_time = _quantize_time(
            float(year_fraction(settlement, exercise_date, spec.day_count)),
            frequency=frequency,
        )
        step = lattice_step_from_time(
            quantized_time,
            dt=dt,
            n_steps=n_steps,
            allow_terminal_step=True,
        )
        if step is not None:
            exercise_steps.append(step)
    return resolve_lattice_exercise_policy_from_control_style(
        "holder_max",
        exercise_steps=exercise_steps,
        exercise_style="bermudan",
    )


def compile_bermudan_swaption_contract_spec(
    lattice: RecombiningLattice,
    *,
    spec: BermudanSwaptionSpecLike,
    settlement: date,
) -> lattice_algebra.LatticeContractSpec:
    """Compile Bermudan swaption exercise into a lattice contract."""
    exercise_policy = build_bermudan_swaption_exercise_policy(
        spec,
        settlement=settlement,
        dt=lattice.dt,
        n_steps=lattice.n_steps,
    )
    if not exercise_policy.exercise_steps:
        return lattice_algebra.LatticeContractSpec(
            claim=lattice_algebra.LatticeLinearClaimSpec(terminal_payoff=lambda step, node, lattice_, obs: 0.0),
            control=None,
        )

    effective_exercise_dates = _effective_exercise_dates(spec, settlement)
    if not effective_exercise_dates:
        return lattice_algebra.LatticeContractSpec(
            claim=lattice_algebra.LatticeLinearClaimSpec(terminal_payoff=lambda step, node, lattice_, obs: 0.0),
            control=None,
        )
    swap_start = min(effective_exercise_dates)
    coupon_by_step = build_bermudan_swaption_coupon_map(
        spec,
        settlement=settlement,
        swap_start=swap_start,
        dt=lattice.dt,
        n_steps=lattice.n_steps,
    )
    valid_exercise_steps = tuple(
        step for step in exercise_policy.exercise_steps if 0 <= step < lattice.n_steps
    )
    if not valid_exercise_steps:
        return lattice_algebra.LatticeContractSpec(
            claim=lattice_algebra.LatticeLinearClaimSpec(terminal_payoff=lambda step, node, lattice_, obs: 0.0),
            control=None,
        )

    frequency_per_year = _frequency_per_year(spec.swap_frequency)
    tenor = float(year_fraction(swap_start, spec.swap_end, spec.day_count))
    quantized_tenor = round(tenor * frequency_per_year) / frequency_per_year
    first_exercise_step = min(valid_exercise_steps)
    swap_end_step = min(
        int(round((first_exercise_step * lattice.dt + quantized_tenor) / lattice.dt)),
        lattice.n_steps,
    )
    payer_swap_values = {
        step: _compute_payer_swap_values_at_step(
            lattice,
            exercise_step=step,
            swap_end_step=swap_end_step,
            coupon_by_step=coupon_by_step,
            principal=float(spec.notional),
        )
        for step in valid_exercise_steps
        if step < swap_end_step
    }
    signed_swap_values = payer_swap_values if bool(spec.is_payer) else {
        step: -values for step, values in payer_swap_values.items()
    }

    def exercise_value(step: int, node: int, lattice_, obs) -> float:
        del lattice_, obs
        values = signed_swap_values.get(step)
        if values is None:
            return 0.0
        return max(float(values[node]), 0.0)

    return lattice_algebra.LatticeContractSpec(
        claim=lattice_algebra.LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice_, obs: 0.0,
            observable_requirements=("rate",),
        ),
        control=lattice_algebra.LatticeControlSpec(
            objective="holder_max",
            exercise_steps=tuple(sorted(signed_swap_values)),
            exercise_value_fn=exercise_value,
        ),
        metadata={"signed_swap_values": signed_swap_values},
    )


def price_bermudan_swaption_on_lattice(
    lattice: RecombiningLattice,
    *,
    spec: BermudanSwaptionSpecLike | None = None,
    settlement: date | None = None,
    contract_spec: lattice_algebra.LatticeContractSpec | None = None,
) -> float:
    """Price a Bermudan swaption on a pre-built lattice.

    This intentionally mirrors the checked-in T04 Bermudan tree reference: the
    option is a Bermudan claim on node-wise swap values, and the underlying swap
    keeps the same quantized tenor measured from the first exercise date.
    """
    if contract_spec is None:
        if spec is None or settlement is None:
            raise ValueError("Provide either contract_spec or both spec and settlement")
        contract_spec = compile_bermudan_swaption_contract_spec(
            lattice,
            spec=spec,
            settlement=settlement,
        )
    return float(price_on_lattice(lattice, contract_spec))


def price_bermudan_swaption_tree(
    market_state: BermudanSwaptionTreeMarketStateLike,
    spec: BermudanSwaptionSpecLike,
    *,
    model: str = "hull_white",
    mean_reversion: float | None = None,
    sigma: float | None = None,
    n_steps: int | None = None,
) -> float:
    """Build the requested tree and return the Bermudan swaption PV."""
    resolved = resolve_bermudan_swaption_tree_inputs(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        sigma=sigma,
        n_steps=n_steps,
    )
    lattice = build_bermudan_swaption_lattice(
        market_state,
        spec,
        model=model,
        mean_reversion=mean_reversion,
        sigma=sigma,
        n_steps=resolved.n_steps,
    )
    return price_bermudan_swaption_on_lattice(
        lattice,
        spec=spec,
        settlement=resolved.settlement,
    )


def _fixed_leg_bond_values(
    lattice: RecombiningLattice,
    *,
    coupon_by_step: dict[int, float],
    principal: float,
    principal_step: int,
) -> dict[int, object]:
    """Return fixed-leg bond values at every lattice step."""
    np = get_numpy()
    values_by_step: dict[int, object] = {}
    terminal_coupon = float(coupon_by_step.get(principal_step, 0.0))
    values = np.full(lattice.n_nodes(principal_step), float(principal) + terminal_coupon)
    values_by_step[principal_step] = values

    for step in range(principal_step - 1, -1, -1):
        n_nodes = lattice.n_nodes(step)
        new_values = np.zeros(n_nodes)
        coupon = float(coupon_by_step.get(step, 0.0))
        for node in range(n_nodes):
            discount = lattice.get_discount(step, node)
            probs = lattice.get_probabilities(step, node)
            children = lattice.child_indices(step, node)
            continuation = discount * sum(
                float(prob) * float(values[child])
                for prob, child in zip(probs, children)
            )
            new_values[node] = continuation + coupon
        values = new_values
        values_by_step[step] = values
    return values_by_step


def _compute_payer_swap_values_at_step(
    lattice: RecombiningLattice,
    *,
    exercise_step: int,
    swap_end_step: int,
    coupon_by_step: dict[int, float],
    principal: float,
):
    """Return payer swap values at one exercise step."""
    np = get_numpy()
    terminal_coupon = float(coupon_by_step.get(swap_end_step, 0.0))
    values = np.full(lattice.n_nodes(swap_end_step), float(principal) + terminal_coupon)

    for step in range(swap_end_step - 1, exercise_step - 1, -1):
        n_nodes = lattice.n_nodes(step)
        new_values = np.zeros(n_nodes)
        coupon = float(coupon_by_step.get(step, 0.0)) if step > exercise_step else 0.0
        for node in range(n_nodes):
            discount = lattice.get_discount(step, node)
            probs = lattice.get_probabilities(step, node)
            children = lattice.child_indices(step, node)
            continuation = discount * sum(
                float(prob) * float(values[child])
                for prob, child in zip(probs, children)
            )
            new_values[node] = continuation + coupon
        values = new_values
    return float(principal) - values


def _normalized_exercise_dates(exercise_dates: ContractTimeline | Iterable[date | str]) -> tuple[date, ...]:
    return normalize_explicit_dates(exercise_dates)


def _effective_exercise_dates(spec: BermudanSwaptionSpecLike, settlement: date) -> tuple[date, ...]:
    """Return the live Bermudan exercise dates after settlement and before swap end."""
    return tuple(
        exercise_date
        for exercise_date in _normalized_exercise_dates(spec.exercise_dates)
        if settlement < exercise_date < spec.swap_end
    )


def _settlement_date(market_state, spec) -> date:
    settlement = getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None)
    if settlement is not None:
        return settlement
    exercise_dates = _normalized_exercise_dates(spec.exercise_dates)
    if exercise_dates:
        return min(exercise_dates)
    return spec.swap_end


def _frequency_per_year(frequency) -> int:
    value = getattr(frequency, "value", frequency)
    return max(int(value), 1)


def _infer_schedule_frequency(schedule_dates: tuple[date, ...] | list[date], day_count) -> int:
    if len(schedule_dates) < 2:
        return 1
    spacings = [
        max(float(year_fraction(schedule_dates[i - 1], schedule_dates[i], day_count)), 1e-6)
        for i in range(1, len(schedule_dates))
    ]
    average_spacing = sum(spacings) / len(spacings)
    if average_spacing >= 0.75:
        return 1
    if average_spacing >= 0.375:
        return 2
    if average_spacing >= 0.15:
        return 4
    return 12


def _quantize_time(value: float, *, frequency: int) -> float:
    return round(float(value) * float(frequency)) / float(frequency)


__all__ = [
    "BermudanSwaptionTreeSpec",
    "ResolvedBermudanSwaptionTreeInputs",
    "build_bermudan_swaption_coupon_map",
    "build_bermudan_swaption_exercise_policy",
    "build_bermudan_swaption_lattice",
    "compile_bermudan_swaption_contract_spec",
    "price_bermudan_swaption_on_lattice",
    "price_bermudan_swaption_tree",
    "resolve_bermudan_swaption_tree_inputs",
]
