"""Simulation and Monte Carlo visitors for route-free execution IR artifacts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as raw_np

from trellis.agent.family_lowering_ir import (
    ConditionalValuationSpec,
    ControlProgramIR,
    EventProgramIR,
    FactorStateSimulationIR,
    MCMeasureSpec,
    ObservationProgramSpec,
    SemanticEventSpec,
    SemanticEventTimeSpec,
    SimulatedMarketProjectionSpec,
    SimulationFactorSpec,
    SimulationProcessBundleSpec,
    SimulationStateSpec,
    StateTransitionSpec,
)
from trellis.book import FutureValueCube, FutureValueCubeMetadata
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.execution.admission import admit_execution_capabilities
from trellis.execution.ir import ContractExecutionIR, CouponLegExecution
from trellis.instruments.swap import SwapSpec
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.event_state import event_step_indices
from trellis.models.monte_carlo.lsm import longstaff_schwartz_multistate_result
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.trees.product_lattice import (
    build_product_spot_lattice_2d,
    rollback_product_lattice_2d,
)


def compile_swap_spec_from_execution_ir(ir: ContractExecutionIR) -> SwapSpec:
    """Compile a bounded fixed-float swap execution IR back into `SwapSpec`."""
    _require_fixed_float_swap_execution_ir(ir)
    coupon_legs = tuple(
        obligation
        for obligation in ir.obligations
        if isinstance(obligation, CouponLegExecution)
    )
    if len(coupon_legs) != 2:
        raise ValueError(
            "fixed_float_swap future-value bridge requires exactly two coupon legs"
        )

    fixed_leg = _select_coupon_leg(coupon_legs, formula_kind="fixed")
    floating_leg = _select_coupon_leg(coupon_legs, formula_kind="floating")
    fixed_metadata = _metadata_dict(fixed_leg.metadata)
    floating_metadata = _metadata_dict(floating_leg.metadata)

    fixed_notional = float(fixed_metadata.get("notional", 0.0))
    floating_notional = float(floating_metadata.get("notional", 0.0))
    if abs(fixed_notional - floating_notional) > 1e-12:
        raise ValueError(
            "fixed_float_swap future-value bridge requires matching constant notionals"
        )

    fixed_periods = tuple(fixed_metadata.get("periods") or ())
    floating_periods = tuple(floating_metadata.get("periods") or ())
    if not fixed_periods or not floating_periods:
        raise ValueError(
            "fixed_float_swap future-value bridge requires explicit fixed and floating periods"
        )

    fixed_start = _coerce_date(fixed_periods[0][0], "fixed_start_date")
    fixed_end = _coerce_date(fixed_periods[-1][1], "fixed_end_date")
    floating_start = _coerce_date(floating_periods[0][0], "floating_start_date")
    floating_end = _coerce_date(floating_periods[-1][1], "floating_end_date")
    if fixed_start != floating_start or fixed_end != floating_end:
        raise ValueError(
            "fixed_float_swap future-value bridge requires aligned fixed and floating schedules"
        )

    fixed_direction = str(fixed_metadata.get("direction") or "").strip().lower()
    if fixed_direction not in {"pay", "receive"}:
        raise ValueError(
            "fixed_float_swap future-value bridge requires a fixed-leg direction"
        )

    rate_index = str(floating_metadata.get("rate_index") or "").strip()
    if not rate_index:
        raise ValueError(
            "fixed_float_swap future-value bridge requires a floating rate index"
        )

    return SwapSpec(
        notional=fixed_notional,
        fixed_rate=float(fixed_metadata.get("fixed_rate", 0.0)),
        start_date=fixed_start,
        end_date=fixed_end,
        fixed_frequency=_frequency_enum(fixed_metadata.get("payment_frequency")),
        float_frequency=_frequency_enum(floating_metadata.get("payment_frequency")),
        fixed_day_count=_day_count_enum(fixed_metadata.get("day_count")),
        float_day_count=_day_count_enum(floating_metadata.get("day_count")),
        rate_index=rate_index,
        is_payer=fixed_direction == "pay",
    )


def compile_factor_state_simulation_ir_from_execution_ir(
    ir: ContractExecutionIR,
) -> FactorStateSimulationIR:
    """Project a bounded execution IR onto the reusable simulation substrate."""
    spec = compile_swap_spec_from_execution_ir(ir)
    observables = tuple(sorted(ir.observables, key=lambda item: item.observable_id))
    fixing_transitions = tuple(
        StateTransitionSpec(
            transition_kind="record_fixing",
            observable_id=_floating_observable_id(ir),
            state_binding=_event_state_binding(event),
            phase=str(event.phase or "fixing"),
        )
        for event in ir.event_plan.events
        if event.event_kind == "fixing"
    )

    return FactorStateSimulationIR(
        route_id="simulation_substrate",
        route_family="simulation",
        engine_family="simulation",
        product_instrument="interest_rate_swap",
        payoff_family="conditional_valuation",
        required_input_ids=tuple(sorted(ir.requirement_hints.market_inputs)),
        market_data_requirements=frozenset(ir.requirement_hints.market_inputs),
        timeline_roles=frozenset(ir.requirement_hints.timeline_roles),
        requested_outputs=("future_value_cube",),
        reporting_currency=_currency_from_settlement(ir),
        state_spec=SimulationStateSpec(
            dimension=1,
            state_layout="scalar",
            state_tags=("markov", "factor_state", "short_rate"),
            coordinate_chart="physical",
            factors=(
                SimulationFactorSpec(
                    factor_name="short_rate",
                    factor_role="discount_curve_state",
                    units="rate",
                ),
            ),
        ),
        process_spec=SimulationProcessBundleSpec(
            process_family="hull_white_1f",
            factor_dimension=1,
            simulation_method="exact",
            calibration_symbol="resolve_hull_white_monte_carlo_process_inputs",
            process_tags=("rates", "future_value"),
        ),
        projection_spec=SimulatedMarketProjectionSpec(
            projection_family="hull_white_1f_rate_projection",
            output_kind="projected_market_view",
            target_market="interest_rate_curves",
            state_fields=("short_rate",),
        ),
        observation_program=ObservationProgramSpec(
            observable_ids=tuple(observable.observable_id for observable in observables),
            observable_kinds=tuple(observable.observable_kind for observable in observables),
            state_transitions=fixing_transitions,
            phase_sequence=("observation", "fixing", "payment", "valuation"),
            terminal_value_symbol="clean_future_value",
        ),
        conditional_valuation=ConditionalValuationSpec(
            valuation_style="exact",
            model_family="pathwise_projection",
            basis_family="swap_cashflow_discounting",
            supports_exact=True,
            requires_train_eval_split=False,
        ),
        event_program=_compile_simulation_event_program(ir, spec),
        control_program=ControlProgramIR(
            control_style="identity",
            controller_role="none",
        ),
        measure_spec=MCMeasureSpec(
            measure_family="risk_neutral",
            numeraire_binding="discount_curve",
        ),
        helper_symbol="price_interest_rate_swap_future_value_cube",
        market_mapping="execution_ir_to_hull_white_swap_future_value",
    )


def build_future_value_cube_from_execution_ir(
    ir: ContractExecutionIR,
    market_state: MarketState,
    *,
    position_name: str | None = None,
    n_paths: int = 10_000,
    n_steps: int = 120,
    seed: int | None = None,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> FutureValueCube:
    """Build a bounded future-value cube directly from execution IR."""
    family_ir = compile_factor_state_simulation_ir_from_execution_ir(ir)
    spec = compile_swap_spec_from_execution_ir(ir)
    resolved_name = str(
        position_name
        or ir.source_track.semantic_id
        or ir.source_track.product_family
        or "execution_position"
    ).strip()
    if not resolved_name:
        resolved_name = "execution_position"

    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_future_value_cube,
    )

    cube = price_interest_rate_swap_future_value_cube(
        name=resolved_name,
        spec=spec,
        market_state=market_state,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    compute_plan = cube.compute_plan
    compute_plan.update(
        {
            "bridge_family": "execution_ir",
            "execution_source_kind": ir.source_track.source_kind,
            "execution_product_family": ir.source_track.product_family,
            "execution_semantic_id": ir.source_track.semantic_id,
            "simulation_helper_symbol": family_ir.helper_symbol,
        }
    )
    position_provenance = cube.position_provenance
    payload = dict(position_provenance.get(resolved_name, {}))
    payload.update(
        {
            "execution_source_kind": ir.source_track.source_kind,
            "execution_product_family": ir.source_track.product_family,
            "execution_semantic_id": ir.source_track.semantic_id,
            "bridge_family": "execution_ir",
        }
    )
    position_provenance[resolved_name] = payload

    return FutureValueCube(
        cube.values,
        position_names=cube.position_names,
        observation_times=cube.observation_times,
        observation_dates=cube.observation_dates,
        metadata=FutureValueCubeMetadata(
            measure=cube.measure,
            numeraire=cube.numeraire,
            value_semantics=cube.value_semantics,
            phase_semantics=cube.phase_semantics,
            state_names=cube.state_names,
            process_family=cube.process_family,
            compute_plan=compute_plan,
            position_provenance=position_provenance,
        ),
    )


@dataclass(frozen=True)
class BermudanBestOfBasketMCInputs:
    """Explicit market inputs consumed by the Bermudan basket MC visitor."""

    valuation_date: date
    spot_values: Mapping[str, float] | tuple[tuple[str, float], ...]
    volatilities: Mapping[str, float] | tuple[tuple[str, float], ...]
    correlation_matrix: Sequence[Sequence[float]]
    risk_free_rate: float
    carry_rates: Mapping[str, float] | tuple[tuple[str, float], ...] = ()
    day_count: DayCountConvention = DayCountConvention.ACT_365

    def __post_init__(self) -> None:
        if not isinstance(self.valuation_date, date):
            raise TypeError("valuation_date must be a datetime.date")
        object.__setattr__(self, "spot_values", _float_mapping(self.spot_values))
        object.__setattr__(self, "volatilities", _float_mapping(self.volatilities))
        object.__setattr__(self, "carry_rates", _float_mapping(self.carry_rates))
        object.__setattr__(
            self,
            "correlation_matrix",
            _float_matrix(self.correlation_matrix),
        )
        object.__setattr__(self, "risk_free_rate", float(self.risk_free_rate))


@dataclass(frozen=True)
class BermudanBestOfBasketMCControls:
    """Simulation controls for the route-free Bermudan basket visitor."""

    n_paths: int = 50_000
    n_steps: int = 252
    seed: int | None = 42
    method: str = "exact"

    def __post_init__(self) -> None:
        n_paths = int(self.n_paths)
        n_steps = int(self.n_steps)
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        method = str(self.method or "exact").strip().lower()
        if method not in {"exact", "euler"}:
            raise ValueError("method must be 'exact' or 'euler'")
        object.__setattr__(self, "n_paths", n_paths)
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        object.__setattr__(self, "method", method)


@dataclass(frozen=True)
class BermudanBestOfBasketLatticeControls:
    """Product-state lattice controls for the Bermudan basket visitor."""

    n_steps: int = 96

    def __post_init__(self) -> None:
        n_steps = int(self.n_steps)
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        object.__setattr__(self, "n_steps", n_steps)


@dataclass(frozen=True)
class BermudanBestOfBasketMCResult:
    """Audited MC visitor result for a Bermudan best-of basket execution IR."""

    price: float
    lower_bound: float
    currency: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", float(self.price))
        object.__setattr__(self, "lower_bound", float(self.lower_bound))
        object.__setattr__(self, "currency", str(self.currency or "").strip().upper())
        object.__setattr__(self, "diagnostics", _mapping_proxy(self.diagnostics))
        object.__setattr__(self, "provenance", _mapping_proxy(self.provenance))

    def with_provenance(
        self,
        values: Mapping[str, object],
    ) -> "BermudanBestOfBasketMCResult":
        """Return a copy with additional provenance fields."""
        provenance = dict(self.provenance)
        provenance.update(dict(values))
        return replace(self, provenance=provenance)


@dataclass(frozen=True)
class BermudanBestOfBasketLatticeResult:
    """Audited lattice visitor result for a Bermudan best-of basket IR."""

    price: float
    currency: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", float(self.price))
        object.__setattr__(self, "currency", str(self.currency or "").strip().upper())
        object.__setattr__(self, "diagnostics", _mapping_proxy(self.diagnostics))
        object.__setattr__(self, "provenance", _mapping_proxy(self.provenance))

    def with_provenance(
        self,
        values: Mapping[str, object],
    ) -> "BermudanBestOfBasketLatticeResult":
        """Return a copy with additional provenance fields."""
        provenance = dict(self.provenance)
        provenance.update(dict(values))
        return replace(self, provenance=provenance)


def price_bermudan_best_of_basket_monte_carlo(
    ir: ContractExecutionIR,
    inputs: BermudanBestOfBasketMCInputs,
    *,
    controls: BermudanBestOfBasketMCControls | None = None,
) -> BermudanBestOfBasketMCResult:
    """Price a Bermudan best-of basket by visiting route-free execution IR."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    controls = controls or BermudanBestOfBasketMCControls()
    admission = admit_execution_capabilities(ir, method="monte_carlo")
    if not admission.admitted:
        blocker_ids = tuple(blocker.blocker_id for blocker in admission.blockers)
        raise ValueError(
            "execution IR is not admitted for Monte Carlo; "
            f"blockers={blocker_ids!r}"
        )

    source_metadata = _metadata_dict(ir.source_track.source_metadata)
    underliers = _ordered_underliers(ir)
    strike_metadata = source_metadata.get("strike")
    if strike_metadata in {None, ""}:
        strike_metadata = _strike_from_settlement(ir)
    strike = float(strike_metadata)
    notional = float(source_metadata.get("notional", 1.0))
    currency = str(source_metadata.get("currency", _currency_from_settlement(ir)) or "USD")
    expiry_date = _expiry_date(ir)
    exercise_dates = _exercise_dates(ir) or (expiry_date,)
    observation_dates = _observation_dates(ir)

    spot_vector = _aligned_vector(inputs.spot_values, underliers, "spot_values")
    vol_vector = _aligned_vector(inputs.volatilities, underliers, "volatilities")
    carry_vector = tuple(
        float(inputs.carry_rates.get(underlier, 0.0))
        for underlier in underliers
    )
    correlation_matrix = _validated_correlation_matrix(
        inputs.correlation_matrix,
        len(underliers),
    )
    maturity = year_fraction(inputs.valuation_date, expiry_date, inputs.day_count)
    if maturity <= 0.0:
        intrinsic = notional * max(max(spot_vector) - strike, 0.0)
        return BermudanBestOfBasketMCResult(
            price=intrinsic,
            lower_bound=intrinsic,
            currency=currency,
            diagnostics={
                "policy_class": "intrinsic",
                "exercise_dates_count": 0,
                "exercised_paths_fraction": 0.0,
                "regression_failures": 0,
            },
            provenance=_provenance(
                ir,
                controls,
                underliers=underliers,
                observation_dates=observation_dates,
                exercise_dates=exercise_dates,
                expiry_date=expiry_date,
                maturity=maturity,
                admission=admission,
                policy_class="intrinsic",
            ),
        )

    process = CorrelatedGBM(
        mu=[inputs.risk_free_rate for _ in underliers],
        sigma=vol_vector,
        corr=correlation_matrix,
        dividend_yield=carry_vector,
    )
    engine = MonteCarloEngine(
        process,
        n_paths=controls.n_paths,
        n_steps=controls.n_steps,
        seed=controls.seed,
        method=controls.method,
    )
    paths = engine.simulate(spot_vector, maturity)
    exercise_times = tuple(
        year_fraction(inputs.valuation_date, exercise_date, inputs.day_count)
        for exercise_date in exercise_dates
        if exercise_date > inputs.valuation_date
    )
    exercise_steps = event_step_indices(exercise_times, maturity, controls.n_steps)

    def payoff(state):
        values = raw_np.asarray(state, dtype=float)
        if values.ndim == 1:
            best = values
        else:
            best = raw_np.max(values, axis=1)
        return raw_np.maximum(best - strike, 0.0)

    policy_result = longstaff_schwartz_multistate_result(
        paths,
        list(exercise_steps),
        payoff,
        inputs.risk_free_rate,
        maturity / controls.n_steps,
    )
    policy_diagnostics = _policy_diagnostics(policy_result.diagnostics)
    price = notional * float(policy_result.price_lower)
    policy_class = str(policy_diagnostics.get("policy_class") or policy_result.policy_class)
    return BermudanBestOfBasketMCResult(
        price=price,
        lower_bound=price,
        currency=currency,
        diagnostics=policy_diagnostics,
        provenance=_provenance(
            ir,
            controls,
            underliers=underliers,
            observation_dates=observation_dates,
            exercise_dates=exercise_dates,
            expiry_date=expiry_date,
            maturity=maturity,
            admission=admission,
            policy_class=policy_class,
        ),
    )


def price_bermudan_best_of_basket_lattice(
    ir: ContractExecutionIR,
    inputs: BermudanBestOfBasketMCInputs,
    *,
    controls: BermudanBestOfBasketLatticeControls | None = None,
) -> BermudanBestOfBasketLatticeResult:
    """Price a two-underlier Bermudan best-of basket on a product-state grid."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    controls = controls or BermudanBestOfBasketLatticeControls()
    admission = admit_execution_capabilities(
        ir,
        method="lattice",
        available_primitives=("multi_asset_bermudan_state_grid",),
    )
    if not admission.admitted:
        blocker_ids = tuple(blocker.blocker_id for blocker in admission.blockers)
        raise ValueError(
            "execution IR is not admitted for lattice; "
            f"blockers={blocker_ids!r}"
        )

    source_metadata = _metadata_dict(ir.source_track.source_metadata)
    underliers = _ordered_underliers(ir)
    if len(underliers) != 2:
        raise ValueError("product-state lattice visitor currently requires exactly two underliers")
    strike_metadata = source_metadata.get("strike")
    if strike_metadata in {None, ""}:
        strike_metadata = _strike_from_settlement(ir)
    strike = float(strike_metadata)
    notional = float(source_metadata.get("notional", 1.0))
    currency = str(source_metadata.get("currency", _currency_from_settlement(ir)) or "USD")
    expiry_date = _expiry_date(ir)
    exercise_dates = _exercise_dates(ir) or (expiry_date,)
    observation_dates = _observation_dates(ir)

    spot_vector = _aligned_vector(inputs.spot_values, underliers, "spot_values")
    vol_vector = _aligned_vector(inputs.volatilities, underliers, "volatilities")
    carry_vector = tuple(
        float(inputs.carry_rates.get(underlier, 0.0))
        for underlier in underliers
    )
    correlation_matrix = _validated_correlation_matrix(
        inputs.correlation_matrix,
        len(underliers),
    )
    maturity = year_fraction(inputs.valuation_date, expiry_date, inputs.day_count)
    if maturity <= 0.0:
        intrinsic = notional * max(max(spot_vector) - strike, 0.0)
        return BermudanBestOfBasketLatticeResult(
            price=intrinsic,
            currency=currency,
            diagnostics={
                "rollback_policy": "intrinsic",
                "exercise_steps_count": 0.0,
                "exercised_nodes": 0.0,
            },
            provenance=_lattice_provenance(
                ir,
                controls,
                underliers=underliers,
                observation_dates=observation_dates,
                exercise_dates=exercise_dates,
                expiry_date=expiry_date,
                maturity=maturity,
                admission=admission,
            ),
        )

    exercise_times = tuple(
        year_fraction(inputs.valuation_date, exercise_date, inputs.day_count)
        for exercise_date in exercise_dates
        if exercise_date > inputs.valuation_date
    )
    exercise_steps = event_step_indices(exercise_times, maturity, controls.n_steps)
    lattice, build_diagnostics = build_product_spot_lattice_2d(
        spots=(spot_vector[0], spot_vector[1]),
        rate=inputs.risk_free_rate,
        sigmas=(vol_vector[0], vol_vector[1]),
        maturity=maturity,
        n_steps=controls.n_steps,
        correlation=float(correlation_matrix[0][1]),
        carry=(carry_vector[0], carry_vector[1]),
    )

    def best_of_call(_step, _node, _lattice, state):
        return max(max(float(state[0]), float(state[1])) - strike, 0.0)

    lattice_price, rollback_diagnostics = rollback_product_lattice_2d(
        lattice,
        terminal_payoff=best_of_call,
        exercise_value=best_of_call,
        exercise_steps=exercise_steps,
    )
    diagnostics = {
        **build_diagnostics,
        **rollback_diagnostics,
        "exercise_steps": exercise_steps,
    }
    return BermudanBestOfBasketLatticeResult(
        price=notional * lattice_price,
        currency=currency,
        diagnostics=diagnostics,
        provenance=_lattice_provenance(
            ir,
            controls,
            underliers=underliers,
            observation_dates=observation_dates,
            exercise_dates=exercise_dates,
            expiry_date=expiry_date,
            maturity=maturity,
            admission=admission,
        ),
    )


def _float_mapping(values: object) -> Mapping[str, float]:
    if isinstance(values, Mapping):
        items = values.items()
    else:
        items = values or ()
    normalized = {
        str(key).strip(): float(value)
        for key, value in items
        if str(key).strip()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


def _float_matrix(values: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(cell) for cell in row) for row in values)


def _mapping_proxy(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(sorted(dict(values or {}).items())))


def _require_fixed_float_swap_execution_ir(ir: ContractExecutionIR) -> None:
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    if ir.source_track.source_kind != "static_leg_contract_ir":
        raise ValueError(
            "future-value bridge currently requires a static_leg_contract_ir execution IR"
        )
    if ir.source_track.product_family != "fixed_float_swap":
        raise ValueError(
            "future-value bridge currently admits only fixed_float_swap execution IR"
        )
    if ir.execution_metadata.unsupported_reasons:
        raise ValueError(
            "Cannot bridge unsupported execution IR: "
            f"{ir.execution_metadata.unsupported_reasons!r}"
        )


def _metadata_dict(items: object) -> dict[str, object]:
    try:
        return dict(items or ())
    except (TypeError, ValueError):
        return {}


def _select_coupon_leg(
    coupon_legs: tuple[CouponLegExecution, ...],
    *,
    formula_kind: str,
) -> CouponLegExecution:
    matches = [
        leg
        for leg in coupon_legs
        if str(_metadata_dict(leg.metadata).get("formula_kind") or "").strip().lower()
        == formula_kind
    ]
    if len(matches) != 1:
        raise ValueError(
            "fixed_float_swap future-value bridge requires exactly one "
            f"{formula_kind} coupon leg"
        )
    return matches[0]


def _frequency_enum(value: object) -> Frequency:
    normalized = str(value or "").strip().lower().replace("-", "_")
    mapping = {
        "annual": Frequency.ANNUAL,
        "semiannual": Frequency.SEMI_ANNUAL,
        "quarterly": Frequency.QUARTERLY,
        "monthly": Frequency.MONTHLY,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported future-value bridge payment frequency {value!r}"
        ) from exc


def _day_count_enum(value: object) -> DayCountConvention:
    normalized = str(value or "").strip().upper().replace("_", "/")
    mapping = {
        "ACT/360": DayCountConvention.ACT_360,
        "ACT/365": DayCountConvention.ACT_365,
        "ACT/ACT": DayCountConvention.ACT_ACT,
        "30/360": DayCountConvention.THIRTY_360,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported future-value bridge day count {value!r}"
        ) from exc


def _floating_observable_id(ir: ContractExecutionIR) -> str:
    for observable in ir.observables:
        if observable.observable_kind == "forward_rate":
            return observable.observable_id
    raise ValueError(
        "fixed_float_swap future-value bridge requires a forward-rate observable"
    )


def _event_state_binding(event) -> str:
    metadata = _metadata_dict(event.metadata)
    leg_id = str(metadata.get("leg_id") or "").strip()
    period_index = int(metadata.get("period_index", 0))
    return f"{leg_id}:{period_index}"


def _compile_simulation_event_program(
    ir: ContractExecutionIR,
    spec: SwapSpec,
) -> EventProgramIR:
    by_date: "OrderedDict[date, dict[str, object]]" = OrderedDict()
    start_date = spec.start_date
    start_bucket = by_date.setdefault(
        start_date,
        {
            "schedule_roles": [],
            "phase_sequence": [],
            "events": [],
        },
    )
    _append_unique(start_bucket["schedule_roles"], "observation_dates")
    _append_unique(start_bucket["phase_sequence"], "observation")
    start_bucket["events"].append(
        SemanticEventSpec(
            event_name="observation:start",
            event_kind="observation",
            schedule_role="observation_dates",
            phase="observation",
            value_semantics="future_value_anchor",
            state_binding="short_rate",
        )
    )

    phase_rank = {
        phase: index
        for index, phase in enumerate(ir.event_plan.phase_order)
    }
    ordered_events = sorted(
        ir.event_plan.events,
        key=lambda event: (
            _coerce_date(event.event_date, "event_date"),
            phase_rank.get(str(event.phase or ""), len(phase_rank)),
            event.event_id,
        ),
    )
    for event in ordered_events:
        event_date = _coerce_date(event.event_date, "event_date")
        bucket = by_date.setdefault(
            event_date,
            {
                "schedule_roles": [],
                "phase_sequence": [],
                "events": [],
            },
        )
        _append_unique(bucket["schedule_roles"], str(event.schedule_role or "").strip())
        _append_unique(bucket["phase_sequence"], str(event.phase or "").strip())
        bucket["events"].append(
            SemanticEventSpec(
                event_name=event.event_id,
                event_kind=event.event_kind,
                schedule_role=event.schedule_role,
                phase=event.phase,
                value_semantics=_value_semantics(event.event_kind),
                state_binding=_event_state_binding(event),
            )
        )

    timeline = []
    for event_date, payload in by_date.items():
        timeline.append(
            SemanticEventTimeSpec(
                event_date=event_date.isoformat(),
                schedule_roles=tuple(
                    item for item in payload["schedule_roles"] if item
                ),
                phase_sequence=tuple(
                    item for item in payload["phase_sequence"] if item
                ),
                events=tuple(payload["events"]),
            )
        )
    return EventProgramIR(timeline=tuple(timeline))


def _append_unique(values: list[str], value: str) -> None:
    text = str(value or "").strip()
    if text and text not in values:
        values.append(text)


def _value_semantics(event_kind: str) -> str:
    mapping = {
        "observation": "future_value_anchor",
        "fixing": "projected_forward_reset",
        "payment": "scheduled_cashflow",
    }
    return mapping.get(str(event_kind or "").strip().lower(), "execution_event")


def _ordered_underliers(ir: ContractExecutionIR) -> tuple[str, ...]:
    candidates: list[tuple[int, str]] = []
    for observable in ir.observables:
        if observable.observable_kind != "spot":
            continue
        metadata = _metadata_dict(observable.metadata)
        name = str(
            metadata.get("underlier")
            or _suffix(observable.source_ref)
            or _suffix(observable.observable_id)
        ).strip()
        if not name:
            continue
        index = int(metadata.get("vector_index", len(candidates)))
        candidates.append((index, name))
    underliers = tuple(name for _, name in sorted(candidates))
    if len(underliers) < 2:
        raise ValueError(
            "Bermudan best-of basket MC visitor requires at least two spot underliers"
        )
    if len(set(underliers)) != len(underliers):
        raise ValueError("Bermudan best-of basket MC visitor requires unique underlier names")
    return underliers


def _suffix(value: object) -> str:
    text = str(value or "").strip()
    return text.rsplit(":", 1)[-1] if text else ""


def _aligned_vector(
    values: Mapping[str, float],
    underliers: tuple[str, ...],
    label: str,
) -> tuple[float, ...]:
    missing = tuple(underlier for underlier in underliers if underlier not in values)
    if missing:
        raise ValueError(f"{label} missing values for {missing!r}")
    return tuple(float(values[underlier]) for underlier in underliers)


def _validated_correlation_matrix(
    matrix: tuple[tuple[float, ...], ...],
    n_assets: int,
) -> tuple[tuple[float, ...], ...]:
    if len(matrix) != n_assets or any(len(row) != n_assets for row in matrix):
        raise ValueError(
            f"correlation_matrix must have shape ({n_assets}, {n_assets})"
        )
    arr = raw_np.asarray(matrix, dtype=float)
    if not raw_np.allclose(arr, arr.T, atol=1e-12, rtol=0.0):
        raise ValueError("correlation_matrix must be symmetric")
    if not raw_np.allclose(raw_np.diag(arr), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("correlation_matrix diagonal entries must equal 1")
    return matrix


def _coerce_date(value: object, label: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return date.fromisoformat(text)


def _event_dates(ir: ContractExecutionIR, event_kind: str) -> tuple[date, ...]:
    return tuple(
        _coerce_date(event.event_date, event_kind)
        for event in ir.event_plan.events
        if event.event_kind == event_kind
    )


def _expiry_date(ir: ContractExecutionIR) -> date:
    settlement_dates = _event_dates(ir, "settlement")
    if settlement_dates:
        return settlement_dates[-1]
    raise ValueError("Bermudan best-of basket MC visitor requires a settlement event")


def _exercise_dates(ir: ContractExecutionIR) -> tuple[date, ...]:
    return _event_dates(ir, "decision")


def _observation_dates(ir: ContractExecutionIR) -> tuple[date, ...]:
    return _event_dates(ir, "observation")


def _currency_from_settlement(ir: ContractExecutionIR) -> str:
    for step in ir.settlement_program.steps:
        if step.currency:
            return step.currency
    return "USD"


def _strike_from_settlement(ir: ContractExecutionIR) -> float:
    for step in ir.settlement_program.steps:
        text = step.expression
        if " - " in text:
            try:
                return float(text.split(" - ", 1)[1].split(",", 1)[0])
            except (ValueError, IndexError):
                continue
    raise ValueError("Bermudan best-of basket MC visitor requires strike metadata")


def _policy_diagnostics(diagnostics: object) -> Mapping[str, object]:
    if diagnostics is None:
        return {}
    return {
        "policy_class": getattr(diagnostics, "policy_class", ""),
        "exercise_dates_count": int(getattr(diagnostics, "exercise_dates_count", 0)),
        "exercised_paths_fraction": float(
            getattr(diagnostics, "exercised_paths_fraction", 0.0)
        ),
        "regression_failures": int(getattr(diagnostics, "regression_failures", 0)),
        "estimator_name": getattr(diagnostics, "estimator_name", None),
    }


def _provenance(
    ir: ContractExecutionIR,
    controls: BermudanBestOfBasketMCControls,
    *,
    underliers: tuple[str, ...],
    observation_dates: tuple[date, ...],
    exercise_dates: tuple[date, ...],
    expiry_date: date,
    maturity: float,
    admission: object,
    policy_class: str,
) -> Mapping[str, object]:
    return {
        "source_semantic_id": ir.source_track.semantic_id,
        "source_ref": ir.source_track.source_ref,
        "product_family": ir.source_track.product_family,
        "method": "monte_carlo",
        "engine_family": "monte_carlo",
        "process_family": "correlated_gbm",
        "policy_class": policy_class,
        "pricing_authority": "execution_ir_visitor",
        "underliers": underliers,
        "observation_dates": observation_dates,
        "exercise_dates": exercise_dates,
        "expiry_date": expiry_date,
        "maturity": float(maturity),
        "n_paths": controls.n_paths,
        "n_steps": controls.n_steps,
        "seed": controls.seed,
        "simulation_method": controls.method,
        "route_ids": (),
        "required_capabilities": tuple(getattr(admission, "required_capabilities", ())),
        "matched_capabilities": tuple(getattr(admission, "matched_capabilities", ())),
    }


def _lattice_provenance(
    ir: ContractExecutionIR,
    controls: BermudanBestOfBasketLatticeControls,
    *,
    underliers: tuple[str, ...],
    observation_dates: tuple[date, ...],
    exercise_dates: tuple[date, ...],
    expiry_date: date,
    maturity: float,
    admission: object,
) -> Mapping[str, object]:
    return {
        "source_semantic_id": ir.source_track.semantic_id,
        "source_ref": ir.source_track.source_ref,
        "product_family": ir.source_track.product_family,
        "method": "lattice",
        "engine_family": "lattice",
        "primitive": "multi_asset_bermudan_state_grid",
        "pricing_authority": "execution_ir_visitor",
        "underliers": underliers,
        "observation_dates": observation_dates,
        "exercise_dates": exercise_dates,
        "expiry_date": expiry_date,
        "maturity": float(maturity),
        "n_steps": controls.n_steps,
        "route_ids": (),
        "required_capabilities": tuple(getattr(admission, "required_capabilities", ())),
        "matched_capabilities": tuple(getattr(admission, "matched_capabilities", ())),
    }


__all__ = [
    "BermudanBestOfBasketLatticeControls",
    "BermudanBestOfBasketLatticeResult",
    "BermudanBestOfBasketMCControls",
    "BermudanBestOfBasketMCInputs",
    "BermudanBestOfBasketMCResult",
    "build_future_value_cube_from_execution_ir",
    "compile_factor_state_simulation_ir_from_execution_ir",
    "compile_swap_spec_from_execution_ir",
    "price_bermudan_best_of_basket_lattice",
    "price_bermudan_best_of_basket_monte_carlo",
]
