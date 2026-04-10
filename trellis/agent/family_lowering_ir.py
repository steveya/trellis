"""Typed family-level lowering IRs for migrated semantic routes."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from trellis.core.types import TimelineRole


def _tuple_unique(values) -> tuple[str, ...]:
    """Return a deduplicated tuple of non-empty string values."""
    result: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class BaseFamilyLoweringIR:
    """Shared typed summary passed from semantic compilation to route lowering."""

    route_id: str
    route_family: str
    product_instrument: str
    payoff_family: str
    required_input_ids: tuple[str, ...] = ()
    market_data_requirements: frozenset[str] = frozenset()
    timeline_roles: frozenset[TimelineRole] = frozenset()
    requested_outputs: tuple[str, ...] = ()
    reporting_currency: str = ""


@dataclass(frozen=True)
class SemanticEventSpec:
    """One semantic event inside the universal event/control program."""

    event_name: str = ""
    event_kind: str = ""
    schedule_role: str = ""
    phase: str = ""
    value_semantics: str = ""
    state_binding: str = ""
    transform_kind: str = ""


@dataclass(frozen=True)
class SemanticEventTimeSpec:
    """One event-time bucket in the universal event program."""

    event_date: str = ""
    schedule_roles: tuple[str, ...] = ()
    phase_sequence: tuple[str, ...] = ()
    events: tuple[SemanticEventSpec, ...] = ()


@dataclass(frozen=True)
class EventProgramIR:
    """Universal semantic event timeline shared across family lowerings."""

    timeline: tuple[SemanticEventTimeSpec, ...] = ()

    @property
    def event_dates(self) -> tuple[str, ...]:
        """Return stable event dates across the semantic event program."""
        dates: list[str] = []
        for bucket in self.timeline:
            event_date = str(bucket.event_date or "").strip()
            if event_date and event_date not in dates:
                dates.append(event_date)
        return tuple(dates)

    @property
    def event_kinds(self) -> tuple[str, ...]:
        """Return stable event kinds across the semantic event program."""
        kinds: list[str] = []
        for bucket in self.timeline:
            for event in bucket.events:
                kind = str(event.event_kind or "").strip()
                if kind and kind not in kinds:
                    kinds.append(kind)
        return tuple(kinds)

    @property
    def transform_kinds(self) -> tuple[str, ...]:
        """Return stable transform kinds across the semantic event program."""
        kinds: list[str] = []
        for bucket in self.timeline:
            for event in bucket.events:
                kind = str(event.transform_kind or "").strip()
                if kind and kind not in kinds:
                    kinds.append(kind)
        return tuple(kinds)


@dataclass(frozen=True)
class ControlProgramIR:
    """Universal semantic control contract shared across family lowerings."""

    control_style: str = "identity"
    controller_role: str = "none"
    decision_phase: str = "decision"
    schedule_role: str = ""
    exercise_style: str = ""


@dataclass(frozen=True)
class AnalyticalBlack76IR(BaseFamilyLoweringIR):
    """Typed lowering payload for vanilla analytical Black76 routes."""

    option_type: str = "call"
    kernel_symbol: str = "black76_call"
    market_mapping: str = "spot_discount_vol_to_forward"


@dataclass(frozen=True)
class TransformStateSpec:
    """Typed state-space contract for one bounded transform-pricing problem."""

    state_variable: str = ""
    dimension: int = 1
    state_tags: tuple[str, ...] = ()
    coordinate_chart: str = "spot"


@dataclass(frozen=True)
class TransformCharacteristicSpec:
    """Typed characteristic-function contract for one transform pricing lane."""

    model_family: str = ""
    characteristic_family: str = ""
    supported_methods: tuple[str, ...] = ("fft", "cos")
    backend_capability: str = "raw_kernel_only"


@dataclass(frozen=True)
class TransformControlSpec:
    """Typed control contract for one bounded transform pricing lane."""

    control_style: str = "identity"
    controller_role: str = "holder"


@dataclass(frozen=True)
class TransformPricingIR(BaseFamilyLoweringIR):
    """Typed lowering payload for bounded transform-pricing routes."""

    state_spec: TransformStateSpec = field(default_factory=TransformStateSpec)
    characteristic_spec: TransformCharacteristicSpec = field(
        default_factory=TransformCharacteristicSpec
    )
    control_program: ControlProgramIR = field(default_factory=ControlProgramIR)
    control_spec: TransformControlSpec = field(default_factory=TransformControlSpec)
    terminal_payoff_kind: str = ""
    strike_semantics: str = ""
    quote_semantics: str = ""
    helper_symbol: str = ""
    market_mapping: str = ""
    compatibility_wrapper: str = ""


@dataclass(frozen=True)
class MCStateSpec:
    """Typed state-space contract for one event-aware Monte Carlo problem."""

    state_variable: str = ""
    dimension: int = 1
    state_tags: tuple[str, ...] = ()
    state_layout: str = "scalar"


@dataclass(frozen=True)
class MCProcessSpec:
    """Typed stochastic-process contract for one Monte Carlo problem."""

    process_family: str = ""
    simulation_scheme: str = ""
    process_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCEventSpec:
    """One typed event replayed against Monte Carlo state."""

    event_name: str = ""
    event_kind: str = ""
    schedule_role: str = ""
    phase: str = ""
    value_semantics: str = ""
    state_binding: str = ""


@dataclass(frozen=True)
class MCEventTimeSpec:
    """One event-time bucket compiled from typed semantic schedules."""

    event_date: str = ""
    schedule_roles: tuple[str, ...] = ()
    phase_sequence: tuple[str, ...] = ()
    events: tuple[MCEventSpec, ...] = ()


@dataclass(frozen=True)
class MCPathRequirementSpec:
    """Typed reduced-state requirement for one Monte Carlo family instance."""

    requirement_kind: str = "terminal_only"
    snapshot_schedule_role: str = ""
    reducer_kinds: tuple[str, ...] = ()
    replay_mode: str = "none"
    stored_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCPayoffReducerSpec:
    """Typed payoff-aggregation contract for one Monte Carlo family instance."""

    reducer_kind: str = ""
    reducer_symbol: str = ""
    output_semantics: str = ""
    event_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCControlSpec:
    """Typed control semantics for one Monte Carlo family instance."""

    control_style: str = "identity"
    controller_role: str = "none"


@dataclass(frozen=True)
class MCMeasureSpec:
    """Typed measure/numeraire contract for one Monte Carlo problem."""

    measure_family: str = "risk_neutral"
    numeraire_binding: str = "discount_curve"


@dataclass(frozen=True)
class MCCalibrationBindingSpec:
    """Typed quote/calibration prerequisites for one Monte Carlo route."""

    model_family: str = ""
    quote_family: str = ""
    required_parameters: tuple[str, ...] = ()
    requires_quote_normalization: bool = False


@dataclass(frozen=True)
class EventAwareMonteCarloIR(BaseFamilyLoweringIR):
    """Typed lowering payload for bounded event-aware Monte Carlo routes."""

    state_spec: MCStateSpec = field(default_factory=MCStateSpec)
    process_spec: MCProcessSpec = field(default_factory=MCProcessSpec)
    event_program: EventProgramIR = field(default_factory=EventProgramIR)
    control_program: ControlProgramIR = field(default_factory=ControlProgramIR)
    path_requirement_spec: MCPathRequirementSpec = field(default_factory=MCPathRequirementSpec)
    payoff_reducer_spec: MCPayoffReducerSpec = field(default_factory=MCPayoffReducerSpec)
    control_spec: MCControlSpec = field(default_factory=MCControlSpec)
    measure_spec: MCMeasureSpec = field(default_factory=MCMeasureSpec)
    calibration_binding: MCCalibrationBindingSpec | None = None
    event_timeline: tuple[MCEventTimeSpec, ...] = ()
    event_specs: tuple[MCEventSpec, ...] = ()
    helper_symbol: str = ""
    market_mapping: str = ""
    compatibility_wrapper: str = ""

    @property
    def event_kinds(self) -> tuple[str, ...]:
        """Return the stable event kinds encoded on this Monte Carlo IR."""
        kinds: list[str] = []
        for event in self.event_specs:
            kind = str(event.event_kind or "").strip()
            if kind and kind not in kinds:
                kinds.append(kind)
        for bucket in self.event_timeline:
            for event in bucket.events:
                kind = str(event.event_kind or "").strip()
                if kind and kind not in kinds:
                    kinds.append(kind)
        return tuple(kinds)

    @property
    def event_dates(self) -> tuple[str, ...]:
        """Return the stable event dates carried by the typed event timeline."""
        dates: list[str] = []
        for bucket in self.event_timeline:
            event_date = str(bucket.event_date or "").strip()
            if event_date and event_date not in dates:
                dates.append(event_date)
        return tuple(dates)

    @property
    def reducer_kinds(self) -> tuple[str, ...]:
        """Return the stable reducer kinds required by the Monte Carlo IR."""
        kinds: list[str] = []
        for kind in getattr(self.path_requirement_spec, "reducer_kinds", ()) or ():
            text = str(kind or "").strip()
            if text and text not in kinds:
                kinds.append(text)
        reducer_kind = str(getattr(self.payoff_reducer_spec, "reducer_kind", "") or "").strip()
        if reducer_kind and reducer_kind not in kinds:
            kinds.append(reducer_kind)
        return tuple(kinds)


@dataclass(frozen=True)
class PDEStateSpec:
    """Typed state-space contract for one event-aware PDE problem."""

    state_variable: str = ""
    dimension: int = 1
    state_tags: tuple[str, ...] = ()
    coordinate_chart: str = "physical"


@dataclass(frozen=True)
class PDEOperatorSpec:
    """Typed generator/solver contract for one event-aware PDE problem."""

    operator_family: str = ""
    solver_family: str = "theta_method"
    stepping_scheme: str = ""


@dataclass(frozen=True)
class PDEEventTransformSpec:
    """One typed event transform applied on the rollback timeline."""

    transform_kind: str = ""
    schedule_role: str = ""
    value_semantics: str = ""
    state_mapping_symbol: str = ""


@dataclass(frozen=True)
class PDEEventTimeSpec:
    """One event-time bucket compiled from typed semantic schedules."""

    event_date: str = ""
    schedule_roles: tuple[str, ...] = ()
    phase_sequence: tuple[str, ...] = ()
    transforms: tuple[PDEEventTransformSpec, ...] = ()


@dataclass(frozen=True)
class PDEControlSpec:
    """Typed control/obstacle semantics for one PDE rollback."""

    control_style: str = "identity"
    controller_role: str = "none"


@dataclass(frozen=True)
class PDEBoundarySpec:
    """Typed boundary and terminal-condition contract for one PDE rollback."""

    terminal_condition_kind: str = ""
    lower_boundary_kind: str = "default"
    upper_boundary_kind: str = "default"


@dataclass(frozen=True)
class EventAwarePDEIR(BaseFamilyLoweringIR):
    """Typed lowering payload for bounded event-aware 1D PDE routes."""

    state_spec: PDEStateSpec = field(default_factory=PDEStateSpec)
    operator_spec: PDEOperatorSpec = field(default_factory=PDEOperatorSpec)
    event_program: EventProgramIR = field(default_factory=EventProgramIR)
    control_program: ControlProgramIR = field(default_factory=ControlProgramIR)
    control_spec: PDEControlSpec = field(default_factory=PDEControlSpec)
    boundary_spec: PDEBoundarySpec = field(default_factory=PDEBoundarySpec)
    event_timeline: tuple[PDEEventTimeSpec, ...] = ()
    event_transforms: tuple[PDEEventTransformSpec, ...] = ()
    helper_symbol: str = ""
    market_mapping: str = ""
    compatibility_wrapper: str = ""

    @property
    def event_transform_kinds(self) -> tuple[str, ...]:
        """Return the stable event-transform kinds encoded on this PDE IR."""
        kinds: list[str] = []
        for transform in self.event_transforms:
            kind = str(transform.transform_kind or "").strip()
            if kind and kind not in kinds:
                kinds.append(kind)
        for bucket in self.event_timeline:
            for transform in bucket.transforms:
                kind = str(transform.transform_kind or "").strip()
                if kind and kind not in kinds:
                    kinds.append(kind)
        return tuple(kinds)

    @property
    def event_dates(self) -> tuple[str, ...]:
        """Return the stable event dates carried by the typed event timeline."""
        dates: list[str] = []
        for bucket in self.event_timeline:
            event_date = str(bucket.event_date or "").strip()
            if event_date and event_date not in dates:
                dates.append(event_date)
        return tuple(dates)


@dataclass(frozen=True)
class VanillaEquityPDEIR(EventAwarePDEIR):
    """Transitional compatibility wrapper for vanilla equity theta-method PDE routes.

    End state: once downstream trace/readers and route review surfaces stop
    keying on the legacy family-IR type, the vanilla route should emit a plain
    ``EventAwarePDEIR`` instance directly.
    """

    state_spec: PDEStateSpec = field(
        default_factory=lambda: PDEStateSpec(
            state_variable="spot",
            dimension=1,
            state_tags=("terminal_markov", "recombining_safe"),
            coordinate_chart="spot",
        )
    )
    operator_spec: PDEOperatorSpec = field(
        default_factory=lambda: PDEOperatorSpec(
            operator_family="black_scholes_1d",
            solver_family="theta_method",
            stepping_scheme="theta_0.5",
        )
    )
    control_spec: PDEControlSpec = field(
        default_factory=lambda: PDEControlSpec(
            control_style="identity",
            controller_role="holder",
        )
    )
    boundary_spec: PDEBoundarySpec = field(
        default_factory=lambda: PDEBoundarySpec(
            terminal_condition_kind="expiry_payoff",
            lower_boundary_kind="vanilla_dirichlet",
            upper_boundary_kind="vanilla_linear_asymptote",
        )
    )
    option_type: str = "call"
    theta: float = 0.5
    helper_symbol: str = "price_vanilla_equity_option_pde"
    market_mapping: str = "equity_spot_discount_black_vol"
    compatibility_wrapper: str = "VanillaEquityPDEIR"


@dataclass(frozen=True)
class ExerciseLatticeIR(BaseFamilyLoweringIR):
    """Typed lowering payload for schedule-dependent strategic-rights lattice routes."""

    control_style: str = "identity"
    controller_role: str = "none"
    exercise_style: str = "none"
    event_program: EventProgramIR = field(default_factory=EventProgramIR)
    control_program: ControlProgramIR = field(default_factory=ControlProgramIR)
    helper_symbol: str = ""
    control_symbol: str = "resolve_lattice_exercise_policy_from_control_style"
    market_mapping: str = ""
    schedule_role: str = ""
    decision_phase: str = "decision"
    observable_ids: tuple[str, ...] = ()
    observable_types: tuple[str, ...] = ()
    state_field_names: tuple[str, ...] = ()
    derived_quantities: tuple[str, ...] = ()
    decision_dates: tuple[str, ...] = ()
    settlement_dates: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorrelatedBasketMonteCarloIR(BaseFamilyLoweringIR):
    """Typed lowering payload for ranked-observation basket Monte Carlo routes."""

    helper_symbol: str = "price_ranked_observation_basket_monte_carlo"
    market_binding_symbol: str = "resolve_basket_semantics"
    market_mapping: str = "basket_spots_vols_discount_correlation_to_resolved_semantics"
    controller_style: str = "identity"
    constituent_names: tuple[str, ...] = ()
    observation_dates: tuple[str, ...] = ()
    observable_ids: tuple[str, ...] = ()
    observable_types: tuple[str, ...] = ()
    automatic_event_names: tuple[str, ...] = ()
    state_field_names: tuple[str, ...] = ()
    state_tags: tuple[str, ...] = ()
    selection_rule: str = ""
    lock_rule: str = ""
    aggregation_rule: str = ""
    selection_count: int = 1
    schedule_role: str = "observation_dates"
    path_requirement_kind: str = "observation_snapshot_state"
    required_fixing_schedule: tuple[str, ...] = ()
    binding_sources: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CreditDefaultSwapIR(BaseFamilyLoweringIR):
    """Typed lowering payload for single-name CDS helper-backed routes."""

    pricing_mode: str = "analytical"
    schedule_builder_symbol: str = "build_cds_schedule"
    helper_symbol: str = "price_cds_analytical"
    market_mapping: str = "discount_curve_credit_curve_to_cds_legs"
    schedule_role: str = "payment_dates"
    leg_semantics: tuple[str, ...] = ("premium_leg", "protection_leg")
    payment_dates: tuple[str, ...] = ()
    observable_ids: tuple[str, ...] = ()
    observable_types: tuple[str, ...] = ()
    state_field_names: tuple[str, ...] = ()
    state_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class NthToDefaultIR(BaseFamilyLoweringIR):
    """Typed lowering payload for nth-to-default helper-backed copula routes."""

    helper_symbol: str = "price_nth_to_default_basket"
    copula_symbol: str = "GaussianCopula"
    schedule_role: str = "observation_dates"
    trigger_rank: int = 1
    reference_entities: tuple[str, ...] = ()
    observation_dates: tuple[str, ...] = ()
    observable_ids: tuple[str, ...] = ()
    observable_types: tuple[str, ...] = ()
    state_field_names: tuple[str, ...] = ()
    state_tags: tuple[str, ...] = ()
    automatic_event_names: tuple[str, ...] = ()


FamilyLoweringIR = (
    AnalyticalBlack76IR
    | TransformPricingIR
    | EventAwareMonteCarloIR
    | EventAwarePDEIR
    | VanillaEquityPDEIR
    | ExerciseLatticeIR
    | CorrelatedBasketMonteCarloIR
    | CreditDefaultSwapIR
    | NthToDefaultIR
)


def build_family_lowering_ir(
    contract,
    *,
    route_id: str,
    route_family: str,
    product_ir,
    valuation_context=None,
    market_binding_spec=None,
) -> FamilyLoweringIR | None:
    """Build a typed family IR for migrated routes, else return ``None``."""
    common_kwargs = _common_kwargs(
        contract,
        product_ir=product_ir,
        valuation_context=valuation_context,
        market_binding_spec=market_binding_spec,
        route_id=route_id,
        route_family=route_family,
    )

    if _is_vanilla_european_contract(contract, product_ir):
        if route_id == "analytical_black76":
            option_type = _option_type_for_contract(contract)
            return AnalyticalBlack76IR(
                option_type=option_type,
                kernel_symbol="black76_put" if option_type == "put" else "black76_call",
                **common_kwargs,
            )

        if route_id == "transform_fft":
            return _build_transform_pricing_ir(
                contract,
                product_ir=product_ir,
                **common_kwargs,
            )

        if route_id == "vanilla_equity_theta_pde":
            control_program = _build_control_program(contract.product)
            return VanillaEquityPDEIR(
                option_type=_option_type_for_contract(contract),
                theta=0.5,
                control_program=control_program,
                operator_spec=PDEOperatorSpec(
                    operator_family="black_scholes_1d",
                    solver_family="theta_method",
                    stepping_scheme="theta_0.5",
                ),
                **common_kwargs,
            )
    if route_id == "pde_theta_1d":
        return _build_event_aware_pde_ir(
            contract,
            product_ir=product_ir,
            **common_kwargs,
        )
    if route_id in {"monte_carlo_paths", "local_vol_monte_carlo"}:
        return _build_event_aware_monte_carlo_ir(
            contract,
            product_ir=product_ir,
            **common_kwargs,
        )

    if route_id == "exercise_lattice":
        return _build_exercise_lattice_ir(contract, product_ir=product_ir, **common_kwargs)
    if route_id == "correlated_basket_monte_carlo":
        return _build_correlated_basket_monte_carlo_ir(
            contract,
            product_ir=product_ir,
            market_binding_spec=market_binding_spec,
            **common_kwargs,
        )
    if route_id in {"credit_default_swap_analytical", "credit_default_swap_monte_carlo"}:
        return _build_credit_default_swap_ir(
            contract,
            product_ir=product_ir,
            **common_kwargs,
        )
    if route_id == "nth_to_default_monte_carlo":
        return _build_nth_to_default_ir(
            contract,
            product_ir=product_ir,
            **common_kwargs,
        )

    return None


def _common_kwargs(
    contract,
    *,
    product_ir,
    valuation_context,
    market_binding_spec,
    route_id: str,
    route_family: str,
) -> dict[str, object]:
    required_input_ids = _required_input_ids(contract, market_binding_spec)
    return dict(
        route_id=route_id,
        route_family=route_family,
        product_instrument=str(getattr(product_ir, "instrument", "")),
        payoff_family=str(getattr(product_ir, "payoff_family", "")),
        required_input_ids=required_input_ids,
        market_data_requirements=frozenset(required_input_ids),
        timeline_roles=_timeline_roles_for_contract(contract),
        requested_outputs=_requested_outputs(valuation_context, market_binding_spec),
        reporting_currency=_reporting_currency(contract, valuation_context, market_binding_spec),
    )


def _build_control_program(product) -> ControlProgramIR:
    """Build the universal semantic control contract for one product."""
    protocol = getattr(product, "controller_protocol", None)
    return ControlProgramIR(
        control_style=_normalized_control_style(
            str(getattr(protocol, "controller_style", "identity") or "identity")
        ),
        controller_role=str(getattr(protocol, "controller_role", "") or "none").strip() or "none",
        decision_phase=str(getattr(protocol, "decision_phase", "") or "decision").strip() or "decision",
        schedule_role=str(getattr(protocol, "schedule_role", "") or "").strip(),
        exercise_style=str(getattr(product, "exercise_style", "") or "").strip(),
    )


def _build_event_program(
    product,
    *,
    control_program: ControlProgramIR,
) -> EventProgramIR:
    """Compile the universal semantic event timeline for one product."""
    buckets: dict[str, dict[str, object]] = {}

    def ensure_bucket(event_date: str) -> dict[str, object]:
        return buckets.setdefault(
            event_date,
            {
                "roles": [],
                "phases": [],
                "events": [],
                "seen": set(),
            },
        )

    def add_event(event_date: str, event: SemanticEventSpec) -> None:
        if not event_date:
            return
        bucket = ensure_bucket(event_date)
        if event.schedule_role and event.schedule_role not in bucket["roles"]:
            bucket["roles"].append(event.schedule_role)
        if event.phase and event.phase not in bucket["phases"]:
            bucket["phases"].append(event.phase)
        event_key = (
            str(event.event_name or ""),
            str(event.event_kind or ""),
            str(event.schedule_role or ""),
            str(event.phase or ""),
            str(event.value_semantics or ""),
            str(event.transform_kind or ""),
        )
        if event_key not in bucket["seen"]:
            bucket["events"].append(event)
            bucket["seen"].add(event_key)

    for observable in getattr(product, "observables", ()) or ():
        schedule_role = str(getattr(observable, "schedule_role", "") or "observation_dates").strip()
        phase = str(getattr(observable, "availability_phase", "") or "").strip()
        event_kind = _semantic_event_kind_for_observable(observable)
        if not phase:
            phase = _semantic_phase_for_event_kind(event_kind)
        transform_kind = (
            "add_cashflow"
            if event_kind == "coupon"
            else ""
        )
        for event_date in _timeline_dates_for_role(product, schedule_role):
            add_event(
                event_date,
                SemanticEventSpec(
                    event_name=str(
                        getattr(observable, "observable_id", "")
                        or getattr(observable, "observable_type", "")
                        or "observable"
                    ),
                    event_kind=event_kind,
                    schedule_role=schedule_role,
                    phase=phase,
                    value_semantics=str(
                        getattr(observable, "observable_id", "")
                        or getattr(observable, "observable_type", "")
                        or "observation"
                    ),
                    state_binding=_mc_state_binding_for_observable(
                        product,
                        str(getattr(observable, "observable_id", "") or ""),
                    ),
                    transform_kind=transform_kind,
                ),
            )

    if control_program.control_style in {"holder_max", "issuer_min"}:
        transform_kind = (
            "project_max"
            if control_program.control_style == "holder_max"
            else "project_min"
        )
        value_semantics = (
            "holder_exercise_projection"
            if control_program.control_style == "holder_max"
            else "issuer_call_projection"
        )
        schedule_role = control_program.schedule_role or "decision_dates"
        phase = control_program.decision_phase or "decision"
        for event_date in _timeline_dates_for_role(product, schedule_role):
            add_event(
                event_date,
                SemanticEventSpec(
                    event_name=value_semantics,
                    event_kind="exercise",
                    schedule_role=schedule_role,
                    phase=phase,
                    value_semantics=value_semantics,
                    state_binding="",
                    transform_kind=transform_kind,
                ),
            )

    for transition in getattr(getattr(product, "event_machine", None), "transitions", ()) or ():
        event_kind = _semantic_event_kind_for_transition(transition)
        schedule_role = _semantic_schedule_role_for_transition(
            product,
            event_kind=event_kind,
            control_program=control_program,
        )
        phase = _semantic_phase_for_event_kind(
            event_kind,
            default_phase=control_program.decision_phase or "decision",
        )
        action = str(
            getattr(getattr(transition, "action", None), "action_type", "")
            or getattr(transition, "name", "")
            or ""
        ).strip()
        transform_kind = _semantic_transform_kind_for_transition(
            transition,
            control_program=control_program,
        )
        for event_date in _timeline_dates_for_role(product, schedule_role):
            add_event(
                event_date,
                SemanticEventSpec(
                    event_name=str(getattr(transition, "name", "") or action or "event_transition"),
                    event_kind=event_kind,
                    schedule_role=schedule_role,
                    phase=phase,
                    value_semantics=action or str(getattr(transition, "name", "") or "event_transition"),
                    state_binding=str(getattr(transition, "to_state", "") or ""),
                    transform_kind=transform_kind,
                ),
            )

    for obligation in getattr(product, "obligations", ()) or ():
        for event_date in _timeline_dates_for_role(product, "settlement_dates"):
            add_event(
                event_date,
                SemanticEventSpec(
                    event_name=str(getattr(obligation, "obligation_id", "") or "settlement"),
                    event_kind="settlement",
                    schedule_role="settlement_dates",
                    phase="settlement",
                    value_semantics=str(
                        getattr(obligation, "amount_expression", "")
                        or getattr(obligation, "obligation_id", "")
                        or "settlement"
                    ),
                    state_binding=str(getattr(obligation, "obligation_id", "") or ""),
                    transform_kind="",
                ),
            )

    timeline: list[SemanticEventTimeSpec] = []
    for event_date in sorted(buckets.keys()):
        bucket = buckets[event_date]
        timeline.append(
            SemanticEventTimeSpec(
                event_date=event_date,
                schedule_roles=tuple(bucket["roles"]),
                phase_sequence=tuple(bucket["phases"]),
                events=tuple(bucket["events"]),
            )
        )
    return EventProgramIR(timeline=tuple(timeline))


def _build_transform_pricing_ir(
    contract,
    *,
    product_ir,
    route_id: str,
    route_family: str,
    product_instrument: str,
    payoff_family: str,
    required_input_ids: tuple[str, ...],
    market_data_requirements: frozenset[str],
    timeline_roles: frozenset[TimelineRole],
    requested_outputs: tuple[str, ...],
    reporting_currency: str,
) -> TransformPricingIR | None:
    """Build a typed transform family IR for bounded single-state terminal claims."""
    product = contract.product
    state_spec = _transform_state_spec_for_product(product)
    characteristic_spec = _transform_characteristic_spec_for_product(product)
    if not state_spec.state_variable or not characteristic_spec.characteristic_family:
        return None

    control_program = _build_control_program(product)
    if not _transform_supports_control_program(control_program):
        return None
    controller_role = str(control_program.controller_role or "holder").strip() or "holder"

    return TransformPricingIR(
        route_id=route_id,
        route_family=route_family,
        product_instrument=product_instrument,
        payoff_family=payoff_family,
        required_input_ids=required_input_ids,
        market_data_requirements=market_data_requirements,
        timeline_roles=timeline_roles,
        requested_outputs=requested_outputs,
        reporting_currency=reporting_currency,
        state_spec=state_spec,
        characteristic_spec=characteristic_spec,
        control_program=control_program,
        control_spec=TransformControlSpec(
            control_style="identity",
            controller_role=controller_role,
        ),
        terminal_payoff_kind=_transform_terminal_payoff_kind_for_product(product),
        strike_semantics="vanilla_strike",
        quote_semantics=_transform_quote_semantics_for_product(characteristic_spec),
        helper_symbol=_transform_helper_symbol_for_product(product, characteristic_spec),
        market_mapping=_transform_market_mapping_for_product(product, characteristic_spec),
        compatibility_wrapper="",
    )


def _transform_supports_control_program(control_program: ControlProgramIR) -> bool:
    """Return whether the bounded transform lane can safely lower one control program."""
    control_style = str(control_program.control_style or "identity").strip().lower()
    controller_role = str(control_program.controller_role or "none").strip().lower()
    if control_style not in {"identity", "holder_max"}:
        return False
    return controller_role in {"", "none", "holder"}


def _transform_state_spec_for_product(product) -> TransformStateSpec:
    """Infer the bounded transform state contract from semantic product metadata."""
    model_family = str(getattr(product, "model_family", "") or "").strip().lower()
    if model_family in {"equity", "equity_diffusion", "stochastic_volatility"}:
        return TransformStateSpec(
            state_variable="spot",
            dimension=1,
            state_tags=("terminal_markov",),
            coordinate_chart="spot",
        )
    return TransformStateSpec()


def _transform_characteristic_spec_for_product(
    product,
) -> TransformCharacteristicSpec:
    """Infer the bounded transform characteristic-function contract for one product."""
    model_family = str(getattr(product, "model_family", "") or "").strip().lower()
    if model_family in {"equity", "equity_diffusion"}:
        return TransformCharacteristicSpec(
            model_family="equity_diffusion",
            characteristic_family="gbm_log_spot",
            supported_methods=("fft", "cos"),
            backend_capability="helper_backed",
        )
    if model_family == "stochastic_volatility":
        return TransformCharacteristicSpec(
            model_family="stochastic_volatility",
            characteristic_family="heston_log_spot",
            supported_methods=("fft", "cos"),
            backend_capability="raw_kernel_only",
        )
    return TransformCharacteristicSpec()


def _transform_terminal_payoff_kind_for_product(product) -> str:
    """Return the bounded terminal payoff kind for one transform family IR."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    if payoff_family == "vanilla_option":
        return "vanilla_terminal_payoff"
    return "compiled_terminal_payoff"


def _transform_quote_semantics_for_product(
    characteristic_spec: TransformCharacteristicSpec,
) -> str:
    """Return a readable quote-semantics label for one transform family IR."""
    if characteristic_spec.model_family == "equity_diffusion":
        return "equity_black_vol_surface"
    if characteristic_spec.model_family == "stochastic_volatility":
        return "stochastic_vol_model_parameters"
    return "compiled_transform_quote_inputs"


def _transform_helper_symbol_for_product(
    product,
    characteristic_spec: TransformCharacteristicSpec,
) -> str:
    """Return the bounded helper symbol for transform routes when one exists."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    if (
        payoff_family == "vanilla_option"
        and characteristic_spec.model_family == "equity_diffusion"
    ):
        return "price_vanilla_equity_option_transform"
    return ""


def _transform_market_mapping_for_product(
    product,
    characteristic_spec: TransformCharacteristicSpec,
) -> str:
    """Return a readable market-binding label for one transform family IR."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    if (
        payoff_family == "vanilla_option"
        and characteristic_spec.model_family == "equity_diffusion"
    ):
        return "single_state_diffusion_transform_inputs"
    if characteristic_spec.model_family == "stochastic_volatility":
        return "stochastic_vol_transform_inputs"
    return "compiled_transform_inputs"


def _build_event_aware_monte_carlo_ir(
    contract,
    *,
    product_ir,
    route_id: str,
    route_family: str,
    product_instrument: str,
    payoff_family: str,
    required_input_ids: tuple[str, ...],
    market_data_requirements: frozenset[str],
    timeline_roles: frozenset[TimelineRole],
    requested_outputs: tuple[str, ...],
    reporting_currency: str,
) -> EventAwareMonteCarloIR | None:
    """Build a typed event-aware Monte Carlo IR for bounded single-state contracts."""
    product = contract.product
    if bool(getattr(product, "multi_asset", False)):
        return None

    control_program = _build_control_program(product)
    control_spec = _mc_control_spec_for_product(product)
    if control_spec is None or control_spec.control_style != "identity":
        return None

    state_spec = _mc_state_spec_for_product(product)
    process_spec = _mc_process_spec_for_product(product, route_id=route_id)
    if not state_spec.state_variable or not process_spec.process_family:
        return None

    event_program = EventProgramIR()
    if not _mc_uses_terminal_only_contract(product):
        event_program = _build_event_program(product, control_program=control_program)
    event_timeline = _project_event_program_to_mc_timeline(event_program)
    if bool(getattr(product, "schedule_dependence", False)) and not event_timeline:
        return None

    path_requirement_spec = _mc_path_requirement_spec_for_product(
        product,
        event_timeline=event_timeline,
    )
    payoff_reducer_spec = _mc_payoff_reducer_spec_for_product(
        product,
        event_timeline=event_timeline,
    )

    return EventAwareMonteCarloIR(
        route_id=route_id,
        route_family=route_family,
        product_instrument=product_instrument,
        payoff_family=payoff_family,
        required_input_ids=required_input_ids,
        market_data_requirements=market_data_requirements,
        timeline_roles=timeline_roles,
        requested_outputs=requested_outputs,
        reporting_currency=reporting_currency,
        state_spec=state_spec,
        process_spec=process_spec,
        event_program=event_program,
        control_program=control_program,
        path_requirement_spec=path_requirement_spec,
        payoff_reducer_spec=payoff_reducer_spec,
        control_spec=control_spec,
        measure_spec=MCMeasureSpec(
            measure_family="risk_neutral",
            numeraire_binding="discount_curve",
        ),
        event_timeline=event_timeline,
        event_specs=tuple(
            event
            for bucket in event_timeline
            for event in bucket.events
        ),
        helper_symbol=_mc_helper_symbol_for_product(product, process_spec),
        market_mapping=_mc_market_mapping_for_product(product, process_spec),
        compatibility_wrapper="",
    )


def _mc_uses_terminal_only_contract(product) -> bool:
    """Return whether the product should lower onto a terminal-only MC contract."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    exercise_style = str(getattr(product, "exercise_style", "") or "").strip().lower()
    return payoff_family == "vanilla_option" and exercise_style == "european"


def _mc_control_spec_for_product(product) -> MCControlSpec | None:
    """Return the bounded control contract for event-aware Monte Carlo lowering."""
    protocol = getattr(product, "controller_protocol", None)
    controller_role = str(getattr(protocol, "controller_role", "") or "none").strip() or "none"
    exercise_style = str(getattr(product, "exercise_style", "") or "").strip().lower()
    if exercise_style == "european":
        return MCControlSpec(
            control_style="identity",
            controller_role=controller_role or "holder",
        )
    control_style = _normalized_control_style(
        str(getattr(protocol, "controller_style", "identity") or "identity")
    )
    if control_style != "identity":
        return None
    return MCControlSpec(
        control_style="identity",
        controller_role=controller_role,
    )


def _mc_state_spec_for_product(product) -> MCStateSpec:
    """Infer the bounded Monte Carlo state contract from semantic metadata."""
    model_family = str(getattr(product, "model_family", "") or "").strip().lower()
    state_tags = _tuple_unique(("terminal_markov", *_state_tags(product)))
    if model_family in {"interest_rate", "short_rate"}:
        return MCStateSpec(
            state_variable="short_rate",
            dimension=1,
            state_tags=state_tags,
            state_layout="scalar",
        )
    if model_family in {"equity", "equity_diffusion"}:
        return MCStateSpec(
            state_variable="spot",
            dimension=1,
            state_tags=state_tags,
            state_layout="scalar",
        )
    return MCStateSpec()


def _mc_process_spec_for_product(product, *, route_id: str) -> MCProcessSpec:
    """Infer the bounded Monte Carlo process contract from semantic metadata."""
    if route_id == "local_vol_monte_carlo":
        return MCProcessSpec(
            process_family="local_vol_1d",
            simulation_scheme="euler_local_vol",
        )
    model_family = str(getattr(product, "model_family", "") or "").strip().lower()
    if model_family in {"interest_rate", "short_rate"}:
        return MCProcessSpec(
            process_family="hull_white_1f",
            simulation_scheme="exact_ou",
        )
    if model_family in {"equity", "equity_diffusion"}:
        return MCProcessSpec(
            process_family="gbm_1d",
            simulation_scheme="exact_lognormal",
        )
    return MCProcessSpec()


def _mc_market_mapping_for_product(product, process_spec: MCProcessSpec) -> str:
    """Return a readable market-binding label for the bounded MC family IR."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    if process_spec.process_family == "hull_white_1f" and payoff_family == "swaption":
        return "discount_curve_forward_curve_black_vol_to_short_rate_mc"
    if process_spec.process_family == "local_vol_1d":
        return "equity_spot_discount_local_vol_to_mc"
    if process_spec.process_family == "gbm_1d":
        return "equity_spot_discount_black_vol_to_mc"
    return "compiled_event_aware_monte_carlo_inputs"


def _mc_helper_symbol_for_product(product, process_spec: MCProcessSpec) -> str:
    """Return a family-level helper symbol when one comprehensive kit exists."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    exercise_style = str(getattr(product, "exercise_style", "") or "").strip().lower()
    if (
        process_spec.process_family == "gbm_1d"
        and payoff_family == "vanilla_option"
        and exercise_style == "european"
    ):
        return "price_vanilla_equity_option_monte_carlo"
    if process_spec.process_family == "hull_white_1f" and payoff_family == "swaption":
        return "price_swaption_monte_carlo"
    return ""


def _build_mc_event_timeline(product) -> tuple[MCEventTimeSpec, ...]:
    """Backward-compatible MC event projection from the universal event program."""
    event_program = _build_event_program(
        product,
        control_program=_build_control_program(product),
    )
    return _project_event_program_to_mc_timeline(event_program)


def _timeline_dates_for_role(product, schedule_role: str) -> tuple[str, ...]:
    """Return the event dates attached to one semantic schedule role."""
    if not schedule_role:
        return ()
    timeline = getattr(product, "timeline", None)
    dates = tuple(getattr(timeline, schedule_role, ()) or ())
    if dates:
        return dates
    if schedule_role == "observation_dates":
        return tuple(getattr(product, "observation_schedule", ()) or ())
    primary_role = str(getattr(getattr(product, "implementation_hints", None), "primary_schedule_role", "") or "").strip()
    if schedule_role == primary_role:
        return tuple(getattr(product, "observation_schedule", ()) or ())
    return ()


def _semantic_event_kind_for_observable(observable) -> str:
    """Map one semantic observable onto the universal event vocabulary."""
    observable_type = str(getattr(observable, "observable_type", "") or "").strip().lower()
    if observable_type == "cashflow_schedule":
        return "coupon"
    return "observation"


def _mc_state_binding_for_observable(product, observable_id: str) -> str:
    """Return the state field names that derive from one semantic observable."""
    bound_fields = [
        str(getattr(field, "field_name", "") or "").strip()
        for field in getattr(product, "state_fields", ()) or ()
        if observable_id in tuple(getattr(field, "source_observables", ()) or ())
    ]
    return ",".join(field for field in bound_fields if field)


def _semantic_event_kind_for_transition(transition) -> str:
    """Map one semantic event transition onto the universal event vocabulary."""
    action = str(getattr(getattr(transition, "action", None), "action_type", "") or "").strip().lower()
    label = str(getattr(transition, "name", "") or "").strip().lower()
    declared_kind = str(getattr(transition, "event_kind", "") or "").strip().lower()
    text = action or label or declared_kind
    if "settle" in text:
        return "settlement"
    if "coupon" in text:
        return "coupon"
    if "exercise" in text or text.startswith("price_"):
        return "exercise"
    if "barrier" in text:
        return "barrier"
    if declared_kind and declared_kind != "observation":
        return declared_kind
    return "observation"


def _semantic_schedule_role_for_transition(
    product,
    *,
    event_kind: str,
    control_program: ControlProgramIR,
) -> str:
    """Return the semantic schedule role used for one transition event kind."""
    if event_kind == "settlement":
        return "settlement_dates"
    if event_kind == "coupon":
        return _cashflow_schedule_role(product)
    if event_kind == "exercise":
        return control_program.schedule_role or "decision_dates"
    if str(getattr(product, "exercise_style", "") or "").strip().lower() == "european":
        return "observation_dates"
    primary_role = str(getattr(getattr(product, "implementation_hints", None), "primary_schedule_role", "") or "").strip()
    return primary_role or "observation_dates"


def _semantic_phase_for_event_kind(
    event_kind: str,
    *,
    default_phase: str = "observation",
) -> str:
    """Return the universal timeline phase used for one semantic event kind."""
    if event_kind == "settlement":
        return "settlement"
    if event_kind == "coupon":
        return "determination"
    if event_kind == "exercise":
        return default_phase or "decision"
    return "observation"


def _semantic_transform_kind_for_transition(
    transition,
    *,
    control_program: ControlProgramIR,
) -> str:
    """Return any universal transform implied by one semantic transition."""
    action = str(getattr(getattr(transition, "action", None), "action_type", "") or "").strip().lower()
    name = str(getattr(transition, "name", "") or "").strip().lower()
    text = action or name
    if text in {"record_barrier_hit", "state_remap", "reset_state"}:
        return "state_remap"
    if "coupon" in text:
        return "add_cashflow"
    return ""


def _project_event_program_to_mc_timeline(
    event_program: EventProgramIR,
) -> tuple[MCEventTimeSpec, ...]:
    """Project the universal event program onto the bounded MC event timeline."""
    timeline: list[MCEventTimeSpec] = []
    for bucket in event_program.timeline:
        events = tuple(
            MCEventSpec(
                event_name=event.event_name,
                event_kind=event.event_kind,
                schedule_role=event.schedule_role,
                phase=event.phase,
                value_semantics=event.value_semantics,
                state_binding=event.state_binding,
            )
            for event in bucket.events
            if str(event.transform_kind or "").strip() not in {"project_max", "project_min"}
        )
        if not events:
            continue
        phases = tuple(
            phase
            for phase in bucket.phase_sequence
            if phase != "decision"
        )
        schedule_roles = tuple(
            role
            for role in bucket.schedule_roles
            if role != "decision_dates"
        )
        timeline.append(
            MCEventTimeSpec(
                event_date=bucket.event_date,
                schedule_roles=schedule_roles or bucket.schedule_roles,
                phase_sequence=phases or bucket.phase_sequence,
                events=events,
            )
        )
    return tuple(timeline)


def _mc_path_requirement_spec_for_product(
    product,
    *,
    event_timeline: tuple[MCEventTimeSpec, ...],
) -> MCPathRequirementSpec:
    """Infer the reduced-state contract needed by the bounded MC family."""
    if event_timeline:
        requirement_kind = "event_replay"
        replay_mode = "deterministic_timeline"
    else:
        requirement_kind = "terminal_only"
        replay_mode = "none"
    primary_role = str(getattr(getattr(product, "implementation_hints", None), "primary_schedule_role", "") or "").strip()
    reducer_kinds = _mc_reducer_kinds_for_product(product)
    return MCPathRequirementSpec(
        requirement_kind=requirement_kind,
        snapshot_schedule_role=primary_role or "observation_dates",
        reducer_kinds=reducer_kinds,
        replay_mode=replay_mode,
        stored_fields=tuple(
            str(getattr(field, "field_name", "") or "").strip()
            for field in getattr(product, "state_fields", ()) or ()
            if str(getattr(field, "field_name", "") or "").strip()
        ),
    )


def _mc_reducer_kinds_for_product(product) -> tuple[str, ...]:
    """Return bounded reduced-state helper names for one semantic product."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    if payoff_family == "swaption":
        return ("discounted_swap_pv",)
    if payoff_family == "vanilla_option":
        return ("terminal_payoff",)
    return ()


def _mc_payoff_reducer_spec_for_product(
    product,
    *,
    event_timeline: tuple[MCEventTimeSpec, ...],
) -> MCPayoffReducerSpec:
    """Infer the payoff-reducer contract for one bounded MC family instance."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    if payoff_family == "swaption":
        dependencies = tuple(
            event.event_name
            for bucket in event_timeline
            for event in bucket.events
        )
        return MCPayoffReducerSpec(
            reducer_kind="swaption_exercise_payoff",
            output_semantics="swaption_exercise_payoff",
            event_dependencies=dependencies,
        )
    if payoff_family == "vanilla_option":
        return MCPayoffReducerSpec(
            reducer_kind="terminal_payoff",
            output_semantics="vanilla_option_payoff",
        )
    return MCPayoffReducerSpec(
        reducer_kind="compiled_schedule_payoff",
        output_semantics=payoff_family or "compiled_payoff",
    )


def _build_event_aware_pde_ir(
    contract,
    *,
    product_ir,
    route_id: str,
    route_family: str,
    product_instrument: str,
    payoff_family: str,
    required_input_ids: tuple[str, ...],
    market_data_requirements: frozenset[str],
    timeline_roles: frozenset[TimelineRole],
    requested_outputs: tuple[str, ...],
    reporting_currency: str,
) -> EventAwarePDEIR | None:
    """Build a typed event-aware PDE IR for bounded schedule-driven contracts."""
    product = contract.product
    if not bool(getattr(product, "schedule_dependence", False)):
        return None

    control_program = _build_control_program(product)
    state_spec = _pde_state_spec_for_product(product)
    operator_spec = _pde_operator_spec_for_product(product)
    if not state_spec.state_variable or not operator_spec.operator_family:
        return None

    control_style = control_program.control_style
    event_program = _build_event_program(product, control_program=control_program)
    event_timeline = _project_event_program_to_pde_timeline(event_program)
    if not event_timeline:
        return None

    return EventAwarePDEIR(
        route_id=route_id,
        route_family=route_family,
        product_instrument=product_instrument,
        payoff_family=payoff_family,
        required_input_ids=required_input_ids,
        market_data_requirements=market_data_requirements,
        timeline_roles=timeline_roles,
        requested_outputs=requested_outputs,
        reporting_currency=reporting_currency,
        state_spec=state_spec,
        operator_spec=operator_spec,
        event_program=event_program,
        control_program=control_program,
        control_spec=PDEControlSpec(
            control_style=control_style,
            controller_role=str(getattr(getattr(product, "controller_protocol", None), "controller_role", "none") or "none"),
        ),
        boundary_spec=_pde_boundary_spec_for_product(product),
        event_timeline=event_timeline,
        event_transforms=tuple(
            transform
            for bucket in event_timeline
            for transform in bucket.transforms
        ),
        helper_symbol=_pde_helper_symbol_for_product(product, operator_spec, control_style),
        market_mapping=_pde_market_mapping_for_product(product, operator_spec),
        compatibility_wrapper="",
    )


def _pde_state_spec_for_product(product) -> PDEStateSpec:
    """Infer a bounded PDE state contract from semantic product metadata."""
    model_family = str(getattr(product, "model_family", "") or "").strip().lower()
    if model_family in {"interest_rate", "short_rate"}:
        return PDEStateSpec(
            state_variable="short_rate",
            dimension=1,
            state_tags=("terminal_markov", "recombining_safe"),
            coordinate_chart="short_rate",
        )
    if model_family in {"equity", "equity_diffusion"}:
        return PDEStateSpec(
            state_variable="spot",
            dimension=1,
            state_tags=("terminal_markov", "recombining_safe"),
            coordinate_chart="spot",
        )
    return PDEStateSpec()


def _pde_operator_spec_for_product(product) -> PDEOperatorSpec:
    """Infer a bounded PDE operator contract from semantic product metadata."""
    model_family = str(getattr(product, "model_family", "") or "").strip().lower()
    if model_family in {"interest_rate", "short_rate"}:
        return PDEOperatorSpec(
            operator_family="hull_white_1f",
            solver_family="theta_method",
            stepping_scheme="theta_0.5",
        )
    if model_family in {"equity", "equity_diffusion"}:
        return PDEOperatorSpec(
            operator_family="black_scholes_1d",
            solver_family="theta_method",
            stepping_scheme="theta_0.5",
        )
    return PDEOperatorSpec()


def _pde_boundary_spec_for_product(product) -> PDEBoundarySpec:
    """Infer the bounded terminal/boundary contract for one PDE family IR."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    if payoff_family in {"callable_fixed_income", "bond"}:
        return PDEBoundarySpec(
            terminal_condition_kind="cashflow_terminal_value",
            lower_boundary_kind="short_rate_linear",
            upper_boundary_kind="short_rate_linear",
        )
    if payoff_family == "vanilla_option":
        return PDEBoundarySpec(
            terminal_condition_kind="expiry_payoff",
            lower_boundary_kind="vanilla_dirichlet",
            upper_boundary_kind="vanilla_linear_asymptote",
        )
    return PDEBoundarySpec(
        terminal_condition_kind="compiled_terminal_value",
        lower_boundary_kind="default",
        upper_boundary_kind="default",
    )


def _pde_market_mapping_for_product(product, operator_spec: PDEOperatorSpec) -> str:
    """Return a readable market-binding label for the bounded PDE family IR."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    if operator_spec.operator_family == "hull_white_1f" and payoff_family == "callable_fixed_income":
        return "discount_curve_coupon_schedule_to_short_rate_pde"
    if operator_spec.operator_family == "black_scholes_1d":
        return "equity_spot_discount_black_vol"
    return "compiled_event_aware_pde_inputs"


def _pde_helper_symbol_for_product(
    product,
    operator_spec: PDEOperatorSpec,
    control_style: str,
) -> str:
    """Return a bounded helper symbol for migrated event-aware PDE proof slices."""
    payoff_family = str(getattr(product, "payoff_family", "") or "").strip().lower()
    if (
        payoff_family == "vanilla_option"
        and operator_spec.operator_family == "black_scholes_1d"
        and control_style == "holder_max"
    ):
        return "price_event_aware_equity_option_pde"
    if (
        payoff_family == "callable_fixed_income"
        and operator_spec.operator_family == "hull_white_1f"
        and control_style == "issuer_min"
    ):
        return "price_callable_bond_pde"
    return ""


def _build_pde_event_timeline(
    product,
    *,
    control_style: str,
) -> tuple[PDEEventTimeSpec, ...]:
    """Backward-compatible PDE event projection from the universal event program."""
    event_program = _build_event_program(
        product,
        control_program=ControlProgramIR(control_style=control_style),
    )
    return _project_event_program_to_pde_timeline(event_program)


def _project_event_program_to_pde_timeline(
    event_program: EventProgramIR,
) -> tuple[PDEEventTimeSpec, ...]:
    """Project the universal event program onto the bounded PDE event timeline."""
    timeline: list[PDEEventTimeSpec] = []
    for bucket in event_program.timeline:
        transforms = tuple(
            PDEEventTransformSpec(
                transform_kind=event.transform_kind,
                schedule_role=event.schedule_role,
                value_semantics=event.value_semantics,
                state_mapping_symbol=event.state_binding,
            )
            for event in bucket.events
            if str(event.transform_kind or "").strip()
        )
        if not transforms:
            continue
        timeline.append(
            PDEEventTimeSpec(
                event_date=bucket.event_date,
                schedule_roles=tuple(
                    role
                    for role in bucket.schedule_roles
                    if any(
                        str(event.schedule_role or "").strip() == role
                        and str(event.transform_kind or "").strip()
                        for event in bucket.events
                    )
                ) or bucket.schedule_roles,
                phase_sequence=tuple(
                    phase
                    for phase in bucket.phase_sequence
                    if any(
                        str(event.phase or "").strip() == phase
                        and str(event.transform_kind or "").strip()
                        for event in bucket.events
                    )
                ) or bucket.phase_sequence,
                transforms=transforms,
            )
        )
    return tuple(timeline)


def _cashflow_schedule_role(product) -> str:
    """Return the semantic schedule role used for deterministic cashflow events."""
    for observable in getattr(product, "observables", ()) or ():
        if str(getattr(observable, "observable_type", "") or "").strip().lower() == "cashflow_schedule":
            schedule_role = str(getattr(observable, "schedule_role", "") or "").strip()
            if schedule_role:
                return schedule_role
    return "determination_dates"


def _build_exercise_lattice_ir(
    contract,
    *,
    product_ir,
    route_id: str,
    route_family: str,
    product_instrument: str,
    payoff_family: str,
    required_input_ids: tuple[str, ...],
    market_data_requirements: frozenset[str],
    timeline_roles: frozenset[TimelineRole],
    requested_outputs: tuple[str, ...],
    reporting_currency: str,
) -> ExerciseLatticeIR | None:
    """Build the tranche-1 exercise-lattice family IR."""
    if not _is_tranche_one_exercise_lattice_contract(contract, product_ir):
        return None

    product = contract.product
    protocol = product.controller_protocol
    control_style = _normalized_control_style(protocol.controller_style)
    control_program = _build_control_program(product)
    event_program = _build_event_program(product, control_program=control_program)
    _validate_exercise_lattice_control(product.exercise_style, control_style)
    _validate_exercise_lattice_timing(product)

    observable_ids = tuple(item.observable_id for item in product.observables)
    observable_types = tuple(item.observable_type for item in product.observables)
    state_field_names = tuple(item.field_name for item in product.state_fields)

    if product.instrument_class == "callable_bond":
        _require_observables(
            product,
            required_types=("discount_curve", "cashflow_schedule"),
            route_name="Exercise-lattice",
        )
        _require_required_inputs(
            required_input_ids,
            required=("discount_curve", "black_vol_surface"),
            route_name="Exercise-lattice",
        )
        return ExerciseLatticeIR(
            route_id=route_id,
            route_family=route_family,
            product_instrument=product_instrument,
            payoff_family=payoff_family,
            required_input_ids=required_input_ids,
            market_data_requirements=market_data_requirements,
            timeline_roles=timeline_roles,
            requested_outputs=requested_outputs,
            reporting_currency=reporting_currency,
            control_style=control_style,
            controller_role=str(protocol.controller_role or ""),
            exercise_style=str(product.exercise_style or ""),
            event_program=event_program,
            control_program=control_program,
            helper_symbol="price_callable_bond_tree",
            market_mapping="discount_curve_black_vol_coupon_schedule_to_callable_tree",
            schedule_role=str(protocol.schedule_role or ""),
            decision_phase=str(protocol.decision_phase or "decision"),
            observable_ids=observable_ids,
            observable_types=observable_types,
            state_field_names=state_field_names,
            derived_quantities=(
                "call_schedule_steps",
                "coupon_accrual_fractions",
                "settlement_discount_factors",
            ),
            decision_dates=tuple(product.timeline.decision_dates),
            settlement_dates=tuple(product.timeline.settlement_dates),
        )

    _require_observables(
        product,
        required_types=("forward_rate", "discount_curve"),
        route_name="Exercise-lattice",
    )
    _require_required_inputs(
        required_input_ids,
        required=("discount_curve", "forward_curve", "black_vol_surface"),
        route_name="Exercise-lattice",
    )
    return ExerciseLatticeIR(
        route_id=route_id,
        route_family=route_family,
        product_instrument=product_instrument,
        payoff_family=payoff_family,
        required_input_ids=required_input_ids,
        market_data_requirements=market_data_requirements,
        timeline_roles=timeline_roles,
        requested_outputs=requested_outputs,
        reporting_currency=reporting_currency,
        control_style=control_style,
        controller_role=str(protocol.controller_role or ""),
        exercise_style=str(product.exercise_style or ""),
        event_program=event_program,
        control_program=control_program,
        helper_symbol="price_bermudan_swaption_tree",
        market_mapping="discount_curve_forward_par_rate_schedule_to_lattice",
        schedule_role=str(protocol.schedule_role or ""),
        decision_phase=str(protocol.decision_phase or "decision"),
        observable_ids=observable_ids,
        observable_types=observable_types,
        state_field_names=state_field_names,
        derived_quantities=(
            "exercise_schedule_steps",
            "schedule_bound_forward_fixings",
            "par_rate_bindings",
            "swap_accrual_fractions",
            "settlement_discount_factors",
        ),
        decision_dates=tuple(product.timeline.decision_dates),
        settlement_dates=tuple(product.timeline.settlement_dates),
    )


def _build_correlated_basket_monte_carlo_ir(
    contract,
    *,
    product_ir,
    market_binding_spec,
    route_id: str,
    route_family: str,
    product_instrument: str,
    payoff_family: str,
    required_input_ids: tuple[str, ...],
    market_data_requirements: frozenset[str],
    timeline_roles: frozenset[TimelineRole],
    requested_outputs: tuple[str, ...],
    reporting_currency: str,
) -> CorrelatedBasketMonteCarloIR | None:
    """Build the tranche-1 ranked-observation basket Monte Carlo family IR."""
    if not _is_ranked_observation_basket_contract(contract, product_ir):
        return None

    product = contract.product
    control_style = _normalized_control_style(product.controller_protocol.controller_style)
    if control_style != "identity":
        raise ValueError(
            "Ranked-observation basket Monte Carlo semantics must remain event-driven "
            "and cannot declare a strategic controller."
        )
    _validate_ranked_observation_basket_timing(product)
    _require_observables(
        product,
        required_types=("spot_vector", "simple_return"),
        route_name="Ranked-observation basket Monte Carlo",
    )
    _require_required_inputs(
        required_input_ids,
        required=("discount_curve", "underlier_spots", "black_vol_surface", "correlation_matrix"),
        route_name="Ranked-observation basket Monte Carlo",
    )
    _require_state_tags(
        product,
        required=("pathwise_only", "remaining_pool", "locked_cashflow_state"),
        route_name="Ranked-observation basket Monte Carlo",
    )
    automatic_event_names = _automatic_event_names(product)
    if not automatic_event_names:
        raise ValueError(
            "Ranked-observation basket Monte Carlo semantics require typed event-machine "
            "transitions for automatic basket state updates."
        )

    return CorrelatedBasketMonteCarloIR(
        route_id=route_id,
        route_family=route_family,
        product_instrument=product_instrument,
        payoff_family=payoff_family,
        required_input_ids=required_input_ids,
        market_data_requirements=market_data_requirements,
        timeline_roles=timeline_roles,
        requested_outputs=requested_outputs,
        reporting_currency=reporting_currency,
        controller_style=control_style,
        constituent_names=tuple(product.constituents),
        observation_dates=tuple(product.timeline.observation_dates or product.observation_schedule),
        observable_ids=tuple(item.observable_id for item in product.observables),
        observable_types=tuple(item.observable_type for item in product.observables),
        automatic_event_names=automatic_event_names,
        state_field_names=tuple(item.field_name for item in product.state_fields),
        state_tags=_state_tags(product),
        selection_rule=str(product.selection_operator or ""),
        lock_rule=str(product.lock_rule or ""),
        aggregation_rule=str(product.aggregation_rule or ""),
        selection_count=max(int(product.selection_count or 1), 1),
        required_fixing_schedule=tuple(product.timeline.observation_dates or product.observation_schedule),
        binding_sources=_binding_sources(market_binding_spec, required_input_ids),
    )


def _build_credit_default_swap_ir(
    contract,
    *,
    product_ir,
    route_id: str,
    route_family: str,
    product_instrument: str,
    payoff_family: str,
    required_input_ids: tuple[str, ...],
    market_data_requirements: frozenset[str],
    timeline_roles: frozenset[TimelineRole],
    requested_outputs: tuple[str, ...],
    reporting_currency: str,
) -> CreditDefaultSwapIR | None:
    """Build the typed CDS family IR for helper-backed analytical and MC routes."""
    if not _is_credit_default_swap_contract(contract, product_ir):
        return None

    product = contract.product
    control_style = _normalized_control_style(product.controller_protocol.controller_style)
    if control_style != "identity":
        raise ValueError(
            "Credit-default-swap semantics must remain automatic and cannot declare a strategic controller."
        )
    _require_observables(
        product,
        required_types=("credit_curve", "cashflow_schedule"),
        route_name="Credit-default-swap",
    )
    _require_required_inputs(
        required_input_ids,
        required=("discount_curve", "credit_curve"),
        route_name="Credit-default-swap",
    )

    helper_symbol = (
        "price_cds_monte_carlo"
        if route_id == "credit_default_swap_monte_carlo"
        else "price_cds_analytical"
    )
    pricing_mode = "monte_carlo" if route_id == "credit_default_swap_monte_carlo" else "analytical"
    state_tags = (
        ("pathwise_only", "schedule_state")
        if pricing_mode == "monte_carlo"
        else ("terminal_markov", "schedule_state")
    )

    return CreditDefaultSwapIR(
        route_id=route_id,
        route_family=route_family,
        product_instrument=product_instrument,
        payoff_family=payoff_family,
        required_input_ids=required_input_ids,
        market_data_requirements=market_data_requirements,
        timeline_roles=timeline_roles,
        requested_outputs=requested_outputs,
        reporting_currency=reporting_currency,
        pricing_mode=pricing_mode,
        helper_symbol=helper_symbol,
        payment_dates=tuple(product.timeline.observation_dates or product.observation_schedule),
        observable_ids=tuple(item.observable_id for item in product.observables),
        observable_types=tuple(item.observable_type for item in product.observables),
        state_field_names=tuple(item.field_name for item in product.state_fields),
        state_tags=state_tags,
    )


def _build_nth_to_default_ir(
    contract,
    *,
    product_ir,
    route_id: str,
    route_family: str,
    product_instrument: str,
    payoff_family: str,
    required_input_ids: tuple[str, ...],
    market_data_requirements: frozenset[str],
    timeline_roles: frozenset[TimelineRole],
    requested_outputs: tuple[str, ...],
    reporting_currency: str,
) -> NthToDefaultIR | None:
    """Build the typed nth-to-default family IR for helper-backed copula routes."""
    if not _is_nth_to_default_contract(contract, product_ir):
        return None

    product = contract.product
    control_style = _normalized_control_style(product.controller_protocol.controller_style)
    if control_style != "identity":
        raise ValueError(
            "Nth-to-default semantics must remain automatic and cannot declare a strategic controller."
        )
    _require_observables(
        product,
        required_types=("credit_curve",),
        route_name="Nth-to-default",
    )
    _require_required_inputs(
        required_input_ids,
        required=("discount_curve", "credit_curve"),
        route_name="Nth-to-default",
    )
    _require_state_tags(
        product,
        required=("pathwise_only", "remaining_pool", "schedule_state"),
        route_name="Nth-to-default",
    )

    reference_entities = tuple(product.constituents)
    if len(reference_entities) < 2:
        raise ValueError(
            "Nth-to-default semantics require at least two reference entities."
        )
    trigger_rank = max(int(product.selection_count or 1), 1)
    if trigger_rank > len(reference_entities):
        raise ValueError(
            "Nth-to-default trigger rank cannot exceed the reference-entity pool."
        )

    automatic_event_names = _automatic_event_names(product)
    if not automatic_event_names:
        raise ValueError(
            "Nth-to-default semantics require typed event-machine transitions for default-order tracking."
        )

    return NthToDefaultIR(
        route_id=route_id,
        route_family=route_family,
        product_instrument=product_instrument,
        payoff_family=payoff_family,
        required_input_ids=required_input_ids,
        market_data_requirements=market_data_requirements,
        timeline_roles=timeline_roles,
        requested_outputs=requested_outputs,
        reporting_currency=reporting_currency,
        trigger_rank=trigger_rank,
        reference_entities=reference_entities,
        observation_dates=tuple(product.timeline.observation_dates or product.observation_schedule),
        observable_ids=tuple(item.observable_id for item in product.observables),
        observable_types=tuple(item.observable_type for item in product.observables),
        state_field_names=tuple(item.field_name for item in product.state_fields),
        state_tags=_state_tags(product),
        automatic_event_names=automatic_event_names,
    )


def _is_vanilla_european_contract(contract, product_ir) -> bool:
    """Whether the semantic contract fits the tranche-1 vanilla family IR slice."""
    instrument = str(getattr(product_ir, "instrument", ""))
    payoff_family = str(getattr(product_ir, "payoff_family", ""))
    exercise_style = str(getattr(product_ir, "exercise_style", ""))
    return (
        instrument == "european_option"
        and payoff_family == "vanilla_option"
        and exercise_style == "european"
        and str(getattr(contract.product, "instrument_class", "")) == "european_option"
    )


def _is_tranche_one_exercise_lattice_contract(contract, product_ir) -> bool:
    """Whether the semantic contract fits the tranche-1 exercise-lattice slice."""
    instrument = str(getattr(product_ir, "instrument", ""))
    exercise_style = str(getattr(product_ir, "exercise_style", ""))
    product_instrument = str(getattr(contract.product, "instrument_class", ""))
    if instrument == "callable_bond" and product_instrument == "callable_bond":
        return exercise_style == "issuer_call"
    if instrument == "swaption" and product_instrument == "swaption":
        return exercise_style == "bermudan"
    return False


def _is_ranked_observation_basket_contract(contract, product_ir) -> bool:
    """Whether the semantic contract fits the tranche-1 ranked-observation basket slice."""
    instrument = str(getattr(product_ir, "instrument", ""))
    payoff_family = str(getattr(product_ir, "payoff_family", ""))
    product_instrument = str(getattr(contract.product, "instrument_class", ""))
    product_payoff_family = str(getattr(contract.product, "payoff_family", ""))
    payoff_traits = {str(item).strip().lower() for item in getattr(product_ir, "payoff_traits", ())}
    return (
        instrument == "basket_path_payoff"
        and payoff_family == "basket_path_payoff"
        and product_instrument == "basket_path_payoff"
        and product_payoff_family == "basket_path_payoff"
        and "ranked_observation" in payoff_traits
    )


def _is_credit_default_swap_contract(contract, product_ir) -> bool:
    """Whether the semantic contract fits the single-name CDS family IR slice."""
    instrument = str(getattr(product_ir, "instrument", ""))
    payoff_family = str(getattr(product_ir, "payoff_family", ""))
    product_instrument = str(getattr(contract.product, "instrument_class", ""))
    product_payoff_family = str(getattr(contract.product, "payoff_family", ""))
    return (
        instrument == "cds"
        and payoff_family == "credit_default_swap"
        and product_instrument == "cds"
        and product_payoff_family == "credit_default_swap"
    )


def _is_nth_to_default_contract(contract, product_ir) -> bool:
    """Whether the semantic contract fits the nth-to-default family IR slice."""
    instrument = str(getattr(product_ir, "instrument", ""))
    payoff_family = str(getattr(product_ir, "payoff_family", ""))
    product_instrument = str(getattr(contract.product, "instrument_class", ""))
    product_payoff_family = str(getattr(contract.product, "payoff_family", ""))
    return (
        instrument == "nth_to_default"
        and payoff_family == "nth_to_default"
        and product_instrument == "nth_to_default"
        and product_payoff_family == "nth_to_default"
    )


def _required_input_ids(contract, market_binding_spec) -> tuple[str, ...]:
    if market_binding_spec is not None and getattr(market_binding_spec, "bindings", None):
        return tuple(binding.input_id for binding in market_binding_spec.bindings if not getattr(binding, "optional", False))
    return tuple(item.input_id for item in contract.market_data.required_inputs)


def _requested_outputs(valuation_context, market_binding_spec) -> tuple[str, ...]:
    if market_binding_spec is not None and getattr(market_binding_spec, "requested_outputs", ()):
        return tuple(market_binding_spec.requested_outputs)
    if valuation_context is not None and getattr(valuation_context, "requested_outputs", ()):
        return tuple(valuation_context.requested_outputs)
    return ()


def _reporting_currency(contract, valuation_context, market_binding_spec) -> str:
    if market_binding_spec is not None and getattr(market_binding_spec, "reporting_currency", ""):
        return str(market_binding_spec.reporting_currency)
    if valuation_context is not None and getattr(valuation_context, "reporting_policy", None) is not None:
        currency = getattr(valuation_context.reporting_policy, "reporting_currency", "")
        if currency:
            return str(currency)
    conventions = getattr(contract.product, "conventions", None)
    if conventions is None:
        return ""
    return str(
        getattr(conventions, "reporting_currency", "")
        or getattr(conventions, "payment_currency", "")
        or ""
    )


def _timeline_roles_for_contract(contract) -> frozenset[TimelineRole]:
    """Infer the route-relevant timeline roles for one semantic contract."""
    roles: set[TimelineRole] = set()
    product = contract.product
    if (
        product.schedule_dependence
        or product.observation_schedule
        or tuple(getattr(getattr(product, "timeline", None), "observation_dates", ()) or ())
        or tuple(getattr(product, "observables", ()) or ())
    ):
        roles.add(TimelineRole.OBSERVATION)
    if (
        product.exercise_style
        and product.exercise_style != "none"
    ) or tuple(getattr(getattr(product, "timeline", None), "decision_dates", ()) or ()):
        roles.add(TimelineRole.EXERCISE)
    if _typed_settlement_rules(product) or tuple(getattr(getattr(product, "timeline", None), "settlement_dates", ()) or ()):
        roles.add(TimelineRole.SETTLEMENT)
    if _has_typed_payment_semantics(product):
        roles.add(TimelineRole.PAYMENT)
    return frozenset(roles)


def _option_type_for_contract(contract) -> str:
    """Resolve a coarse call/put label from the semantic contract."""
    product = contract.product
    if hasattr(product, "option_type"):
        option_type = str(getattr(product, "option_type")).strip().lower()
        if option_type in {"call", "put"}:
            return option_type
    description = str(getattr(contract, "description", "")).lower()
    if re.search(r"\bput\b", description):
        return "put"
    return "call"


def _normalized_control_style(control_style: str) -> str:
    style = str(control_style or "identity").strip().lower()
    if style not in {"identity", "holder_max", "issuer_min"}:
        raise ValueError(
            "Exercise-lattice family lowering only supports controller styles "
            "'identity', 'holder_max', and 'issuer_min'."
        )
    return style


def _validate_exercise_lattice_control(exercise_style: str, control_style: str) -> None:
    style = str(exercise_style or "none").strip().lower()
    expected = {
        "issuer_call": "issuer_min",
        "bermudan": "holder_max",
        "european": "identity",
        "none": "identity",
    }.get(style)
    if expected is None:
        raise ValueError(
            f"Unsupported exercise-lattice exercise style '{exercise_style}'."
        )
    if control_style != expected:
        raise ValueError(
            "Exercise-lattice control style does not match product exercise style: "
            f"{control_style!r} vs expected {expected!r} for {style!r}."
        )


def _validate_exercise_lattice_timing(product) -> None:
    phase_order = tuple(str(item).strip().lower() for item in product.timeline.phase_order)
    phase_positions = {phase: idx for idx, phase in enumerate(phase_order)}
    observation_phase = _observable_phase(product)
    decision_phase = str(product.controller_protocol.decision_phase or "decision").strip().lower()
    settlement_phase = "settlement"
    for phase in (observation_phase, decision_phase, settlement_phase):
        if phase not in phase_positions:
            raise ValueError(
                f"Exercise-lattice phase order is missing required phase '{phase}'."
            )
    if phase_positions[observation_phase] > phase_positions[decision_phase]:
        raise ValueError(
            "Exercise-lattice semantics require observation/fixing before strategic decision."
        )
    if phase_positions[decision_phase] > phase_positions[settlement_phase]:
        raise ValueError(
            "Exercise-lattice semantics require decision/notice before settlement."
        )

    decision_dates = tuple(product.timeline.decision_dates)
    settlement_dates = tuple(product.timeline.settlement_dates)
    if decision_dates and settlement_dates and min(settlement_dates) < min(decision_dates):
        raise ValueError(
            "Exercise-lattice semantics require settlement on or after the first decision date."
        )


def _validate_ranked_observation_basket_timing(product) -> None:
    phase_order = tuple(str(item).strip().lower() for item in product.timeline.phase_order)
    phase_positions = {phase: idx for idx, phase in enumerate(phase_order)}
    required_phases = ("observation", "determination", "settlement", "state_update")
    for phase in required_phases:
        if phase not in phase_positions:
            raise ValueError(
                f"Ranked-observation basket phase order is missing required phase '{phase}'."
            )
    if phase_positions["observation"] > phase_positions["determination"]:
        raise ValueError(
            "Ranked-observation basket semantics require observation before determination."
        )
    if phase_positions["determination"] > phase_positions["settlement"]:
        raise ValueError(
            "Ranked-observation basket semantics require determination before settlement."
        )
    if phase_positions["settlement"] > phase_positions["state_update"]:
        raise ValueError(
            "Ranked-observation basket semantics require settlement before state update."
        )
    if tuple(product.timeline.decision_dates):
        raise ValueError(
            "Ranked-observation basket semantics cannot declare strategic decision dates."
        )


def _observable_phase(product) -> str:
    if product.observables:
        return str(product.observables[0].availability_phase or "observation").strip().lower()
    return "observation"


def _require_observables(
    product,
    *,
    required_types: tuple[str, ...],
    route_name: str = "Semantic route",
) -> None:
    available = {str(item.observable_type or "").strip().lower() for item in product.observables}
    missing = [item for item in required_types if item not in available]
    if missing:
        raise ValueError(
            f"{route_name} semantics are missing required typed observables: "
            + ", ".join(missing)
        )


def _require_required_inputs(
    required_input_ids: tuple[str, ...],
    *,
    required: tuple[str, ...],
    route_name: str = "Semantic route",
) -> None:
    available = {str(item).strip() for item in required_input_ids}
    missing = [item for item in required if item not in available]
    if missing:
        raise ValueError(
            f"{route_name} semantics are missing required market inputs: "
            + ", ".join(missing)
        )


def _automatic_event_names(product) -> tuple[str, ...]:
    if getattr(product, "event_machine", None) is None:
        return ()
    transitions = getattr(product.event_machine, "transitions", ())
    names = [str(getattr(item, "name", "")).strip() for item in transitions if getattr(item, "name", "")]
    return tuple(name for name in names if name)


def _typed_settlement_rules(product) -> tuple[str, ...]:
    """Return typed settlement rules emitted by obligations, deduplicated in order."""
    rules: list[str] = []
    for obligation in getattr(product, "obligations", ()) or ():
        rule = str(getattr(obligation, "settle_date_rule", "")).strip()
        if rule and rule not in rules:
            rules.append(rule)
    return tuple(rules)


def _has_typed_payment_semantics(product) -> bool:
    """Return whether typed observables or obligations imply payment timing."""
    observable_types = {
        str(getattr(item, "observable_type", "")).strip().lower()
        for item in getattr(product, "observables", ()) or ()
    }
    if "cashflow_schedule" in observable_types:
        return True
    payoff_traits = {
        str(item).strip().lower()
        for item in getattr(product, "payoff_traits", ()) or ()
    }
    if "fixed_coupons" in payoff_traits or "floating_coupons" in payoff_traits:
        return True
    return any(
        "coupon" in str(getattr(obligation, "amount_expression", "")).strip().lower()
        for obligation in getattr(product, "obligations", ()) or ()
    )


def _state_tags(product) -> tuple[str, ...]:
    tags: list[str] = []
    for state_field in product.state_fields:
        for tag in getattr(state_field, "tags", ()):
            text = str(tag).strip()
            if text and text not in tags:
                tags.append(text)
    return tuple(tags)


def _require_state_tags(product, *, required: tuple[str, ...], route_name: str) -> None:
    available = set(_state_tags(product))
    missing = [item for item in required if item not in available]
    if missing:
        raise ValueError(
            f"{route_name} semantics are missing required state tags: " + ", ".join(missing)
        )


def _binding_sources(market_binding_spec, required_input_ids: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    if market_binding_spec is None or not getattr(market_binding_spec, "bindings", None):
        return tuple((input_id, "") for input_id in required_input_ids)
    by_id = {
        str(binding.input_id): str(getattr(binding, "binding_source", "") or "")
        for binding in market_binding_spec.bindings
        if not getattr(binding, "optional", False)
    }
    return tuple((input_id, by_id.get(input_id, "")) for input_id in required_input_ids)
