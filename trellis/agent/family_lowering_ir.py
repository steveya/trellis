"""Typed family-level lowering IRs for migrated semantic routes."""

from __future__ import annotations

from dataclasses import dataclass
import re

from trellis.core.types import TimelineRole


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
class AnalyticalBlack76IR(BaseFamilyLoweringIR):
    """Typed lowering payload for vanilla analytical Black76 routes."""

    option_type: str = "call"
    kernel_symbol: str = "black76_call"
    market_mapping: str = "spot_discount_vol_to_forward"


@dataclass(frozen=True)
class VanillaEquityPDEIR(BaseFamilyLoweringIR):
    """Typed lowering payload for vanilla equity theta-method PDE routes."""

    option_type: str = "call"
    theta: float = 0.5
    helper_symbol: str = "price_vanilla_equity_option_pde"
    market_mapping: str = "equity_spot_discount_black_vol"


@dataclass(frozen=True)
class ExerciseLatticeIR(BaseFamilyLoweringIR):
    """Typed lowering payload for schedule-dependent strategic-rights lattice routes."""

    control_style: str = "identity"
    controller_role: str = "none"
    exercise_style: str = "none"
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


FamilyLoweringIR = (
    AnalyticalBlack76IR
    | VanillaEquityPDEIR
    | ExerciseLatticeIR
    | CorrelatedBasketMonteCarloIR
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

        if route_id == "vanilla_equity_theta_pde":
            return VanillaEquityPDEIR(
                option_type=_option_type_for_contract(contract),
                theta=0.5,
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
