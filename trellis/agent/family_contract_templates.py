"""Checked-in family-contract templates for known product families."""

from __future__ import annotations

from trellis.agent.family_contracts import (
    BlueprintHints,
    FamilyContract,
    MarketDataContract,
    MarketInputSpec,
    MethodContract,
    ProductSemantics,
    SensitivityContract,
    ValidationContract,
)
from trellis.agent.semantic_contracts import (
    SemanticBlueprintHints,
    SemanticContract,
    SemanticMarketDataContract,
    SemanticMarketInputSpec,
    SemanticMethodContract,
    SemanticProductSemantics,
    SemanticValidationContract,
)


_QUANTO_OPTION_TEMPLATE = FamilyContract(
    product=ProductSemantics(
        family_id="quanto_option",
        family_version="c0",
        instrument="quanto_option",
        instrument_aliases=("quanto", "quanto_option", "single_underlier_quanto"),
        payoff_family="vanilla_option",
        payoff_traits=("vanilla_option", "cross_currency", "fx_linked"),
        exercise_style="european",
        path_dependence="terminal_markov",
        schedule_dependence=False,
        state_dependence="terminal_markov",
        model_family="equity_diffusion",
        schedule_semantics=("single_expiry",),
        state_variables=("underlier_spot", "fx_rate"),
        event_transitions=(
            "observe_underlier_and_fx_at_expiry",
            "settle_in_domestic_currency",
        ),
    ),
    market_data=MarketDataContract(
        required_inputs=(
            MarketInputSpec(
                input_id="domestic_discount_curve",
                description="Discounting curve for the payout currency.",
                capability="discount_curve",
                aliases=("discount_curve", "domestic_curve"),
                connector_hint="Resolve the payout-currency discount curve from the selected market-data connector.",
            ),
            MarketInputSpec(
                input_id="foreign_discount_curve",
                description="Discount or carry curve for the underlying currency.",
                capability="forward_curve",
                aliases=("forecast_curve", "foreign_curve", "carry_curve"),
                connector_hint="Resolve the foreign discount/carry input or explicit forecast-curve bridge.",
                derivable_from=("domestic_discount_curve", "fx_spot"),
                allowed_provenance=("observed", "derived"),
            ),
            MarketInputSpec(
                input_id="underlier_spot",
                description="Current spot for the foreign underlier.",
                capability="spot",
                aliases=("spot", "underlier_spot"),
                connector_hint="Resolve the underlying spot from the market snapshot or equity connector.",
            ),
            MarketInputSpec(
                input_id="fx_spot",
                description="FX spot linking underlier and domestic payout currencies.",
                capability="fx_rates",
                aliases=("fx_rate", "fx_rates", "fx_spot"),
                connector_hint="Resolve the FX spot for the contract currency pair.",
            ),
            MarketInputSpec(
                input_id="underlier_vol",
                description="Volatility input for the underlier process.",
                capability="black_vol_surface",
                aliases=("vol_surface", "underlier_vol_surface"),
                connector_hint="Resolve the underlier volatility input from the available vol surface.",
            ),
            MarketInputSpec(
                input_id="fx_vol",
                description="Volatility input for the FX process.",
                capability="black_vol_surface",
                aliases=("fx_vol_surface", "fx_vol"),
                connector_hint="Resolve the FX volatility input for the contract currency pair.",
            ),
            MarketInputSpec(
                input_id="underlier_fx_correlation",
                description="Cross-correlation between the underlier and FX drivers.",
                capability="model_parameters",
                aliases=("correlation", "quanto_correlation"),
                connector_hint="Resolve observed correlation if available, or block unless an explicit estimation policy allows derivation.",
            ),
        ),
        optional_inputs=(
            MarketInputSpec(
                input_id="dividend_yield",
                description="Optional carry input for the underlying asset.",
                capability="forward_curve",
                aliases=("dividend", "borrow_cost"),
                connector_hint="Resolve dividend or carry data when available.",
                allowed_provenance=("observed", "derived"),
            ),
            MarketInputSpec(
                input_id="local_vol_surface",
                description="Optional local-vol input for later extensions.",
                capability="local_vol_surface",
                aliases=("local_vol",),
                connector_hint="Optional future extension for surface-driven quanto routes.",
            ),
        ),
        derivable_inputs=("foreign_discount_curve",),
        estimation_policy=(
            "Do not fabricate fx_vol or underlier_fx_correlation.",
            "Foreign discounting may use an explicit forecast/discount bridge when the runtime declares that bridge.",
        ),
        provenance_requirements=(
            "Every runtime input must be labeled as observed, derived, estimated, or user-supplied.",
        ),
        missing_data_error_policy=(
            "Missing required quanto inputs must raise a family-specific error naming the missing field.",
        ),
    ),
    methods=MethodContract(
        candidate_methods=("analytical", "monte_carlo"),
        reference_methods=("analytical",),
        production_methods=("analytical", "monte_carlo"),
        unsupported_variants=(
            "early_exercise_quanto",
            "path_dependent_quanto",
            "basket_quanto",
            "stochastic_rates_quanto",
        ),
        method_limitations=(
            "Analytical route is limited to single-underlier European quanto semantics.",
            "Monte Carlo route requires explicit underlier/FX joint state semantics.",
        ),
        preferred_method="analytical",
    ),
    sensitivities=SensitivityContract(
        support_level="bump_only",
        supported_measures=("dv01", "duration", "convexity", "key_rate_durations", "vega"),
        stability_notes=(
            "Native closed-form Greeks are not yet a unified runtime contract.",
            "Cross-currency sensitivity support should stay honest about repricing-based implementation.",
        ),
    ),
    validation=ValidationContract(
        bundle_hints=("quanto_option",),
        universal_checks=("check_non_negativity", "check_price_sanity"),
        family_checks=("check_quanto_required_inputs", "check_quanto_cross_currency_semantics"),
        comparison_targets=("analytical_vs_mc",),
    ),
    blueprint=BlueprintHints(
        target_modules=(
            "trellis.agent.planner",
            "trellis.agent.quant",
            "trellis.agent.codegen_guardrails",
            "trellis.models.analytical.quanto",
            "trellis.models.processes.correlated_gbm",
        ),
        primitive_families=("quanto_adjustment_analytical", "correlated_gbm_monte_carlo"),
        adapter_obligations=(
            "bind_domestic_and_foreign_curves",
            "bind_underlier_and_fx_vols",
            "bind_underlier_fx_correlation",
        ),
        proving_tasks=("T105",),
        spec_schema_hints=("QuantoOptionAnalyticalPayoff", "QuantoOptionMonteCarloPayoff"),
    ),
    description="Single-underlier European quanto option contract template.",
)


_TEMPLATES = {
    "quanto_option": _QUANTO_OPTION_TEMPLATE,
}


def get_family_contract_template(family_id: str) -> FamilyContract:
    """Return a checked-in template for one known family id."""
    key = str(family_id).strip().lower()
    try:
        return _TEMPLATES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown family contract template: {family_id!r}") from exc


def list_family_contract_templates() -> tuple[str, ...]:
    """Return the checked-in template ids."""
    return tuple(sorted(_TEMPLATES))


def _family_market_input_to_semantic(spec: MarketInputSpec) -> SemanticMarketInputSpec:
    """Convert a family MarketInputSpec to its semantic equivalent."""
    return SemanticMarketInputSpec(
        input_id=spec.input_id,
        description=spec.description,
        capability=spec.capability,
        aliases=spec.aliases,
        connector_hint=spec.connector_hint,
        derivable_from=spec.derivable_from,
        allowed_provenance=spec.allowed_provenance,
    )


def _sensitivity_as_method_limitations(
    sensitivities: SensitivityContract,
) -> tuple[str, ...]:
    """Encode family sensitivity contract info as method-limitation strings.

    Sensitivity metadata from family contracts is folded into the semantic
    method_limitations tuple so it survives the conversion without requiring
    a new field on SemanticContract.
    """
    lines: list[str] = []
    if sensitivities.support_level and sensitivities.support_level != "unsupported":
        lines.append(f"sensitivity_support_level:{sensitivities.support_level}")
    if sensitivities.supported_measures:
        measures = ",".join(sensitivities.supported_measures)
        lines.append(f"sensitivity_supported_measures:{measures}")
    for note in sensitivities.stability_notes:
        lines.append(f"sensitivity_note:{note}")
    return tuple(lines)


def _family_semantic_overrides(family_id: str) -> dict:
    """Return per-family semantic-product fields that are required by the
    semantic validator but absent from the family contract schema.

    These are checked-in enrichments that bridge the gap between the two
    schemas until family contracts are fully retired.
    """
    if family_id == "quanto_option":
        return {
            "underlier_structure": "cross_currency_single_underlier",
            "payoff_rule": "quanto_adjusted_vanilla_payoff",
            "settlement_rule": "cash_settle_at_expiry_after_fx_conversion",
            "observation_schedule": ("single_expiry",),
            "constituents": ("underlier",),
        }
    return {}


def _family_contract_to_semantic(fc: FamilyContract) -> SemanticContract:
    """Convert a FamilyContract into a pre-resolved SemanticContract."""
    overrides = _family_semantic_overrides(fc.product.family_id)
    product = SemanticProductSemantics(
        semantic_id=fc.product.family_id,
        semantic_version=fc.product.family_version,
        instrument_class=fc.product.instrument,
        instrument_aliases=fc.product.instrument_aliases,
        payoff_family=fc.product.payoff_family,
        underlier_structure=overrides.get("underlier_structure", ""),
        payoff_rule=overrides.get("payoff_rule", ""),
        settlement_rule=overrides.get("settlement_rule", ""),
        payoff_traits=fc.product.payoff_traits,
        exercise_style=fc.product.exercise_style,
        path_dependence=fc.product.path_dependence,
        schedule_dependence=fc.product.schedule_dependence,
        state_dependence=fc.product.state_dependence,
        model_family=fc.product.model_family,
        observation_schedule=overrides.get("observation_schedule", ()),
        constituents=overrides.get("constituents", ()),
        state_variables=fc.product.state_variables,
        event_transitions=fc.product.event_transitions,
    )
    market_data = SemanticMarketDataContract(
        required_inputs=tuple(
            _family_market_input_to_semantic(inp)
            for inp in fc.market_data.required_inputs
        ),
        optional_inputs=tuple(
            _family_market_input_to_semantic(inp)
            for inp in fc.market_data.optional_inputs
        ),
        derivable_inputs=fc.market_data.derivable_inputs,
        estimation_policy=fc.market_data.estimation_policy,
        provenance_requirements=fc.market_data.provenance_requirements,
        missing_data_error_policy=fc.market_data.missing_data_error_policy,
    )
    sensitivity_limitations = _sensitivity_as_method_limitations(fc.sensitivities)
    methods = SemanticMethodContract(
        candidate_methods=fc.methods.candidate_methods,
        reference_methods=fc.methods.reference_methods,
        production_methods=fc.methods.production_methods,
        unsupported_variants=fc.methods.unsupported_variants,
        method_limitations=(*fc.methods.method_limitations, *sensitivity_limitations),
        preferred_method=fc.methods.preferred_method,
    )
    validation = SemanticValidationContract(
        bundle_hints=fc.validation.bundle_hints,
        universal_checks=fc.validation.universal_checks,
        semantic_checks=fc.validation.family_checks,
        comparison_targets=fc.validation.comparison_targets,
        reduction_cases=fc.validation.reduction_cases,
    )
    blueprint = SemanticBlueprintHints(
        target_modules=fc.blueprint.target_modules,
        primitive_families=fc.blueprint.primitive_families,
        adapter_obligations=fc.blueprint.adapter_obligations,
        proving_tasks=fc.blueprint.proving_tasks,
        blocked_by=fc.blueprint.blocked_by,
        spec_schema_hints=fc.blueprint.spec_schema_hints,
    )
    return SemanticContract(
        product=product,
        market_data=market_data,
        methods=methods,
        validation=validation,
        blueprint=blueprint,
        description=fc.description,
    )


def family_template_as_semantic_contract(family_id: str) -> SemanticContract | None:
    """Convert a family contract template to a pre-resolved SemanticContract.

    This bridges the family fast-path into the unified semantic compilation
    pipeline.  Returns None if the family_id has no template.
    """
    key = str(family_id).strip().lower()
    fc = _TEMPLATES.get(key)
    if fc is None:
        return None
    return _family_contract_to_semantic(fc)
