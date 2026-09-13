"""The existing caplet helper is admitted without implying a joint-forward model."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from trellis.agent.build_gate import evaluate_pre_generation_gate
from trellis.agent.route_registry import evaluate_route_admissibility, find_route_by_id
from trellis.agent.semantic_contract_compiler import compile_semantic_contract
from trellis.agent.semantic_contracts import make_period_rate_option_strip_contract


def _blueprint(instrument_class="cap"):
    return compile_semantic_contract(
        make_period_rate_option_strip_contract(
            description="Uninformative label",
            instrument_class=instrument_class,
            observation_schedule=("2025-02-15", "2025-05-15"),
            preferred_method="monte_carlo",
        ),
        preferred_method="monte_carlo",
    )


@pytest.mark.parametrize("instrument_class", ["cap", "floor"])
def test_forward_marginal_cap_floor_passes_ordinary_route_and_generation_gate(
    instrument_class,
):
    blueprint = _blueprint(instrument_class)
    route = find_route_by_id("monte_carlo_paths")
    decision = evaluate_route_admissibility(route, semantic_blueprint=blueprint)
    assert decision.ok, decision.failures
    plan = SimpleNamespace(primitive_plan=SimpleNamespace(route=route.id))
    gate = evaluate_pre_generation_gate(None, plan, semantic_blueprint=blueprint)
    assert gate.decision == "proceed", gate.reason


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("product", "instrument_class", "swaption"),
        ("product", "exercise_style", "bermudan"),
        ("product", "model_family", "equity_diffusion"),
        ("product", "payoff_family", "path_dependent_rate_option"),
        ("product", "term_fields", {"mc_distribution": "joint_forward_process"}),
        ("product", "term_fields", {"model": "normal"}),
        ("product", "term_fields", {"sampling": "pseudo_random"}),
        ("family", "product_instrument", "floor"),
        ("family", "helper_symbol", "price_event_aware_monte_carlo"),
        ("family", "calibration_binding", object()),
        ("state_spec", "state_variable", "short_rate"),
        ("state_spec", "dimension", 2),
        ("state_spec", "state_tags", ("terminal_markov",)),
        ("state_spec", "state_layout", "vector"),
        ("process_spec", "process_family", "hull_white_1f"),
        ("process_spec", "simulation_scheme", "exact_ou"),
        ("process_spec", "process_tags", ()),
        ("path_requirement_spec", "requirement_kind", "full_path"),
        ("path_requirement_spec", "snapshot_schedule_role", "decision_dates"),
        ("path_requirement_spec", "replay_mode", "deterministic_timeline"),
        ("path_requirement_spec", "reducer_kinds", ("running_average",)),
        ("path_requirement_spec", "stored_fields", ("short_rate",)),
        ("payoff_reducer_spec", "reducer_kind", "compiled_schedule_payoff"),
        ("payoff_reducer_spec", "output_semantics", "joint_forward_payoff"),
        ("control_spec", "control_style", "issuer_call"),
        ("measure_spec", "measure_family", "risk_neutral"),
        ("measure_spec", "numeraire_binding", "discount_curve"),
        ("blueprint", "calibration_step", object()),
    ],
)
def test_forward_marginal_profile_mismatches_honestly_block_before_generation(
    section, field, value
):
    blueprint = _blueprint()
    if section == "product":
        blueprint = replace(
            blueprint,
            contract=replace(
                blueprint.contract,
                product=replace(blueprint.contract.product, **{field: value}),
            ),
        )
    elif section == "blueprint":
        blueprint = replace(blueprint, **{field: value})
    else:
        family = blueprint.dsl_lowering.family_ir
        if section == "family":
            family = replace(family, **{field: value})
        else:
            family = replace(
                family, **{section: replace(getattr(family, section), **{field: value})}
            )
        blueprint = replace(
            blueprint, dsl_lowering=replace(blueprint.dsl_lowering, family_ir=family)
        )
    route = find_route_by_id("monte_carlo_paths")
    decision = evaluate_route_admissibility(route, semantic_blueprint=blueprint)
    assert not decision.ok
    assert any(
        failure.startswith("unsupported_forward_marginal_profile:")
        for failure in decision.failures
    )
    plan = SimpleNamespace(primitive_plan=SimpleNamespace(route=route.id))
    gate = evaluate_pre_generation_gate(None, plan, semantic_blueprint=blueprint)
    assert gate.decision == "block"
    assert gate.route_admissibility_failures == decision.failures
