"""Typed comparison-target contracts and execution-artifact coherence tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace

import pytest


def _proof_tasks():
    from trellis.agent.task_manifests import load_task_manifest

    return {
        task["id"]: task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
    }


def _contracts_for(task_id: str):
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    return {
        target.target_id: target.contract
        for target in build_comparison_harness_plan(_proof_tasks()[task_id]).targets
    }


def test_target_projection_accepts_structured_variant_metadata():
    from trellis.agent.comparison_target_contracts import (
        ComparisonTargetContract,
        project_product_ir_for_comparison_target,
    )
    from trellis.agent.knowledge.decompose import decompose_to_ir

    product_ir = decompose_to_ir(
        "European two-asset basket option",
        instrument_type="basket_option",
    )
    contract = ComparisonTargetContract(
        target_id="transform",
        method="fft_pricing",
        payoff_family="basket_option",
        observation_style="terminal",
        model_family="multi_asset_diffusion",
        variant_parameters={
            "dimensions": 2,
            "transform": "hurd_zhou",
            "grid_controls": {"size": 256, "frequency_step": 0.25},
        },
    )

    projected = project_product_ir_for_comparison_target(product_ir, contract)

    assert projected.candidate_engine_families == ("fft_pricing",)
    assert "two_asset_terminal_basket" in projected.payoff_traits
    assert "spread" in projected.payoff_traits


def test_t04_declares_tree_reference_and_european_lower_bound_relation():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    plan = build_comparison_harness_plan(_proof_tasks()["T04"])
    targets = {target.target_id: target for target in plan.targets}

    assert plan.reference_target == "hw_tree_bermudan"
    assert targets["hw_tree_bermudan"].is_reference is True
    assert targets["black76_european_lower_bound"].relation == "<="


def test_declared_option_comparison_targets_preserve_method_variants_and_axes():
    t27 = _contracts_for("T27")
    assert t27["polynomial"].method == "monte_carlo"
    assert t27["polynomial"].variant_parameters == {
        "regression_basis": "polynomial"
    }
    assert t27["laguerre"].variant_parameters["regression_basis"] == "laguerre"
    assert t27["hermite"].variant_parameters["regression_basis"] == "hermite"
    assert t27["chebyshev"].variant_parameters["regression_basis"] == "chebyshev"
    assert t27["high_step_tree_2000"].method == "rate_tree"
    assert t27["high_step_tree_2000"].variant_parameters["tree_steps"] == 2000
    assert t27["high_step_tree_2000"].spec_overrides["tree_steps"] == 2000
    assert t27["polynomial"].exercise_style == "american"
    assert t27["polynomial"].payoff_family == "vanilla_option"

    t28 = _contracts_for("T28")
    assert t28["euler_heston"].variant_parameters["scheme"] == "euler"
    assert t28["qe_heston"].variant_parameters["scheme"] == "heston_qe"
    assert t28["qe_heston"].model_family == "stochastic_volatility"
    assert t28["heston_fft"].method == "fft_pricing"
    assert t28["heston_fft"].route_id == "transform_fft"
    assert t28["heston_fft"].variant_parameters["characteristic_family"] == (
        "heston_log_spot"
    )

    t29 = _contracts_for("T29")
    assert t29["mc_asian"].method == "monte_carlo"
    assert t29["mc_asian"].observation_style == "fixed_schedule"
    assert t29["mc_asian"].variant_parameters["observation_frequency"] == "monthly"
    assert t29["turnbull_wakeman_approx"].method == "analytical"
    assert t29["turnbull_wakeman_approx"].variant_parameters["approximation"] == (
        "turnbull_wakeman"
    )


def test_declared_basket_comparison_targets_preserve_method_and_sampling_identity():
    t35 = _contracts_for("T35")
    assert t35["mc_basket"].method == "monte_carlo"
    assert t35["mc_basket"].variant_parameters["sampling"] == "pseudo_random"
    assert t35["mc_basket_antithetic"].spec_overrides["antithetic"] is True

    t47 = _contracts_for("T47")
    assert t47["spread_fft_2d"].method == "fft_pricing"
    assert t47["spread_fft_2d"].variant_parameters == {
        "dimensions": 2,
        "transform": "hurd_zhou",
    }
    assert t47["spread_mc"].method == "monte_carlo"

    t126 = _contracts_for("T126")
    assert t126["kirk_spread"].method == "analytical"
    assert t126["kirk_spread"].variant_parameters["approximation"] == "kirk"
    assert t126["mc_spread_2d"].method == "monte_carlo"
    assert t126["fft_spread_2d"].method == "fft_pricing"


def test_callable_fixed_income_targets_have_explicit_executable_contracts():
    tasks = _proof_tasks()

    t02 = _contracts_for("T02")
    assert set(t02) == {"bdt_tree", "hull_white_tree"}
    assert t02["bdt_tree"].explicit is True
    assert t02["bdt_tree"].route_id == "exercise_lattice"
    assert t02["bdt_tree"].route_family == "rate_lattice"
    assert t02["bdt_tree"].backend_binding_id == (
        "trellis.models.trees.algebra.price_on_lattice"
    )
    assert t02["bdt_tree"].validation_bundle_id == "rate_tree:callable_bond"
    assert t02["bdt_tree"].variant_parameters == {"lattice_model": "bdt"}
    assert t02["bdt_tree"].spec_overrides == {}
    assert t02["hull_white_tree"].variant_parameters == {
        "lattice_model": "hull_white"
    }
    assert t02["hull_white_tree"].exercise_style == "issuer_call"
    assert t02["hull_white_tree"].observation_style == "exercise_schedule"

    # The old callable-tree "symmetry" label was not an independent
    # implementation.  T05 is narrowed to the puttable target whose standard
    # validation bundle proves the meaningful straight-bond lower bound.
    assert tasks["T05"]["cross_validate"]["internal"] == ["puttable_tree"]
    t05 = _contracts_for("T05")
    assert set(t05) == {"puttable_tree"}
    assert t05["puttable_tree"].explicit is True
    assert t05["puttable_tree"].route_id == "exercise_lattice"
    assert t05["puttable_tree"].backend_binding_id == (
        "trellis.models.trees.algebra.price_on_lattice"
    )
    assert t05["puttable_tree"].validation_bundle_id == "rate_tree:puttable_bond"
    assert t05["puttable_tree"].payoff_family == "puttable_fixed_income"
    assert t05["puttable_tree"].exercise_style == "holder_put"

    t17 = _contracts_for("T17")
    assert set(t17) == {"hw_pde_theta", "hw_rate_tree"}
    assert t17["hw_pde_theta"].method == "pde_solver"
    assert t17["hw_pde_theta"].route_id == "pde_theta_1d"
    assert t17["hw_pde_theta"].route_family == "pde_solver"
    assert t17["hw_pde_theta"].backend_binding_id == (
        "trellis.models.callable_bond_pde.price_callable_bond_pde"
    )
    assert t17["hw_pde_theta"].validation_bundle_id == (
        "pde_solver:callable_bond"
    )
    assert t17["hw_pde_theta"].variant_parameters == {
        "pricing_method": "pde_solver",
        "theta": 0.5,
        "model_parameter_set": "t17_hull_white_comparison:hull_white",
        "mean_reversion": 0.1,
        "sigma": 0.01,
    }
    assert t17["hw_rate_tree"].method == "rate_tree"
    assert t17["hw_rate_tree"].backend_binding_id == (
        "trellis.models.trees.algebra.price_on_lattice"
    )
    assert t17["hw_rate_tree"].variant_parameters == {
        "pricing_method": "rate_tree",
        "lattice_model": "hull_white",
        "model_parameter_set": "t17_hull_white_comparison:hull_white",
        "mean_reversion": 0.1,
        "sigma": 0.01,
    }


def test_comparison_execution_binding_uses_validation_contract_axis_fallbacks():
    from trellis.agent.executor import _comparison_execution_binding_metadata

    binding = _comparison_execution_binding_metadata(
        pricing_plan=SimpleNamespace(method="pde_solver"),
        generation_plan=SimpleNamespace(
            lowering_route_id="pde_theta_1d",
            backend_route_family="",
            backend_binding_id="callable-pde-binding",
            validation_bundle_id="",
            primitive_plan=None,
        ),
        product_ir=SimpleNamespace(
            payoff_family="callable_fixed_income",
            exercise_style="issuer_call",
            model_family="interest_rate",
            schedule_dependence=True,
            state_dependence="terminal_markov",
            underlying_asset_class="rate",
            option_type="call",
        ),
        request_metadata={},
        validation_contract=SimpleNamespace(
            route_id="pde_theta_1d",
            route_family="pde_solver",
            backend_binding_id="callable-pde-binding",
            bundle_id="pde_solver:callable_bond",
        ),
    )

    assert binding["selected_route_family"] == "pde_solver"
    assert binding["selected_validation_bundle_id"] == "pde_solver:callable_bond"
    assert binding["selected_semantic_axes"]["observation_style"] == (
        "exercise_schedule"
    )


def test_legacy_comparison_target_fallback_is_typed_and_explicitly_unresolved():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    plan = build_comparison_harness_plan(
        {
            "id": "LEGACY",
            "construct": ["monte_carlo", "transforms"],
            "cross_validate": {"internal": ["plain_mc", "fft"]},
        }
    )

    contracts = {target.target_id: target.contract for target in plan.targets}
    assert contracts["plain_mc"].method == "monte_carlo"
    assert contracts["fft"].method == "fft_pricing"
    assert contracts["plain_mc"].resolution_source == "legacy_target_inference"
    assert contracts["plain_mc"].explicit is False
    with pytest.raises(TypeError):
        contracts["plain_mc"].variant_parameters["sampling"] = "antithetic"


def test_comparison_target_contract_rejects_unknown_method_family():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract

    with pytest.raises(ValueError, match="unknown method"):
        ComparisonTargetContract(target_id="mystery", method="magic_solver")


def test_comparison_target_description_carries_typed_executable_obligations():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _description_for_comparison_target,
    )

    contract = ComparisonTargetContract(
        target_id="qe_heston",
        method="monte_carlo",
        variant_parameters={"scheme": "heston_qe"},
    )

    description = _description_for_comparison_target(
        "Price a Heston option.",
        ComparisonBuildTarget(contract=contract),
    )

    assert '"target_id":"qe_heston"' in description
    assert '"scheme":"heston_qe"' in description
    assert "exercise every declared variant" in description
    assert "__trellis_comparison_bindings__" in description
    assert '"target_contract"' in description


def test_comparison_target_description_serializes_non_json_native_values():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _description_for_comparison_target,
    )

    contract = ComparisonTargetContract(
        target_id="scheduled_mc",
        method="monte_carlo",
        variant_parameters={"observation_date": date(2026, 7, 15)},
    )

    description = _description_for_comparison_target(
        "Price a scheduled option.",
        ComparisonBuildTarget(contract=contract),
    )

    assert '"observation_date":"2026-07-15"' in description


def test_build_result_payload_separates_requested_contract_from_artifact_evidence():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.knowledge.autonomous import BuildResult
    from trellis.agent.task_runtime import _build_result_payload

    requested = ComparisonTargetContract(
        target_id="qe_heston",
        method="monte_carlo",
        variant_parameters={"scheme": "heston_qe"},
    )
    result = BuildResult(
        payoff_cls=None,
        success=False,
        attempts=1,
        comparison_target_contract={},
        comparison_binding_evidence_source="cached_artifact_declaration_missing",
    )

    payload = _build_result_payload(
        result,
        preferred_method="monte_carlo",
        comparison_target_contract=requested,
    )

    assert payload["comparison_target_contract"] == {}
    assert payload["requested_comparison_target_contract"] == requested.to_payload()
    assert payload["comparison_binding_evidence_source"] == (
        "cached_artifact_declaration_missing"
    )


def test_assisted_comparison_retry_preserves_requested_target_contract():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.intra_run_learning import RecoveryMode
    from trellis.agent.knowledge.autonomous import BuildResult
    from trellis.agent.task_runtime import (
        _build_result_payload,
        _maybe_retry_with_intra_run_learning,
    )

    contract = ComparisonTargetContract(
        target_id="qe_heston",
        method="monte_carlo",
        route_id="monte_carlo_paths",
        variant_parameters={"scheme": "heston_qe"},
    )
    failed = BuildResult(
        payoff_cls=None,
        success=False,
        attempts=1,
        failures=[
            "resolve_double_barrier_inputs() got an unexpected keyword argument "
            "'spot'"
        ],
        reflection={
            "lesson_captured": "comparison_retry_contract",
            "gaps_identified": ["Bind the declared Heston scheme explicitly"],
        },
    )
    initial_payload = _build_result_payload(
        failed,
        preferred_method=contract.method,
        comparison_target_contract=contract,
    )
    recovered = BuildResult(
        payoff_cls=type("RecoveredHestonPayoff", (), {}),
        success=True,
        attempts=1,
        comparison_target_contract=contract.to_payload(),
    )
    calls = []

    def retry_build(**kwargs):
        calls.append(kwargs)
        return recovered

    _, retry_payload, record = _maybe_retry_with_intra_run_learning(
        build_fn=retry_build,
        build_kwargs={
            "request_metadata": {
                "comparison_target_contract": contract.to_payload()
            }
        },
        initial_result=failed,
        initial_payload=initial_payload,
        target_id=contract.target_id,
        preferred_method=contract.method,
        reference_target=False,
        task_kind="pricing",
        instrument_type="heston_option",
        recovery_mode=RecoveryMode.ASSISTED,
        comparison_target_contract=contract,
    )

    assert len(calls) == 1
    assert calls[0]["request_metadata"]["comparison_target_contract"] == (
        contract.to_payload()
    )
    assert retry_payload["requested_comparison_target_contract"] == (
        contract.to_payload()
    )
    assert record["recovered"] is True


def test_comparison_harness_rejects_unreferenced_target_declarations():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    with pytest.raises(ValueError, match="unreferenced comparison target contracts"):
        build_comparison_harness_plan(
            {
                "id": "BAD",
                "construct": "monte_carlo",
                "cross_validate": {
                    "internal": ["plain_mc"],
                    "target_contracts": {
                        "plain_mc": {"method": "monte_carlo"},
                        "unused_mc": {"method": "monte_carlo"},
                    },
                },
            }
        )


def test_comparison_harness_rejects_explicit_contracts_that_omit_construct_method():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    with pytest.raises(ValueError, match="do not cover construct method families: fft_pricing"):
        build_comparison_harness_plan(
            {
                "id": "BAD",
                "construct": ["monte_carlo", "fft_pricing"],
                "cross_validate": {
                    "internal": ["plain_mc", "nominal_fft"],
                    "target_contracts": {
                        "plain_mc": {"method": "monte_carlo"},
                        "nominal_fft": {"method": "monte_carlo"},
                    },
                },
            }
        )


def test_comparison_harness_rejects_partial_explicit_contract_set():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    with pytest.raises(ValueError, match="missing comparison target contracts: nominal_fft"):
        build_comparison_harness_plan(
            {
                "id": "BAD",
                "construct": ["monte_carlo", "fft_pricing"],
                "cross_validate": {
                    "internal": ["plain_mc", "nominal_fft"],
                    "target_contracts": {
                        "plain_mc": {"method": "monte_carlo"},
                    },
                },
            }
        )


def test_comparison_harness_rejects_empty_explicit_target_declaration():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    with pytest.raises(ValueError, match="must not be empty"):
        build_comparison_harness_plan(
            {
                "id": "BAD",
                "construct": "monte_carlo",
                "cross_validate": {
                    "target_contracts": {"monte_carlo": {}},
                },
            }
        )


def test_construct_derived_target_can_be_the_explicit_reference():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    plan = build_comparison_harness_plan(
        {
            "id": "REFERENCE",
            "construct": ["monte_carlo", "analytical"],
            "cross_validate": {"reference_target": "analytical"},
        }
    )

    assert plan.reference_target == "analytical"
    assert [target.target_id for target in plan.targets if target.is_reference] == [
        "analytical"
    ]


def test_non_introspectable_legacy_payoff_factory_falls_back_to_three_arguments(
    monkeypatch,
):
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import _call_comparison_payoff_factory

    monkeypatch.setattr(
        "trellis.agent.task_runtime.inspect.signature",
        lambda _factory: (_ for _ in ()).throw(ValueError("no signature")),
    )
    calls = []

    def legacy_factory(payoff_cls, spec_schema, settle):
        calls.append((payoff_cls, spec_schema, settle))
        return "legacy-payoff"

    contract = ComparisonTargetContract(target_id="mc", method="monte_carlo")
    result = _call_comparison_payoff_factory(
        legacy_factory,
        object,
        "schema",
        "settle",
        contract,
    )

    assert result == "legacy-payoff"
    assert calls == [(object, "schema", "settle")]


def test_non_introspectable_payoff_factory_does_not_swallow_internal_type_error(
    monkeypatch,
):
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import _call_comparison_payoff_factory

    monkeypatch.setattr(
        "trellis.agent.task_runtime.inspect.signature",
        lambda _factory: (_ for _ in ()).throw(ValueError("no signature")),
    )

    def broken_factory(_payoff_cls, _spec_schema, _settle, _contract):
        raise TypeError("factory implementation failed")

    contract = ComparisonTargetContract(target_id="mc", method="monte_carlo")
    with pytest.raises(TypeError, match="factory implementation failed"):
        _call_comparison_payoff_factory(
            broken_factory,
            object,
            "schema",
            "settle",
            contract,
        )


def test_comparison_harness_rejects_unknown_reference_target():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    with pytest.raises(ValueError, match="unknown comparison reference target"):
        build_comparison_harness_plan(
            {
                "id": "BAD",
                "construct": "monte_carlo",
                "cross_validate": {
                    "internal": ["plain_mc", "antithetic_mc"],
                    "reference_target": "missing_reference",
                },
            }
        )


def test_comparison_harness_rejects_multiple_reference_targets():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    with pytest.raises(ValueError, match="exactly one comparison reference target"):
        build_comparison_harness_plan(
            {
                "id": "BAD",
                "construct": "monte_carlo",
                "cross_validate": {
                    "internal": ["plain_mc"],
                    "analytical": "black_scholes",
                    "reference_target": "plain_mc",
                },
            }
        )


def test_comparison_harness_rejects_duplicate_target_ids():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    with pytest.raises(ValueError, match="duplicate comparison target ids"):
        build_comparison_harness_plan(
            {
                "id": "BAD",
                "construct": "monte_carlo",
                "cross_validate": {"internal": ["plain_mc", "plain_mc"]},
            }
        )


def test_comparison_harness_rejects_duplicate_contract_ids():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    with pytest.raises(ValueError, match="duplicate comparison target contract ids"):
        build_comparison_harness_plan(
            {
                "id": "BAD",
                "construct": "monte_carlo",
                "cross_validate": {
                    "internal": ["plain_mc", "antithetic_mc"],
                    "target_contracts": {
                        "plain_mc": {
                            "contract_id": "shared-contract",
                            "method": "monte_carlo",
                        },
                        "antithetic_mc": {
                            "contract_id": "shared-contract",
                            "method": "monte_carlo",
                        },
                    },
                },
            }
        )


def test_comparison_harness_rejects_malformed_target_declaration():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    with pytest.raises(ValueError, match="must be a mapping"):
        build_comparison_harness_plan(
            {
                "id": "BAD",
                "construct": "monte_carlo",
                "cross_validate": {
                    "internal": ["plain_mc"],
                    "target_contracts": {"plain_mc": "monte_carlo"},
                },
            }
        )


def test_cross_validation_rejects_unique_artifact_that_does_not_bind_variant():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    class UniquePayoff:
        pass

    contract = ComparisonTargetContract(
        target_id="qe_heston",
        method="monte_carlo",
        variant_parameters={"scheme": "heston_qe"},
    )
    result = SimpleNamespace(
        success=True,
        payoff_cls=UniquePayoff,
        selected_method="monte_carlo",
        comparison_target_contract=contract.to_payload(),
    )
    price_calls = []

    comparison = _cross_validate_comparison_task(
        [ComparisonBuildTarget(contract=contract)],
        {contract.target_id: result},
        market_state=object(),
        configured_targets={},
        payoff_factory=lambda *_args: UniquePayoff(),
        price_fn=lambda *_args: price_calls.append(True) or 10.0,
    )

    assert comparison["status"] == "semantic_artifact_mismatch"
    assert comparison["prices"] == {}
    assert price_calls == []
    report = comparison["artifact_coherence"][contract.target_id]
    assert report["failures"][-1]["code"] == "missing_variant_execution_evidence"


def test_cross_validation_allows_unique_artifact_with_declared_variant_binding():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    contract = ComparisonTargetContract(
        target_id="qe_heston",
        method="monte_carlo",
        variant_parameters={"scheme": "heston_qe"},
    )

    class UniquePayoff:
        __trellis_comparison_bindings__ = {
            "qe_heston": {"target_contract": contract.to_payload()}
        }
    result = SimpleNamespace(
        success=True,
        payoff_cls=UniquePayoff,
        selected_method="monte_carlo",
        comparison_target_contract=contract.to_payload(),
    )

    comparison = _cross_validate_comparison_task(
        [ComparisonBuildTarget(contract=contract)],
        {contract.target_id: result},
        market_state=object(),
        configured_targets={},
        payoff_factory=lambda *_args: UniquePayoff(),
        price_fn=lambda *_args: 10.0,
    )

    assert comparison["status"] == "insufficient_results"
    assert comparison["prices"] == {contract.target_id: 10.0}
    report = comparison["artifact_coherence"][contract.target_id]
    assert report["status"] == "bound_unique_artifact"
    assert report["binding_declaration"]["target_contract"][
        "variant_parameters"
    ] == {"scheme": "heston_qe"}


def test_cross_validation_requires_executed_validation_bundle_evidence():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    contract = ComparisonTargetContract(
        target_id="plain_mc",
        method="monte_carlo",
        validation_bundle_id="monte_carlo:european_option",
    )

    class UniquePayoff:
        __trellis_comparison_bindings__ = {
            "plain_mc": {"target_contract": contract.to_payload()}
        }

    result = SimpleNamespace(
        success=True,
        payoff_cls=UniquePayoff,
        selected_method="monte_carlo",
        selected_validation_bundle_id=contract.validation_bundle_id,
        comparison_target_contract=contract.to_payload(),
        validation_binding_evidence_source="",
    )

    comparison = _cross_validate_comparison_task(
        [ComparisonBuildTarget(contract=contract)],
        {contract.target_id: result},
        market_state=object(),
        configured_targets={},
        payoff_factory=lambda *_args: UniquePayoff(),
        price_fn=lambda *_args: 10.0,
    )

    assert comparison["status"] == "semantic_artifact_mismatch"
    report = comparison["artifact_coherence"][contract.target_id]
    assert report["failures"][-1]["code"] == "validation_bundle_not_executed"


def test_cross_validation_requires_full_declaration_for_explicit_target():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    class UndeclaredPayoff:
        pass

    contract = ComparisonTargetContract(
        target_id="plain_mc",
        method="monte_carlo",
    )
    result = SimpleNamespace(
        success=True,
        payoff_cls=UndeclaredPayoff,
        selected_method=contract.method,
        comparison_target_contract=contract.to_payload(),
    )

    comparison = _cross_validate_comparison_task(
        [ComparisonBuildTarget(contract=contract)],
        {contract.target_id: result},
        market_state=object(),
        configured_targets={},
        payoff_factory=lambda *_args: UndeclaredPayoff(),
        price_fn=lambda *_args: 10.0,
    )

    assert comparison["status"] == "semantic_artifact_mismatch"
    report = comparison["artifact_coherence"][contract.target_id]
    assert report["failures"][0]["code"] == (
        "missing_artifact_target_declaration"
    )


def test_cross_validation_cannot_pass_without_declared_reference_result():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    def legacy_contract(target_id: str) -> ComparisonTargetContract:
        return ComparisonTargetContract(
            target_id=target_id,
            method="monte_carlo",
            resolution_source="legacy_target_inference",
            explicit=False,
        )

    contracts = {
        target_id: legacy_contract(target_id)
        for target_id in ("plain", "antithetic", "reference")
    }
    plain_payoff = type("PlainPayoff", (), {"price": 10.0})
    antithetic_payoff = type("AntitheticPayoff", (), {"price": 10.1})
    results = {
        "plain": SimpleNamespace(
            success=True,
            payoff_cls=plain_payoff,
            selected_method="monte_carlo",
            comparison_target_contract=contracts["plain"].to_payload(),
        ),
        "antithetic": SimpleNamespace(
            success=True,
            payoff_cls=antithetic_payoff,
            selected_method="monte_carlo",
            comparison_target_contract=contracts["antithetic"].to_payload(),
        ),
        "reference": SimpleNamespace(success=False, payoff_cls=None),
    }
    targets = [
        ComparisonBuildTarget(contract=contracts["plain"]),
        ComparisonBuildTarget(contract=contracts["antithetic"]),
        ComparisonBuildTarget(
            contract=contracts["reference"],
            is_reference=True,
        ),
    ]

    comparison = _cross_validate_comparison_task(
        targets,
        results,
        market_state=object(),
        configured_targets={"tolerance_pct": 5.0},
        payoff_factory=lambda payoff_cls, *_args: payoff_cls(),
        price_fn=lambda payoff, _market: payoff.price,
    )

    assert comparison["prices"] == {"plain": 10.0, "antithetic": 10.1}
    assert comparison["reference_target"] == "reference"
    assert comparison["reference_price"] is None
    assert comparison["status"] == "insufficient_results"


def test_cross_validation_rejects_same_contract_id_with_different_payload():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    class UniquePayoff:
        pass

    requested = ComparisonTargetContract(
        target_id="plain_mc",
        method="monte_carlo",
        contract_id="stable-id",
        route_id="monte_carlo_a",
    )
    persisted = ComparisonTargetContract(
        target_id="plain_mc",
        method="monte_carlo",
        contract_id="stable-id",
        route_id="monte_carlo_b",
    )
    result = SimpleNamespace(
        success=True,
        payoff_cls=UniquePayoff,
        selected_method="monte_carlo",
        selected_route_id="monte_carlo_a",
        comparison_target_contract=persisted.to_payload(),
    )

    comparison = _cross_validate_comparison_task(
        [ComparisonBuildTarget(contract=requested)],
        {requested.target_id: result},
        market_state=object(),
        configured_targets={},
        payoff_factory=lambda *_args: UniquePayoff(),
        price_fn=lambda *_args: 10.0,
    )

    report = comparison["artifact_coherence"][requested.target_id]
    assert comparison["status"] == "semantic_artifact_mismatch"
    assert report["failures"][-1]["code"] == "target_contract_payload_mismatch"


def test_cross_validation_rejects_equivalence_group_for_different_contracts():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    contracts = [
        ComparisonTargetContract(
            target_id="mc",
            method="monte_carlo",
            equivalence_group="same-result",
        ),
        ComparisonTargetContract(
            target_id="fft",
            method="fft_pricing",
            equivalence_group="same-result",
        ),
    ]
    shared_payoff = type(
        "SharedPayoff",
        (),
        {
            "__trellis_comparison_bindings__": {
                contract.target_id: {"target_contract": contract.to_payload()}
                for contract in contracts
            }
        },
    )
    results = {
        contract.target_id: SimpleNamespace(
            success=True,
            payoff_cls=shared_payoff,
            selected_method=contract.method,
            comparison_target_contract=contract.to_payload(),
        )
        for contract in contracts
    }

    comparison = _cross_validate_comparison_task(
        [ComparisonBuildTarget(contract=contract) for contract in contracts],
        results,
        market_state=object(),
        configured_targets={},
        payoff_factory=lambda *_args: shared_payoff(),
        price_fn=lambda *_args: 10.0,
    )

    assert comparison["status"] == "semantic_artifact_mismatch"
    assert all(
        report["status"] == "unbound_shared_artifact"
        for report in comparison["artifact_coherence"].values()
    )


def test_generated_artifact_identity_prefers_build_source_over_mutable_module_file():
    from trellis.agent.task_runtime import _comparison_artifact_identity

    class GeneratedPayoff:
        pass

    first_identity, first_artifact = _comparison_artifact_identity(
        SimpleNamespace(payoff_cls=GeneratedPayoff, code="first generated target")
    )
    second_identity, second_artifact = _comparison_artifact_identity(
        SimpleNamespace(payoff_cls=GeneratedPayoff, code="second generated target")
    )

    assert first_artifact["module_name"] == second_artifact["module_name"]
    assert first_artifact["class_name"] == second_artifact["class_name"]
    assert first_artifact["code_hash"] != second_artifact["code_hash"]
    assert first_identity != second_identity


def test_cross_validation_uses_persisted_contract_method_for_legacy_result():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    class UniquePayoff:
        pass

    contract = ComparisonTargetContract(
        target_id="plain_mc",
        method="monte_carlo",
        resolution_source="legacy_target_inference",
        explicit=False,
    )
    result = SimpleNamespace(
        success=True,
        payoff_cls=UniquePayoff,
        comparison_target_contract=contract.to_payload(),
    )

    comparison = _cross_validate_comparison_task(
        [ComparisonBuildTarget(contract=contract)],
        {contract.target_id: result},
        market_state=object(),
        configured_targets={},
        payoff_factory=lambda *_args: UniquePayoff(),
        price_fn=lambda *_args: 10.0,
    )

    assert comparison["status"] == "insufficient_results"
    assert comparison["prices"] == {contract.target_id: 10.0}


def test_cross_validation_rejects_one_unbound_artifact_for_heterogeneous_targets():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    contracts = [
        ComparisonTargetContract(target_id="mc", method="monte_carlo"),
        ComparisonTargetContract(target_id="fft", method="fft_pricing"),
    ]
    shared_payoff = type(
        "SharedPayoff",
        (),
        {
            "price": 10.0,
            "__trellis_comparison_bindings__": {
                contract.target_id: {"target_contract": contract.to_payload()}
                for contract in contracts
            },
        },
    )
    targets = [
        ComparisonBuildTarget(contract=contracts[0]),
        ComparisonBuildTarget(
            contract=contracts[1],
            is_reference=True,
        ),
    ]

    def result_for(target):
        return SimpleNamespace(
            success=True,
            payoff_cls=shared_payoff,
            selected_method=target.preferred_method,
            comparison_target_contract=target.contract.to_payload(),
        )

    price_calls = []
    result = _cross_validate_comparison_task(
        targets,
        {target.target_id: result_for(target) for target in targets},
        market_state=object(),
        configured_targets={"tolerance_pct": 5.0},
        payoff_factory=lambda payoff_cls, _schema, _settle: payoff_cls(),
        price_fn=lambda payoff, _market: price_calls.append(payoff) or payoff.price,
    )

    assert result["status"] == "semantic_artifact_mismatch"
    assert result["prices"] == {}
    assert price_calls == []
    assert set(result["artifact_coherence"]) == {"mc", "fft"}
    assert all(
        report["status"] == "unbound_shared_artifact"
        for report in result["artifact_coherence"].values()
    )


def test_cross_validation_rejects_unique_artifact_that_drops_spec_overrides():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    @dataclass(frozen=True)
    class SamplingSpec:
        sampling: str

    class UniquePayoff:
        def __init__(self):
            self.spec = SamplingSpec(sampling="pseudo_random")

    contract = ComparisonTargetContract(
        target_id="antithetic",
        method="monte_carlo",
        variant_parameters={"sampling": "antithetic"},
        spec_overrides={"sampling": "antithetic"},
    )
    UniquePayoff.__trellis_comparison_bindings__ = {
        contract.target_id: {"target_contract": contract.to_payload()}
    }
    target = ComparisonBuildTarget(contract=contract)
    result = SimpleNamespace(
        success=True,
        payoff_cls=UniquePayoff,
        selected_method="monte_carlo",
        comparison_target_contract=contract.to_payload(),
    )
    price_calls = []

    comparison = _cross_validate_comparison_task(
        [target],
        {"antithetic": result},
        market_state=object(),
        configured_targets={},
        payoff_factory=lambda *_args: UniquePayoff(),
        price_fn=lambda *_args: price_calls.append(True) or 10.0,
    )

    assert comparison["status"] == "semantic_artifact_mismatch"
    assert comparison["prices"] == {}
    assert price_calls == []
    report = comparison["artifact_coherence"]["antithetic"]
    assert report["status"] == "unbound_artifact"
    assert report["exercised_spec_overrides"] == {
        "sampling": "pseudo_random"
    }
    assert report["failures"][-1]["code"] == (
        "target_spec_overrides_not_exercised"
    )


def test_cross_validation_allows_shared_class_with_exercised_typed_variants():
    from trellis.agent.comparison_target_contracts import ComparisonTargetContract
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
    )

    @dataclass(frozen=True)
    class SamplingSpec:
        sampling: str

    contracts = [
        ComparisonTargetContract(
            target_id="plain",
            method="monte_carlo",
            variant_parameters={"sampling": "pseudo_random"},
            spec_overrides={"sampling": "pseudo_random"},
        ),
        ComparisonTargetContract(
            target_id="antithetic",
            method="monte_carlo",
            variant_parameters={"sampling": "antithetic"},
            spec_overrides={"sampling": "antithetic"},
        ),
    ]

    class SharedMonteCarloPayoff:
        __trellis_comparison_bindings__ = {
            contract.target_id: {"target_contract": contract.to_payload()}
            for contract in contracts
        }

        def __init__(self, spec):
            self.spec = spec
    targets = [
        ComparisonBuildTarget(contract=contracts[0]),
        ComparisonBuildTarget(contract=contracts[1], is_reference=True),
    ]

    def result_for(target):
        return SimpleNamespace(
            success=True,
            payoff_cls=SharedMonteCarloPayoff,
            selected_method="monte_carlo",
            comparison_target_contract=target.contract.to_payload(),
        )

    result = _cross_validate_comparison_task(
        targets,
        {target.target_id: result_for(target) for target in targets},
        market_state=object(),
        configured_targets={"tolerance_pct": 100.0},
        payoff_factory=lambda payoff_cls, _schema, _settle, contract: payoff_cls(
            SamplingSpec(**dict(contract.spec_overrides))
        ),
        price_fn=lambda payoff, _market: (
            10.0 if payoff.spec.sampling == "pseudo_random" else 10.1
        ),
    )

    assert result["status"] == "passed"
    assert result["prices"] == {"plain": 10.0, "antithetic": 10.1}
    assert all(
        report["status"] == "bound_shared_variant"
        for report in result["artifact_coherence"].values()
    )
