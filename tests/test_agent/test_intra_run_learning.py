from __future__ import annotations

from trellis.agent.intra_run_learning import (
    RecoveryMode,
    build_knowledge_patch_candidate,
    normalize_recovery_mode,
    render_knowledge_overlay,
)


def test_knowledge_patch_candidate_requires_non_strict_failure_with_evidence():
    strict_candidate = build_knowledge_patch_candidate(
        target_id="heston_adi_pde",
        preferred_method="heston_adi_pde",
        instrument_type="heston_option",
        recovery_mode=RecoveryMode.STRICT,
        payload={
            "success": False,
            "failures": ["helper() got an unexpected keyword argument 'resolved'"],
            "reflection": {"gaps_identified": ["Missing helper signature contract"]},
        },
    )
    assert strict_candidate is None

    success_candidate = build_knowledge_patch_candidate(
        target_id="heston_adi_pde",
        preferred_method="heston_adi_pde",
        instrument_type="heston_option",
        recovery_mode=RecoveryMode.ASSISTED,
        payload={"success": True, "failures": []},
    )
    assert success_candidate is None

    candidate = build_knowledge_patch_candidate(
        target_id="heston_adi_pde",
        preferred_method="heston_adi_pde",
        instrument_type="heston_option",
        recovery_mode=RecoveryMode.ASSISTED,
        payload={
            "success": False,
            "failures": [
                "price_heston_option_adi_pde_result() got an unexpected "
                "keyword argument 'resolved'"
            ],
            "reflection": {
                "lesson_captured": "fd_044",
                "gaps_identified": [
                    "Exact callable signature and argument contract for Heston ADI"
                ],
            },
            "agent_observations": [
                {
                    "agent": "quant",
                    "kind": "decision",
                    "summary": "Selected pricing method `pde_solver`",
                }
            ],
        },
    )

    assert candidate is not None
    assert candidate.recovery_mode == RecoveryMode.ASSISTED
    assert candidate.patch_type == "cookbook_patch"
    assert candidate.confidence >= 0.45
    assert "unexpected keyword argument" in "\n".join(candidate.evidence)
    assert "Exact callable signature" in "\n".join(candidate.guidance)


def test_knowledge_patch_candidate_skips_provider_failures():
    candidate = build_knowledge_patch_candidate(
        target_id="mc_autocall",
        preferred_method="monte_carlo",
        instrument_type="autocallable",
        recovery_mode="assisted",
        payload={
            "success": False,
            "failures": ["OpenAI request failed after 3 attempts"],
            "reflection": {"gaps_identified": ["Missing autocallable cookbook"]},
        },
    )

    assert candidate is None


def test_knowledge_patch_candidate_records_callable_signature_evidence():
    candidate = build_knowledge_patch_candidate(
        target_id="pde_double_barrier",
        preferred_method="pde_solver",
        instrument_type="barrier_option",
        recovery_mode="assisted",
        payload={
            "success": False,
            "failures": [
                "BlackScholesOperator.__init__() got an unexpected keyword argument 'grid'"
            ],
            "reflection": {"gaps_identified": ["PDE operator constructor mismatch"]},
        },
    )

    assert candidate is not None
    signature_records = [
        item
        for item in candidate.structured_evidence
        if item["kind"] == "callable_signature"
    ]
    assert signature_records
    record = signature_records[0]
    assert record["symbol"] == "BlackScholesOperator"
    assert record["module"] == "trellis.models.pde.operator"
    assert record["unexpected_keyword"] == "grid"
    assert record["available"] is True
    assert "sigma_fn" in record["signature"]
    assert "r_fn" in record["signature"]
    assert "grid" in "\n".join(candidate.guidance)


def test_knowledge_patch_candidate_records_required_primitive_obligation():
    candidate = build_knowledge_patch_candidate(
        target_id="mc_autocall",
        preferred_method="monte_carlo",
        instrument_type="autocallable",
        recovery_mode="assisted",
        payload={
            "success": False,
            "failures": [
                "assembly.required_primitive_missing: generated code did not use "
                "trellis.models.monte_carlo.variance_reduction.sobol_normals"
            ],
            "reflection": {"gaps_identified": ["QMC primitive obligation drift"]},
        },
    )

    assert candidate is not None
    primitive_records = [
        item
        for item in candidate.structured_evidence
        if item["kind"] == "required_primitive"
    ]
    assert primitive_records
    record = primitive_records[0]
    assert record["primitive"] == (
        "trellis.models.monte_carlo.variance_reduction.sobol_normals"
    )
    assert record["module"] == "trellis.models.monte_carlo.variance_reduction"
    assert record["symbol"] == "sobol_normals"
    assert record["available"] is True
    assert "n_paths" in record["signature"]
    assert any(
        item["kind"] == "required_primitive"
        for item in candidate.repair_obligations
    )


def test_knowledge_patch_candidate_records_comparison_contract_evidence():
    candidate = build_knowledge_patch_candidate(
        target_id="heston_adi_pde",
        preferred_method="pde_solver",
        instrument_type="heston_option",
        recovery_mode="assisted",
        payload={
            "success": False,
            "failures": ["cross-validation failed: Heston PDE disagrees with MC"],
            "comparison": {
                "status": "failed",
                "method_prices": {"heston_adi_pde": 10.2, "heston_mc": 9.7},
                "reference_target": "heston_mc",
                "tolerance": 0.05,
            },
            "runtime_contract": {
                "selected_route": "heston_adi_pde",
                "binding": "heston:equity:pde",
            },
            "validation": {"bundle": "heston:equity:comparison"},
            "payoff_class": "HestonEuropeanCallPayoff",
            "payoff_module": "trellis.instruments._agent.hestonoption",
            "reflection": {"gaps_identified": ["Comparison evidence is needed"]},
        },
    )

    assert candidate is not None
    comparison_records = [
        item
        for item in candidate.structured_evidence
        if item["kind"] == "comparison_contract"
    ]
    assert comparison_records
    record = comparison_records[0]
    assert record["method_prices"] == {
        "heston_adi_pde": 10.2,
        "heston_mc": 9.7,
    }
    assert record["reference_target"] == "heston_mc"
    assert record["tolerance"] == 0.05
    assert record["selected_route"] == "heston_adi_pde"
    assert record["binding"] == "heston:equity:pde"
    assert record["validation_bundle"] == "heston:equity:comparison"
    assert record["payoff_class"] == "HestonEuropeanCallPayoff"


def test_knowledge_patch_candidate_marks_prose_only_candidate_not_retryable():
    candidate = build_knowledge_patch_candidate(
        target_id="mc_autocall",
        preferred_method="monte_carlo",
        instrument_type="autocallable",
        recovery_mode="assisted",
        payload={
            "success": False,
            "failures": ["generated adapter failed post-build validation"],
            "reflection": {
                "gaps_identified": [
                    "Missing cookbook guidance for autocallable event branching"
                ]
            },
        },
    )

    assert candidate is not None
    assert candidate.retryable is False
    assert candidate.contract_completeness == 0.0
    assert "missing_structured_repair_obligation" in candidate.skip_reasons


def test_knowledge_patch_candidate_marks_contract_backed_candidate_retryable():
    candidate = build_knowledge_patch_candidate(
        target_id="pde_double_barrier",
        preferred_method="pde_solver",
        instrument_type="barrier_option",
        recovery_mode="assisted",
        payload={
            "success": False,
            "failures": [
                "BlackScholesOperator.__init__() got an unexpected keyword argument 'grid'"
            ],
            "reflection": {"gaps_identified": ["PDE operator constructor mismatch"]},
        },
    )

    assert candidate is not None
    assert candidate.retryable is True
    assert candidate.contract_completeness >= 0.5
    assert candidate.skip_reasons == ()


def test_render_knowledge_overlay_is_ephemeral_and_actionable():
    candidate = build_knowledge_patch_candidate(
        target_id="pde_double_barrier",
        preferred_method="pde_solver",
        instrument_type="barrier_option",
        recovery_mode=normalize_recovery_mode("remediation"),
        payload={
            "success": False,
            "failures": [
                "resolve_double_barrier_inputs() got an unexpected keyword "
                "argument 'spot'"
            ],
            "reflection": {
                "gaps_identified": [
                    "Missing cookbook guidance for adapting market-state "
                    "fields into barrier helper inputs"
                ]
            },
        },
    )

    text = render_knowledge_overlay([candidate])

    assert "Intra-Run Candidate Knowledge Overlay" in text
    assert "not canonical" in text
    assert "pde_double_barrier" in text
    assert "resolve_double_barrier_inputs" in text
    assert "barrier helper inputs" in text
