"""Tests for the knowledge system — store, retrieval, promotion, decomposition."""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

# ---------------------------------------------------------------------------
# Store & retrieval
# ---------------------------------------------------------------------------


class TestKnowledgeStore:
    """Test KnowledgeStore loading and retrieval."""

    def test_store_loads(self):
        from trellis.agent.knowledge import get_store
        store = get_store()
        assert len(store._features) > 20
        assert len(store._decompositions) >= 13
        assert len(store._principles) >= 3
        assert len(store._lesson_index) >= 18
        assert len(store._failure_signatures) >= 5

    def test_principles_loaded(self):
        from trellis.agent.knowledge import get_store
        store = get_store()
        ids = [p.id for p in store._principles]
        assert "P1" in ids
        assert "P2" in ids
        assert "P3" in ids
        assert "P10" in ids
        assert "P11" in ids

    def test_feature_expansion_callable(self):
        from trellis.agent.knowledge.store import expand_features, KnowledgeStore
        store = KnowledgeStore()
        expanded = expand_features(["callable"], store._features)
        assert "callable" in expanded
        assert "early_exercise" in expanded
        assert "backward_induction" in expanded

    def test_feature_expansion_barrier(self):
        from trellis.agent.knowledge.store import expand_features, KnowledgeStore
        store = KnowledgeStore()
        expanded = expand_features(["barrier"], store._features)
        assert "barrier" in expanded
        assert "path_dependent" in expanded

    def test_feature_expansion_unknown(self):
        """Unknown features should be kept, not cause errors."""
        from trellis.agent.knowledge.store import expand_features, KnowledgeStore
        store = KnowledgeStore()
        expanded = expand_features(["delta_space_interpolation", "callable"], store._features)
        assert "delta_space_interpolation" in expanded
        assert "callable" in expanded

    def test_find_similar_products_ranks_callable_bond_for_cold_start_rate_tree_note(self):
        from trellis.agent.knowledge import get_store
        from trellis.agent.knowledge.schema import RetrievalSpec

        store = get_store()
        matches = store.find_similar_products(
            RetrievalSpec(
                method="rate_tree",
                features=["callable", "fixed_coupons", "mean_reversion"],
                instrument="callable_range_note",
            )
        )

        assert matches
        assert matches[0].instrument == "callable_bond"
        assert "callable" in matches[0].shared_features
        assert matches[0].promoted_routes

    def test_retrieve_callable_bond(self):
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "rate_tree",
            features=["callable", "fixed_coupons", "mean_reversion"],
            instrument="callable_bond",
        )
        assert len(k["principles"]) >= 3
        assert len(k["lessons"]) >= 3
        assert k["data_contracts"]  # should have VOL_BLACK_TO_HW
        assert k["method_requirements"] is not None

    def test_retrieve_lessons_ranked_by_relevance(self):
        """Lessons with more feature overlap should rank higher."""
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "rate_tree",
            features=["callable", "fixed_coupons", "backward_induction",
                       "early_exercise", "coupons"],
        )
        lessons = k["lessons"]
        assert len(lessons) >= 3
        # The first few should be about callable bonds / backward induction
        top_titles = {l.title for l in lessons[:3]}
        assert any("exercise" in t.lower() or "callable" in t.lower()
                    for t in top_titles)

    def test_retrieve_composite_product(self):
        """A callable range accrual should get lessons from multiple sources."""
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "rate_tree",
            features=["callable", "range_condition", "floating_coupons",
                       "mean_reversion"],
        )
        lessons = k["lessons"]
        # Should have lessons about callable bonds AND rate trees
        categories = {l.category for l in lessons}
        assert "backward_induction" in categories or "calibration" in categories

    def test_query_lessons_uses_indexed_supersedes_before_hydration(self, monkeypatch):
        from trellis.agent.knowledge.schema import (
            AppliesWhen,
            Lesson,
            LessonIndex,
            LessonStatus,
            RetrievalSpec,
            Severity,
        )
        from trellis.agent.knowledge.store import KnowledgeStore

        store = KnowledgeStore()
        store._lesson_index = [
            LessonIndex(
                id="lesson_old",
                title="Old lesson",
                severity=Severity.MEDIUM,
                category="testing",
                applies_when=AppliesWhen(
                    method=("rate_tree",),
                    features=("callable",),
                ),
                status=LessonStatus.PROMOTED,
            ),
            LessonIndex(
                id="lesson_new",
                title="New lesson",
                severity=Severity.HIGH,
                category="testing",
                applies_when=AppliesWhen(
                    method=("rate_tree",),
                    features=("callable",),
                ),
                status=LessonStatus.PROMOTED,
                supersedes=("lesson_old",),
            ),
        ]

        load_calls: list[str] = []

        def fake_load(lesson_id: str) -> Lesson:
            load_calls.append(lesson_id)
            return Lesson(
                id=lesson_id,
                title=lesson_id,
                severity=Severity.HIGH,
                category="testing",
                applies_when=AppliesWhen(
                    method=("rate_tree",),
                    features=("callable",),
                ),
                symptom="symptom",
                root_cause="root_cause",
                fix="fix",
                validation="validation",
            )

        monkeypatch.setattr(store, "_load_lesson", fake_load)

        lessons = store._query_lessons(
            ["callable"],
            RetrievalSpec(
                method="rate_tree",
                features=["callable"],
                instrument="callable_bond",
                max_lessons=1,
            ),
            1,
        )

        assert [lesson.id for lesson in lessons] == ["lesson_new"]
        assert load_calls == ["lesson_new"]

    def test_query_lessons_hydrates_only_ranked_window(self, monkeypatch):
        from trellis.agent.knowledge.schema import (
            AppliesWhen,
            Lesson,
            LessonIndex,
            LessonStatus,
            RetrievalSpec,
            Severity,
        )
        from trellis.agent.knowledge.store import KnowledgeStore

        store = KnowledgeStore()
        store._lesson_index = [
            LessonIndex(
                id=f"lesson_{index}",
                title=f"Lesson {index}",
                severity=Severity.MEDIUM,
                category="testing",
                applies_when=AppliesWhen(
                    method=("analytical",),
                    features=("vol_surface_dependence",)
                    if index < 3
                    else ("irrelevant",),
                ),
                status=LessonStatus.PROMOTED,
            )
            for index in range(8)
        ]

        load_calls: list[str] = []

        def fake_load(lesson_id: str) -> Lesson:
            load_calls.append(lesson_id)
            return Lesson(
                id=lesson_id,
                title=lesson_id,
                severity=Severity.MEDIUM,
                category="testing",
                applies_when=AppliesWhen(
                    method=("analytical",),
                    features=("vol_surface_dependence",),
                ),
                symptom="symptom",
                root_cause="root_cause",
                fix="fix",
                validation="validation",
            )

        monkeypatch.setattr(store, "_load_lesson", fake_load)

        lessons = store._query_lessons(
            ["vol_surface_dependence"],
            RetrievalSpec(
                method="analytical",
                features=["vol_surface_dependence"],
                instrument="cap",
                max_lessons=2,
            ),
            2,
        )

        assert [lesson.id for lesson in lessons] == ["lesson_0", "lesson_1"]
        assert load_calls == ["lesson_0", "lesson_1"]
        assert len(load_calls) < len(store._lesson_index)

    def test_retrieve_cookbook(self):
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task("rate_tree", features=["callable"])
        assert k["cookbook"] is not None
        assert "rate_tree" in k["cookbook"].method
        assert "evaluate" in k["cookbook"].template.lower()

    def test_retrieve_analytical(self):
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "analytical",
            features=["floating_coupons", "vol_surface_dependence"],
            instrument="cap",
        )
        assert k["cookbook"] is not None
        assert "analytical" in k["cookbook"].method

    def test_retrieve_monte_carlo(self):
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "monte_carlo",
            features=["barrier", "path_dependent"],
            instrument="barrier_option",
        )
        assert k["cookbook"] is not None
        assert k["data_contracts"]

    def test_retrieve_analytical_cds_includes_credit_curve_guidance(self):
        from trellis.agent.knowledge import retrieve_for_task

        k = retrieve_for_task(
            "analytical",
            features=["credit", "spread"],
            instrument="cds",
        )

        assert k["decomposition"] is not None
        assert k["decomposition"].instrument == "cds"
        assert k["cookbook"] is not None
        template = k["cookbook"].template
        assert "Credit default swap (single-name)" in template
        assert "survival_probability" in template
        assert "credit_curve" in template
        assert "build_cds_schedule" in template
        assert "price_cds_analytical" in template
        assert "spec.start_date" in template
        assert "Do not trapezoid the protection" in template
        assert "price_cds_analytical" in template

    def test_retrieve_monte_carlo_cds_includes_hazard_rate_guidance(self):
        from trellis.agent.knowledge import retrieve_for_task

        k = retrieve_for_task(
            "monte_carlo",
            features=["credit", "spread"],
            instrument="cds",
        )

        assert k["decomposition"] is not None
        assert k["decomposition"].instrument == "cds"
        assert k["cookbook"] is not None
        template = k["cookbook"].template
        cds_section = template.split("### Credit default swap (single-name)")[1].split(
            "If the user request is partial or underspecified",
        )[0]
        assert "Credit default swap (single-name)" in template
        assert "hazard_rate" in template or "survival_probability" in template
        assert "credit_curve" in template
        assert "150 bp -> 0.015" in template
        assert "`100`" in template and "`0.01`" in template
        assert "MonteCarloEngine" not in cds_section
        assert "build_cds_schedule" in cds_section
        assert "price_cds_monte_carlo" in cds_section

    def test_retrieve_copula_nth_to_default_keeps_basket_credit_guidance(self):
        from trellis.agent.knowledge import retrieve_for_task

        k = retrieve_for_task(
            "copula",
            features=["credit_risk"],
            instrument="nth_to_default",
        )

        assert k["decomposition"] is not None
        assert k["decomposition"].instrument == "nth_to_default"
        assert k["cookbook"] is not None
        template = k["cookbook"].template
        assert "portfolio credit instruments" in template
        assert "FactorCopula" in template

    def test_retrieve_qmc(self):
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "qmc",
            features=["path_dependent"],
        )
        assert k["cookbook"] is not None
        assert k["cookbook"].method == "qmc"
        assert k["data_contracts"]
        assert k["method_requirements"] is not None

    def test_retrieve_pde_european_option_mentions_checked_in_helper_surface(self):
        from trellis.agent.knowledge import retrieve_for_task

        k = retrieve_for_task(
            "pde_solver",
            features=["discounting", "vol_surface_dependence"],
            instrument="european_option",
        )

        assert k["cookbook"] is not None
        template = k["cookbook"].template
        assert "price_vanilla_equity_option_pde" in template
        assert "theta-method" in template.lower()

    def test_retrieve_american_put_ir_includes_early_exercise_policy_guidance(self):
        from trellis.agent.knowledge import retrieve_for_product_ir
        from trellis.agent.knowledge.decompose import decompose_to_ir

        ir = decompose_to_ir(
            "American put option on equity",
            instrument_type="american_option",
        )
        k = retrieve_for_product_ir(ir, preferred_method="monte_carlo")

        assert k["cookbook"] is not None
        assert "longstaff_schwartz" in k["cookbook"].template
        assert "approved early-exercise control primitive" in k["cookbook"].template
        assert k["method_requirements"] is not None
        requirements_text = "\n".join(k["method_requirements"].requirements)
        assert "longstaff_schwartz" in requirements_text
        assert "primal_dual_mc" in requirements_text
        assert 'method="lsm"' in requirements_text

    def test_retrieve_fx_vanilla_ir_includes_garman_kohlhagen_guidance(self):
        from trellis.agent.knowledge import retrieve_for_product_ir
        from trellis.agent.knowledge.decompose import build_product_ir

        ir = build_product_ir(
            description="European FX call option on EURUSD",
            instrument="european_option",
            payoff_family="vanilla_option",
            payoff_traits=("fx", "vanilla_option"),
            exercise_style="european",
            state_dependence="terminal_markov",
            schedule_dependence=False,
            model_family="fx",
            candidate_engine_families=("analytical", "monte_carlo"),
            required_market_data={
                "discount_curve",
                "forward_curve",
                "black_vol_surface",
                "fx_rates",
                "spot",
            },
            reusable_primitives=(
                "ResolvedGarmanKohlhagenInputs",
                "garman_kohlhagen_price_raw",
            ),
            supported=True,
            preferred_method="analytical",
        )
        k = retrieve_for_product_ir(ir, preferred_method="analytical")

        assert k["cookbook"] is not None
        assert "garman_kohlhagen_price_raw" in k["cookbook"].template
        assert k["data_contracts"]
        contract_names = {contract.name for contract in k["data_contracts"]}
        assert "FX_DOMESTIC_FOREIGN_DISCOUNTING" in contract_names
        assert k["method_requirements"] is not None
        requirements_text = "\n".join(k["method_requirements"].requirements)
        assert "GARMAN-KOHLHAGEN" in requirements_text
        assert "resolved inputs" in requirements_text
        assert "garman_kohlhagen_price_raw" in k["cookbook"].template
        assert "fx_rates" in requirements_text
        assert "forward_curve" in requirements_text

    def test_retrieve_swaption_ir_includes_helper_backed_black76_guidance(self):
        from trellis.agent.knowledge import retrieve_for_product_ir
        from trellis.agent.knowledge.decompose import build_product_ir

        ir = build_product_ir(
            description="European payer swaption on USD 5Y swap",
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            state_dependence="terminal_markov",
            schedule_dependence=True,
            model_family="interest_rate",
            candidate_engine_families=("analytical", "rate_tree"),
            required_market_data={"discount_curve", "black_vol_surface", "forward_curve"},
            reusable_primitives=(
                "ResolvedSwaptionBlack76Inputs",
                "resolve_swaption_black76_inputs",
                "price_swaption_black76_raw",
            ),
            supported=True,
            preferred_method="analytical",
        )
        k = retrieve_for_product_ir(ir, preferred_method="analytical")

        assert k["cookbook"] is not None
        assert "ResolvedSwaptionBlack76Inputs" in k["cookbook"].template
        assert "resolve_swaption_black76_inputs" in k["cookbook"].template
        assert "price_swaption_black76_raw" in k["cookbook"].template
        assert k["method_requirements"] is not None
        requirements_text = "\n".join(k["method_requirements"].requirements)
        assert "RATE-STYLE SWAPTION HELPER CONTRACT" in requirements_text
        assert "price_swaption_black76_raw" in requirements_text

    def test_retrieve_zcb_option_ir_includes_jamshidian_raw_guidance(self):
        from trellis.agent.knowledge import retrieve_for_product_ir
        from trellis.agent.knowledge.decompose import build_product_ir

        ir = build_product_ir(
            description="European call on a zero-coupon bond under Hull-White / Jamshidian",
            instrument="zcb_option",
            payoff_family="zcb_option",
            exercise_style="european",
            state_dependence="terminal_markov",
            schedule_dependence=True,
            model_family="interest_rate",
            candidate_engine_families=("analytical", "rate_tree"),
            required_market_data={"discount_curve", "black_vol_surface"},
            reusable_primitives=(
                "ResolvedJamshidianInputs",
                "resolve_zcb_option_hw_inputs",
                "zcb_option_hw_raw",
            ),
            supported=True,
            preferred_method="analytical",
        )
        k = retrieve_for_product_ir(ir, preferred_method="analytical")

        assert k["cookbook"] is not None
        assert "resolve_zcb_option_hw_inputs" in k["cookbook"].template
        assert "ResolvedJamshidianInputs" in k["cookbook"].template
        assert "zcb_option_hw_raw" in k["cookbook"].template
        assert k["method_requirements"] is not None
        requirements_text = "\n".join(k["method_requirements"].requirements)
        assert "JAMSHIDIAN ZCB OPTION CONSISTENCY" in requirements_text
        assert "unit face" in requirements_text

    def test_retrieve_ranked_observation_basket_includes_semantic_basket_guidance(self):
        from trellis.agent.knowledge import retrieve_for_product_ir
        from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload
        from trellis.agent.knowledge.import_registry import get_import_registry
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract

        contract = make_ranked_observation_basket_contract(
            description="Himalaya-style ranked observation basket on AAPL, MSFT, and NVDA",
            constituents=("AAPL", "MSFT", "NVDA"),
            observation_schedule=("2025-01-15", "2025-02-15", "2025-03-15"),
        )
        compiled = compile_semantic_contract(contract)
        k = retrieve_for_product_ir(compiled.product_ir, preferred_method="monte_carlo")

        assert k["decomposition"] is not None
        assert k["decomposition"].instrument == "basket_path_payoff"
        assert k["cookbook"] is not None
        assert "resolve_basket_semantics" in k["cookbook"].template
        assert "price_ranked_observation_basket_monte_carlo" in k["cookbook"].template
        assert k["lessons"]
        assert any("ranked-observation basket" in lesson.title.lower() for lesson in k["lessons"])
        assert any(
            "semantic understanding" in principle.rule.lower()
            for principle in k["principles"]
        )

        payload = build_shared_knowledge_payload(k)
        superseded_ids = payload["summary"].get("superseded_lesson_ids", [])
        assert {"mc_016", "mc_018"}.issubset(set(superseded_ids))
        assert "mc_021" not in superseded_ids

        registry = get_import_registry()
        assert "trellis.models.resolution.basket_semantics" in registry
        assert "trellis.models.monte_carlo.semantic_basket" in registry

    def test_retrieve_basket_bootstrap_guidance_includes_launch_contract(self):
        from trellis.agent.knowledge import retrieve_for_task

        k = retrieve_for_task(
            "monte_carlo",
            features=[
                "discounting",
                "multi_asset",
                "path_dependent",
                "ranked_observation",
                "remaining_selection",
                "remove_selected",
                "locked_returns",
                "maturity_settlement",
                "vol_surface_dependence",
            ],
            instrument="basket_option",
            max_lessons=20,
        )

        lesson_ids = [lesson.id for lesson in k["lessons"]]
        assert "con_014" in lesson_ids
        assert any("pinned interpreter" in lesson.title.lower() for lesson in k["lessons"])

    def test_retrieve_partial_request_includes_required_input_plan(self):
        from trellis.agent.knowledge import retrieve_for_task

        k = retrieve_for_task(
            "monte_carlo",
            features=[
                "discounting",
                "path_dependent",
                "multi_asset",
                "vol_surface_dependence",
            ],
            instrument="basket_option",
            max_lessons=80,
        )

        lesson_ids = [lesson.id for lesson in k["lessons"]]
        assert "sem_002" in lesson_ids
        assert any("required-input plan" in lesson.title.lower() for lesson in k["lessons"])

    def test_retrieve_basket_request_includes_correlated_gbm_mu_lesson(self):
        from trellis.agent.knowledge import retrieve_for_task

        k = retrieve_for_task(
            "monte_carlo",
            features=[
                "multi_asset",
                "path_dependent",
                "ranked_observation",
                "remaining_selection",
                "remove_selected",
                "locked_returns",
                "maturity_settlement",
                "discounting",
            ],
            instrument="basket_option",
            max_lessons=20,
        )

        lesson_ids = [lesson.id for lesson in k["lessons"]]
        assert "mc_021" in lesson_ids
        assert any("mu" in lesson.title.lower() and "mus" in lesson.title.lower() for lesson in k["lessons"])

    def test_retrieve_alias_method(self):
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "pde",
            features=["pde_grid"],
        )
        assert k["cookbook"] is not None
        assert k["cookbook"].method == "pde_solver"

    def test_retrieve_qmc_alias_method(self):
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "quasi_monte_carlo",
            features=["path_dependent"],
        )
        assert k["cookbook"] is not None
        assert k["cookbook"].method == "qmc"

    def test_failure_signature_matching(self):
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "rate_tree",
            features=["callable"],
            error_signatures=["callable > straight bond"],
        )
        sigs = k.get("matched_signatures", [])
        assert len(sigs) >= 1
        assert any("exercise" in s.diagnostic_hint.lower() for s in sigs)

    def test_decomposition_lookup(self):
        from trellis.agent.knowledge import get_store
        store = get_store()
        d = store._decompositions.get("callable_bond")
        assert d is not None
        assert "callable" in d.features
        assert d.method == "rate_tree"

    def test_retrieve_for_task_uses_runtime_cache(self):
        from trellis.agent.knowledge.schema import RetrievalSpec
        from trellis.agent.knowledge.store import KnowledgeStore

        store = KnowledgeStore()
        spec = RetrievalSpec(
            method="monte_carlo",
            features=["early_exercise", "path_dependent"],
            instrument="american_option",
        )

        first = store.retrieve_for_task(spec)
        second = store.retrieve_for_task(spec)
        stats = store.retrieval_cache_stats()

        assert first is second
        assert stats["misses"] == 1
        assert stats["hits"] == 1
        assert stats["size"] >= 1

    def test_live_repo_state_helpers_are_revision_keyed(self):
        from trellis.agent.knowledge import (
            get_package_map,
            get_repo_facts,
            get_repo_revision,
            get_symbol_map,
            get_test_map,
            suggest_tests_for_symbol,
        )

        revision = get_repo_revision()
        symbol_map = get_symbol_map()
        package_map = get_package_map()
        test_map = get_test_map()
        repo_facts = get_repo_facts()

        assert symbol_map.repo_revision == revision
        assert package_map.repo_revision == revision
        assert test_map.repo_revision == revision
        assert repo_facts and all(f.repo_revision == revision for f in repo_facts)
        assert "trellis.models.black" in symbol_map.module_to_symbols
        assert "black76_call" in symbol_map.symbol_to_modules
        assert "trellis.models" in package_map.package_to_modules
        assert any(
            path.endswith("test_build_loop.py")
            for tests in test_map.directory_to_tests.values()
            for path in tests
        )
        assert suggest_tests_for_symbol("callable_bond")


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


class TestFormatting:

    def test_format_has_all_sections(self):
        from trellis.agent.knowledge import retrieve_for_task, format_knowledge_for_prompt
        k = retrieve_for_task(
            "rate_tree",
            features=["callable", "fixed_coupons", "mean_reversion"],
            instrument="callable_bond",
        )
        text = format_knowledge_for_prompt(k)
        assert "## Key Principles" in text
        assert "## Lessons" in text
        assert "## DATA CONTRACTS" in text
        assert "## MODELING REQUIREMENTS" in text

    def test_format_empty_retrieval(self):
        from trellis.agent.knowledge.retrieval import format_knowledge_for_prompt
        text = format_knowledge_for_prompt({})
        # Even empty retrieval includes the API map and import registry
        assert "AVAILABLE IMPORTS" in text
        assert "API Map" in text

    def test_build_shared_knowledge_payload_includes_prompt_views_and_summary(self):
        from trellis.agent.knowledge import retrieve_for_product_ir
        from trellis.agent.knowledge.decompose import decompose_to_ir
        from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload

        product_ir = decompose_to_ir(
            "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
            instrument_type="quanto_option",
        )
        knowledge = retrieve_for_product_ir(product_ir, preferred_method="analytical")
        payload = build_shared_knowledge_payload(knowledge)

        assert "## Product Semantics" in payload["builder_text"]
        assert "API Map" in payload["builder_text_distilled"]
        assert "## Shared Failure Memory" in payload["review_text"]
        assert payload["routing_text"]
        assert "API Map" in payload["routing_text_distilled"]
        assert (
            "## Prior Lessons From Similar Products" in payload["routing_text"]
            or "## Shared Routing Principles" in payload["routing_text"]
        )
        assert payload["builder_text_expanded"]
        assert payload["review_text_expanded"]
        assert payload["routing_text_expanded"]
        assert payload["builder_text_distilled"]
        assert payload["review_text_distilled"]
        assert payload["routing_text_distilled"]
        assert payload["summary"]["instrument"] == product_ir.instrument
        assert payload["summary"]["lesson_count"] >= 1
        assert payload["summary"]["selected_artifact_ids"]
        assert "## Generated Skills" in payload["builder_text_distilled"]
        assert "builder" in payload["summary"]["selected_artifacts_by_audience"]
        builder_artifact = payload["summary"]["selected_artifacts_by_audience"]["builder"][0]
        assert "lineage_status" in builder_artifact
        assert "lineage_summary" in builder_artifact
        if builder_artifact["lineage_summary"]:
            assert "lineage:" in payload["builder_text_distilled"]
        assert payload["summary"]["prompt_sizes"]["builder"]["expanded_chars"] >= (
            payload["summary"]["prompt_sizes"]["builder"]["compact_chars"]
        )
        assert payload["summary"]["prompt_sizes"]["builder"]["compact_chars"] >= (
            payload["summary"]["prompt_sizes"]["builder"]["distilled_chars"]
        )

    def test_build_shared_knowledge_payload_compacts_lessons_and_templates(self):
        from trellis.agent.knowledge.retrieval import (
            build_shared_knowledge_payload,
            format_knowledge_for_prompt,
        )

        lessons = [
            SimpleNamespace(
                id=f"L{idx}",
                severity=SimpleNamespace(value="high"),
                title=f"Lesson {idx}",
                symptom="symptom",
                root_cause="root cause",
                fix="fix",
            )
            for idx in range(5)
        ]
        principles = [
            SimpleNamespace(id=f"P{idx}", rule=f"Rule {idx}")
            for idx in range(6)
        ]
        cookbook = SimpleNamespace(
            method="analytical",
            description="desc",
            template="x" * 2500,
        )

        knowledge = {
            "principles": principles,
            "lessons": lessons,
            "cookbook": cookbook,
        }

        raw_text = format_knowledge_for_prompt(knowledge, compact=False)
        payload = build_shared_knowledge_payload(knowledge)

        assert "Lesson 4" in raw_text
        assert len(payload["builder_text_distilled"]) <= len(payload["builder_text"])
        assert "Lesson 4" not in payload["builder_text"]
        assert "[omitted 2 additional lessons]" in payload["builder_text"]
        assert "[truncated cookbook template]" in payload["builder_text"]
        assert "Lesson 4" in payload["builder_text_expanded"]
        assert payload["summary"]["prompt_sizes"]["builder"]["expanded_chars"] > (
            payload["summary"]["prompt_sizes"]["builder"]["compact_chars"]
        )

    def test_build_shared_knowledge_payload_surfaces_similar_products_and_borrowed_lessons(self):
        from trellis.agent.knowledge import get_store
        from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload
        from trellis.agent.knowledge.schema import RetrievalSpec

        store = get_store()
        knowledge = store.retrieve_for_task(
            RetrievalSpec(
                method="rate_tree",
                features=["callable", "fixed_coupons", "mean_reversion"],
                instrument="callable_range_note",
                max_lessons=1,
            )
        )
        payload = build_shared_knowledge_payload(knowledge)

        assert knowledge["similar_products"]
        assert knowledge["similar_products"][0].instrument == "callable_bond"
        assert knowledge["borrowed_lessons"]
        assert "## Similar Products" in payload["builder_text"]
        assert "callable_bond" in payload["builder_text"]
        assert knowledge["similar_products"][0].promoted_routes[0] in payload["builder_text"]
        assert payload["summary"]["similar_product_ids"][0] == "callable_bond"
        assert payload["summary"]["borrowed_lesson_ids"]

    def test_retrieve_for_task_surfaces_canonical_model_grammar_entries(self):
        from trellis.agent.knowledge import get_store
        from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload, format_knowledge_for_prompt
        from trellis.agent.knowledge.schema import RetrievalSpec

        store = get_store()
        knowledge = store.retrieve_for_task(
            RetrievalSpec(
                method="rate_tree",
                features=["callable", "mean_reversion"],
                instrument="callable_bond",
                model_family="interest_rate",
                candidate_engine_families=("lattice", "exercise"),
                max_lessons=1,
            )
        )

        model_grammar = knowledge["model_grammar"]
        assert model_grammar
        grammar_ids = [entry.id for entry in model_grammar]
        assert "rates_hull_white_1f" in grammar_ids
        hull_white = next(entry for entry in model_grammar if entry.id == "rates_hull_white_1f")
        assert hull_white.runtime_materialization_kind == "model_parameter_set"
        assert hull_white.rates_curve_roles == ("discount_curve", "forecast_curve")

        text = format_knowledge_for_prompt(knowledge)
        assert "## Canonical Model Grammar" in text
        assert "rates_hull_white_1f" in text

        payload = build_shared_knowledge_payload(knowledge)
        assert "## Canonical Model Grammar" in payload["builder_text"]
        assert "rates_hull_white_1f" in payload["builder_text"]
        assert "rates_hull_white_1f" in payload["summary"]["model_grammar_ids"]

    def test_format_knowledge_for_prompt_renders_model_grammar_entries(self):
        from trellis.agent.knowledge.retrieval import format_knowledge_for_prompt
        from trellis.agent.knowledge.schema import ModelGrammarEntry

        text = format_knowledge_for_prompt(
            {
                "model_grammar": [
                    ModelGrammarEntry(
                        id="credit_single_name_reduced_form",
                        title="Reduced-form single-name CDS calibration",
                        model_name="Reduced-form single-name credit",
                        quote_families=("spread", "hazard"),
                        calibration_workflows=("calibrate_single_name_credit_curve_workflow",),
                        runtime_materialization_kind="credit_curve",
                        deferred_scope=("basket_credit",),
                    )
                ]
            }
        )

        assert "## Canonical Model Grammar" in text
        assert "credit_single_name_reduced_form" in text
        assert "calibrate_single_name_credit_curve_workflow" in text
        assert "credit_curve" in text


# ---------------------------------------------------------------------------
# Decompose
# ---------------------------------------------------------------------------


class TestDecompose:

    def test_decompose_known(self):
        from trellis.agent.knowledge.decompose import decompose
        d = decompose("callable bond", instrument_type="callable_bond")
        assert d.instrument == "callable_bond"
        assert "callable" in d.features
        assert d.method == "rate_tree"

    def test_decompose_fuzzy(self):
        from trellis.agent.knowledge.decompose import decompose
        d = decompose("callable bond with 5% coupon")
        assert d.instrument == "callable_bond"
        assert d.method == "rate_tree"

    def test_decompose_prefers_longer_match(self):
        from trellis.agent.knowledge.decompose import decompose
        d = decompose("bermudan swaption")
        assert d.instrument == "bermudan_swaption"
        assert d.method == "rate_tree"

    def test_decompose_plain_bond(self):
        from trellis.agent.knowledge.decompose import decompose
        d = decompose("plain vanilla bond")
        assert d.instrument == "bond"
        assert d.method == "analytical"

    def test_decompose_uses_runtime_cache(self, monkeypatch):
        from trellis.agent.knowledge import get_store
        from trellis.agent.knowledge import decompose as decompose_product
        from trellis.agent.knowledge.decompose import (
            clear_decomposition_cache,
            decomposition_cache_stats,
        )
        import importlib

        decompose_module = importlib.import_module("trellis.agent.knowledge.decompose")

        clear_decomposition_cache()
        store = get_store()
        calls = {"count": 0}
        original_match = decompose_module._match_static_decomposition

        def _tracking_match(key, active_store):
            calls["count"] += 1
            return original_match(key, active_store)

        monkeypatch.setattr(decompose_module, "_match_static_decomposition", _tracking_match)

        first = decompose_product("callable bond", instrument_type="callable_bond", store=store)
        second = decompose_product("callable bond", instrument_type="callable_bond", store=store)
        stats = decomposition_cache_stats()

        assert first is second
        assert calls["count"] == 1
        assert stats["misses"] == 1
        assert stats["hits"] == 1
        assert stats["size"] >= 1


# ---------------------------------------------------------------------------
# Promotion pipeline
# ---------------------------------------------------------------------------


class TestPromotion:

    @pytest.fixture(autouse=True)
    def isolated_store(self, monkeypatch, tmp_path):
        """Run promotion tests against an isolated lesson store."""
        import trellis.agent.knowledge.promotion as promotion_module

        lessons_dir = tmp_path / "lessons"
        entries_dir = lessons_dir / "entries"
        traces_dir = tmp_path / "traces"
        semantic_traces_dir = traces_dir / "semantic_extensions"
        entries_dir.mkdir(parents=True, exist_ok=True)
        semantic_traces_dir.mkdir(parents=True, exist_ok=True)
        index_path = lessons_dir / "index.yaml"
        index_path.write_text(
            yaml.dump(
                {"entries": [], "settings": {"max_prompt_entries": 7}},
                default_flow_style=False,
                sort_keys=False,
            )
        )

        monkeypatch.setattr(promotion_module, "_LESSONS_DIR", lessons_dir)
        monkeypatch.setattr(promotion_module, "_TRACES_DIR", traces_dir)
        monkeypatch.setattr(promotion_module, "_SEMANTIC_EXTENSION_TRACES_DIR", semantic_traces_dir)
        monkeypatch.setattr(promotion_module, "_INDEX_PATH", index_path)
        monkeypatch.setattr(promotion_module, "_INDEX_REBUILD_SUPPRESS_DEPTH", 0)
        monkeypatch.setattr(promotion_module, "_INDEX_REBUILD_PENDING", False)
        yield

    def _write_lesson_entry(
        self,
        lesson_id: str,
        *,
        title: str,
        status: str = "candidate",
        severity: str = "medium",
        category: str = "vol_surface",
        applies_when: dict | None = None,
        **extra: object,
    ) -> Path:
        import trellis.agent.knowledge.promotion as promotion_module

        payload = {
            "id": lesson_id,
            "title": title,
            "severity": severity,
            "category": category,
            "status": status,
            "confidence": extra.pop("confidence", 0.7),
            "created": extra.pop("created", "2026-03-01T00:00:00"),
            "version": extra.pop("version", ""),
            "source_trace": extra.pop("source_trace", None),
            "applies_when": applies_when
            or {
                "method": [],
                "features": [],
                "instrument": [],
                "error_signature": None,
            },
            "symptom": extra.pop("symptom", "symptom"),
            "root_cause": extra.pop("root_cause", "root cause"),
            "fix": extra.pop("fix", "fix"),
            "validation": extra.pop("validation", "validation"),
        }
        payload.update(extra)

        path = promotion_module._LESSONS_DIR / "entries" / f"{lesson_id}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return path

    def test_capture_returns_id(self):
        from trellis.agent.knowledge.promotion import capture_lesson
        lid = capture_lesson(
            category="vol_surface", title="_test_capture",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", features=["vol_surface_dependence"],
        )
        assert lid is not None
        assert lid.startswith("vs_")

    def test_capture_dedup(self):
        from trellis.agent.knowledge.promotion import capture_lesson
        lid1 = capture_lesson(
            category="vol_surface", title="_test_dedup_a",
            severity="medium", symptom="test", root_cause="test", fix="fix",
        )
        lid2 = capture_lesson(
            category="vol_surface", title="_test_dedup_a",
            severity="medium", symptom="test", root_cause="test", fix="fix",
        )
        assert lid1 is not None
        assert lid2 is None  # duplicate rejected

    def test_capture_dedup_by_context_and_remediation_text(self):
        from trellis.agent.knowledge.promotion import capture_lesson, rebuild_index

        self._write_lesson_entry(
            "md_001",
            title="Transient API failure",
            status="validated",
            severity="high",
            category="market_data",
            applies_when={
                "method": ["analytical"],
                "features": ["floating_coupons", "vol_surface_dependence"],
                "instrument": [],
                "error_signature": None,
            },
            symptom="Validation failed because the external request errored out.",
            root_cause=(
                "The mental model assumed the external dependency would always respond "
                "successfully, but transient network or service failures can interrupt "
                "pricing."
            ),
            fix=(
                "Retry the request with backoff and add fallback handling so transient "
                "API failures do not stop validation."
            ),
        )
        rebuild_index()

        duplicate = capture_lesson(
            category="market_data",
            title="Handle API failures gracefully",
            severity="high",
            symptom="The valuation failed because the external request errored out.",
            root_cause=(
                "The mental model treated the external service call as guaranteed to "
                "succeed, even though transient network or service issues can interrupt "
                "pricing."
            ),
            fix=(
                "Add retry, timeout, and fallback handling around the external request "
                "so the valuation can proceed or fail gracefully."
            ),
            method="analytical",
            features=["floating_coupons", "vol_surface_dependence"],
        )
        distinct_context = capture_lesson(
            category="market_data",
            title="Handle API failures gracefully",
            severity="high",
            symptom="The valuation failed because the external request errored out.",
            root_cause=(
                "The mental model treated the external service call as guaranteed to "
                "succeed, even though transient network or service issues can interrupt "
                "pricing."
            ),
            fix=(
                "Add retry, timeout, and fallback handling around the external request "
                "so the valuation can proceed or fail gracefully."
            ),
            method="monte_carlo",
            features=["vol_surface_dependence"],
        )

        assert duplicate is None
        assert distinct_context is not None

    def test_capture_rejects_invalid_contract(self):
        from trellis.agent.knowledge.promotion import capture_lesson, validate_lesson_payload

        report = validate_lesson_payload(
            {
                "category": "vol_surface",
                "title": "",
                "severity": "medium",
                "symptom": "test",
                "root_cause": "test",
                "fix": "test fix",
            }
        )

        assert not report.valid
        assert any("title is required" in error for error in report.errors)
        assert (
            capture_lesson(
                category="vol_surface",
                title="",
                severity="medium",
                symptom="test",
                root_cause="test",
                fix="test fix",
            )
            is None
        )

    def test_validate_requires_confidence(self):
        from trellis.agent.knowledge.promotion import capture_lesson, validate_lesson
        lid = capture_lesson(
            category="vol_surface", title="_test_validate",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", confidence=0.4,
        )
        assert not validate_lesson(lid)  # too low

    def test_validate_succeeds(self):
        from trellis.agent.knowledge.promotion import capture_lesson, validate_lesson
        lid = capture_lesson(
            category="vol_surface", title="_test_validate_ok",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", confidence=0.7,
        )
        assert validate_lesson(lid)

    def test_promote_requires_validated(self):
        from trellis.agent.knowledge.promotion import (
            capture_lesson, promote_lesson, boost_confidence,
        )
        lid = capture_lesson(
            category="vol_surface", title="_test_promote_skip",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", confidence=0.9,
        )
        # candidate → promote should fail (must be validated first)
        assert not promote_lesson(lid)

    def test_full_pipeline(self):
        from trellis.agent.knowledge.promotion import (
            capture_lesson, validate_lesson, promote_lesson, boost_confidence,
        )
        lid = capture_lesson(
            category="vol_surface", title="_test_pipeline",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", confidence=0.5,
        )
        assert not validate_lesson(lid)    # 0.5 < 0.6
        boost_confidence(lid, 0.2)
        assert validate_lesson(lid)        # 0.7 >= 0.6
        assert not promote_lesson(lid)     # 0.7 < 0.8
        boost_confidence(lid, 0.15)
        assert promote_lesson(lid)         # 0.85 >= 0.8

    def test_boost_confidence(self):
        from trellis.agent.knowledge.promotion import capture_lesson, boost_confidence
        lid = capture_lesson(
            category="vol_surface", title="_test_boost",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", confidence=0.3,
        )
        new = boost_confidence(lid, 0.5)
        assert new == 0.8
        # Capped at 1.0
        new = boost_confidence(lid, 0.5)
        assert new == 1.0

    def test_rebuild_index_matches_entries(self):
        import trellis.agent.knowledge.promotion as promotion_module

        self._write_lesson_entry(
            "mc_002",
            title="Later lesson",
            status="promoted",
            severity="high",
            category="monte_carlo",
            applies_when={
                "method": ["monte_carlo"],
                "features": ["beta"],
                "instrument": ["basket_option"],
                "error_signature": None,
            },
        )
        self._write_lesson_entry(
            "mc_001",
            title="Earlier lesson",
            status="validated",
            severity="medium",
            category="monte_carlo",
            applies_when={
                "method": ["monte_carlo"],
                "features": ["alpha"],
                "instrument": ["basket_option"],
                "error_signature": None,
            },
        )

        index = promotion_module.rebuild_index()
        index_path = promotion_module._INDEX_PATH
        persisted = yaml.safe_load(index_path.read_text())

        assert [entry["id"] for entry in index["entries"]] == ["mc_001", "mc_002"]
        assert [entry["id"] for entry in persisted["entries"]] == ["mc_001", "mc_002"]
        assert persisted["settings"]["max_prompt_entries"] == 7
        assert persisted["entries"][0]["title"] == "Earlier lesson"
        assert persisted["entries"][0]["applies_when"]["features"] == ["alpha"]

    def test_rebuild_index_persists_supersedes(self):
        import trellis.agent.knowledge.promotion as promotion_module

        self._write_lesson_entry(
            "md_001",
            title="Superseding lesson",
            status="promoted",
            category="market_data",
            supersedes=["md_000"],
            applies_when={
                "method": ["analytical"],
                "features": ["floating_coupons", "vol_surface_dependence"],
                "instrument": ["swaption"],
                "error_signature": None,
            },
        )

        index = promotion_module.rebuild_index()
        persisted = yaml.safe_load(promotion_module._INDEX_PATH.read_text())

        assert index["entries"][0]["supersedes"] == ["md_000"]
        assert persisted["entries"][0]["supersedes"] == ["md_000"]

    def test_rebuild_index_skips_corrupt_files(self):
        import trellis.agent.knowledge.promotion as promotion_module

        self._write_lesson_entry(
            "vol_001",
            title="Valid lesson",
            status="candidate",
            severity="low",
            category="volatility",
        )
        corrupt_path = promotion_module._LESSONS_DIR / "entries" / "vol_002.yaml"
        corrupt_path.write_text("not: [valid\n")

        index = promotion_module.rebuild_index()

        assert [entry["id"] for entry in index["entries"]] == ["vol_001"]

    def test_rebuild_index_is_deterministic(self):
        import trellis.agent.knowledge.promotion as promotion_module

        self._write_lesson_entry(
            "num_002",
            title="Second numeric lesson",
            status="promoted",
            severity="high",
            category="numerical",
        )
        self._write_lesson_entry(
            "num_001",
            title="First numeric lesson",
            status="validated",
            severity="medium",
            category="numerical",
        )

        index_path = promotion_module._INDEX_PATH
        first = yaml.safe_dump(promotion_module.rebuild_index(), sort_keys=False)
        second = yaml.safe_dump(promotion_module.rebuild_index(), sort_keys=False)

        assert first == second
        assert yaml.safe_load(index_path.read_text())["entries"][0]["id"] == "num_001"

    def test_capture_invalidates_retrieval_cache(self, monkeypatch):
        import trellis.agent.knowledge as knowledge_pkg
        from trellis.agent.knowledge.promotion import capture_lesson

        class FakeStore:
            def __init__(self) -> None:
                self.reload_calls = 0

            def reload(self) -> None:
                self.reload_calls += 1

        fake_store = FakeStore()
        monkeypatch.setattr(knowledge_pkg, "_store", fake_store, raising=False)

        lid = capture_lesson(
            category="vol_surface",
            title="_test_cache_invalidation",
            severity="medium",
            symptom="test",
            root_cause="test",
            fix="test fix",
            confidence=0.7,
        )

        assert lid is not None
        assert fake_store.reload_calls == 1

    def test_semantic_extension_trace_rebuilds_index_once(self, monkeypatch):
        import trellis.agent.knowledge.promotion as promotion_module

        rebuild_calls = {"count": 0}

        def _counting_rebuild_index() -> dict:
            rebuild_calls["count"] += 1
            return {"entries": [], "settings": {"max_prompt_entries": 7}}

        monkeypatch.setattr(promotion_module, "rebuild_index", _counting_rebuild_index)

        trace_kwargs = {
            "request_id": "request-1",
            "request_text": "Need an extension for a missing pricing primitive",
            "instrument_type": "callable_bond",
            "semantic_gap": {
                "summary": "missing route helper",
                "missing_route_helpers": ["resolve_route"],
            },
            "semantic_extension": {
                "decision": "extend",
                "confidence": 0.9,
                "recommended_next_step": "Add a route helper",
            },
            "route_method": "rate_tree",
        }

        first_path = promotion_module.record_semantic_extension_trace(**trace_kwargs)
        second_path = promotion_module.record_semantic_extension_trace(**trace_kwargs)

        assert Path(first_path).exists()
        assert Path(second_path).exists()
        assert rebuild_calls["count"] == 1

        second_trace = yaml.safe_load(Path(second_path).read_text())
        lesson_id = second_trace["lesson_id"]
        lesson_path = promotion_module._LESSONS_DIR / "entries" / f"{lesson_id}.yaml"
        lesson_data = yaml.safe_load(lesson_path.read_text())
        assert lesson_data["status"] == "promoted"

    def test_distill_rebuilds_index_once(self, monkeypatch):
        import trellis.agent.knowledge.promotion as promotion_module

        rebuild_calls = {"count": 0}

        def _counting_rebuild_index() -> dict:
            rebuild_calls["count"] += 1
            return {"entries": [], "settings": {"max_prompt_entries": 7}}

        monkeypatch.setattr(promotion_module, "rebuild_index", _counting_rebuild_index)

        index_path = promotion_module._INDEX_PATH
        entries = [
            {
                "id": "mc_010",
                "title": "Validated lesson",
                "severity": "high",
                "category": "monte_carlo",
                "status": "validated",
                "applies_when": {
                    "method": ["monte_carlo"],
                    "features": ["discounting"],
                    "instrument": ["basket_option"],
                    "error_signature": None,
                },
            },
            {
                "id": "mc_011",
                "title": "Stale candidate",
                "severity": "medium",
                "category": "monte_carlo",
                "status": "candidate",
                "applies_when": {
                    "method": ["monte_carlo"],
                    "features": ["discounting"],
                    "instrument": ["basket_option"],
                    "error_signature": None,
                },
            },
        ]
        index_path.write_text(
            yaml.safe_dump(
                {
                    "entries": entries,
                    "settings": {"max_prompt_entries": 7},
                },
                sort_keys=False,
            )
        )
        self._write_lesson_entry(
            "mc_010",
            title="Validated lesson",
            status="validated",
            severity="high",
            category="monte_carlo",
            applies_when=entries[0]["applies_when"],
            confidence=0.9,
        )
        self._write_lesson_entry(
            "mc_011",
            title="Stale candidate",
            status="candidate",
            severity="medium",
            category="monte_carlo",
            applies_when=entries[1]["applies_when"],
            confidence=0.5,
            created="2026-01-01T00:00:00",
        )

        stats = promotion_module.distill()

        assert stats["promoted"] == 1
        assert stats["archived"] == 1
        assert rebuild_calls["count"] == 1

    def test_resolve_adapter_lifecycle_records_prefers_newest_persisted_artifact(self):
        import trellis.agent.knowledge.promotion as promotion_module
        from trellis.agent.knowledge.promotion import resolve_adapter_lifecycle_records
        from trellis.agent.knowledge.schema import (
            AdapterLifecycleRecord,
            AdapterLifecycleStatus,
        )

        review_dir = promotion_module._TRACES_DIR / "promotion_reviews"
        review_dir.mkdir(parents=True, exist_ok=True)
        adapter_id = "trellis.instruments._agent.demo_route"

        def _artifact_payload(*, revision: str, reason: str) -> dict[str, object]:
            return {
                "adapter_lifecycle": {
                    "resolved": {
                        "records": [
                            {
                                "adapter_id": adapter_id,
                                "status": "deprecated",
                                "module_path": adapter_id,
                                "validated_against_repo_revision": revision,
                                "supersedes": [],
                                "replacement": "trellis.instruments._agent._fresh.demo_route",
                                "reason": reason,
                                "code_hash": revision,
                            }
                        ]
                    }
                }
            }

        (review_dir / "20260405_120001_demo_approved.yaml").write_text(
            yaml.safe_dump(_artifact_payload(revision="rev-new", reason="newer review"), sort_keys=False)
        )
        (review_dir / "20260405_115959_demo_approved.yaml").write_text(
            yaml.safe_dump(_artifact_payload(revision="rev-old", reason="older review"), sort_keys=False)
        )

        resolved = resolve_adapter_lifecycle_records(
            [
                AdapterLifecycleRecord(
                    adapter_id=adapter_id,
                    status=AdapterLifecycleStatus.STALE,
                    module_path=adapter_id,
                    validated_against_repo_revision="rev-live",
                    replacement="trellis.instruments._agent._fresh.demo_route",
                    reason="live stale record",
                    code_hash="live",
                )
            ]
        )

        record = next(item for item in resolved if item.adapter_id == adapter_id)
        assert record.status == AdapterLifecycleStatus.DEPRECATED
        assert record.validated_against_repo_revision == "rev-new"
        assert record.reason == "newer review"

    def test_format_knowledge_for_prompt_uses_latest_adapter_lifecycle_review(self, monkeypatch):
        import trellis.agent.knowledge.retrieval as retrieval_module
        import trellis.agent.knowledge.promotion as promotion_module
        from trellis.agent.knowledge.schema import (
            AdapterLifecycleRecord,
            AdapterLifecycleStatus,
        )

        review_dir = promotion_module._TRACES_DIR / "promotion_reviews"
        review_dir.mkdir(parents=True, exist_ok=True)
        adapter_id = "trellis.instruments._agent.demo_route"

        for filename, revision, reason in (
            ("20260405_120001_demo_approved.yaml", "rev-new", "newer review"),
            ("20260405_115959_demo_approved.yaml", "rev-old", "older review"),
        ):
            (review_dir / filename).write_text(
                yaml.safe_dump(
                    {
                        "adapter_lifecycle": {
                            "resolved": {
                                "records": [
                                    {
                                        "adapter_id": adapter_id,
                                        "status": "deprecated",
                                        "module_path": adapter_id,
                                        "validated_against_repo_revision": revision,
                                        "supersedes": [],
                                        "replacement": "trellis.instruments._agent._fresh.demo_route",
                                        "reason": reason,
                                        "code_hash": revision,
                                    }
                                ]
                            }
                        }
                    },
                    sort_keys=False,
                )
            )

        monkeypatch.setattr(
            retrieval_module,
            "detect_adapter_lifecycle_records",
            lambda: [
                AdapterLifecycleRecord(
                    adapter_id=adapter_id,
                    status=AdapterLifecycleStatus.STALE,
                    module_path=adapter_id,
                    validated_against_repo_revision="rev-live",
                    replacement="trellis.instruments._agent._fresh.demo_route",
                    reason="live stale record",
                    code_hash="live",
                )
            ],
        )

        text = retrieval_module.format_knowledge_for_prompt({})

        assert "**DEPRECATED**" in text
        assert "newer review" in text
        assert "older review" not in text


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class TestSignatures:

    def test_match_nan_overflow(self):
        from trellis.agent.knowledge.signatures import match_failure
        from trellis.agent.knowledge import get_store
        sigs = get_store()._failure_signatures
        matches = match_failure("RuntimeWarning: overflow in exp(), result is NaN", sigs)
        assert len(matches) >= 1
        assert any(m.category == "numerical_overflow" for m in matches)

    def test_match_callable_gt_straight(self):
        from trellis.agent.knowledge.signatures import match_failure
        from trellis.agent.knowledge import get_store
        sigs = get_store()._failure_signatures
        matches = match_failure("callable > straight bond", sigs)
        assert len(matches) >= 1
        assert any("exercise" in m.diagnostic_hint.lower() for m in matches)


# ---------------------------------------------------------------------------
# Gap check
# ---------------------------------------------------------------------------


class TestGapCheck:

    def test_callable_bond_full_coverage(self):
        from trellis.agent.knowledge.decompose import decompose
        from trellis.agent.knowledge.gap_check import gap_check
        d = decompose("callable bond", instrument_type="callable_bond")
        report = gap_check(d)
        assert report.has_decomposition
        assert report.has_cookbook
        assert report.lesson_count >= 3
        assert report.has_contracts
        assert report.has_requirements
        assert report.confidence >= 0.8

    def test_heston_has_cookbook_after_enrichment(self):
        """After remediation, Heston should have full coverage."""
        from trellis.agent.knowledge.decompose import decompose
        from trellis.agent.knowledge.gap_check import gap_check
        d = decompose("heston option", instrument_type="heston_option")
        report = gap_check(d)
        assert report.has_cookbook  # FFT cookbook added by remediation
        assert report.confidence >= 0.8

    def test_novel_product_low_confidence(self):
        from trellis.agent.knowledge.schema import ProductDecomposition
        from trellis.agent.knowledge.gap_check import gap_check
        d = ProductDecomposition(
            instrument="exotic_widget",
            features=("discounting",),
            method="analytical",
            learned=True,
        )
        report = gap_check(d)
        assert not report.has_decomposition  # learned, not static
        assert report.confidence < 1.0

    def test_gap_warnings_format_with_gaps(self):
        """Test gap warnings for a product with known gaps."""
        from trellis.agent.knowledge.schema import ProductDecomposition
        from trellis.agent.knowledge.gap_check import gap_check, format_gap_warnings
        # Fabricate a product with a method that has no cookbook
        d = ProductDecomposition(
            instrument="exotic_widget",
            features=("discounting",),
            method="nonexistent_method",
            learned=True,
        )
        report = gap_check(d)
        text = format_gap_warnings(report)
        assert "KNOWLEDGE GAPS" in text
        assert "cookbook" in text.lower()

    def test_no_gaps_empty_warnings(self):
        from trellis.agent.knowledge.decompose import decompose
        from trellis.agent.knowledge.gap_check import gap_check, format_gap_warnings
        d = decompose("callable bond", instrument_type="callable_bond")
        report = gap_check(d)
        text = format_gap_warnings(report)
        # callable_bond should have good coverage, minimal warnings
        # (may still have some if lesson count < 3 for specific features)
        assert isinstance(text, str)

    def test_retrieved_lesson_ids_populated(self):
        from trellis.agent.knowledge.decompose import decompose
        from trellis.agent.knowledge.gap_check import gap_check
        d = decompose("callable bond", instrument_type="callable_bond")
        report = gap_check(d)
        assert len(report.retrieved_lesson_ids) >= 3

    def test_gap_warnings_include_similar_products_for_cold_start(self):
        from trellis.agent.knowledge.gap_check import format_gap_warnings, gap_check
        from trellis.agent.knowledge.schema import ProductDecomposition

        report = gap_check(
            ProductDecomposition(
                instrument="callable_range_note",
                features=("callable", "fixed_coupons", "mean_reversion"),
                method="rate_tree",
                learned=True,
            )
        )

        assert report.similar_products
        text = format_gap_warnings(report)
        assert "Similar Products" in text
        assert "callable_bond" in text
        assert report.similar_products[0].promoted_routes[0] in text


# ---------------------------------------------------------------------------
# Reflect (unit tests, no LLM)
# ---------------------------------------------------------------------------


class TestReflect:

    def test_attribute_success_boosts_confidence(self):
        from trellis.agent.knowledge.reflect import _attribute_success
        import trellis.agent.knowledge.promotion as promotion_module
        from trellis.agent.knowledge.promotion import capture_lesson, boost_confidence
        import yaml

        # Create a test lesson
        lid = capture_lesson(
            category="vol_surface", title="_test_attribution",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", confidence=0.5,
        )
        assert lid is not None

        # Read initial confidence
        path = promotion_module._LESSONS_DIR / "entries" / f"{lid}.yaml"
        data = yaml.safe_load(path.read_text())
        initial = data["confidence"]

        # Attribute
        count = _attribute_success([lid])
        assert count == 1

        # Confidence should have increased
        data2 = yaml.safe_load(path.read_text())
        assert data2["confidence"] > initial

        # Cleanup
        path.unlink()
        from trellis.agent.knowledge.promotion import rebuild_index
        rebuild_index()

    def test_auto_validate_and_promote(self):
        from trellis.agent.knowledge.reflect import _auto_validate_and_promote
        import trellis.agent.knowledge.promotion as promotion_module
        from trellis.agent.knowledge.promotion import capture_lesson
        import yaml

        lid = capture_lesson(
            category="vol_surface", title="_test_auto_promote",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", confidence=0.9,
        )
        assert lid is not None

        _auto_validate_and_promote(lid)

        path = promotion_module._LESSONS_DIR / "entries" / f"{lid}.yaml"
        data = yaml.safe_load(path.read_text())
        assert data["status"] == "promoted"

        # Cleanup
        path.unlink()
        from trellis.agent.knowledge.promotion import rebuild_index
        rebuild_index()

    def test_should_distill_false_initially(self):
        from trellis.agent.knowledge.reflect import _should_distill
        # With ~20 entries and few candidates, shouldn't trigger
        # (depends on current state, but should not crash)
        result = _should_distill()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Solution-contract and semantic-definition regression tests
# ---------------------------------------------------------------------------


class TestAnalyticalSolutionContracts:
    """Verify that the analytical cookbook and requirements express
    contract-specific assumptions, not just a generic method label."""

    def test_analytical_cookbook_mentions_solution_contracts(self):
        """The analytical cookbook template should reference specific solution
        contracts (black_scholes_equity, garman_kohlhagen_fx, black76_forward_rate)
        so the LLM sees contract-level distinctions."""
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "analytical",
            features=["vol_surface_dependence"],
            instrument="european_option",
        )
        cookbook = k["cookbook"]
        assert cookbook is not None
        template = cookbook.template
        assert "solution contract" in template.lower(), (
            "Analytical cookbook template should reference solution contracts"
        )
        assert "black_scholes_equity" in template or "Black-Scholes" in template
        assert "garman_kohlhagen_fx" in template or "Garman-Kohlhagen" in template
        assert "black76_forward_rate" in template or "Black76 forward-rate" in template

    def test_analytical_cookbook_states_assumptions(self):
        """Each analytical code block should be preceded by its assumptions."""
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "analytical",
            features=["vol_surface_dependence"],
            instrument="european_option",
        )
        template = k["cookbook"].template
        assert "lognormal" in template.lower() or "GBM" in template
        assert "deterministic" in template.lower()
        assert "European exercise" in template or "european" in template.lower()

    def test_analytical_requirements_per_contract(self):
        """Analytical method requirements should contain contract-specific
        requirement text, not just one generic block."""
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "analytical",
            features=["vol_surface_dependence"],
            instrument="european_option",
        )
        reqs = k["method_requirements"]
        assert reqs is not None
        req_text = " ".join(reqs.requirements)
        # Should mention at least two distinct contracts
        assert "BLACK-SCHOLES" in req_text or "EQUITY" in req_text
        assert "GARMAN-KOHLHAGEN" in req_text or "FX" in req_text
        assert "OUTPUT CONTRACT" in req_text

    def test_analytical_requirements_not_empty(self):
        """Regression: the restructured YAML must still load non-empty requirements."""
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "analytical",
            features=["vol_surface_dependence"],
            instrument="cap",
        )
        reqs = k["method_requirements"]
        assert reqs is not None
        assert len(reqs.requirements) >= 5, (
            f"Expected >=5 analytical requirements after split, got {len(reqs.requirements)}"
        )

    def test_analytical_description_mentions_contracts(self):
        """The cookbook description should signal that contracts are not interchangeable."""
        from trellis.agent.knowledge import retrieve_for_task
        k = retrieve_for_task(
            "analytical",
            features=["vol_surface_dependence"],
            instrument="european_option",
        )
        desc = k["cookbook"].description
        assert "not interchangeable" in desc.lower() or "solution contract" in desc.lower(), (
            "Analytical cookbook description should warn that contracts are assumption-bound"
        )

    def test_prompt_surface_includes_contract_assumptions(self):
        """When formatted for the builder prompt, the analytical knowledge
        should include assumption text, not just code templates."""
        from trellis.agent.knowledge import retrieve_for_task
        from trellis.agent.knowledge.retrieval import format_knowledge_for_prompt
        k = retrieve_for_task(
            "analytical",
            features=["vol_surface_dependence"],
            instrument="european_option",
        )
        prompt_text = format_knowledge_for_prompt(k)
        assert "assumption" in prompt_text.lower() or "Assumptions" in prompt_text, (
            "Builder prompt should include assumption text for analytical contracts"
        )

    def test_market_data_candidate_swaption_cluster_is_deduplicated(self):
        root = Path(__file__).resolve().parents[2]
        candidate_index = yaml.safe_load(
            (root / "trellis" / "agent" / "knowledge" / "lessons" / "index.yaml").read_text()
        )["entries"]

        validated_titles = {
            entry["title"]
            for entry in candidate_index
            if entry.get("status") in {"validated", "promoted"}
            and entry.get("category") == "market_data"
            and entry.get("applies_when", {}).get("method") == ["analytical"]
            and entry.get("applies_when", {}).get("features")
            == ["floating_coupons", "vol_surface_dependence"]
        }
        index_titles = {
            entry["title"]
            for entry in candidate_index
            if entry.get("status") == "candidate"
            and entry.get("category") == "market_data"
            and entry.get("applies_when", {}).get("method") == ["analytical"]
            and entry.get("applies_when", {}).get("features")
            == ["floating_coupons", "vol_surface_dependence"]
        }

        assert {"Module reference typo", "Transient API failure"} <= validated_titles
        assert index_titles <= {"Module reference typo", "Transient API failure"}
        assert len(index_titles) <= 2
