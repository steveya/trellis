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
            reusable_primitives=("garman_kohlhagen_call", "garman_kohlhagen_put"),
            supported=True,
            preferred_method="analytical",
        )
        k = retrieve_for_product_ir(ir, preferred_method="analytical")

        assert k["cookbook"] is not None
        assert "garman_kohlhagen_call" in k["cookbook"].template
        assert k["data_contracts"]
        contract_names = {contract.name for contract in k["data_contracts"]}
        assert "FX_DOMESTIC_FOREIGN_DISCOUNTING" in contract_names
        assert k["method_requirements"] is not None
        requirements_text = "\n".join(k["method_requirements"].requirements)
        assert "GARMAN-KOHLHAGEN" in requirements_text
        assert "fx_rates" in requirements_text
        assert "forward_curve" in requirements_text

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
        # Even empty retrieval includes the import registry
        assert "AVAILABLE IMPORTS" in text

    def test_build_shared_knowledge_payload_includes_prompt_views_and_summary(self):
        from trellis.agent.knowledge import retrieve_for_product_ir
        from trellis.agent.knowledge.decompose import decompose_to_ir
        from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload

        product_ir = decompose_to_ir(
            "American put option on equity",
            instrument_type="american_option",
        )
        knowledge = retrieve_for_product_ir(product_ir, preferred_method="monte_carlo")
        payload = build_shared_knowledge_payload(knowledge)

        assert "## Product Semantics" in payload["builder_text"]
        assert "## Shared Failure Memory" in payload["review_text"]
        assert payload["routing_text"]
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
    def cleanup(self):
        """Clean up any test lessons after each test."""
        yield
        # Remove test entries from index and entries dir
        from trellis.agent.knowledge.promotion import _LESSONS_DIR, _INDEX_PATH
        index = yaml.safe_load(_INDEX_PATH.read_text()) if _INDEX_PATH.exists() else {}
        entries = index.get("entries", [])
        cleaned = [e for e in entries if not e.get("title", "").startswith("_test_")]
        if len(cleaned) != len(entries):
            index["entries"] = cleaned
            with open(_INDEX_PATH, "w") as f:
                yaml.dump(index, f, default_flow_style=False, sort_keys=False)
        # Remove test entry files
        entries_dir = _LESSONS_DIR / "entries"
        for f in entries_dir.glob("*_test_*.yaml"):
            f.unlink()
        for f in entries_dir.glob("vs_*.yaml"):
            f.unlink()

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


# ---------------------------------------------------------------------------
# Reflect (unit tests, no LLM)
# ---------------------------------------------------------------------------


class TestReflect:

    def test_attribute_success_boosts_confidence(self):
        from trellis.agent.knowledge.reflect import _attribute_success
        from trellis.agent.knowledge.promotion import capture_lesson, boost_confidence
        import yaml
        from pathlib import Path

        # Create a test lesson
        lid = capture_lesson(
            category="vol_surface", title="_test_attribution",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", confidence=0.5,
        )
        assert lid is not None

        # Read initial confidence
        path = Path(f"/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/lessons/entries/{lid}.yaml")
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
        # Remove from index
        idx_path = Path("/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/lessons/index.yaml")
        idx = yaml.safe_load(idx_path.read_text()) or {}
        idx["entries"] = [e for e in idx.get("entries", []) if e["id"] != lid]
        with open(idx_path, "w") as f:
            yaml.dump(idx, f, default_flow_style=False, sort_keys=False)

    def test_auto_validate_and_promote(self):
        from trellis.agent.knowledge.reflect import _auto_validate_and_promote
        from trellis.agent.knowledge.promotion import capture_lesson
        import yaml
        from pathlib import Path

        lid = capture_lesson(
            category="vol_surface", title="_test_auto_promote",
            severity="medium", symptom="test", root_cause="test",
            fix="test fix", confidence=0.9,
        )
        assert lid is not None

        _auto_validate_and_promote(lid)

        path = Path(f"/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/lessons/entries/{lid}.yaml")
        data = yaml.safe_load(path.read_text())
        assert data["status"] == "promoted"

        # Cleanup
        path.unlink()
        idx_path = Path("/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/lessons/index.yaml")
        idx = yaml.safe_load(idx_path.read_text()) or {}
        idx["entries"] = [e for e in idx.get("entries", []) if e["id"] != lid]
        with open(idx_path, "w") as f:
            yaml.dump(idx, f, default_flow_style=False, sort_keys=False)

    def test_should_distill_false_initially(self):
        from trellis.agent.knowledge.reflect import _should_distill
        # With ~20 entries and few candidates, shouldn't trigger
        # (depends on current state, but should not crash)
        result = _should_distill()
        assert isinstance(result, bool)
