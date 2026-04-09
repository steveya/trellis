"""Tests for the two-stage route scorer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.route_registry import RouteAdmissibilitySpec, RouteSpec, load_route_registry
from trellis.agent.route_scorer import (
    LearnedModel,
    RouteScorer,
    ScoringContext,
    extract_scoring_features,
    get_scorer,
    model_metadata,
    _save_model,
    _load_model,
)


@pytest.fixture(scope="module")
def registry():
    return load_route_registry()


@pytest.fixture(scope="module")
def scorer(registry):
    return RouteScorer(registry)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_basic_features(self, registry):
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        ir = ProductIR(instrument="swaption", payoff_family="swaption", exercise_style="european")
        ctx = ScoringContext(
            product_ir=ir,
            route_spec=spec,
            pricing_plan=None,
            blockers=[],
        )
        features = extract_scoring_features(ctx)
        assert features["bias"] == 1.0
        assert features["blocker_count"] == 0.0
        assert features["route:analytical_black76"] == 1.0
        assert features["engine_family:analytical"] == 1.0
        assert features["exercise:european"] == 1.0
        assert features["payoff:swaption"] == 1.0
        assert features["family_capability_ok"] == 1.0

    def test_blocker_features(self, registry):
        spec = [r for r in registry.routes if r.id == "monte_carlo_paths"][0]
        ir = ProductIR(instrument="exotic", payoff_family="exotic")
        ctx = ScoringContext(
            product_ir=ir,
            route_spec=spec,
            pricing_plan=None,
            blockers=["missing_module:foo", "missing_symbol:bar"],
        )
        features = extract_scoring_features(ctx)
        assert features["blocker_count"] == 2.0
        assert features["blocker:missing_module:foo"] == 1.0

    def test_capability_failure_features(self):
        spec = RouteSpec(
            id="synthetic_terminal_mc",
            engine_family="monte_carlo",
            route_family="monte_carlo",
            status="promoted",
            confidence=1.0,
            match_methods=("monte_carlo",),
            match_instruments=None,
            exclude_instruments=(),
            match_payoff_family=None,
            match_payoff_traits=None,
            match_exercise=None,
            exclude_exercise=(),
            match_required_market_data=None,
            exclude_required_market_data=None,
            primitives=(),
            conditional_primitives=(),
            conditional_route_family=None,
            adapters=(),
            notes=(),
            admissibility=RouteAdmissibilitySpec(
                supported_state_tags=("terminal_markov",),
            ),
        )
        ir = ProductIR(
            instrument="cds",
            payoff_family="credit_default_swap",
            schedule_dependence=True,
            state_dependence="pathwise_only",
            candidate_engine_families=("monte_carlo",),
            route_families=("credit_default_swap",),
        )
        ctx = ScoringContext(
            product_ir=ir,
            route_spec=spec,
            pricing_plan=None,
            blockers=[],
        )

        features = extract_scoring_features(ctx)

        assert features["family_capability_ok"] == 0.0
        assert features["capability_failure:family_identity_mismatch"] == 1.0
        assert features["capability_failure:schedule_dependence_unsupported"] == 1.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_heuristic_fallback_when_no_model(self, scorer, registry):
        assert not scorer.has_trained_model
        spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
        ir = ProductIR(
            instrument="swaption",
            payoff_family="swaption",
            exercise_style="european",
            candidate_engine_families=("analytical",),
        )
        ctx = ScoringContext(
            product_ir=ir, route_spec=spec, pricing_plan=None, blockers=[],
        )
        result = scorer.score_route(ctx)
        assert result.scoring_method == "heuristic_fallback"
        assert result.final_score > 0

    def test_scores_match_heuristic_exactly(self, scorer, registry):
        from trellis.agent.codegen_guardrails import _route_score
        spec = [r for r in registry.routes if r.id == "exercise_lattice"][0]
        ir = ProductIR(
            instrument="callable_bond",
            payoff_family="callable_bond",
            exercise_style="issuer_call",
            model_family="interest_rate",
            candidate_engine_families=("lattice",),
            route_families=("rate_lattice",),
        )
        from trellis.agent.route_registry import resolve_route_family
        rf = resolve_route_family(spec, ir)
        blockers = []

        ctx = ScoringContext(
            product_ir=ir, route_spec=spec, pricing_plan=None,
            blockers=blockers, route_family=rf,
        )
        scorer_result = scorer.score_route(ctx)
        heuristic_result = _route_score(spec, ir, blockers, route_family=rf)
        assert scorer_result.final_score == heuristic_result
        assert scorer_result.heuristic_score == heuristic_result

    def test_rank_routes_returns_sorted(self, scorer, registry):
        from trellis.agent.route_registry import resolve_route_family
        specs = [r for r in registry.routes if r.id in ("exercise_lattice", "rate_tree_backward_induction")]
        ir = ProductIR(
            instrument="callable_bond",
            payoff_family="callable_bond",
            exercise_style="issuer_call",
            model_family="interest_rate",
            candidate_engine_families=("lattice",),
            route_families=("rate_lattice",),
        )
        contexts = [
            ScoringContext(
                product_ir=ir, route_spec=s, pricing_plan=None, blockers=[],
                route_family=resolve_route_family(s, ir),
            )
            for s in specs
        ]
        ranked = scorer.rank_routes(contexts)
        assert len(ranked) == 2
        assert ranked[0].final_score >= ranked[1].final_score

    def test_capability_mismatch_penalizes_heuristic_score(self, scorer):
        matching = RouteSpec(
            id="family_mc",
            engine_family="monte_carlo",
            route_family="credit_default_swap",
            status="promoted",
            confidence=1.0,
            match_methods=("monte_carlo",),
            match_instruments=None,
            exclude_instruments=(),
            match_payoff_family=None,
            match_payoff_traits=None,
            match_exercise=None,
            exclude_exercise=(),
            match_required_market_data=None,
            exclude_required_market_data=None,
            primitives=(),
            conditional_primitives=(),
            conditional_route_family=None,
            adapters=(),
            notes=(),
            admissibility=RouteAdmissibilitySpec(
                supported_state_tags=("schedule_state", "pathwise_only"),
            ),
        )
        mismatched = RouteSpec(
            id="generic_mc",
            engine_family="monte_carlo",
            route_family="monte_carlo",
            status="promoted",
            confidence=1.0,
            match_methods=("monte_carlo",),
            match_instruments=None,
            exclude_instruments=(),
            match_payoff_family=None,
            match_payoff_traits=None,
            match_exercise=None,
            exclude_exercise=(),
            match_required_market_data=None,
            exclude_required_market_data=None,
            primitives=(),
            conditional_primitives=(),
            conditional_route_family=None,
            adapters=(),
            notes=(),
            admissibility=RouteAdmissibilitySpec(
                supported_state_tags=("terminal_markov",),
            ),
        )
        ir = ProductIR(
            instrument="cds",
            payoff_family="credit_default_swap",
            schedule_dependence=True,
            state_dependence="pathwise_only",
            candidate_engine_families=("monte_carlo",),
            route_families=("credit_default_swap",),
        )

        matching_score = scorer.score_route(
            ScoringContext(product_ir=ir, route_spec=matching, pricing_plan=None, blockers=[]),
        )
        mismatched_score = scorer.score_route(
            ScoringContext(product_ir=ir, route_spec=mismatched, pricing_plan=None, blockers=[]),
        )

        assert matching_score.final_score > mismatched_score.final_score


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_save_and_load_roundtrip(self):
        model = LearnedModel(
            feature_names=("bias", "blocker_count"),
            weights=(1.5, -3.0),
            training_size=20,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            _save_model(model, path)
            loaded = _load_model(path)
            assert loaded is not None
            assert loaded.feature_names == model.feature_names
            assert loaded.weights == model.weights
            assert loaded.training_size == 20

    def test_load_returns_none_when_missing(self):
        loaded = _load_model(Path("/nonexistent/model.json"))
        assert loaded is None

    def test_model_metadata_no_model(self):
        meta = model_metadata()
        # May or may not have a model — just verify it returns a dict
        assert isinstance(meta, dict)
        assert "trained" in meta


# ---------------------------------------------------------------------------
# Trained model scoring
# ---------------------------------------------------------------------------

class TestTrainedModelScoring:
    def test_trained_model_overrides_heuristic(self, registry):
        model = LearnedModel(
            feature_names=("bias", "route:analytical_black76"),
            weights=(10.0, 5.0),
            training_size=50,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            _save_model(model, path)
            scorer = RouteScorer(registry, model_path=path)
            assert scorer.has_trained_model

            spec = [r for r in registry.routes if r.id == "analytical_black76"][0]
            ir = ProductIR(instrument="swaption", payoff_family="swaption")
            ctx = ScoringContext(
                product_ir=ir, route_spec=spec, pricing_plan=None, blockers=[],
            )
            result = scorer.score_route(ctx)
            assert result.scoring_method == "linear"
            assert result.linear_score == pytest.approx(15.0)  # 10.0 * 1 + 5.0 * 1
