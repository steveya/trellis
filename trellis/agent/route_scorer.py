"""Two-stage route scorer: linear model + optional LLM re-rank.

Stage 1 (always runs, <1ms): A fitted linear scorer over feature maps.
When no trained model exists, falls back to the YAML-driven heuristic
in ``codegen_guardrails._route_score``.

Stage 2 (optional, off by default): When the top-2 linear scores are
within ``llm_rerank_margin``, an LLM call breaks the tie.  Results are
NOT cached in the generation plan cache (non-deterministic).

Training data comes from ``task_run_store`` outcome records.  The model
is persisted as a JSON file of feature names and weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.route_registry import (
    RouteRegistry,
    RouteSpec,
    evaluate_route_capability_match,
    resolve_route_primitives,
)
from trellis.core.differentiable import get_numpy

np = get_numpy()


_MODEL_PATH = (
    Path(__file__).resolve().parent / "knowledge" / "canonical" / "route_model.json"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringContext:
    """All signals available at scoring time."""

    product_ir: ProductIR | None
    route_spec: RouteSpec
    pricing_plan: Any  # PricingPlan — avoid circular import
    blockers: list[str]
    route_family: str = ""
    knowledge_gap_confidence: float = 1.0
    lesson_count: int = 0


@dataclass(frozen=True)
class RouteScore:
    """Scored route with provenance."""

    route_id: str
    linear_score: float
    heuristic_score: float
    llm_rerank_score: float | None = None
    final_score: float = 0.0
    confidence: str = "high"
    scoring_method: str = "heuristic_fallback"


# ---------------------------------------------------------------------------
# Learned model persistence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LearnedModel:
    """Fitted linear weights + metadata."""

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    training_size: int
    ridge: float = 1e-6

    def score(self, feature_map: dict[str, float]) -> float:
        return float(sum(
            w * float(feature_map.get(name, 0.0))
            for name, w in zip(self.feature_names, self.weights)
        ))


def _load_model(path: Path | None = None) -> LearnedModel | None:
    """Load a trained model from JSON.  Returns None if file doesn't exist."""
    path = path or _MODEL_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return LearnedModel(
            feature_names=tuple(data["feature_names"]),
            weights=tuple(float(w) for w in data["weights"]),
            training_size=int(data.get("training_size", 0)),
            ridge=float(data.get("ridge", 1e-6)),
        )
    except Exception:
        return None


def _save_model(model: LearnedModel, path: Path | None = None) -> None:
    """Persist a trained model to JSON."""
    path = path or _MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "feature_names": list(model.feature_names),
        "weights": [float(w) for w in model.weights],
        "training_size": model.training_size,
        "ridge": model.ridge,
    }
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def extract_scoring_features(ctx: ScoringContext) -> dict[str, float]:
    """Build a feature map from scoring context.

    Extends ``route_learning.extract_route_feature_map`` with outcome-derived
    and registry-derived signals.
    """
    spec = ctx.route_spec
    ir = ctx.product_ir
    route_family = ctx.route_family or spec.route_family

    features: dict[str, float] = {
        "bias": 1.0,
        "blocker_count": float(len(ctx.blockers)),
        "knowledge_gap_confidence": ctx.knowledge_gap_confidence,
        "lesson_count": float(min(ctx.lesson_count, 10)),
        "route_confidence": spec.confidence,
        "successful_builds": float(min(spec.successful_builds, 20)),
    }
    resolved_primitives = tuple(resolve_route_primitives(spec, ir))
    resolved_roles = {primitive.role for primitive in resolved_primitives}
    exact_surface_roles = {"route_helper", "pricing_kernel", "cashflow_engine"}

    features["binding_role_count"] = float(len(resolved_roles))
    features["binding_has_exact_surface"] = 1.0 if resolved_roles.intersection(exact_surface_roles) else 0.0
    for role in resolved_roles:
        features[f"binding_role:{role}"] = 1.0
    if ir is not None:
        model_family = str(getattr(ir, "model_family", "") or "").strip()
        model_support_roles = {
            "credit_copula": {"default_time_sampler", "loss_distribution"},
        }.get(model_family, set())
        for role in resolved_roles.intersection(model_support_roles):
            features[f"model_support_role:{model_family}:{role}"] = 1.0

    if ir is not None:
        capability = evaluate_route_capability_match(spec, ir)
        features["product_supported"] = 1.0 if ir.supported else 0.0
        features["schedule_dependence"] = 1.0 if ir.schedule_dependence else 0.0
        features["has_unresolved_primitives"] = 1.0 if ir.unresolved_primitives else 0.0
        features["engine_family_matches_ir"] = (
            1.0 if spec.engine_family in ir.candidate_engine_families else 0.0
        )
        features["family_capability_ok"] = 1.0 if capability.ok else 0.0
        features["family_capability_match_count"] = float(len(capability.matched_predicates))
        features[f"exercise:{ir.exercise_style}"] = 1.0
        features[f"state:{ir.state_dependence}"] = 1.0
        features[f"model:{ir.model_family}"] = 1.0
        features[f"payoff:{ir.payoff_family}"] = 1.0
        for predicate in capability.matched_predicates:
            features[f"capability:{predicate}"] = 1.0
        for failure in capability.failures:
            features[f"capability_failure:{failure}"] = 1.0

    features[f"engine_family:{spec.engine_family}"] = 1.0
    features[f"status:{spec.status}"] = 1.0

    for blocker in ctx.blockers:
        features[f"blocker:{blocker}"] = 1.0

    return features


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class RouteScorer:
    """Two-stage route scorer with heuristic fallback."""

    def __init__(
        self,
        registry: RouteRegistry,
        model_path: Path | None = None,
    ):
        self._registry = registry
        self._model = _load_model(model_path)
        self._has_model = self._model is not None

    @property
    def has_trained_model(self) -> bool:
        return self._has_model

    def score_route(
        self,
        ctx: ScoringContext,
        *,
        llm_rerank: bool = False,
    ) -> RouteScore:
        """Score a single route.

        Uses the trained linear model if available; otherwise falls back
        to the YAML-driven heuristic.
        """
        from trellis.agent.codegen_guardrails import _route_score

        heuristic = _route_score(
            ctx.route_spec,
            ctx.product_ir,
            ctx.blockers,
            route_family=ctx.route_family,
            pricing_plan=ctx.pricing_plan,
        )

        if self._model is not None:
            features = extract_scoring_features(ctx)
            linear = self._model.score(features)
            return RouteScore(
                route_id=ctx.route_spec.id,
                linear_score=linear,
                heuristic_score=heuristic,
                final_score=linear,
                scoring_method="linear",
            )

        return RouteScore(
            route_id=ctx.route_spec.id,
            linear_score=heuristic,
            heuristic_score=heuristic,
            final_score=heuristic,
            scoring_method="heuristic_fallback",
        )

    def rank_routes(
        self,
        contexts: list[ScoringContext],
        *,
        llm_rerank: bool = False,
        llm_rerank_margin: float = 1.5,
    ) -> list[RouteScore]:
        """Score and rank multiple route candidates.

        Stage 1: Score all candidates (linear or heuristic).
        Stage 2: If llm_rerank=True and top-2 within margin, ask LLM.
        """
        scored = [self.score_route(ctx) for ctx in contexts]
        scored.sort(key=lambda s: (-s.final_score, s.route_id))

        if (
            llm_rerank
            and len(scored) >= 2
            and abs(scored[0].final_score - scored[1].final_score) < llm_rerank_margin
        ):
            reranked = self._llm_rerank(
                scored[0], scored[1],
                contexts[0].product_ir,
            )
            if reranked is not None:
                scored = reranked

        return scored

    def _llm_rerank(
        self,
        a: RouteScore,
        b: RouteScore,
        product_ir: ProductIR | None,
    ) -> list[RouteScore] | None:
        """Ask an LLM to break a tie between two close-scoring routes."""
        try:
            from trellis.agent.config import get_model_for_stage
            from trellis.agent.prompts import llm_generate_json

            model = get_model_for_stage("route_selection")
            ir_summary = ""
            if product_ir is not None:
                ir_summary = (
                    f"instrument={product_ir.instrument}, "
                    f"payoff_family={product_ir.payoff_family}, "
                    f"exercise={product_ir.exercise_style}, "
                    f"model_family={product_ir.model_family}"
                )

            prompt = (
                f"Two candidate pricing routes scored similarly for this product:\n"
                f"Product: {ir_summary}\n\n"
                f"Route A: {a.route_id} (score: {a.final_score:.2f})\n"
                f"Route B: {b.route_id} (score: {b.final_score:.2f})\n\n"
                f"Which route is more likely to produce correct pricing?\n"
                f'Return JSON: {{"selected": "A" or "B", "reason": "brief reason"}}'
            )

            result = llm_generate_json(prompt, model=model)
            if result and isinstance(result, dict):
                selected = result.get("selected", "").upper()
                if selected == "A":
                    return [
                        RouteScore(
                            route_id=a.route_id,
                            linear_score=a.linear_score,
                            heuristic_score=a.heuristic_score,
                            llm_rerank_score=1.0,
                            final_score=a.final_score + 0.5,
                            scoring_method="linear+llm",
                        ),
                        b,
                    ]
                elif selected == "B":
                    return [
                        RouteScore(
                            route_id=b.route_id,
                            linear_score=b.linear_score,
                            heuristic_score=b.heuristic_score,
                            llm_rerank_score=1.0,
                            final_score=b.final_score + 0.5,
                            scoring_method="linear+llm",
                        ),
                        a,
                    ]
        except Exception:
            pass
        return None

    def train_from_outcomes(self, task_run_dir: Path) -> LearnedModel | None:
        """Fit a linear model from historical task_run_store data.

        Scans JSON run files in ``task_run_dir``, extracts (route, success)
        pairs, builds feature maps, and fits a ridge regression.
        """
        rows = _extract_training_rows(task_run_dir, self._registry)
        if len(rows) < 5:
            return None

        feature_names = tuple(sorted({
            name for features, _ in rows for name in features
        }))
        x = np.array([
            [float(features.get(name, 0.0)) for name in feature_names]
            for features, _ in rows
        ])
        y = np.array([target for _, target in rows], dtype=float)

        ridge = 1e-6
        xtx = x.T @ x
        reg = np.eye(xtx.shape[0]) * ridge
        weights = np.linalg.solve(xtx + reg, x.T @ y)

        model = LearnedModel(
            feature_names=feature_names,
            weights=tuple(float(w) for w in weights),
            training_size=len(rows),
            ridge=ridge,
        )
        _save_model(model)
        self._model = model
        self._has_model = True
        return model


# ---------------------------------------------------------------------------
# Training data extraction
# ---------------------------------------------------------------------------

def _extract_training_rows(
    task_run_dir: Path,
    registry: RouteRegistry,
) -> list[tuple[dict[str, float], float]]:
    """Extract (feature_map, target) pairs from task run JSON files."""
    rows: list[tuple[dict[str, float], float]] = []

    if not task_run_dir.exists():
        return rows

    for run_file in sorted(task_run_dir.glob("*.json")):
        try:
            data = json.loads(run_file.read_text())
            _extract_from_run(data, registry, rows)
        except Exception:
            continue

    return rows


def _extract_from_run(
    data: dict,
    registry: RouteRegistry,
    rows: list[tuple[dict[str, float], float]],
) -> None:
    """Extract training rows from a single task run record."""
    success = data.get("success", False)
    method_results = data.get("method_results", {})

    for method_key, method_data in method_results.items():
        route_name = method_data.get("route")
        if not route_name:
            # Try to extract from build observability
            obs = method_data.get("build_observability", {})
            route_name = obs.get("route") or obs.get("primitive_plan_route")
        if not route_name:
            continue

        # Find the route spec
        spec = None
        for r in registry.routes:
            if r.id == route_name or route_name in r.aliases:
                spec = r
                break
        if spec is None:
            continue

        method_success = method_data.get("success", False)
        attempts = method_data.get("attempts", 1)
        gap_confidence = method_data.get("gap_confidence", 1.0)

        # Build a synthetic ScoringContext feature map
        ir = _ir_from_method_data(method_data)
        ctx = ScoringContext(
            product_ir=ir,
            route_spec=spec,
            pricing_plan=None,
            blockers=[],
            knowledge_gap_confidence=gap_confidence,
        )
        features = extract_scoring_features(ctx)
        features["historical_attempts"] = float(min(attempts, 5))

        # Target: positive for success, negative for failure
        if method_success:
            target = max(2.0, 6.0 - attempts)  # Higher for fewer attempts
        else:
            target = -4.0

        rows.append((features, target))


def _ir_from_method_data(method_data: dict) -> ProductIR | None:
    """Reconstruct a minimal ProductIR from task run method data."""
    try:
        instrument = method_data.get("instrument", "")
        payoff_family = method_data.get("payoff_family", instrument)
        exercise = method_data.get("exercise_style", "none")
        return ProductIR(
            instrument=instrument,
            payoff_family=payoff_family,
            exercise_style=exercise,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_scorer(
    registry: RouteRegistry | None = None,
    model_path: Path | None = None,
) -> RouteScorer:
    """Create a scorer, loading the registry and model if not provided."""
    if registry is None:
        from trellis.agent.route_registry import load_route_registry
        registry = load_route_registry()
    return RouteScorer(registry, model_path)


def model_metadata() -> dict[str, Any]:
    """Return metadata about the currently loaded model."""
    model = _load_model()
    if model is None:
        return {"trained": False}
    return {
        "trained": True,
        "feature_count": len(model.feature_names),
        "training_size": model.training_size,
        "ridge": model.ridge,
    }
