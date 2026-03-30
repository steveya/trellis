"""Offline route-learning helpers for primitive-route ranking.

Phase 8 keeps all legality checks deterministic. The learned layer only ranks
candidate routes after:

- ProductIR construction
- candidate-route enumeration
- primitive/blocker detection

This module builds a small offline dataset from synthetic and known products,
fits a simple linear scorer, and evaluates route choices under the same hard
blocker semantics as the live build path.
"""

from __future__ import annotations

from dataclasses import dataclass

from trellis.agent.codegen_guardrails import PrimitivePlan, rank_primitive_routes
from trellis.agent.knowledge.decompose import decompose_to_ir
from trellis.agent.quant import PricingPlan
from trellis.core.differentiable import get_numpy


np = get_numpy()


@dataclass(frozen=True)
class SyntheticProductCase:
    """Deterministic product case used for offline route-learning data."""

    description: str
    instrument_type: str | None
    preferred_method: str


@dataclass(frozen=True)
class RouteTrainingRow:
    """Single supervised row for route-ranking experiments."""

    description: str
    instrument: str
    preferred_method: str
    route: str
    heuristic_score: float
    target: float
    decision: str
    blockers: tuple[str, ...]
    feature_map: dict[str, float]


@dataclass(frozen=True)
class LearnedRouteRanker:
    """Simple fitted linear scorer over route feature maps."""

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    ridge: float = 1e-6

    def score_feature_map(self, feature_map: dict[str, float]) -> float:
        """Score a feature map with the fitted linear weights."""
        return float(sum(
            weight * float(feature_map.get(name, 0.0))
            for name, weight in zip(self.feature_names, self.weights)
        ))


@dataclass(frozen=True)
class LearnedRouteCandidate:
    """Candidate route scored by the learned ranker."""

    route: str
    learned_score: float
    heuristic_score: float
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class LearnedRouteDecision:
    """Top-level decision produced by the learned ranker under hard gates."""

    decision: str
    selected_route: str | None
    blockers: tuple[str, ...]
    candidates: tuple[LearnedRouteCandidate, ...]


def default_synthetic_product_cases() -> tuple[SyntheticProductCase, ...]:
    """Return a deterministic corpus of supported and blocked product cases."""
    return (
        SyntheticProductCase(
            description="European payer swaption",
            instrument_type="swaption",
            preferred_method="analytical",
        ),
        SyntheticProductCase(
            description="American put option on equity",
            instrument_type="american_option",
            preferred_method="monte_carlo",
        ),
        SyntheticProductCase(
            description="Callable bond with semiannual coupon and call schedule",
            instrument_type="callable_bond",
            preferred_method="rate_tree",
        ),
        SyntheticProductCase(
            description="Bermudan swaption: tree vs LSM MC",
            instrument_type="bermudan_swaption",
            preferred_method="rate_tree",
        ),
        SyntheticProductCase(
            description="Build a pricer for: Geometric Asian option: closed-form vs MC",
            instrument_type="asian_option",
            preferred_method="monte_carlo",
        ),
        SyntheticProductCase(
            description="Build a pricer for: FX barrier option: analytical vs MC",
            instrument_type="barrier_option",
            preferred_method="monte_carlo",
        ),
        SyntheticProductCase(
            description="Build a pricer for: FFT vs COS: GBM calls/puts across strikes and maturities",
            instrument_type=None,
            preferred_method="fft_pricing",
        ),
        SyntheticProductCase(
            description="American Asian barrier option under Heston with early exercise",
            instrument_type=None,
            preferred_method="monte_carlo",
        ),
    )


def build_route_training_rows(
    cases: tuple[SyntheticProductCase, ...] | list[SyntheticProductCase],
) -> tuple[RouteTrainingRow, ...]:
    """Build route-ranking rows from deterministic product cases."""
    rows: list[RouteTrainingRow] = []
    for case in cases:
        product_ir = decompose_to_ir(
            case.description,
            instrument_type=case.instrument_type,
        )
        pricing_plan = _pricing_plan_for_case(case, product_ir)
        ranked = rank_primitive_routes(
            pricing_plan=pricing_plan,
            product_ir=product_ir,
        )
        for idx, candidate in enumerate(ranked):
            decision, target = _label_candidate(candidate, product_ir, idx)
            rows.append(RouteTrainingRow(
                description=case.description,
                instrument=product_ir.instrument,
                preferred_method=case.preferred_method,
                route=candidate.route,
                heuristic_score=candidate.score,
                target=target,
                decision=decision,
                blockers=candidate.blockers,
                feature_map=extract_route_feature_map(candidate, product_ir),
            ))
    return tuple(rows)


def fit_linear_route_ranker(
    rows: tuple[RouteTrainingRow, ...] | list[RouteTrainingRow],
    *,
    ridge: float = 1e-6,
) -> LearnedRouteRanker:
    """Fit a simple ridge-regularized linear scorer to route training rows."""
    if not rows:
        return LearnedRouteRanker(feature_names=("bias",), weights=(0.0,), ridge=ridge)

    feature_names = tuple(sorted({
        feature_name
        for row in rows
        for feature_name in row.feature_map
    }))
    x = np.array([
        [float(row.feature_map.get(name, 0.0)) for name in feature_names]
        for row in rows
    ])
    y = np.array([row.target for row in rows], dtype=float)
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0]) * ridge
    weights = np.linalg.solve(xtx + reg, x.T @ y)
    return LearnedRouteRanker(
        feature_names=feature_names,
        weights=tuple(float(value) for value in weights),
        ridge=ridge,
    )


def rank_routes_with_learned_model(
    *,
    pricing_plan: PricingPlan,
    product_ir,
    ranker: LearnedRouteRanker,
) -> tuple[LearnedRouteCandidate, ...]:
    """Rank candidate routes using a fitted linear scorer."""
    candidates = rank_primitive_routes(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )
    learned_candidates = [
        LearnedRouteCandidate(
            route=candidate.route,
            learned_score=ranker.score_feature_map(
                extract_route_feature_map(candidate, product_ir)
            ),
            heuristic_score=candidate.score,
            blockers=candidate.blockers,
        )
        for candidate in candidates
    ]
    learned_candidates.sort(
        key=lambda candidate: (
            -candidate.learned_score,
            len(candidate.blockers),
            -candidate.heuristic_score,
            candidate.route,
        )
    )
    return tuple(learned_candidates)


def learned_route_decision(
    *,
    pricing_plan: PricingPlan,
    product_ir,
    ranker: LearnedRouteRanker,
) -> LearnedRouteDecision:
    """Select a learned top route while preserving hard blocker semantics."""
    candidates = rank_routes_with_learned_model(
        pricing_plan=pricing_plan,
        product_ir=product_ir,
        ranker=ranker,
    )
    if not candidates:
        return LearnedRouteDecision(
            decision="block",
            selected_route=None,
            blockers=(),
            candidates=(),
        )

    top = candidates[0]
    if top.blockers and not product_ir.supported:
        return LearnedRouteDecision(
            decision="block",
            selected_route=top.route,
            blockers=top.blockers,
            candidates=candidates,
        )
    return LearnedRouteDecision(
        decision="proceed",
        selected_route=top.route,
        blockers=top.blockers,
        candidates=candidates,
    )


def extract_route_feature_map(candidate: PrimitivePlan, product_ir) -> dict[str, float]:
    """Extract a stable numeric feature map for route learning."""
    route_family = getattr(candidate, "route_family", "") or candidate.engine_family
    route_families = set(getattr(product_ir, "route_families", ()) or ())
    feature_map: dict[str, float] = {
        "bias": 1.0,
        "blocker_count": float(len(candidate.blockers)),
        "product_supported": 1.0 if product_ir.supported else 0.0,
        "schedule_dependence": 1.0 if product_ir.schedule_dependence else 0.0,
        "has_unresolved_primitives": 1.0 if product_ir.unresolved_primitives else 0.0,
        "engine_family_matches_ir": 1.0 if candidate.engine_family in product_ir.candidate_engine_families else 0.0,
        "route_family_matches_ir": 1.0 if route_family in route_families else 0.0,
    }
    feature_map[f"route:{candidate.route}"] = 1.0
    feature_map[f"engine_family:{candidate.engine_family}"] = 1.0
    feature_map[f"route_family:{route_family}"] = 1.0
    feature_map[f"exercise:{product_ir.exercise_style}"] = 1.0
    feature_map[f"state:{product_ir.state_dependence}"] = 1.0
    feature_map[f"model:{product_ir.model_family}"] = 1.0
    for blocker in candidate.blockers:
        feature_map[f"blocker:{blocker}"] = 1.0
    return feature_map


def _pricing_plan_for_case(case: SyntheticProductCase, product_ir) -> PricingPlan:
    """Create a synthetic pricing plan aligned with one offline training case."""
    return PricingPlan(
        method=case.preferred_method,
        method_modules=_method_modules_for(case.preferred_method),
        required_market_data=set(product_ir.required_market_data),
        model_to_build=case.instrument_type,
        reasoning="phase8_synthetic_case",
    )


def _method_modules_for(method: str) -> list[str]:
    """Return representative module evidence for a canonical method label."""
    modules = {
        "analytical": ["trellis.models.black"],
        "rate_tree": ["trellis.models.trees.lattice"],
        "monte_carlo": ["trellis.models.monte_carlo.engine"],
        "qmc": ["trellis.models.qmc"],
        "fft_pricing": ["trellis.models.transforms.fft_pricer"],
    }
    return modules.get(method, [])


def _label_candidate(candidate: PrimitivePlan, product_ir, rank_index: int) -> tuple[str, float]:
    """Assign a supervised label and numeric target to one ranked route candidate."""
    if rank_index == 0 and candidate.blockers and not product_ir.supported:
        return ("block", min(candidate.score, -1.0))
    if rank_index == 0 and not candidate.blockers:
        return ("proceed", candidate.score)
    return ("alternative", candidate.score)
