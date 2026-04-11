"""KnowledgeStore — unified facade over all knowledge tiers.

Hot tier (loaded at init): principles, lesson index, feature taxonomy,
decompositions, failure signatures.

Warm tier (loaded per-task, cached): full lessons, cookbooks, data
contracts, method requirements, benchmarks.

Cold tier (on-demand): run traces, archived lessons.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.knowledge.schema import (
    AppliesWhen,
    BenchmarkCase,
    BenchmarkSuite,
    CookbookEntry,
    DataContractEntry,
    FailureSignature,
    Feature,
    Lesson,
    LessonIndex,
    LessonStatus,
    ModelGrammarEntry,
    MethodRequirements,
    Principle,
    ProductDecomposition,
    RetrievalSpec,
    Severity,
    SimilarProductMatch,
)


_KNOWLEDGE_DIR = Path(__file__).parent

_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}
_BASKET_SUPERSEDED_MARKERS = (
    "trellis.models.processes.correlated_gbm",
    "correlatedgbm as thin adapter",
    "wrap correlatedgbm",
    "imports correlatedgbm",
    "import correlatedgbm",
    "correlated_gbm module path",
    "unapproved correlated_gbm",
)
_BASKET_HELPER_MARKERS = (
    "resolve_basket_semantics",
    "price_ranked_observation_basket_monte_carlo",
    "semantic basket helper",
)

# ---------------------------------------------------------------------------
# Feature expansion
# ---------------------------------------------------------------------------

def expand_features(
    features: list[str],
    taxonomy: dict[str, Feature],
) -> list[str]:
    """Transitively expand a list of feature IDs by following 'implies' links.

    Each feature in the taxonomy may declare that it implies other features.
    For example, 'callable' implies 'early_exercise', which in turn implies
    'backward_induction'.  This function walks those chains and returns the
    full set of features that apply.

    Example: [callable] -> [backward_induction, callable, early_exercise]

    Features not found in the taxonomy are kept as-is (no expansion, no error).
    """
    expanded: set[str] = set()
    stack = list(features)
    while stack:
        fid = stack.pop()
        if fid in expanded:
            continue
        expanded.add(fid)
        feat = taxonomy.get(fid)
        if feat and feat.implies:
            stack.extend(feat.implies)
    return sorted(expanded)


def identify_superseded_basket_lesson_ids(*, root: Path | None = None) -> list[str]:
    """Return basket lessons that should be treated as superseded guidance."""
    knowledge_root = Path(root) if root is not None else _KNOWLEDGE_DIR
    entries_dir = knowledge_root / "lessons" / "entries"
    if not entries_dir.exists():
        return []

    superseded: list[str] = []
    for path in sorted(entries_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if str(data.get("category") or "").strip() != "monte_carlo":
            continue
        status = str(data.get("status") or "").strip().lower()
        if status not in {"promoted", "validated", "archived"}:
            continue

        applies_when = data.get("applies_when") or {}
        if not isinstance(applies_when, dict):
            applies_when = {}
        features = {str(feature).strip() for feature in applies_when.get("features", []) if str(feature).strip()}
        if "ranked_observation" not in features and "multi_asset" not in features:
            continue

        title = str(data.get("title") or "").strip().lower()
        if status == "archived" or title.startswith("superseded -"):
            lesson_id = str(data.get("id") or "").strip()
            if lesson_id:
                superseded.append(lesson_id)
            continue

        text = " ".join(
            str(data.get(field) or "")
            for field in ("title", "symptom", "root_cause", "fix", "validation")
        ).lower()
        normalized_text = text.replace("`", "").replace("-", " ")
        if any(marker in normalized_text for marker in _BASKET_HELPER_MARKERS):
            continue
        has_correlated_gbm = "correlatedgbm" in normalized_text or "correlated_gbm" in normalized_text
        has_direct_guidance = any(
            marker in normalized_text
            for marker in (
                "direct",
                "import",
                "wrap",
                "module path",
                "unapproved",
                "path",
                "delegate",
            )
        )
        if has_correlated_gbm and has_direct_guidance:
            lesson_id = str(data.get("id") or "").strip()
            if lesson_id:
                superseded.append(lesson_id)

    return sorted(set(superseded))


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------

class KnowledgeStore:
    """Singleton-style facade over the four knowledge stores."""

    def __init__(self) -> None:
        """Initialize hot-tier state and empty caches for warm-tier artifacts."""
        # Hot tier
        self._features: dict[str, Feature] = {}
        self._decompositions: dict[str, ProductDecomposition] = {}
        self._principles: list[Principle] = []
        self._lesson_index: list[LessonIndex] = []
        self._failure_signatures: list[FailureSignature] = []

        # Warm tier (lazy, cached)
        self._lessons_cache: dict[str, Lesson] = {}
        self._cookbooks_cache: dict[str, CookbookEntry] | None = None
        self._contracts_cache: dict[str, list[DataContractEntry]] | None = None
        self._requirements_cache: dict[str, MethodRequirements] | None = None
        self._model_grammar_cache: tuple[ModelGrammarEntry, ...] | None = None
        self._benchmarks_cache: dict[str, BenchmarkSuite] | None = None
        self._retrieval_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._basket_superseded_lesson_ids_cache: list[str] | None = None
        self._retrieval_cache_hits = 0
        self._retrieval_cache_misses = 0

        self._load_hot_tier()

    # --------------------------------------------------------------------- #
    # Hot tier loading
    # --------------------------------------------------------------------- #

    def _load_hot_tier(self) -> None:
        """Load the always-resident canonical knowledge files into memory."""
        self._load_features()
        self._load_decompositions()
        self._load_principles()
        self._load_lesson_index()
        self._load_failure_signatures()

    def _load_features(self) -> None:
        """Load the feature taxonomy used for transitive retrieval expansion."""
        path = _KNOWLEDGE_DIR / "canonical" / "features.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or {}
        items = data if isinstance(data, list) else data.get("features", [])
        for f in items:
            self._features[f["id"]] = Feature(
                id=f["id"],
                description=f.get("description", ""),
                implies=tuple(f.get("implies", [])),
                method_hint=f.get("method_hint"),
                market_data=tuple(f.get("market_data", [])),
            )

    def _load_decompositions(self) -> None:
        """Load canonical product decompositions keyed by instrument type."""
        path = _KNOWLEDGE_DIR / "canonical" / "decompositions.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or {}

        # Support both list format and dict format
        if isinstance(data, list):
            items = [(entry["instrument"], entry) for entry in data]
        elif isinstance(data, dict):
            items = list(data.items())
        else:
            return

        for key, entry in items:
            self._decompositions[key] = ProductDecomposition(
                instrument=key,
                features=tuple(entry.get("features", [])),
                method=normalize_method(entry.get("method", "")),
                method_modules=tuple(entry.get("method_modules", [])),
                required_market_data=frozenset(entry.get("required_market_data", [])),
                modeling_requirements=tuple(entry.get("modeling_requirements", [])),
                reasoning=entry.get("reasoning", ""),
                notes=entry.get("notes", ""),
                learned=entry.get("learned", False),
            )

    def _load_principles(self) -> None:
        """Load high-level design principles used across all retrieval tasks."""
        path = _KNOWLEDGE_DIR / "canonical" / "principles.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or []
        for p in data:
            self._principles.append(Principle(
                id=p["id"],
                rule=p["rule"],
                derived_from=tuple(p.get("derived_from", [])),
                category=p.get("category", ""),
            ))

    def _load_lesson_index(self) -> None:
        """Load the lightweight lesson index used for fast relevance scoring."""
        path = _KNOWLEDGE_DIR / "lessons" / "index.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or {}
        for entry in data.get("entries", []):
            aw = entry.get("applies_when", {})
            self._lesson_index.append(LessonIndex(
                id=entry["id"],
                title=entry["title"],
                severity=Severity(entry.get("severity", "low")),
                category=entry.get("category", ""),
                applies_when=AppliesWhen(
                    method=tuple(normalize_method(m) for m in aw.get("method", [])),
                    features=tuple(aw.get("features", [])),
                    instrument=tuple(aw.get("instrument", [])),
                    error_signature=aw.get("error_signature"),
                ),
                status=LessonStatus(entry.get("status", "promoted")),
                supersedes=tuple(entry.get("supersedes", [])),
            ))

    def _load_failure_signatures(self) -> None:
        """Load regex-based failure signatures used to interpret execution errors."""
        path = _KNOWLEDGE_DIR / "canonical" / "failure_signatures.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or []
        for s in data:
            self._failure_signatures.append(FailureSignature(
                pattern=s.get("pattern", ""),
                magnitude=s.get("magnitude", "unknown"),
                category=s.get("category", "unknown"),
                probable_causes=tuple(s.get("probable_causes", [])),
                features=tuple(s.get("features", [])),
                diagnostic_hint=s.get("diagnostic_hint", ""),
            ))

    def _basket_superseded_lesson_ids(self) -> list[str]:
        """Return the basket lessons that should be suppressed as stale guidance."""
        if self._basket_superseded_lesson_ids_cache is None:
            self._basket_superseded_lesson_ids_cache = identify_superseded_basket_lesson_ids(
                root=_KNOWLEDGE_DIR,
            )
        return list(self._basket_superseded_lesson_ids_cache)

    # --------------------------------------------------------------------- #
    # Main retrieval
    # --------------------------------------------------------------------- #

    def retrieve_for_task(self, spec: RetrievalSpec) -> dict[str, Any]:
        """Main entry point: get all relevant knowledge for a task.

        Features are the primary retrieval axis.  Lessons, benchmarks, and
        failure signatures are matched via feature union.
        """
        cache_key = self._retrieval_cache_key(spec)
        cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            self._retrieval_cache_hits += 1
            return cached

        self._retrieval_cache_misses += 1
        expanded = expand_features(spec.features, self._features)
        method = normalize_method(spec.method) if spec.method else None
        basket_superseded_ids = set()
        if spec.instrument == "basket_path_payoff" or (
            "ranked_observation" in expanded and "multi_asset" in expanded
        ):
            basket_superseded_ids = set(self._basket_superseded_lesson_ids())

        result: dict[str, Any] = {
            "principles": list(self._principles),
            "decomposition": self._decompositions.get(spec.instrument) if spec.instrument else None,
            "product_ir": getattr(spec, "product_ir", None),
            "unresolved_primitives": tuple(spec.unresolved_primitives),
            "lessons": self._query_lessons(
                expanded,
                spec,
                spec.max_lessons,
                basket_superseded_ids=basket_superseded_ids,
            ),
            "cookbook": self._load_cookbook(method) if method else None,
            "data_contracts": self._load_contracts(method) if method else [],
            "method_requirements": self._load_requirements(method) if method else None,
            "model_grammar": self._load_model_grammar(spec),
            "similar_products": [],
            "borrowed_lessons": [],
        }

        if spec.include_benchmarks:
            result["benchmarks"] = self._find_benchmarks(expanded, method or "")

        if spec.error_signatures:
            result["matched_signatures"] = self._match_signatures(spec.error_signatures)

        if self._should_surface_similar_products(spec, result):
            similar_products = self.find_similar_products(spec)
            result["similar_products"] = similar_products
            if similar_products:
                result["borrowed_lessons"] = self._borrow_lessons_from_similar_products(
                    similar_products,
                    spec,
                    primary_lessons=result["lessons"],
                )

        self._retrieval_cache[cache_key] = result
        return result

    def find_similar_products(
        self,
        spec: RetrievalSpec,
        *,
        limit: int = 3,
        min_score: float = 0.2,
    ) -> list[SimilarProductMatch]:
        """Return deterministic nearest known products for a sparse retrieval spec."""
        query_features = set(expand_features(list(spec.features), self._features))
        if not query_features:
            return []

        method = normalize_method(spec.method) if spec.method else ""
        matches: list[SimilarProductMatch] = []
        for instrument, decomposition in self._decompositions.items():
            if spec.instrument and instrument == spec.instrument:
                continue

            candidate_features = set(
                expand_features(list(decomposition.features), self._features)
            )
            shared_features = sorted(query_features & candidate_features)
            if not shared_features:
                continue

            union = query_features | candidate_features
            jaccard = len(shared_features) / max(len(union), 1)
            method_bonus = 0.15 if method and normalize_method(decomposition.method) == method else 0.0
            score = round(min(1.0, jaccard + method_bonus), 2)
            if score < min_score:
                continue

            matches.append(
                SimilarProductMatch(
                    instrument=instrument,
                    method=normalize_method(decomposition.method),
                    score=score,
                    shared_features=tuple(shared_features),
                    query_only_features=tuple(sorted(query_features - candidate_features)),
                    candidate_only_features=tuple(sorted(candidate_features - query_features)),
                    promoted_routes=self._promoted_routes_for(
                        instrument=instrument,
                        method=decomposition.method,
                    ),
                )
            )

        matches.sort(
            key=lambda match: (
                -match.score,
                -len(match.shared_features),
                match.instrument,
            )
        )
        return matches[: max(limit, 0)]

    def retrieval_cache_stats(self) -> dict[str, int]:
        """Return lightweight retrieval-cache statistics for testing and traces."""
        return {
            "hits": self._retrieval_cache_hits,
            "misses": self._retrieval_cache_misses,
            "size": len(self._retrieval_cache),
        }

    def clear_runtime_caches(self) -> None:
        """Clear warm/runtime caches without reloading the hot tier from disk."""
        self._lessons_cache.clear()
        self._cookbooks_cache = None
        self._contracts_cache = None
        self._requirements_cache = None
        self._model_grammar_cache = None
        self._benchmarks_cache = None
        self._retrieval_cache.clear()
        self._basket_superseded_lesson_ids_cache = None
        self._retrieval_cache_hits = 0
        self._retrieval_cache_misses = 0

    # --------------------------------------------------------------------- #
    # Lesson retrieval (feature-based union)
    # --------------------------------------------------------------------- #

    def _query_lessons(
        self,
        expanded_features: list[str],
        spec: RetrievalSpec,
        max_n: int,
        *,
        basket_superseded_ids: set[str] | None = None,
    ) -> list[Lesson]:
        """Score all indexed lessons against the current task and load the top matches.

        Scoring uses feature overlap, method match, instrument hint, and
        severity to rank lessons.  Only promoted or validated lessons are
        considered.  Lessons that are superseded by any other active lesson
        are suppressed before ranking.  The top ``max_n`` entries are loaded
        from disk (hydrated) and returned.
        """
        basket_superseded_ids = basket_superseded_ids or set()
        active_entries = [
            idx
            for idx in self._lesson_index
            if idx.status in (LessonStatus.PROMOTED, LessonStatus.VALIDATED)
            and idx.id not in basket_superseded_ids
        ]
        superseded_ids = {
            superseded_id
            for idx in active_entries
            for superseded_id in idx.supersedes
        }

        scored: list[tuple[float, LessonIndex]] = []
        for idx in active_entries:
            if idx.id in superseded_ids:
                continue
            score = self._relevance_score(idx, expanded_features, spec)
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda item: (-item[0], _SEVERITY_ORDER.get(item[1].severity, 3)))

        overfetch_n = max(max_n * 2, max_n + 1)
        candidate_entries = [idx for _, idx in scored[:overfetch_n]]
        lessons: list[Lesson] = []

        for idx in candidate_entries:
            lesson = self._load_lesson(idx.id)
            if lesson is None:
                continue
            lessons.append(lesson)
            if len(lessons) >= max_n:
                return lessons

        for _, idx in scored[overfetch_n:]:
            lesson = self._load_lesson(idx.id)
            if lesson is None:
                continue
            lessons.append(lesson)
            if len(lessons) >= max_n:
                break
        return lessons

    def _should_surface_similar_products(
        self,
        spec: RetrievalSpec,
        result: dict[str, Any],
    ) -> bool:
        """Return whether similar-product suggestions should be added."""
        if not spec.features:
            return False
        if result.get("decomposition") is None:
            return True
        if len(result.get("lessons", []) or []) < min(3, max(spec.max_lessons, 1)):
            return True
        if spec.instrument and not self._promoted_routes_for(instrument=spec.instrument, method=spec.method):
            return True
        return False

    def _borrow_lessons_from_similar_products(
        self,
        matches: list[SimilarProductMatch],
        spec: RetrievalSpec,
        *,
        primary_lessons: list[Lesson],
        max_borrowed: int = 2,
    ) -> list[Lesson]:
        """Borrow a few extra lessons from the nearest known products."""
        borrowed: list[Lesson] = []
        seen_ids = {lesson.id for lesson in primary_lessons}
        for match in matches:
            candidate = self._decompositions.get(match.instrument)
            if candidate is None:
                continue

            candidate_spec = RetrievalSpec(
                method=match.method,
                features=list(candidate.features),
                instrument=match.instrument,
                exercise_style=spec.exercise_style,
                state_dependence=spec.state_dependence,
                schedule_dependence=spec.schedule_dependence,
                model_family=spec.model_family,
                candidate_engine_families=spec.candidate_engine_families,
                max_lessons=2,
            )
            candidate_lessons = self._query_lessons(
                expand_features(list(candidate.features), self._features),
                candidate_spec,
                2,
            )
            for lesson in candidate_lessons:
                if lesson.id in seen_ids:
                    continue
                borrowed.append(lesson)
                seen_ids.add(lesson.id)
                break
            if len(borrowed) >= max_borrowed:
                break
        return borrowed

    @staticmethod
    def _promoted_routes_for(*, instrument: str | None, method: str | None) -> tuple[str, ...]:
        """Return promoted route ids for one canonical decomposition when available."""
        if not instrument or not method:
            return ()
        try:
            from trellis.agent.knowledge.schema import ProductIR
            from trellis.agent.route_registry import (
                load_route_registry,
                match_candidate_routes,
            )

            registry = load_route_registry()
            promoted = match_candidate_routes(
                registry,
                method,
                ProductIR(instrument=instrument, payoff_family=instrument),
                promoted_only=True,
            )
            return tuple(route.id for route in promoted)
        except Exception:
            return ()

    @staticmethod
    def _relevance_score(
        idx: LessonIndex,
        features: list[str],
        spec: RetrievalSpec,
    ) -> float:
        """Compute a heuristic relevance score for one indexed lesson.

        The score sums weighted signals: feature overlap with the current
        task (strongest signal, +2 per overlapping feature), method match
        (+1 exact, +0.5 wildcard), instrument hint (+0.75), exercise/state/
        model bonuses from ProductIR fields, and severity bonus (+1 critical,
        +0.5 high).  A score of 0 means no relevance.
        """
        score = 0.0
        aw = idx.applies_when
        method = normalize_method(spec.method) if spec.method else None

        # Feature overlap — primary signal
        if aw.features:
            overlap = len(set(aw.features) & set(features))
            score += overlap * 2.0

        # Method match
        if aw.method and method:
            if method in aw.method:
                score += 5.0
            elif "any" in aw.method:
                score += 0.25

        # Instrument hint
        if aw.instrument and spec.instrument:
            if spec.instrument in aw.instrument:
                score += 0.75

        # Semantic bonuses from ProductIR-derived retrieval spec.
        score += KnowledgeStore._semantic_bonus(idx, spec)

        # Severity bonus
        severity_bonus = {Severity.CRITICAL: 1.0, Severity.HIGH: 0.5}
        score += severity_bonus.get(idx.severity, 0.0)

        return score

    @staticmethod
    def _semantic_bonus(idx: LessonIndex, spec: RetrievalSpec) -> float:
        """Add semantic boosts derived from exercise style, state dependence, and model family."""
        score = 0.0
        feature_set = set(idx.applies_when.features)

        if spec.exercise_style in {"american", "bermudan", "issuer_call", "holder_put"}:
            if "early_exercise" in feature_set:
                score += 1.5
            if spec.exercise_style == "issuer_call" and "callable" in feature_set:
                score += 0.5
            if spec.exercise_style == "holder_put" and "puttable" in feature_set:
                score += 0.5

        if spec.state_dependence == "path_dependent" and "path_dependent" in feature_set:
            score += 1.0
        if spec.state_dependence == "schedule_dependent" and "backward_induction" in feature_set:
            score += 0.75

        if spec.model_family == "interest_rate" and "mean_reversion" in feature_set:
            score += 1.0
        if spec.model_family == "stochastic_volatility" and "stochastic_vol" in feature_set:
            score += 1.0

        return score

    def _load_lesson(self, lesson_id: str) -> Lesson | None:
        """Load and cache one full lesson entry by identifier."""
        if lesson_id in self._lessons_cache:
            return self._lessons_cache[lesson_id]

        path = _KNOWLEDGE_DIR / "lessons" / "entries" / f"{lesson_id}.yaml"
        if not path.exists():
            return None

        data = yaml.safe_load(path.read_text())
        if not data:
            return None

        aw = data.get("applies_when", {})
        lesson = Lesson(
            id=data["id"],
            title=data["title"],
            severity=Severity(data.get("severity", "low")),
            category=data.get("category", ""),
            applies_when=AppliesWhen(
                method=tuple(normalize_method(m) for m in aw.get("method", [])),
                features=tuple(aw.get("features", [])),
                instrument=tuple(aw.get("instrument", [])),
                error_signature=aw.get("error_signature"),
            ),
            symptom=data.get("symptom", ""),
            root_cause=data.get("root_cause", data.get("explanation", "")),
            fix=data.get("fix", ""),
            validation=data.get("validation", ""),
            confidence=data.get("confidence", 1.0),
            status=LessonStatus(data.get("status", "promoted")),
            version=data.get("version", ""),
            created=data.get("created", ""),
            source_trace=data.get("source_trace"),
            supersedes=tuple(data.get("supersedes", [])),
            derived_principle=data.get("derived_principle"),
        )
        self._lessons_cache[lesson_id] = lesson
        return lesson

    def list_lessons(
        self,
        *,
        lesson_ids: tuple[str, ...] | None = None,
        statuses: tuple[LessonStatus, ...] = (
            LessonStatus.PROMOTED,
            LessonStatus.VALIDATED,
        ),
    ) -> tuple[Lesson, ...]:
        """Return hydrated lessons filtered by id and lifecycle state.

        The ordering is stable for downstream deterministic consumers:
        promoted lessons first, then validated lessons, then by severity and id.
        """
        allowed_statuses = {status for status in statuses}
        selected_ids = {
            str(lesson_id).strip()
            for lesson_id in (lesson_ids or ())
            if str(lesson_id).strip()
        }
        status_priority = {
            LessonStatus.PROMOTED: 0,
            LessonStatus.VALIDATED: 1,
            LessonStatus.CANDIDATE: 2,
            LessonStatus.ARCHIVED: 3,
        }
        eligible = [
            index
            for index in self._lesson_index
            if index.status in allowed_statuses
            and (not selected_ids or index.id in selected_ids)
        ]
        ordered = sorted(
            eligible,
            key=lambda index: (
                status_priority.get(index.status, 99),
                _SEVERITY_ORDER.get(index.severity, 99),
                index.id,
            ),
        )
        hydrated: list[Lesson] = []
        for index in ordered:
            lesson = self._load_lesson(index.id)
            if lesson is None or lesson.status not in allowed_statuses:
                continue
            hydrated.append(lesson)
        return tuple(hydrated)

    # --------------------------------------------------------------------- #
    # Warm tier: cookbooks
    # --------------------------------------------------------------------- #

    def _load_cookbook(self, method: str) -> CookbookEntry | None:
        """Load and cache cookbook templates keyed by normalized method name."""
        if self._cookbooks_cache is None:
            self._cookbooks_cache = {}
            path = _KNOWLEDGE_DIR / "canonical" / "cookbooks.yaml"
            if path.exists():
                data = yaml.safe_load(path.read_text()) or {}
                for key, entry in data.items():
                    canonical = normalize_method(key)
                    self._cookbooks_cache[canonical] = CookbookEntry(
                        method=canonical,
                        template=entry.get("template", ""),
                        description=entry.get("description", ""),
                        applicable_instruments=tuple(
                            entry.get("applicable_instruments", [])
                        ),
                        version=entry.get("version", ""),
                    )
        return self._cookbooks_cache.get(normalize_method(method))

    # --------------------------------------------------------------------- #
    # Warm tier: data contracts
    # --------------------------------------------------------------------- #

    def _load_contracts(self, method: str) -> list[DataContractEntry]:
        """Load and cache method-specific data contracts and conversion guidance."""
        if self._contracts_cache is None:
            self._contracts_cache = {}
            path = _KNOWLEDGE_DIR / "canonical" / "data_contracts.yaml"
            if path.exists():
                data = yaml.safe_load(path.read_text()) or []
                for entry in data:
                    m = normalize_method(entry.get("method", ""))
                    if m not in self._contracts_cache:
                        self._contracts_cache[m] = []
                    self._contracts_cache[m].append(DataContractEntry(
                        name=entry.get("name", ""),
                        method=m,
                        source=entry.get("source", ""),
                        convention=entry.get("convention", ""),
                        typical_range=entry.get("typical_range", ""),
                        model_expects=entry.get("model_expects", ""),
                        conversion=entry.get("conversion", ""),
                        model_range=entry.get("model_range", ""),
                        warning=entry.get("warning", ""),
                    ))
        return self._contracts_cache.get(normalize_method(method), [])

    # --------------------------------------------------------------------- #
    # Warm tier: method requirements
    # --------------------------------------------------------------------- #

    def _load_requirements(self, method: str) -> MethodRequirements | None:
        """Load and cache structural requirements for a pricing method family.

        The YAML structure supports two formats:
        - flat list: ``method: [req1, req2, ...]``
        - contract-keyed dict: ``method: {common: [...], contract_a: [...], ...}``

        The contract-keyed format (used by the analytical family) is flattened
        into a single requirements tuple so downstream code sees no structural
        change.  The contract labels are preserved in the requirement text.
        """
        if self._requirements_cache is None:
            self._requirements_cache = {}
            path = _KNOWLEDGE_DIR / "canonical" / "method_requirements.yaml"
            if path.exists():
                data = yaml.safe_load(path.read_text()) or {}
                for m, reqs in data.items():
                    canonical = normalize_method(m)
                    if isinstance(reqs, list):
                        flat = tuple(reqs)
                    elif isinstance(reqs, dict):
                        # Contract-keyed: flatten common first, then each contract
                        parts: list[str] = []
                        common = reqs.get("common")
                        if isinstance(common, list):
                            parts.extend(common)
                        for key, sub_reqs in reqs.items():
                            if key == "common" or not isinstance(sub_reqs, list):
                                continue
                            parts.extend(sub_reqs)
                        flat = tuple(parts)
                    else:
                        flat = ()
                    self._requirements_cache[canonical] = MethodRequirements(
                        method=canonical,
                        requirements=flat,
                    )
        return self._requirements_cache.get(normalize_method(method))

    @staticmethod
    def _as_token_tuple(values: object) -> tuple[str, ...]:
        """Normalize one YAML sequence-like field into a tuple of lowercase tokens."""
        if values is None:
            return ()
        if isinstance(values, str):
            candidates = (values,)
        elif isinstance(values, (list, tuple, set, frozenset)):
            candidates = tuple(values)
        else:
            return ()
        normalized: list[str] = []
        for value in candidates:
            token = str(value or "").strip().lower().replace(" ", "_")
            if token:
                normalized.append(token)
        return tuple(normalized)

    def _load_model_grammar_catalog(self) -> tuple[ModelGrammarEntry, ...]:
        """Load and cache canonical calibration-layer model-grammar entries."""
        if self._model_grammar_cache is not None:
            return self._model_grammar_cache

        path = _KNOWLEDGE_DIR / "canonical" / "model_grammar.yaml"
        records: list[ModelGrammarEntry] = []
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = data.get("entries", [])
            else:
                entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_id = str(entry.get("id") or "").strip()
                if not entry_id:
                    continue
                records.append(
                    ModelGrammarEntry(
                        id=entry_id,
                        title=str(entry.get("title") or "").strip(),
                        methods=self._as_token_tuple(entry.get("methods")),
                        instruments=self._as_token_tuple(entry.get("instruments")),
                        model_families=self._as_token_tuple(entry.get("model_families")),
                        engine_families=self._as_token_tuple(entry.get("engine_families")),
                        features=self._as_token_tuple(entry.get("features")),
                        model_name=str(entry.get("model_name") or "").strip(),
                        state_semantics=self._as_token_tuple(entry.get("state_semantics")),
                        quote_families=self._as_token_tuple(entry.get("quote_families")),
                        calibration_workflows=tuple(
                            str(item).strip()
                            for item in entry.get("calibration_workflows", [])
                            if str(item).strip()
                        ),
                        runtime_materialization_kind=str(
                            entry.get("runtime_materialization_kind") or ""
                        ).strip(),
                        runtime_materialization_targets=self._as_token_tuple(
                            entry.get("runtime_materialization_targets")
                        ),
                        rates_curve_roles=self._as_token_tuple(entry.get("rates_curve_roles")),
                        required_market_data=self._as_token_tuple(entry.get("required_market_data")),
                        authority_surfaces=tuple(
                            str(item).strip()
                            for item in entry.get("authority_surfaces", [])
                            if str(item).strip()
                        ),
                        deferred_scope=tuple(
                            str(item).strip()
                            for item in entry.get("deferred_scope", [])
                            if str(item).strip()
                        ),
                        notes=str(entry.get("notes") or "").strip(),
                    )
                )
        self._model_grammar_cache = tuple(records)
        return self._model_grammar_cache

    def _load_model_grammar(self, spec: RetrievalSpec) -> list[ModelGrammarEntry]:
        """Return model-grammar entries relevant to one retrieval spec."""
        catalog = self._load_model_grammar_catalog()
        if not catalog:
            return []

        method = normalize_method(spec.method) if spec.method else ""
        instrument = str(spec.instrument or "").strip().lower()
        model_family = str(spec.model_family or "").strip().lower()
        feature_set = {
            str(feature).strip().lower().replace(" ", "_")
            for feature in (spec.features or [])
            if str(feature).strip()
        }
        candidate_engine_families = {
            str(family).strip().lower().replace(" ", "_")
            for family in spec.candidate_engine_families
            if str(family).strip()
        }

        ranked: list[tuple[float, ModelGrammarEntry]] = []
        for entry in catalog:
            score = 0.0
            if method and method in entry.methods:
                score += 2.0
            if instrument and instrument in entry.instruments:
                score += 3.0
            if model_family and model_family in entry.model_families:
                score += 2.5
            if candidate_engine_families and set(entry.engine_families) & candidate_engine_families:
                score += 1.0
            if feature_set and entry.features:
                score += 0.25 * len(feature_set & set(entry.features))
            if score <= 0.0:
                continue
            ranked.append((score, entry))

        ranked.sort(key=lambda item: (-item[0], item[1].id))
        return [entry for _, entry in ranked]

    # --------------------------------------------------------------------- #
    # Warm tier: benchmarks
    # --------------------------------------------------------------------- #

    def _find_benchmarks(
        self,
        features: list[str],
        method: str,
    ) -> list[BenchmarkSuite]:
        """Return benchmark suites that match the feature set or requested method."""
        if self._benchmarks_cache is None:
            self._benchmarks_cache = {}
            reg_path = _KNOWLEDGE_DIR / "benchmarks" / "registry.yaml"
            if reg_path.exists():
                registry = yaml.safe_load(reg_path.read_text()) or []
                for entry in registry:
                    suite_path = _KNOWLEDGE_DIR / "benchmarks" / entry["file"]
                    if suite_path.exists():
                        suite_data = yaml.safe_load(suite_path.read_text()) or {}
                        cases = tuple(
                            BenchmarkCase(
                                params=c.get("params", {}),
                                reference=c["reference"],
                                tolerance_pct=c.get("tolerance_pct", 2.0),
                                tolerance_abs=c.get("tolerance_abs", 0.0),
                                source=c.get("source", ""),
                            )
                            for c in suite_data.get("cases", [])
                        )
                        self._benchmarks_cache[entry["id"]] = BenchmarkSuite(
                            id=entry["id"],
                            title=entry["title"],
                            features=tuple(entry.get("features", [])),
                            method=normalize_method(entry.get("method", "")),
                            source_citation=entry.get("source_citation", ""),
                            cases=cases,
                            setup=suite_data.get("setup", {}),
                        )

        results: list[BenchmarkSuite] = []
        feature_set = set(features)
        for suite in self._benchmarks_cache.values():
            if set(suite.features) & feature_set:
                results.append(suite)
            elif suite.method == method:
                results.append(suite)
        return results

    # --------------------------------------------------------------------- #
    # Failure signature matching
    # --------------------------------------------------------------------- #

    def _match_signatures(
        self,
        error_messages: list[str],
    ) -> list[FailureSignature]:
        """Match raw execution errors against known regex failure signatures."""
        matched: list[FailureSignature] = []
        for sig in self._failure_signatures:
            for msg in error_messages:
                try:
                    if re.search(sig.pattern, msg, re.IGNORECASE):
                        matched.append(sig)
                        break
                except re.error:
                    continue
        return matched

    # --------------------------------------------------------------------- #
    # Decomposition persistence
    # --------------------------------------------------------------------- #

    def save_decomposition(self, decomp: ProductDecomposition) -> None:
        """Cache a successful decomposition and persist to YAML."""
        self._decompositions[decomp.instrument] = decomp

        path = _KNOWLEDGE_DIR / "canonical" / "decompositions.yaml"
        data: dict = {}
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}

        data[decomp.instrument] = {
            "features": list(decomp.features),
            "method": normalize_method(decomp.method),
            "method_modules": list(decomp.method_modules),
            "required_market_data": sorted(decomp.required_market_data),
            "modeling_requirements": list(decomp.modeling_requirements),
            "reasoning": decomp.reasoning,
            "notes": decomp.notes,
            "learned": decomp.learned,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)
        self.clear_runtime_caches()

    # --------------------------------------------------------------------- #
    # Audit helpers
    # --------------------------------------------------------------------- #

    def compute_knowledge_hash(self) -> str:
        """SHA-256 fingerprint of canonical knowledge state.

        Hashes the contents of all canonical YAML files plus the promoted
        lesson count to produce a short fingerprint. Changes to any canonical
        artifact or lesson promotion will change the hash.
        """
        import hashlib

        h = hashlib.sha256()
        canonical_dir = _KNOWLEDGE_DIR / "canonical"
        if canonical_dir.exists():
            for path in sorted(canonical_dir.glob("*.yaml")):
                h.update(path.read_bytes())
        h.update(str(len(self._lesson_index)).encode())
        return h.hexdigest()[:16]

    # --------------------------------------------------------------------- #
    # Reload
    # --------------------------------------------------------------------- #

    def reload(self) -> None:
        """Reload all tiers (after migration or lesson capture)."""
        self.clear_runtime_caches()
        self._features.clear()
        self._decompositions.clear()
        self._principles.clear()
        self._lesson_index.clear()
        self._failure_signatures.clear()
        self._load_hot_tier()

    def _retrieval_cache_key(self, spec: RetrievalSpec) -> tuple[Any, ...]:
        """Build a stable cache key for a retrieval spec."""
        return (
            normalize_method(spec.method) if spec.method else None,
            tuple(spec.features),
            spec.instrument,
            spec.exercise_style,
            spec.state_dependence,
            spec.schedule_dependence,
            spec.model_family,
            tuple(spec.candidate_engine_families),
            tuple(spec.unresolved_primitives),
            tuple(spec.error_signatures),
            spec.max_lessons,
            spec.include_benchmarks,
        )
