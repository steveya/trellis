"""Post-build reflection — the autonomous learning loop.

After every build (success or failure), this module:
1. Attributes success/failure to retrieved lessons
2. Captures new lessons with structured feature tags
3. Detects and records knowledge gaps
4. Enriches missing cookbooks from successful code
5. Saves learned decompositions
6. Triggers periodic distillation
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from trellis.agent.knowledge.schema import ProductDecomposition
from trellis.agent.knowledge.gap_check import GapReport


_KNOWLEDGE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Structured reflection prompt
# ---------------------------------------------------------------------------

_REFLECT_PROMPT = """\
You just {outcome} building a pricing model for: {description}

Method: {method}
Features: {features}
Knowledge confidence before build: {confidence:.0%}
Knowledge gaps: {gaps}

{failure_section}

Retrieved lessons that were available:
{retrieved_lessons}

## Your tasks

Return JSON with these fields:

{{
    "lesson": {{
        "category": "volatility|calibration|backward_induction|finite_differences|monte_carlo|market_data|numerical|convention",
        "title": "Short title (max 10 words)",
        "severity": "critical|high|medium|low",
        "symptom": "Machine-checkable sign that this issue occurred (1 sentence)",
        "root_cause": "Why it happens (1-2 sentences)",
        "fix": "How to fix it (1-2 sentences)",
        "features": {feature_options},
        "method": "{method}"
    }},

    "knowledge_gaps": [
        "description of what knowledge was MISSING that could have prevented failures"
    ],

    "cookbook_extract": "If the method '{method}' had no cookbook template, extract the \
reusable evaluate() pattern from the successful code below. Otherwise null.",

    "route_discovery": {{
        "route_id": "descriptive_snake_case_id",
        "engine_family": "analytical|monte_carlo|rate_tree|pde_solver|fft_pricing|copula|waterfall",
        "match_methods": ["{method}"],
        "match_instruments": ["{instrument}"],
        "primitives_used": [
            {{"module": "trellis.models.xxx", "symbol": "YYY", "role": "pricing_kernel|route_helper|state_process|path_simulation|..."}}
        ],
        "market_data_accessed": ["discount_curve", "black_vol_surface"],
        "parameters_extracted": ["maturity", "strike"],
        "rationale": "Why this route pattern works for this product class"
    }}
}}

Rules:
- "lesson" should be null if no failure was resolved (first-attempt success)
- "features" MUST be from the provided feature list, not invented
- "knowledge_gaps" should list what was missing from the knowledge base
- "cookbook_extract" should be a reusable code template with INSTRUMENT-SPECIFIC markers, or null
- "route_discovery" should be non-null ONLY if the build succeeded AND no known route was used (ad-hoc generation). Otherwise null.

{code_section}

Return ONLY JSON, no markdown fences."""


def reflect_on_build(
    description: str,
    decomposition: ProductDecomposition,
    gap_report: GapReport,
    retrieved_lesson_ids: list[str],
    success: bool,
    failures: list[str],
    code: str,
    attempt: int,
    agent_observations: list[dict[str, Any]] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Post-build reflection — the autonomous learning loop.

    Returns a summary dict of actions taken.
    """
    actions: dict[str, Any] = {
        "success": success,
        "attempt": attempt,
        "lessons_attributed": 0,
        "routes_attributed": 0,
        "lesson_captured": None,
        "lesson_contract": None,
        "lesson_promotion_outcome": None,
        "route_discovered": None,
        "route_discovery_outcome": None,
        "gaps_identified": [],
        "cookbook_enriched": False,
        "cookbook_candidate_saved": None,
        "knowledge_trace_saved": None,
        "knowledge_gap_log_saved": None,
        "decomposition_saved": False,
        "distill_run": False,
        "agent_observation_count": len(agent_observations or []),
        "scorer_retrained": False,
        "validators_promoted": False,
    }
    knowledge_writes_allowed = _knowledge_store_mutations_allowed()

    try:
        # 1. Attribution — boost retrieved lessons and routes on success
        if success and knowledge_writes_allowed:
            actions["lessons_attributed"] = _attribute_success(
                retrieved_lesson_ids
            )
            actions["routes_attributed"] = _attribute_route_success(
                decomposition,
            )

        # 2. Save learned decomposition on success
        if success and decomposition.learned and knowledge_writes_allowed:
            _save_decomposition(decomposition)
            actions["decomposition_saved"] = True

        # 3. LLM reflection — capture lesson + identify gaps + extract cookbook
        if failures or not gap_report.has_cookbook:
            reflection = _llm_reflect(
                description, decomposition, gap_report,
                retrieved_lesson_ids, success, failures, code,
                agent_observations or [], model,
            )

            if reflection:
                # Capture lesson if one was identified
                lesson_data = reflection.get("lesson")
                if lesson_data and isinstance(lesson_data, dict) and knowledge_writes_allowed:
                    lid, lesson_contract, lesson_outcome = _capture_structured_lesson(
                        lesson_data, decomposition, success, attempt,
                    )
                    actions["lesson_captured"] = lid
                    actions["lesson_contract"] = lesson_contract
                    actions["lesson_promotion_outcome"] = lesson_outcome

                # Record knowledge gaps
                gaps = reflection.get("knowledge_gaps", [])
                if gaps:
                    actions["gaps_identified"] = gaps
                    if knowledge_writes_allowed:
                        actions["knowledge_gap_log_saved"] = _record_gaps(
                            gaps,
                            decomposition,
                            gap_report,
                        )

                # Enrich cookbook if missing
                cookbook_extract = reflection.get("cookbook_extract")
                if (
                    cookbook_extract
                    and not gap_report.has_cookbook
                    and success
                    and knowledge_writes_allowed
                ):
                    _enrich_cookbook(decomposition.method, cookbook_extract)
                    actions["cookbook_enriched"] = True

            if (
                success
                and not gap_report.has_cookbook
                and not actions["cookbook_enriched"]
                and knowledge_writes_allowed
            ):
                actions["cookbook_candidate_saved"] = _record_cookbook_candidate(
                    decomposition.method,
                    description,
                    code,
                )

            # Capture discovered route if build was ad-hoc
            if reflection:
                route_data = reflection.get("route_discovery")
                if (
                    route_data
                    and isinstance(route_data, dict)
                    and success
                    and knowledge_writes_allowed
                ):
                    rid, outcome = _capture_discovered_route(
                        route_data, decomposition, attempt,
                    )
                    actions["route_discovered"] = rid
                    actions["route_discovery_outcome"] = outcome

        # 4. Record trace
        if knowledge_writes_allowed:
            actions["knowledge_trace_saved"] = _record_full_trace(
                description, decomposition, gap_report, retrieved_lesson_ids,
                success, failures, code, attempt, actions, agent_observations or [],
            )
            if actions.get("lesson_captured") and actions["knowledge_trace_saved"]:
                _attach_source_trace(
                    actions["lesson_captured"],
                    actions["knowledge_trace_saved"],
                )

        # 5. Maybe distill
        if knowledge_writes_allowed and _should_distill():
            from trellis.agent.knowledge.promotion import distill
            distill()
            actions["distill_run"] = True

        # 6. Maybe retrain route scorer
        if knowledge_writes_allowed:
            actions["scorer_retrained"] = _maybe_retrain_scorer()

        # 7. Maybe promote semantic validators
        if knowledge_writes_allowed:
            actions["validators_promoted"] = _maybe_promote_validators()

    except Exception:
        pass  # Reflection must never block the build

    return actions


def _knowledge_store_mutations_allowed() -> bool:
    """Return False when the current runtime should not mutate the knowledge store."""
    try:
        from trellis.agent.cassette import current_llm_cassette_context
    except Exception:
        return True
    return current_llm_cassette_context() is None


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def _attribute_success(lesson_ids: list[str]) -> int:
    """Boost confidence of lessons that were present during a successful build."""
    from trellis.agent.knowledge.promotion import boost_confidence
    count = 0
    for lid in lesson_ids:
        result = boost_confidence(lid, 0.1)
        if result is not None:
            count += 1
    return count


def _attribute_route_success(decomposition: ProductDecomposition) -> int:
    """Boost confidence of discovered routes that match this product's method+instrument."""
    try:
        from trellis.agent.route_registry import (
            clear_route_registry_cache,
            load_route_registry,
            match_candidate_routes,
        )
        from trellis.agent.knowledge.schema import ProductIR
        from trellis.agent.semantic_tokens import internal_payoff_family_for_surface

        registry = load_route_registry(include_discovered=True)
        minimal_ir = ProductIR(
            instrument=decomposition.instrument,
            payoff_family=internal_payoff_family_for_surface(
                instrument=decomposition.instrument,
            ),
        )
        # Only boost discovered routes (not canonical)
        all_matches = match_candidate_routes(
            registry, decomposition.method, minimal_ir, promoted_only=False,
        )
        count = 0
        for route in all_matches:
            if route.discovered_from is not None and route.status in ("candidate", "validated"):
                _boost_route_confidence(route.id, delta=0.1)
                count += 1
        return count
    except Exception:
        return 0


def _boost_route_confidence(route_id: str, delta: float = 0.1) -> None:
    """Increment a discovered route's confidence and successful_builds count."""
    import yaml
    entries_dir = _KNOWLEDGE_DIR / "routes" / "entries"
    route_path = entries_dir / f"{route_id}.yaml"
    if not route_path.exists():
        return
    try:
        data = yaml.safe_load(route_path.read_text()) or {}
        old_confidence = float(data.get("confidence", 0.5))
        new_confidence = round(min(1.0, old_confidence + delta), 2)
        data["confidence"] = new_confidence
        data["successful_builds"] = int(data.get("successful_builds", 0)) + 1

        # Auto-validate at 0.6
        if new_confidence >= 0.6 and data.get("status") == "candidate":
            data["status"] = "validated"
        # Auto-promote at 0.8
        if new_confidence >= 0.8 and data.get("status") in ("candidate", "validated"):
            data["status"] = "promoted"

        with open(route_path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

        from trellis.agent.route_registry import clear_route_registry_cache
        clear_route_registry_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Route discovery
# ---------------------------------------------------------------------------

_ROUTES_ENTRIES_DIR = _KNOWLEDGE_DIR / "routes" / "entries"


def _capture_discovered_route(
    route_data: dict,
    decomposition: ProductDecomposition,
    attempt: int,
) -> tuple[str | None, str]:
    """Capture a discovered route from LLM reflection output.

    Returns (route_id, outcome) where outcome is one of:
    "captured", "existing_boosted", "equivalent_boosted", "invalid", "error".
    """
    import yaml

    try:
        route_id = route_data.get("route_id", "").strip().lower().replace(" ", "_").replace("-", "_")
        if not route_id or len(route_id) < 3:
            return None, "invalid"

        from trellis.agent.route_registry import load_route_registry, clear_route_registry_cache

        registry = load_route_registry(include_discovered=True)

        # Check if route already exists by ID
        for existing in registry.routes:
            if existing.id == route_id or route_id in existing.aliases:
                _boost_route_confidence(existing.id, delta=0.1)
                return existing.id, "existing_boosted"

        # Check for functional equivalence (same primitives)
        primitives_used = route_data.get("primitives_used", [])
        new_prim_set = frozenset(
            (p.get("module", ""), p.get("symbol", ""), p.get("role", ""))
            for p in primitives_used
            if isinstance(p, dict)
        )
        if new_prim_set:
            equiv_id = _find_equivalent_route(new_prim_set, registry)
            if equiv_id:
                _boost_route_confidence(equiv_id, delta=0.1)
                return equiv_id, "equivalent_boosted"

        # Write new discovered route entry
        _ROUTES_ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
        route_yaml = {
            "id": route_id,
            "engine_family": route_data.get("engine_family", "unknown"),
            "route_family": route_data.get("engine_family", "unknown"),
            "status": "candidate",
            "confidence": 0.5,
            "discovered_from": f"reflect_attempt_{attempt}",
            "successful_builds": 1,
            "match": {
                "methods": route_data.get("match_methods", [decomposition.method]),
                "instruments": route_data.get("match_instruments", [decomposition.instrument]),
            },
            "primitives": primitives_used,
            "market_data_access": {
                "required": {
                    cap: [f"market_state.{cap.replace('_curve', '').replace('black_', '')}"]
                    for cap in route_data.get("market_data_accessed", [])
                },
            },
            "parameter_bindings": {
                "required": route_data.get("parameters_extracted", []),
            },
            "notes": [route_data.get("rationale", "")],
        }

        route_path = _ROUTES_ENTRIES_DIR / f"{route_id}.yaml"
        with open(route_path, "w") as fh:
            yaml.dump(route_yaml, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

        clear_route_registry_cache()
        return route_id, "captured"

    except Exception:
        return None, "error"


def _find_equivalent_route(
    new_prim_set: frozenset[tuple[str, str, str]],
    registry,
) -> str | None:
    """Find a registry route with the same primitive set (module, symbol, role).

    Returns the route ID if an equivalent exists, None otherwise.
    """
    for route in registry.routes:
        existing_set = frozenset(
            (p.module, p.symbol, p.role) for p in route.primitives
        )
        if existing_set == new_prim_set:
            return route.id
    return None


# ---------------------------------------------------------------------------
# Scorer auto-retraining
# ---------------------------------------------------------------------------

def _maybe_retrain_scorer() -> bool:
    """Retrain the route scorer if enough new outcome data has accumulated.

    Triggers when the task run count exceeds the last training size by 10+.
    Returns True if retraining occurred.
    """
    try:
        from trellis.agent.route_scorer import model_metadata, get_scorer
        from trellis.agent.task_run_store import TASK_RUN_LATEST_ROOT

        meta = model_metadata()
        last_size = meta.get("training_size", 0)

        # Count completed runs
        if not TASK_RUN_LATEST_ROOT.exists():
            return False
        current_size = sum(1 for f in TASK_RUN_LATEST_ROOT.glob("*.json"))
        if current_size - last_size < 10:
            return False

        scorer = get_scorer()
        model = scorer.train_from_outcomes(TASK_RUN_LATEST_ROOT)
        return model is not None
    except Exception:
        return False


def _maybe_promote_validators() -> list[str]:
    """Promote semantic validators from warning → blocking based on FP rate.

    A finding is a false positive if the build succeeded despite the warning.
    Promotion threshold: <5% FP rate over 50+ findings.
    Returns list of validator names promoted.
    """
    try:
        from trellis.agent.semantic_validators import set_validator_mode, get_validator_modes
        from trellis.agent.task_run_store import TASK_RUN_LATEST_ROOT

        if not TASK_RUN_LATEST_ROOT.exists():
            return []

        modes = get_validator_modes()
        promoted = []
        for validator_name, current_mode in modes.items():
            if current_mode == "blocking":
                continue  # already promoted
            # Count findings and false positives from task run records
            total_findings = 0
            false_positives = 0
            for run_file in TASK_RUN_LATEST_ROOT.glob("*.json"):
                try:
                    import json
                    data = json.loads(run_file.read_text())
                    sem_findings = data.get("semantic_findings", [])
                    build_ok = data.get("success", False)
                    for f in sem_findings:
                        if f.get("validator") == validator_name:
                            total_findings += 1
                            if build_ok:
                                false_positives += 1
                except Exception:
                    continue

            if total_findings >= 50:
                fp_rate = false_positives / total_findings
                if fp_rate < 0.05:
                    set_validator_mode(validator_name, "blocking")
                    promoted.append(validator_name)

        return promoted
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Structured lesson capture
# ---------------------------------------------------------------------------

def _capture_structured_lesson(
    lesson_data: dict,
    decomposition: ProductDecomposition,
    success: bool,
    attempt: int,
) -> tuple[str | None, dict[str, Any] | None, str]:
    """Capture a lesson with structured feature tags from the decomposition."""
    from trellis.agent.knowledge.promotion import (
        build_lesson_payload,
        capture_lesson,
        defer_index_rebuilds,
        validate_lesson_payload,
    )

    # Use features from lesson_data if valid, else from decomposition
    features = lesson_data.get("features", [])
    if not features or not isinstance(features, list):
        features = list(decomposition.features)

    # Confidence based on how the lesson was discovered
    if success and attempt == 1:
        confidence = 0.6  # first-attempt insight
    elif success:
        confidence = 0.8  # resolved after failure
    else:
        confidence = 0.4  # from unresolved failure

    lesson_payload = build_lesson_payload(
        category=lesson_data.get("category", "unknown"),
        title=lesson_data.get("title", ""),
        severity=lesson_data.get("severity", "medium"),
        symptom=lesson_data.get("symptom", ""),
        root_cause=lesson_data.get("root_cause", ""),
        fix=lesson_data.get("fix", ""),
        validation=f"Discovered during build (attempt {attempt})",
        method=lesson_data.get("method") or decomposition.method,
        features=features,
        confidence=confidence,
        version="",
    )
    lesson_contract = validate_lesson_payload(lesson_payload)
    if not lesson_contract.valid:
        return None, lesson_contract.to_dict(), "invalid_contract"

    lesson_promotion_outcome = "captured"
    with defer_index_rebuilds():
        lid = capture_lesson(
            category=lesson_data.get("category", "unknown"),
            title=lesson_data.get("title", ""),
            severity=lesson_data.get("severity", "medium"),
            symptom=lesson_data.get("symptom", ""),
            root_cause=lesson_data.get("root_cause", ""),
            fix=lesson_data.get("fix", ""),
            validation=f"Discovered during build (attempt {attempt})",
            method=lesson_data.get("method") or decomposition.method,
            features=features,
            confidence=confidence,
            version="",
        )

        if lid:
            promotion_state = _auto_validate_and_promote(lid)
            if promotion_state["promoted"]:
                lesson_promotion_outcome = "promoted"
            elif promotion_state["validated"]:
                lesson_promotion_outcome = "validated"
        else:
            lesson_promotion_outcome = "duplicate"

    return lid, lesson_contract.to_dict(), lesson_promotion_outcome


def _auto_validate_and_promote(lesson_id: str) -> dict[str, bool]:
    """Auto-validate and promote if criteria met."""
    from trellis.agent.knowledge.promotion import (
        defer_index_rebuilds,
        validate_lesson, promote_lesson,
    )
    import yaml

    path = _KNOWLEDGE_DIR / "lessons" / "entries" / f"{lesson_id}.yaml"
    if not path.exists():
        return {"validated": False, "promoted": False}

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return {"validated": False, "promoted": False}

    try:
        conf = float(data.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        conf = 0.0
    validated = False
    promoted = False
    with defer_index_rebuilds():
        if conf >= 0.6:
            validated = validate_lesson(lesson_id)
        if conf >= 0.8:
            promoted = promote_lesson(lesson_id)
    return {"validated": validated, "promoted": promoted}


# ---------------------------------------------------------------------------
# LLM reflection
# ---------------------------------------------------------------------------

def _llm_reflect(
    description: str,
    decomposition: ProductDecomposition,
    gap_report: GapReport,
    retrieved_lesson_ids: list[str],
    success: bool,
    failures: list[str],
    code: str,
    agent_observations: list[dict[str, Any]],
    model: str | None,
) -> dict | None:
    """Ask LLM to reflect on the build and produce structured output."""
    try:
        from trellis.agent.config import llm_generate_json, load_env
        from trellis.agent.knowledge import get_store
        load_env()

        store = get_store()

        # Build retrieved lesson summaries
        lesson_lines = []
        for lid in retrieved_lesson_ids:
            lesson = store._load_lesson(lid)
            if lesson:
                lesson_lines.append(f"- [{lesson.severity.value}] {lesson.title}")
        retrieved_text = "\n".join(lesson_lines) if lesson_lines else "(none)"

        # Feature options for the LLM
        feature_ids = sorted(store._features.keys())

        # Failure section
        if failures:
            failure_section = (
                "Failures encountered:\n"
                + "\n".join(f"- {f}" for f in failures)
                + f"\n\nBuild {'succeeded after retries' if success else 'FAILED (all retries exhausted)'}"
            )
        else:
            failure_section = "Build succeeded on first attempt (no failures)."

        observation_lines = []
        for observation in agent_observations:
            agent = observation.get("agent", "agent")
            kind = observation.get("kind", "note")
            severity = observation.get("severity", "info")
            summary = observation.get("summary", "")
            details = observation.get("details")
            line = f"- [{agent}/{kind}/{severity}] {summary}"
            if details:
                line += f" | details={details}"
            observation_lines.append(line)
        observations_text = "\n".join(observation_lines) if observation_lines else "(none)"

        # Code section (only for cookbook extraction or failure analysis)
        code_section = ""
        if not gap_report.has_cookbook and success:
            code_section = f"## Successful code (for cookbook extraction):\n```python\n{code[:3000]}\n```"
        elif failures:
            code_section = f"## Final code:\n```python\n{code[:2000]}\n```"

        prompt = _REFLECT_PROMPT.format(
            outcome="SUCCEEDED in" if success else "FAILED",
            description=description,
            method=decomposition.method,
            instrument=decomposition.instrument,
            features=list(decomposition.features),
            confidence=gap_report.confidence,
            gaps=gap_report.missing or "(none)",
            failure_section=failure_section + f"\n\nAgent observations:\n{observations_text}",
            retrieved_lessons=retrieved_text,
            feature_options=feature_ids,
            code_section=code_section,
        )

        return llm_generate_json(prompt, model=model)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cookbook enrichment
# ---------------------------------------------------------------------------

def _enrich_cookbook(method: str, pattern: str) -> None:
    """Save an extracted cookbook pattern for a method that had none."""
    import yaml

    path = _KNOWLEDGE_DIR / "canonical" / "cookbooks.yaml"
    data: dict = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}

    if method in data:
        return  # Already exists

    data[method] = {
        "template": pattern,
        "description": f"Auto-extracted from successful build",
        "applicable_instruments": [],
        "version": "auto",
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    # Invalidate the live store caches so the new cookbook is immediately visible.
    from trellis.agent.knowledge.promotion import _clear_loaded_store_runtime_caches
    _clear_loaded_store_runtime_caches()


# ---------------------------------------------------------------------------
# Decomposition persistence
# ---------------------------------------------------------------------------

def _save_decomposition(decomposition: ProductDecomposition) -> None:
    """Save a learned decomposition."""
    from trellis.agent.knowledge import get_store
    get_store().save_decomposition(decomposition)


# ---------------------------------------------------------------------------
# Gap recording
# ---------------------------------------------------------------------------

def _record_gaps(
    gaps: list[str],
    decomposition: ProductDecomposition,
    gap_report: GapReport,
) -> str:
    """Record identified knowledge gaps to a log file for review."""
    import yaml

    log_path = _KNOWLEDGE_DIR / "traces" / "knowledge_gaps.yaml"
    existing = []
    if log_path.exists():
        existing = yaml.safe_load(log_path.read_text()) or []

    existing.append({
        "timestamp": datetime.now().isoformat(),
        "instrument": decomposition.instrument,
        "method": decomposition.method,
        "features": list(decomposition.features),
        "confidence": gap_report.confidence,
        "gaps": gaps,
    })

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    return str(log_path)


def _record_cookbook_candidate(
    method: str,
    description: str,
    code: str,
) -> str | None:
    """Persist a deterministic cookbook candidate without mutating canonical cookbooks."""
    from trellis.agent.assembly_tools import build_cookbook_candidate_payload

    payload = build_cookbook_candidate_payload(
        method=method,
        description=description,
        code=code,
    )
    if payload is None:
        return None

    import yaml

    candidate_dir = _KNOWLEDGE_DIR / "traces" / "cookbook_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{method}.yaml"
    path = candidate_dir / filename
    with open(path, "w") as f:
        yaml.dump(
            {
                "timestamp": datetime.now().isoformat(),
                **payload,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return str(path)


# ---------------------------------------------------------------------------
# Full trace
# ---------------------------------------------------------------------------

def _record_full_trace(
    description: str,
    decomposition: ProductDecomposition,
    gap_report: GapReport,
    retrieved_lesson_ids: list[str],
    success: bool,
    failures: list[str],
    code: str,
    attempt: int,
    actions: dict,
    agent_observations: list[dict[str, Any]],
) -> str:
    """Record a comprehensive trace of the knowledge-aware build."""
    from trellis.agent.knowledge.promotion import record_trace
    trace_filename = record_trace(
        instrument=decomposition.instrument,
        method=decomposition.method,
        description=description,
        pricing_plan={
            "method": decomposition.method,
            "features": list(decomposition.features),
            "confidence": gap_report.confidence,
            "gaps": gap_report.missing,
            "retrieved_lessons": retrieved_lesson_ids,
        },
        attempt=attempt,
        code=code,
        validation_failures=failures,
        diagnosis=actions,
        agent_observations=agent_observations,
        resolved=success,
        lesson_id=actions.get("lesson_captured"),
    )
    return str(_KNOWLEDGE_DIR / "traces" / trace_filename)


def _attach_source_trace(lesson_id: str, trace_path: str) -> None:
    """Backfill ``source_trace`` on a captured lesson after the trace is written."""
    import yaml

    lesson_path = _KNOWLEDGE_DIR / "lessons" / "entries" / f"{lesson_id}.yaml"
    if not lesson_path.exists():
        return

    data = yaml.safe_load(lesson_path.read_text()) or {}
    data["source_trace"] = trace_path
    with open(lesson_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    from trellis.agent.knowledge.promotion import _clear_loaded_store_runtime_caches
    _clear_loaded_store_runtime_caches()


# ---------------------------------------------------------------------------
# Distillation trigger
# ---------------------------------------------------------------------------

def _should_distill() -> bool:
    """Check if we should run distillation."""
    import yaml

    idx_path = _KNOWLEDGE_DIR / "lessons" / "index.yaml"
    if not idx_path.exists():
        return False

    data = yaml.safe_load(idx_path.read_text()) or {}
    entries = data.get("entries", [])

    candidates = sum(1 for e in entries if e.get("status") == "candidate")
    return candidates >= 5 or len(entries) >= 30
