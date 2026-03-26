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
reusable evaluate() pattern from the successful code below. Otherwise null."
}}

Rules:
- "lesson" should be null if no failure was resolved (first-attempt success)
- "features" MUST be from the provided feature list, not invented
- "knowledge_gaps" should list what was missing from the knowledge base
- "cookbook_extract" should be a reusable code template with INSTRUMENT-SPECIFIC markers, or null

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
        "lesson_captured": None,
        "gaps_identified": [],
        "cookbook_enriched": False,
        "cookbook_candidate_saved": None,
        "knowledge_trace_saved": None,
        "knowledge_gap_log_saved": None,
        "decomposition_saved": False,
        "distill_run": False,
        "agent_observation_count": len(agent_observations or []),
    }

    try:
        # 1. Attribution — boost retrieved lessons on success
        if success:
            actions["lessons_attributed"] = _attribute_success(
                retrieved_lesson_ids
            )

        # 2. Save learned decomposition on success
        if success and decomposition.learned:
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
                if lesson_data and isinstance(lesson_data, dict) and lesson_data.get("title"):
                    lid = _capture_structured_lesson(
                        lesson_data, decomposition, success, attempt,
                    )
                    actions["lesson_captured"] = lid

                # Record knowledge gaps
                gaps = reflection.get("knowledge_gaps", [])
                if gaps:
                    actions["gaps_identified"] = gaps
                    actions["knowledge_gap_log_saved"] = _record_gaps(
                        gaps,
                        decomposition,
                        gap_report,
                    )

                # Enrich cookbook if missing
                cookbook_extract = reflection.get("cookbook_extract")
                if cookbook_extract and not gap_report.has_cookbook and success:
                    _enrich_cookbook(decomposition.method, cookbook_extract)
                    actions["cookbook_enriched"] = True

            if success and not gap_report.has_cookbook and not actions["cookbook_enriched"]:
                actions["cookbook_candidate_saved"] = _record_cookbook_candidate(
                    decomposition.method,
                    description,
                    code,
                )

        # 4. Record trace
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
        if _should_distill():
            from trellis.agent.knowledge.promotion import distill
            distill()
            actions["distill_run"] = True

    except Exception:
        pass  # Reflection must never block the build

    return actions


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


# ---------------------------------------------------------------------------
# Structured lesson capture
# ---------------------------------------------------------------------------

def _capture_structured_lesson(
    lesson_data: dict,
    decomposition: ProductDecomposition,
    success: bool,
    attempt: int,
) -> str | None:
    """Capture a lesson with structured feature tags from the decomposition."""
    from trellis.agent.knowledge.promotion import capture_lesson

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
        _auto_validate_and_promote(lid)

    return lid


def _auto_validate_and_promote(lesson_id: str) -> None:
    """Auto-validate and promote if criteria met."""
    from trellis.agent.knowledge.promotion import (
        validate_lesson, promote_lesson,
    )
    import yaml

    path = _KNOWLEDGE_DIR / "lessons" / "entries" / f"{lesson_id}.yaml"
    if not path.exists():
        return

    data = yaml.safe_load(path.read_text())
    if not data:
        return

    conf = data.get("confidence", 0)
    if conf >= 0.6:
        validate_lesson(lesson_id)
    if conf >= 0.8:
        promote_lesson(lesson_id)


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

    # Invalidate the store cache
    from trellis.agent.knowledge import get_store
    store = get_store()
    store._cookbooks_cache = None


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
