from __future__ import annotations

from pathlib import Path


def _seed_minimal_knowledge_dir(root: Path) -> None:
    import yaml

    canonical = root / "canonical"
    lessons = root / "lessons" / "entries"
    traces = root / "traces"
    canonical.mkdir(parents=True, exist_ok=True)
    lessons.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)

    (canonical / "features.yaml").write_text(yaml.safe_dump([
        {"id": "path_dependent", "description": "path dependent"},
        {"id": "vanilla", "description": "vanilla"},
    ]))
    (canonical / "decompositions.yaml").write_text(yaml.safe_dump({
        "european_option": {
            "features": ["vanilla", "path_dependent"],
            "method": "monte_carlo",
            "required_market_data": ["discount_curve", "black_vol_surface", "spot"],
        }
    }))
    (canonical / "principles.yaml").write_text(yaml.safe_dump([]))
    (canonical / "failure_signatures.yaml").write_text(yaml.safe_dump([]))
    (canonical / "cookbooks.yaml").write_text(yaml.safe_dump({
        "monte_carlo": {
            "template": "return 0.0",
            "description": "stub",
            "applicable_instruments": ["european_option"],
        }
    }))
    (canonical / "data_contracts.yaml").write_text(yaml.safe_dump({}))
    (canonical / "method_requirements.yaml").write_text(yaml.safe_dump({
        "monte_carlo": [
            "Use simulated paths.",
        ]
    }))
    (lessons.parent / "index.yaml").write_text(yaml.safe_dump({"entries": []}))


def _patch_knowledge_paths(monkeypatch, root: Path) -> None:
    import trellis.agent.knowledge as knowledge_pkg
    import trellis.agent.knowledge.store as store_mod
    import trellis.agent.knowledge.reflect as reflect_mod
    import trellis.agent.knowledge.promotion as promotion_mod

    monkeypatch.setattr(store_mod, "_KNOWLEDGE_DIR", root)
    monkeypatch.setattr(reflect_mod, "_KNOWLEDGE_DIR", root)
    monkeypatch.setattr(promotion_mod, "_KNOWLEDGE_DIR", root)
    monkeypatch.setattr(promotion_mod, "_LESSONS_DIR", root / "lessons")
    monkeypatch.setattr(promotion_mod, "_TRACES_DIR", root / "traces")
    monkeypatch.setattr(promotion_mod, "_INDEX_PATH", root / "lessons" / "index.yaml")
    monkeypatch.setattr(knowledge_pkg, "_store", None)


def test_reflect_on_build_records_cookbook_candidate_when_llm_fails(monkeypatch, tmp_path):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.reflect import reflect_on_build
    from trellis.agent.knowledge.schema import ProductDecomposition

    knowledge_root = tmp_path / "knowledge"
    _seed_minimal_knowledge_dir(knowledge_root)
    _patch_knowledge_paths(monkeypatch, knowledge_root)

    monkeypatch.setattr("trellis.agent.knowledge.reflect._llm_reflect", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.knowledge.reflect._should_distill", lambda: False)

    actions = reflect_on_build(
        description="European call option",
        decomposition=ProductDecomposition(
            instrument="european_option",
            features=("vanilla",),
            method="analytical",
            learned=False,
        ),
        gap_report=GapReport(has_cookbook=False, confidence=0.2, retrieved_lesson_ids=[]),
        retrieved_lesson_ids=[],
        success=True,
        failures=[],
        code="""class Demo:\n    def evaluate(self, market_state):\n        spec = self._spec\n        return spec.spot\n""",
        attempt=1,
        model=None,
    )

    assert actions["cookbook_enriched"] is False
    assert actions["cookbook_candidate_saved"]
    candidate_path = Path(actions["cookbook_candidate_saved"])
    assert candidate_path.exists()
    assert "analytical" in candidate_path.name
    assert actions["knowledge_trace_saved"]
    knowledge_trace_path = Path(actions["knowledge_trace_saved"])
    assert knowledge_trace_path.exists()
    assert knowledge_trace_path.parent.name == "traces"


def test_closed_loop_failure_captures_lesson_and_next_gap_check_retrieves_it(monkeypatch, tmp_path):
    import trellis.agent.knowledge as knowledge_pkg
    from trellis.agent.knowledge.gap_check import GapReport, gap_check
    from trellis.agent.knowledge.reflect import reflect_on_build
    from trellis.agent.knowledge.schema import ProductDecomposition

    knowledge_root = tmp_path / "knowledge"
    _seed_minimal_knowledge_dir(knowledge_root)
    _patch_knowledge_paths(monkeypatch, knowledge_root)
    monkeypatch.setattr("trellis.agent.knowledge.reflect._should_distill", lambda: False)
    monkeypatch.setattr(
        "trellis.agent.knowledge.reflect._llm_reflect",
        lambda *args, **kwargs: {
            "lesson": {
                "category": "numerical",
                "title": "Use antithetic control variates",
                "severity": "medium",
                "symptom": "Monte Carlo variance is too high",
                "root_cause": "The simulation uses too few effective variance-reduction techniques.",
                "fix": "Use antithetic sampling and control variates for the vanilla leg.",
                "features": ["path_dependent"],
                "method": "monte_carlo",
            },
            "knowledge_gaps": [],
            "cookbook_extract": None,
        },
    )

    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("path_dependent",),
        method="monte_carlo",
        learned=False,
    )
    first_gap = GapReport(
        has_decomposition=True,
        has_cookbook=True,
        has_contracts=True,
        has_requirements=True,
        confidence=0.3,
        retrieved_lesson_ids=[],
    )

    actions = reflect_on_build(
        description="European call option priced with Monte Carlo after one failed retry",
        decomposition=decomposition,
        gap_report=first_gap,
        retrieved_lesson_ids=[],
        success=True,
        failures=["Monte Carlo variance too high"],
        code="return 0.0",
        attempt=2,
        model=None,
    )

    assert actions["lesson_captured"] is not None
    assert actions["lesson_contract"] is not None
    assert actions["lesson_contract"]["valid"] is True
    assert actions["lesson_promotion_outcome"] == "promoted"
    assert actions["knowledge_trace_saved"] is not None

    monkeypatch.setattr(knowledge_pkg, "_store", None)
    second_gap = gap_check(decomposition)

    assert actions["lesson_captured"] in second_gap.retrieved_lesson_ids
    assert second_gap.lesson_count >= 1
    assert second_gap.confidence > first_gap.confidence

    import yaml

    lesson_path = knowledge_root / "lessons" / "entries" / f"{actions['lesson_captured']}.yaml"
    lesson_data = yaml.safe_load(lesson_path.read_text())
    assert lesson_data["source_trace"] == actions["knowledge_trace_saved"]

    trace_path = Path(actions["knowledge_trace_saved"])
    trace_data = yaml.safe_load(trace_path.read_text())
    assert trace_data["diagnosis"]["lesson_contract"]["valid"] is True
    assert trace_data["diagnosis"]["lesson_promotion_outcome"] == "promoted"


def test_reflect_trace_records_agent_observations(monkeypatch, tmp_path):
    import yaml
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.reflect import reflect_on_build
    from trellis.agent.knowledge.schema import ProductDecomposition

    knowledge_root = tmp_path / "knowledge"
    _seed_minimal_knowledge_dir(knowledge_root)
    _patch_knowledge_paths(monkeypatch, knowledge_root)
    monkeypatch.setattr("trellis.agent.knowledge.reflect._llm_reflect", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.knowledge.reflect._should_distill", lambda: False)

    actions = reflect_on_build(
        description="European call option",
        decomposition=ProductDecomposition(
            instrument="european_option",
            features=("vanilla",),
            method="analytical",
            learned=False,
        ),
        gap_report=GapReport(has_cookbook=True, confidence=0.7, retrieved_lesson_ids=[]),
        retrieved_lesson_ids=[],
        success=False,
        failures=["arbiter rejected build"],
        code="return 0.0",
        attempt=2,
        agent_observations=[
            {
                "agent": "critic",
                "kind": "concern",
                "summary": "Potential double discounting",
                "severity": "error",
            }
        ],
        model=None,
    )

    trace_path = Path(actions["knowledge_trace_saved"])
    trace_data = yaml.safe_load(trace_path.read_text())
    assert trace_data["agent_observations"][0]["agent"] == "critic"
    assert trace_data["agent_observations"][0]["summary"] == "Potential double discounting"
