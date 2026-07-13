from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_load_role_orientations_exposes_required_runtime_roles():
    from trellis.agent.role_orientation import load_role_orientations

    orientations = load_role_orientations()

    assert set(orientations) == {"quant", "model_validator"}
    assert orientations["quant"].contract_id == "quant-runtime-navigation"
    assert orientations["model_validator"].contract_id == (
        "model-validator-runtime-navigation"
    )
    assert all(orientation.version == 1 for orientation in orientations.values())
    assert all(orientation.navigation for orientation in orientations.values())


def test_role_orientation_cards_are_bounded_and_role_separated():
    from trellis.agent.role_orientation import (
        get_role_orientation,
        render_role_orientation_card,
    )

    quant = get_role_orientation("quant")
    quant_card = render_role_orientation_card("quant")
    validator = get_role_orientation("model_validator")
    validator_card = render_role_orientation_card("model_validator")

    assert len(quant_card) <= quant.max_render_chars
    assert len(validator_card) <= validator.max_render_chars
    assert "`quant-runtime-navigation@1`" in quant_card
    assert "canonical/decompositions.yaml" in quant_card
    assert "canonical/model_grammar.yaml" in quant_card
    assert "canonical/cookbooks.yaml" in quant_card
    assert "docs/quant/index.rst" in quant_card
    assert "Do not generate code or select import paths" in quant_card
    assert "deterministic_evidence_packet" not in quant_card

    assert "`model-validator-runtime-navigation@1`" in validator_card
    assert "deterministic_evidence_packet" in validator_card
    assert "docs/mathematical/calibration.rst" in validator_card
    assert "canonical/cookbooks.yaml" in validator_card
    assert "LIMITATIONS.md" in validator_card
    assert "Do not repeat deterministic checks" in validator_card
    assert "canonical/decompositions.yaml" not in validator_card


def test_role_orientation_summary_is_trace_safe():
    from trellis.agent.role_orientation import role_orientation_summary

    assert role_orientation_summary("quant") == {
        "role": "quant",
        "contract_id": "quant-runtime-navigation",
        "version": 1,
    }
    assert role_orientation_summary("model_validator") == {
        "role": "model_validator",
        "contract_id": "model-validator-runtime-navigation",
        "version": 1,
    }


def test_role_orientation_file_navigation_targets_exist():
    from trellis.agent.role_orientation import load_role_orientations

    for orientation in load_role_orientations().values():
        for resource in orientation.navigation:
            if resource.path.startswith("runtime:"):
                continue
            assert (ROOT / resource.path).is_file(), (
                f"{orientation.identity} references missing {resource.path}"
            )


def test_load_role_orientations_rejects_incomplete_manifest(tmp_path):
    from trellis.agent.role_orientation import load_role_orientations

    canonical = yaml.safe_load(
        (ROOT / "trellis/agent/knowledge/canonical/agent_orientations.yaml").read_text()
    )
    canonical["orientations"].pop("model_validator")
    path = tmp_path / "agent_orientations.yaml"
    path.write_text(yaml.safe_dump(canonical, sort_keys=False))

    with pytest.raises(ValueError, match="quant.*model_validator"):
        load_role_orientations(path)


def test_unknown_runtime_role_fails_closed():
    from trellis.agent.role_orientation import get_role_orientation

    with pytest.raises(ValueError, match="Unsupported runtime agent role"):
        get_role_orientation("builder")


def test_quant_llm_decomposition_prompt_includes_only_quant_orientation(monkeypatch):
    from trellis.agent.knowledge import get_store
    from trellis.agent.knowledge.decompose import _decompose_via_llm

    captured: dict[str, str] = {}

    monkeypatch.setattr("trellis.agent.config.load_env", lambda: None)

    def fake_generate(prompt, model=None):
        captured["prompt"] = prompt
        return {
            "features": ["discounting"],
            "method": "analytical",
            "method_modules": [],
            "required_market_data": ["discount_curve"],
            "reasoning": "bounded test",
        }

    monkeypatch.setattr("trellis.agent.config.llm_generate_json", fake_generate)

    result = _decompose_via_llm(
        "Novel bounded derivative",
        "novel_bounded_derivative",
        get_store(),
        "fake-model",
    )

    assert result.method == "analytical"
    assert "quant-runtime-navigation@1" in captured["prompt"]
    assert "canonical/decompositions.yaml" in captured["prompt"]
    assert "model-validator-runtime-navigation" not in captured["prompt"]
    assert '"method_modules"' not in captured["prompt"]
