"""Tests for bounded runtime-agent orientation content resolution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trellis.agent.knowledge.schema import (
    CookbookEntry,
    MethodRequirements,
    ModelGrammarEntry,
    Principle,
    ProductDecomposition,
)
from trellis.agent.role_orientation import (
    OrientationResource,
    get_role_orientation,
)


def _write_docs(root: Path) -> None:
    docs = root / "docs" / "quant"
    docs.mkdir(parents=True)
    (docs / "index.rst").write_text(
        """Quant Documentation
===================

.. toctree::
   :maxdepth: 1

   monte_carlo
   calibration
"""
    )
    (docs / "monte_carlo.rst").write_text(
        """Monte Carlo
===========

Path Simulation
---------------

Use bounded path state and explicit observation schedules for path-dependent claims.

Builder Example
---------------

.. code-block:: python

   from trellis.models.product_helper import price_task_specific_claim
"""
    )
    (docs / "calibration.rst").write_text(
        """Calibration
===========

Heston Calibration Consistency
------------------------------

A volatility-surface bump and a Heston parameter bump are different experiments.
Recalibration must be explicit and its target quotes must be recorded.

Unrelated Vanilla Note
----------------------

This section discusses a plain constant-volatility European option.
"""
    )


def _docs_only_orientation(role: str, *, max_context_chars: int = 900):
    return replace(
        get_role_orientation(role),
        max_context_chars=max_context_chars,
        max_resource_chars=360,
        navigation=(
            OrientationResource(
                order=1,
                resource_id="quant_docs_index",
                kind="official_docs_index",
                path="docs/quant/index.rst",
                purpose="Resolve task-relevant quantitative documentation.",
            ),
        ),
    )


def test_resolution_follows_docs_index_ranks_sections_and_is_deterministic(tmp_path):
    from trellis.agent.orientation_resolution import (
        RoleOrientationQuery,
        resolve_role_orientation_packet,
    )

    _write_docs(tmp_path)
    query = RoleOrientationQuery(
        instrument_type="european_option",
        method="pde_solver",
        model_family="heston",
        residual_risks=("calibration_consistency",),
        description="Review Heston recalibration after a volatility surface bump.",
    )
    orientation = _docs_only_orientation("model_validator")

    first = resolve_role_orientation_packet(
        "model_validator",
        query,
        orientation=orientation,
        repo_root=tmp_path,
        knowledge={},
    )
    second = resolve_role_orientation_packet(
        "model_validator",
        query,
        orientation=orientation,
        repo_root=tmp_path,
        knowledge={},
    )

    assert first == second
    assert "Heston Calibration Consistency" in first.context
    assert "volatility-surface bump" in first.context
    assert "price_task_specific_claim" not in first.context
    assert len(first.context) <= orientation.max_context_chars
    assert first.content_digest == second.content_digest
    assert first.excerpts[0].source_path == "docs/quant/calibration.rst"
    assert first.excerpts[0].section == "Heston Calibration Consistency"
    assert len(first.excerpts[0].source_digest) == 64

    summary = first.summary()
    assert summary["role"] == "model_validator"
    assert summary["content_digest"] == first.content_digest
    assert summary["selected_sections"] == [
        "docs/quant/calibration.rst#Heston Calibration Consistency"
    ]
    assert "volatility-surface bump" not in str(summary)


def test_resolution_enforces_context_budget_and_reports_omissions(tmp_path):
    from trellis.agent.orientation_resolution import (
        RoleOrientationQuery,
        resolve_role_orientation_packet,
    )

    _write_docs(tmp_path)
    orientation = _docs_only_orientation("quant", max_context_chars=140)
    packet = resolve_role_orientation_packet(
        "quant",
        RoleOrientationQuery(
            method="monte_carlo",
            features=("path_dependent", "scheduled_observation"),
            description="Path-dependent scheduled-observation Monte Carlo claim.",
        ),
        orientation=orientation,
        repo_root=tmp_path,
        knowledge={},
    )

    assert len(packet.context) <= 140
    assert packet.omitted_count >= 1
    assert "[orientation context truncated" in packet.context


def test_resolution_rejects_resource_paths_outside_repository(tmp_path):
    from trellis.agent.orientation_resolution import (
        RoleOrientationQuery,
        resolve_role_orientation_packet,
    )

    outside = tmp_path.parent / "outside-agent-doc.md"
    outside.write_text("# Outside\n\nThis must never be loaded.\n")
    orientation = replace(
        get_role_orientation("quant"),
        navigation=(
            OrientationResource(
                order=1,
                resource_id="escaping_doc",
                kind="official_docs",
                path="../outside-agent-doc.md",
                purpose="Invalid test resource.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="outside repository root"):
        resolve_role_orientation_packet(
            "quant",
            RoleOrientationQuery(description="outside"),
            orientation=orientation,
            repo_root=tmp_path,
            knowledge={},
        )


def test_quant_projection_uses_canonical_evidence_without_cookbook_code(
    tmp_path,
    monkeypatch,
):
    from trellis.agent.orientation_resolution import (
        RoleOrientationQuery,
        resolve_role_orientation_packet,
    )

    orientation = replace(
        get_role_orientation("quant"),
        navigation=tuple(
            resource
            for resource in get_role_orientation("quant").navigation
            if resource.resource_id
            in {
                "product_decompositions",
                "model_grammar",
                "method_requirements",
                "cookbook_catalog",
            }
        ),
    )
    knowledge = {
        "decomposition": ProductDecomposition(
            instrument="bounded_path_claim",
            features=("path_dependent", "scheduled_observation"),
            method="monte_carlo",
            reasoning="The payoff requires scheduled path state.",
        ),
        "principles": [
            Principle(id="P-test", rule="Keep contractual observation times explicit."),
        ],
        "method_requirements": MethodRequirements(
            method="monte_carlo",
            requirements=("Use a convergence diagnostic for sampled estimators.",),
        ),
        "model_grammar": [
            ModelGrammarEntry(
                id="gbm",
                title="Geometric Brownian motion",
                methods=("monte_carlo",),
                model_families=("black_scholes",),
                state_semantics=("positive scalar spot",),
            )
        ],
        "cookbook": CookbookEntry(
            method="monte_carlo",
            description="Simulation with explicit process, path state, and estimator controls.",
            template="from forbidden.builder_api import task_specific_helper\n",
        ),
        "lessons": [],
    }

    monkeypatch.setattr(
        "trellis.agent.knowledge.skills.select_prompt_skill_artifacts",
        lambda *args, **kwargs: [
            {
                "id": "principle:safe-navigation",
                "kind": "principle",
                "summary": "Challenge model assumptions before selecting a method.",
            },
            {
                "id": "route_hint:forbidden-helper",
                "kind": "route_hint",
                "summary": "Call a task-specific helper.",
            },
        ],
    )

    packet = resolve_role_orientation_packet(
        "quant",
        RoleOrientationQuery(
            instrument_type="bounded_path_claim",
            method="monte_carlo",
            model_family="black_scholes",
            features=("path_dependent", "scheduled_observation"),
        ),
        orientation=orientation,
        repo_root=tmp_path,
        knowledge=knowledge,
    )

    assert "scheduled path state" in packet.context
    assert "convergence diagnostic" in packet.context
    assert "Geometric Brownian motion" in packet.context
    assert "Simulation with explicit process" in packet.context
    assert "Generated skill principle:safe-navigation" in packet.context
    assert "forbidden-helper" not in packet.context
    assert "forbidden.builder_api" not in packet.context
    assert "task_specific_helper" not in packet.context
    assert "from trellis" not in packet.context


def test_default_role_packets_are_role_separated_and_source_bounded():
    from trellis.agent.orientation_resolution import (
        RoleOrientationQuery,
        resolve_role_orientation_packet,
    )

    query = RoleOrientationQuery(
        instrument_type="european_option",
        method="pde_solver",
        model_family="heston",
        residual_risks=("calibration_consistency", "boundary_conditions"),
        description="Validate a Heston PDE price and calibration contract.",
    )
    quant = resolve_role_orientation_packet("quant", query)
    validator = resolve_role_orientation_packet("model_validator", query)

    assert quant.orientation_identity.startswith("quant-runtime-navigation@")
    assert validator.orientation_identity.startswith(
        "model-validator-runtime-navigation@"
    )
    assert "model-validator-runtime-navigation" not in quant.rendered
    assert "quant-runtime-navigation" not in validator.rendered
    assert "deterministic evidence" not in quant.context.lower()
    assert "from trellis" not in quant.context.lower()
    assert "route_helper" not in quant.context.lower()
    assert any(
        "heston" in excerpt.section.lower()
        for excerpt in quant.excerpts
        if excerpt.resource_id == "quant_docs_index"
    )
    assert "calibration" in validator.context.lower()
    assert len(quant.context) <= get_role_orientation("quant").max_context_chars
    assert len(validator.context) <= get_role_orientation(
        "model_validator"
    ).max_context_chars
