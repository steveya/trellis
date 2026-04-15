"""Executor-time fresh-generated boundary enforcement (QUA-872).

QUA-866 added a post-build boundary check in the benchmark runner.  This
file defends the earlier layer: ``_resolve_output_target`` must never let a
fresh build resolve to a path under ``trellis/instruments/_agent/``.  The
existing design redirects such paths to the fresh-build namespace, but a
regression in that redirect logic would silently leak admitted adapters
back into the pilot critical path — and the symptom would surface only
after an LLM call had burned tokens.  This test locks the invariant at the
resolver level, before any LLM is invoked.
"""

from __future__ import annotations

import pytest

from trellis.agent import executor as executor_mod
from trellis.agent.fresh_generated_boundary import FreshGeneratedBoundaryError


def _resolve(module_path: str, *, fresh_build: bool, metadata: dict | None = None):
    return executor_mod._resolve_output_target(
        module_path,
        fresh_build=fresh_build,
        request_metadata=metadata,
    )


def test_non_agent_path_resolves_unchanged_whether_or_not_fresh_build():
    # Step paths come from `PrimitivePlan.steps[*].module_path` and are
    # relative to the `trellis/` package root (`instruments/_agent/...`,
    # `models/black.py`, etc.).  Passing a `trellis/`-prefixed path would
    # produce a double-prefixed import name like `trellis.trellis.models.black`,
    # which is not how the resolver is actually called from `build_payoff`.
    # (PR #590 Copilot review.)
    file_path, module_path, module_name = _resolve(
        "models/black.py",
        fresh_build=False,
    )
    assert module_path == "models/black.py"
    assert module_name == "trellis.models.black"
    assert "_agent" not in str(file_path)

    file_path_fresh, module_path_fresh, module_name_fresh = _resolve(
        "models/black.py",
        fresh_build=True,
    )
    assert module_path_fresh == "models/black.py"
    assert module_name_fresh == "trellis.models.black"
    assert "_agent" not in str(file_path_fresh)


def test_agent_path_with_fresh_build_redirects_to_fresh_namespace():
    file_path, module_path, module_name = _resolve(
        "instruments/_agent/europeanoptionanalytical.py",
        fresh_build=True,
    )
    # Resolver puts the fresh build under either the benchmark artifact root
    # (task_runs/...) or the `_agent/_fresh/` isolation sub-namespace.  In
    # either case, the admitted `_agent/*` tree itself (without `_fresh/`)
    # must not be the target.
    normalized = module_path.replace("\\", "/")
    assert "_fresh" in normalized or "financepy_benchmarks/generated" in normalized
    assert "_fresh" in str(file_path) or "financepy_benchmarks/generated" in str(file_path)


def test_agent_path_with_fresh_build_and_benchmark_metadata_lands_outside_package():
    """Pilot runs supply `task_corpus=benchmark_financepy` metadata and should
    land under `task_runs/financepy_benchmarks/generated/...`, entirely off
    the package tree."""
    file_path, module_path, module_name = _resolve(
        "instruments/_agent/europeanoptionanalytical.py",
        fresh_build=True,
        metadata={
            "task_corpus": "benchmark_financepy",
            "task_id": "F001",
            "preferred_method": "analytical",
        },
    )
    assert "_agent" not in module_path.replace("\\", "/")
    assert "_agent" not in module_name
    assert "financepy_benchmarks/generated/f001" in str(file_path).replace("\\", "/")


def test_agent_path_without_fresh_build_keeps_writing_to_admitted_tree():
    file_path, module_path, _ = _resolve(
        "instruments/_agent/europeanoptionanalytical.py",
        fresh_build=False,
    )
    assert "_agent/" in module_path.replace("\\", "/")
    assert "_agent" in str(file_path)


def test_fresh_build_refuses_when_redirect_returns_agent_path(monkeypatch):
    """If the fresh-build redirect regresses, the resolver must raise before the LLM is called."""

    def broken_redirect(path: str) -> str:
        # Pretend the redirect helper silently returned an _agent path.
        return path

    monkeypatch.setattr(executor_mod, "_fresh_build_module_path", broken_redirect)
    monkeypatch.setattr(executor_mod, "_benchmark_fresh_build_root", lambda metadata: None)

    with pytest.raises(FreshGeneratedBoundaryError) as exc_info:
        _resolve(
            "instruments/_agent/europeanoptionanalytical.py",
            fresh_build=True,
        )
    assert "_agent" in str(exc_info.value)
    assert "QUA-872" in str(exc_info.value) or "QUA-866" in str(exc_info.value)
