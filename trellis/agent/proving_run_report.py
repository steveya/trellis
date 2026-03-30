"""Markdown rendering helpers for arbitrary-derivative proving runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def render_proving_run_report(report: Mapping[str, Any]) -> str:
    """Render a structured proving-run report as Markdown."""
    title = str(report.get("title") or "Arbitrary-derivative proving run").strip()
    lines: list[str] = [f"# {title}", ""]

    lines.extend(_render_summary(report))
    lines.extend(_render_scope_caveats(report))
    lines.extend(_render_prompt(report))
    lines.extend(_render_decisions(report))
    lines.extend(_render_sections(report))
    lines.extend(_render_pricing(report))
    lines.extend(_render_repro(report))
    return "\n".join(lines).rstrip() + "\n"


def _render_summary(report: Mapping[str, Any]) -> list[str]:
    task = dict(report.get("task") or {})
    pricing = dict(report.get("pricing") or {})
    reproducibility = dict(report.get("reproducibility") or {})
    semantic = dict(report.get("semantic") or {})
    seed = pricing.get("seed") or reproducibility.get("seed") or dict(report.get("deterministic_decisions") or {}).get("simulation_seed")
    return [
        "## Summary",
        f"- Task id: `{task.get('id', '')}`",
        f"- Task title: `{task.get('title', '')}`",
        f"- Semantic contract: `{semantic.get('semantic_id', '')}`",
        f"- Run seed: `{seed or ''}`",
        f"- Output file: `{report.get('output_path', '')}`",
        "",
    ]


def _render_scope_caveats(report: Mapping[str, Any]) -> list[str]:
    build_result = dict(report.get("build_result") or {})
    summary = dict(report.get("semantic_trace") or {}).get("summary") or {}
    learning = dict(summary.get("learning") or {})
    pricing = dict(report.get("pricing") or {})
    reproducibility = dict(report.get("reproducibility") or {})
    market = dict(pricing.get("market") or reproducibility.get("market") or {})
    caveats = [
        "This is a proving run, not a full critic/arbiter/codegen build.",
        f"Build payoff class: `{build_result.get('payoff_class', summary.get('payoff_class', ''))}`",
        "The knowledge system did not engage on this run; retrieval and promotion counts are zero.",
        f"Comparison status: `{summary.get('comparison_status', 'null')}`",
        "The mock market data is intentionally simple: flat 20% equity vol, a generic correlation matrix, and no live provider wiring.",
    ]
    if market:
        caveats.append("The stored runtime market snapshot is preserved separately for replay, but it is not the same as the mock pricing market used for the deterministic report.")
    if learning.get("retrieved_lesson_ids") == [] and learning.get("captured_lesson_ids") == []:
        caveats.append("No lessons were retrieved or captured; this report is about deterministic assembly, not learning-loop improvement.")
    return [
        "## Scope and Caveats",
        *[f"- {item}" for item in caveats],
        "",
    ]


def _render_prompt(report: Mapping[str, Any]) -> list[str]:
    prompt = str(report.get("prompt") or "").strip()
    return [
        "## Prompt",
        "```text",
        prompt,
        "```",
        "",
    ]


def _render_decisions(report: Mapping[str, Any]) -> list[str]:
    deterministic = report.get("deterministic_decisions") or {}
    agent = report.get("agent_decisions") or {}
    return [
        "## Deterministic Decisions",
        _json_block(deterministic),
        "",
        "## Agent Decisions",
        _json_block(agent),
        "",
    ]


def _render_sections(report: Mapping[str, Any]) -> list[str]:
    semantic = report.get("semantic") or {}
    product_ir = report.get("product_ir") or {}
    semantic_trace = report.get("semantic_trace") or {}
    assembly = report.get("assembly") or {}
    build_observability = report.get("build_observability") or {}
    return [
        "## Semantic Contract",
        _json_block(semantic),
        "",
        "## ProductIR Decomposition",
        _json_block(product_ir),
        "",
        "## Semantic Trace",
        _json_block(semantic_trace),
        "",
        "## Build Result",
        _json_block(report.get("build_result") or {}),
        "",
        "## Build Observability",
        _json_block(build_observability),
        "",
        "## Pricer Assembly",
        _json_block(assembly),
        "",
    ]


def _render_pricing(report: Mapping[str, Any]) -> list[str]:
    pricing = report.get("pricing") or {}
    greeks = pricing.get("greeks") or {}
    spot_deltas = dict(greeks.get("spot_deltas") or {})
    delta_note = []
    if spot_deltas and all(abs(float(value)) < 1e-12 for value in spot_deltas.values()):
        delta_note = [
            "",
            "- Spot deltas are zero in this setup because the payoff is expressed in relative-return units and is scale-invariant to the starting spot levels.",
        ]
    return [
        "## Mock Pricing Run",
        _json_block({k: v for k, v in pricing.items() if k != "greeks"}),
        "",
        "## Final Price and Greeks",
        _bullet_block(
            [
                f"clean_price: `{_format_number(pricing.get('clean_price'))}`",
                f"dirty_price: `{_format_number(pricing.get('dirty_price'))}`",
                f"accrued_interest: `{_format_number(pricing.get('accrued_interest'))}`",
            ]
        ),
        *delta_note,
        "",
        _json_block(greeks),
        "",
    ]


def _render_repro(report: Mapping[str, Any]) -> list[str]:
    repro = report.get("reproducibility") or {}
    return [
        "## Reproducibility",
        _json_block(repro),
        "",
    ]


def _bullet_block(lines: Sequence[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _json_block(data: Any) -> str:
    return "```json\n" + json.dumps(data, indent=2, sort_keys=True, default=str) + "\n```"


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.10f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


__all__ = ["render_proving_run_report"]
