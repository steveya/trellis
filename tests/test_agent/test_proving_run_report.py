from __future__ import annotations

from trellis.agent.proving_run_report import render_proving_run_report


def test_render_proving_run_report_includes_key_sections():
    report = {
        "title": "QUA-284 Arbitrary-Derivative Proving Run",
        "task": {"id": "T999", "title": "Himalaya ranked observation basket"},
        "prompt": "Build a pricer for: Himalaya ranked observation basket",
        "deterministic_decisions": {"semantic_contract_id": "ranked_observation_basket"},
        "agent_decisions": {"route_family": "family-name-free semantic basket route"},
        "semantic": {"semantic_id": "ranked_observation_basket"},
        "product_ir": {"instrument_class": "basket_path_payoff"},
        "semantic_trace": {"task_id": "T999"},
        "build_observability": {"source_status": "sanitized"},
        "assembly": {"engine": "trellis.models.monte_carlo.engine.MonteCarloEngine"},
        "pricing": {
            "clean_price": 12.345678,
            "dirty_price": 12.345678,
            "accrued_interest": 0.0,
            "greeks": {"spot_deltas": {"AAPL": 0.1}},
            "seed": 20260328,
        },
        "reproducibility": {"seed": 20260328},
        "output_path": "docs/qua-284-arbitrary-derivative-proving-run.md",
    }

    text = render_proving_run_report(report)

    assert "# QUA-284 Arbitrary-Derivative Proving Run" in text
    assert "## Prompt" in text
    assert "## Deterministic Decisions" in text
    assert "## Agent Decisions" in text
    assert "## Semantic Contract" in text
    assert "## ProductIR Decomposition" in text
    assert "## Semantic Trace" in text
    assert "## Build Observability" in text
    assert "## Pricer Assembly" in text
    assert "## Mock Pricing Run" in text
    assert "## Final Price and Greeks" in text
    assert "12.345678" in text
