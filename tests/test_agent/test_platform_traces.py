"""Tests for platform trace persistence and replay metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

def test_platform_trace_round_trip_preserves_simulation_identity(tmp_path):
    from trellis.agent.platform_requests import (
        CompiledPlatformRequest,
        ExecutionPlan,
        PlatformRequest,
    )
    from trellis.agent.platform_traces import load_platform_traces, record_platform_trace

    request = PlatformRequest(
        request_id="executor_build_20260328_seeded",
        request_type="build",
        entry_point="executor",
        description="Price a Monte Carlo basket",
        instrument_type="basket_option",
        metadata={
            "runtime_contract": {
                "simulation_identity": {
                    "seed": 271828,
                    "seed_source": "task.simulation_seed",
                    "sample_source": {
                        "kind": "market_snapshot",
                        "source": "mock",
                    },
                    "sample_indexing": {
                        "kind": "path_index",
                        "ordering": "simulation_generation_order",
                        "start": 0,
                    },
                    "simulation_stream_id": "executor_build_20260328_seeded:abc123",
                    "replay_key": "executor_build_20260328_seeded:abc123",
                },
                "simulation_seed": 271828,
            }
        },
    )
    compiled = CompiledPlatformRequest(
        request=request,
        market_snapshot=None,
        execution_plan=ExecutionPlan(
            action="build_then_price",
            reason="semantic_contract_request",
            route_method="monte_carlo",
            requires_build=True,
        ),
    )

    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )
    traces = load_platform_traces(root=tmp_path)

    assert trace_path.exists()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.simulation_seed == 271828
    assert trace.simulation_identity["seed"] == 271828
    assert trace.simulation_identity["seed_source"] == "task.simulation_seed"
    assert trace.sample_source["kind"] == "market_snapshot"
    assert trace.sample_indexing["kind"] == "path_index"
    assert trace.simulation_stream_id == "executor_build_20260328_seeded:abc123"


def test_platform_trace_writer_normalizes_tuples_for_yaml_round_trip(tmp_path):
    from trellis.agent.platform_traces import load_platform_traces, record_platform_trace

    request = SimpleNamespace(
        request_id="executor_build_tuple_safe",
        request_type="build",
        entry_point="executor",
        metadata={
            "runtime_contract": {
                "evaluation_tags": ("task_runtime", "semantic_contract"),
                "simulation_identity": {
                    "seed": 123,
                    "sample_source": {"kind": "market_snapshot"},
                    "sample_indexing": {"kind": "path_index"},
                },
            }
        },
        instrument_type="basket_option",
    )
    compiled = SimpleNamespace(
        request=request,
        execution_plan=SimpleNamespace(
            action="build_then_price",
            route_method="monte_carlo",
            measures=(),
            requires_build=True,
        ),
        pricing_plan=SimpleNamespace(sensitivity_support=None),
        product_ir=SimpleNamespace(instrument="basket_path_payoff"),
        blocker_report=None,
        knowledge_summary={},
    )

    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )
    raw = Path(trace_path).read_text()
    traces = load_platform_traces(root=tmp_path)

    assert "!!python/tuple" not in raw
    assert len(traces) == 1
    assert traces[0].request_id == "executor_build_tuple_safe"
