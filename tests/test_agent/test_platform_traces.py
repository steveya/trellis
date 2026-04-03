"""Tests for platform trace persistence and replay metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


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


def test_platform_trace_persists_dsl_family_ir_summary(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import record_platform_trace

    compiled = compile_build_request(
        (
            "Single-name CDS on ACME with premium dates "
            "2026-06-20, 2026-09-20, 2026-12-20, 2027-03-20, 2027-06-20"
        ),
        instrument_type="credit_default_swap",
        preferred_method="analytical",
    )

    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )
    raw = Path(trace_path).read_text()

    assert "dsl_family_ir_type: CreditDefaultSwapIR" in raw
    assert "schedule_builder_symbol: build_cds_schedule" in raw


@pytest.mark.parametrize(
    (
        "description",
        "instrument_type",
        "expected_semantic_id",
        "expected_bridge_status",
        "expected_route_id",
        "expected_module",
        "expected_helper",
    ),
    [
        (
            "European equity call on AAPL with strike 120 and expiry 2025-11-15",
            "european_option",
            "vanilla_option",
            "thin_compatibility_wrapper",
            "analytical_black76",
            "trellis.models.black",
            None,
        ),
        (
            "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
            "quanto_option",
            "quanto_option",
            "canonical_semantic",
            "quanto_adjustment_analytical",
            "trellis.models.resolution.quanto",
            "trellis.models.analytical.quanto.price_quanto_option_analytical",
        ),
        (
            "Himalaya-style ranked observation basket on AAPL, MSFT, NVDA with observation dates "
            "2025-01-15, 2025-02-15, 2025-03-15. At each observation choose the best performer "
            "among the remaining constituents, remove it, lock the simple return, and settle the "
            "average locked returns at maturity.",
            "basket_option",
            "ranked_observation_basket",
            "thin_compatibility_wrapper",
            "correlated_basket_monte_carlo",
            "trellis.models.monte_carlo.semantic_basket",
            "trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo",
        ),
    ],
)
def test_platform_trace_persists_semantic_checkpoint_and_generation_boundary(
    tmp_path,
    description,
    instrument_type,
    expected_semantic_id,
    expected_bridge_status,
    expected_route_id,
    expected_module,
    expected_helper,
):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import (
        load_platform_trace_boundary,
        load_platform_traces,
        record_platform_trace,
    )

    compiled = compile_build_request(
        description,
        instrument_type=instrument_type,
    )
    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )

    boundary = load_platform_trace_boundary(trace_path)
    traces = load_platform_traces(root=tmp_path)

    assert len(traces) == 1
    trace = traces[0]
    assert trace.semantic_checkpoint["semantic_id"] == expected_semantic_id
    assert trace.semantic_checkpoint["compatibility_bridge_status"] == expected_bridge_status
    assert boundary["semantic_checkpoint"]["semantic_id"] == expected_semantic_id
    assert boundary["semantic_checkpoint"]["compatibility_bridge_status"] == expected_bridge_status
    assert trace.generation_boundary["lane_plan"]["lane_family"] in {
        "analytical",
        "monte_carlo",
    }
    assert trace.generation_boundary["lane_plan"]["plan_kind"] == "exact_target_binding"
    assert trace.generation_boundary["lowering"]["route_id"] == expected_route_id
    assert boundary["generation_boundary"]["lowering"]["route_id"] == expected_route_id
    assert trace.generation_boundary["route_binding_authority"]["route_id"] == expected_route_id
    assert boundary["generation_boundary"]["route_binding_authority"]["route_id"] == expected_route_id
    assert trace.generation_boundary["route_binding_authority"]["authority_kind"] == "exact_backend_fit"
    assert (
        trace.generation_boundary["route_binding_authority"]["validation_bundle_id"]
        == trace.validation_contract["bundle_id"]
    )
    assert expected_module in trace.generation_boundary["approved_modules"]
    assert expected_module in boundary["generation_boundary"]["approved_modules"]
    if expected_helper is not None:
        assert expected_helper in trace.generation_boundary["lowering"]["helper_refs"]
        assert expected_helper in boundary["generation_boundary"]["lowering"]["helper_refs"]
        assert expected_helper in trace.generation_boundary["route_binding_authority"]["helper_refs"]
