"""Tests for platform trace persistence and replay metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


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
    assert traces[0].events == ()


def test_platform_trace_writes_summary_yaml_and_append_only_events_log(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import (
        load_platform_trace_events,
        load_platform_trace_payload,
        load_platform_traces,
        record_platform_trace,
    )

    compiled = compile_build_request(
        "Build a pricer for: American put option on equity",
        instrument_type="american_option",
        preferred_method="monte_carlo",
    )

    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )

    summary = yaml.safe_load(Path(trace_path).read_text())
    events_path = Path(trace_path).with_suffix(".events.ndjson")
    payload = load_platform_trace_payload(trace_path)
    traces = load_platform_traces(root=tmp_path)
    events = load_platform_trace_events(trace_path)

    assert summary.get("events") in (None, [])
    assert events_path.exists()
    assert len(events_path.read_text().splitlines()) == 1
    assert [event.event for event in events] == ["request_succeeded"]
    assert [event["event"] for event in payload["events"]] == ["request_succeeded"]
    assert traces[0].events == ()


def test_platform_trace_payload_reads_legacy_inline_events(tmp_path):
    from trellis.agent.platform_traces import (
        load_platform_trace_events,
        load_platform_trace_payload,
    )

    trace_path = tmp_path / "legacy_trace.yaml"
    trace_path.write_text(
        yaml.safe_dump(
            {
                "request_id": "legacy_trace",
                "request_type": "price",
                "entry_point": "session",
                "action": "price_existing_instrument",
                "status": "failed",
                "outcome": "price_failed",
                "success": False,
                "timestamp": "2026-04-04T12:00:00+00:00",
                "updated_at": "2026-04-04T12:00:01+00:00",
                "events": [
                    {
                        "event": "request_compiled",
                        "status": "ok",
                        "timestamp": "2026-04-04T12:00:00+00:00",
                        "details": {"action": "price_existing_instrument"},
                    },
                    {
                        "event": "request_failed",
                        "status": "error",
                        "timestamp": "2026-04-04T12:00:01+00:00",
                        "details": {"error": "boom"},
                    },
                ],
            },
            sort_keys=False,
        )
    )

    payload = load_platform_trace_payload(trace_path)
    events = load_platform_trace_events(trace_path)

    assert [event["event"] for event in payload["events"]] == [
        "request_compiled",
        "request_failed",
    ]
    assert [event.event for event in events] == [
        "request_compiled",
        "request_failed",
    ]


def test_platform_trace_writer_migrates_legacy_inline_events_on_append(tmp_path):
    from trellis.agent.platform_traces import (
        append_platform_trace_event,
        load_platform_trace_payload,
    )

    trace_path = tmp_path / "executor_build_legacy.yaml"
    trace_path.write_text(
        yaml.safe_dump(
            {
                "request_id": "executor_build_legacy",
                "request_type": "build",
                "entry_point": "executor",
                "action": "build_then_price",
                "status": "running",
                "outcome": "",
                "success": None,
                "timestamp": "2026-04-04T12:00:00+00:00",
                "updated_at": "2026-04-04T12:00:00+00:00",
                "events": [
                    {
                        "event": "request_compiled",
                        "status": "ok",
                        "timestamp": "2026-04-04T12:00:00+00:00",
                        "details": {"action": "build_then_price"},
                    }
                ],
            },
            sort_keys=False,
        )
    )

    compiled = SimpleNamespace(
        request=SimpleNamespace(
            request_id="executor_build_legacy",
            request_type="build",
            entry_point="executor",
            instrument_type="european_option",
            metadata={},
        ),
        execution_plan=SimpleNamespace(
            action="build_then_price",
            route_method="analytical",
            measures=(),
            requires_build=True,
        ),
        pricing_plan=SimpleNamespace(sensitivity_support=None),
        product_ir=SimpleNamespace(instrument="european_option"),
        blocker_report=None,
        knowledge_summary={},
    )

    append_platform_trace_event(
        compiled,
        "request_failed",
        status="error",
        success=False,
        outcome="request_failed",
        details={"error": "boom"},
        root=tmp_path,
    )

    payload = load_platform_trace_payload(trace_path)
    summary = yaml.safe_load(Path(trace_path).read_text())
    events_path = trace_path.with_suffix(".events.ndjson")

    assert summary.get("events") in (None, [])
    assert events_path.exists()
    assert [event["event"] for event in payload["events"]] == [
        "request_compiled",
        "request_failed",
    ]


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


def test_platform_trace_summarizes_event_aware_pde_family_ir(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import load_platform_trace_boundary, record_platform_trace

    compiled = compile_build_request(
        (
            "Build a pricer for: Callable bond: HW rate PDE (PSOR) vs HW tree\n\n"
            "Price a 10-year callable bond paying a 5% semi-annual coupon, par $100,\n"
            "callable at par on any coupon date after year 3.\n"
            "Use the USD OIS discount curve from the market snapshot (as_of 2024-11-15).\n"
            "Hull-White model: mean reversion a=0.05, short-rate vol sigma=0.01.\n"
            "Method 1: solve the HW rate PDE backward in time using PSOR to\n"
            "enforce the call constraint (issuer calls when continuation value > par)."
        ),
        instrument_type="callable_bond",
        preferred_method="pde_solver",
    )

    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )
    boundary = load_platform_trace_boundary(trace_path)
    lowering = boundary["generation_boundary"]["lowering"]
    construction_identity = boundary["generation_boundary"]["construction_identity"]
    family_ir_summary = lowering["family_ir_summary"]

    assert lowering["family_ir_type"] == "EventAwarePDEIR"
    assert construction_identity["primary_kind"] == "backend_binding"
    assert construction_identity["route_alias"] == "pde_theta_1d"
    assert construction_identity["backend_binding_id"] == "trellis.models.callable_bond_pde.price_callable_bond_pde"
    assert family_ir_summary["semantic_control_style"] == "issuer_min"
    assert family_ir_summary["helper_symbol"] == "price_callable_bond_pde"
    assert family_ir_summary["semantic_transform_kinds"] == ["add_cashflow", "project_min"]
    assert family_ir_summary["compatibility_status"] == "native_event_aware"


def test_platform_trace_marks_vanilla_pde_wrapper_as_transitional(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import load_platform_trace_boundary, record_platform_trace

    compiled = compile_build_request(
        "European equity call on AAPL with strike 120 and expiry 2025-11-15",
        instrument_type="european_option",
        preferred_method="pde_solver",
    )

    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )
    boundary = load_platform_trace_boundary(trace_path)
    lowering = boundary["generation_boundary"]["lowering"]
    family_ir_summary = lowering["family_ir_summary"]

    assert lowering["family_ir_type"] == "VanillaEquityPDEIR"
    assert family_ir_summary["operator_family"] == "black_scholes_1d"
    assert family_ir_summary["control_style"] == "identity"
    assert family_ir_summary["semantic_control_style"] == "holder_max"
    assert family_ir_summary["compatibility_wrapper"] == "VanillaEquityPDEIR"
    assert family_ir_summary["compatibility_status"] == "transitional_wrapper"
    assert family_ir_summary["end_state"] == "migrate_to_plain_EventAwarePDEIR"


def test_platform_trace_summarizes_terminal_only_event_aware_monte_carlo_family_ir(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import load_platform_trace_boundary, record_platform_trace

    compiled = compile_build_request(
        "European equity call on AAPL with strike 120 and expiry 2025-11-15",
        instrument_type="european_option",
        preferred_method="monte_carlo",
    )

    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )
    boundary = load_platform_trace_boundary(trace_path)
    lowering = boundary["generation_boundary"]["lowering"]
    family_ir_summary = lowering["family_ir_summary"]

    assert lowering["family_ir_type"] == "EventAwareMonteCarloIR"
    assert family_ir_summary["process_family"] == "gbm_1d"
    assert family_ir_summary["control_style"] == "identity"
    assert family_ir_summary["semantic_control_style"] == "holder_max"
    assert family_ir_summary["path_requirement_kind"] == "terminal_only"
    assert family_ir_summary["reducer_kind"] == "terminal_payoff"
    assert family_ir_summary["event_kinds"] == []
    assert family_ir_summary["semantic_event_kinds"] == []
    assert family_ir_summary["compatibility_status"] == "native_event_aware"


def test_platform_trace_summary_reads_legacy_top_level_route_binding_fields():
    from trellis.agent.platform_traces import _construction_identity_summary, _generation_boundary_summary

    lowering = _generation_boundary_summary(
        SimpleNamespace(
            request_metadata={
                "semantic_blueprint": {},
                "route_binding_authority": {
                    "route_id": "analytical_black76",
                    "route_family": "",
                    "engine_family": "analytical",
                    "exact_backend_fit": True,
                },
            }
        ),
        request_metadata={
            "semantic_blueprint": {},
            "route_binding_authority": {
                "route_id": "analytical_black76",
                "route_family": "",
                "engine_family": "analytical",
                "exact_backend_fit": True,
            },
        },
    )["lowering"]
    construction_identity = _construction_identity_summary(
        lane_plan={},
        lowering=lowering,
        route_binding_authority={
            "route_id": "analytical_black76",
            "engine_family": "analytical",
            "exact_backend_fit": True,
            "backend_binding": {
                "binding_id": "trellis.models.rate_style_swaption.price_swaption_black76",
            },
        },
    )

    assert lowering["route_family"] == "analytical"
    assert construction_identity["primary_kind"] == "backend_binding"
    assert construction_identity["backend_exact_fit"] is True
    assert construction_identity["backend_engine_family"] == "analytical"


def test_platform_trace_keeps_route_less_semantic_requests_truthful(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import (
        load_platform_trace_boundary,
        load_platform_traces,
        record_platform_trace,
    )

    compiled = compile_build_request(
        (
            "Range accrual note on SOFR paying 5.25% when SOFR stays between 1.50% "
            "and 3.25% on 2026-01-15, 2026-04-15, 2026-07-15, and 2026-10-15."
        ),
        instrument_type="range_accrual",
    )

    trace_path = record_platform_trace(
        compiled,
        success=False,
        outcome="request_blocked",
        root=tmp_path,
    )
    boundary = load_platform_trace_boundary(trace_path)
    traces = load_platform_traces(root=tmp_path)

    assert len(traces) == 1
    trace = traces[0]
    assert trace.route_method == "analytical"
    assert trace.generation_boundary["method"] == "analytical"
    assert trace.generation_boundary["lowering"]["route_id"] is None
    assert boundary["generation_boundary"]["lowering"]["route_id"] is None
    assert trace.generation_boundary["lowering"]["route_family"] is None
    assert trace.generation_boundary["construction_identity"]["primary_kind"] == "lane_family"
    assert trace.generation_boundary["construction_identity"]["primary_label"] == "analytical"
    assert trace.generation_boundary["primitive_plan"] == {}
    assert trace.generation_boundary["route_binding_authority"] == {}
    assert trace.validation_contract["route_id"] is None


@pytest.mark.parametrize(
    (
        "description",
        "instrument_type",
        "expected_semantic_id",
        "expected_bridge_status",
        "expected_route_id",
        "expected_route_alias",
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
            None,
            "trellis.models.black",
            None,
        ),
        (
            "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
            "quanto_option",
            "quanto_option",
            "canonical_semantic",
            "quanto_adjustment_analytical",
            None,
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
    expected_route_alias,
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
    construction_identity = trace.generation_boundary["construction_identity"]
    assert construction_identity["primary_kind"] == "backend_binding"
    assert construction_identity["backend_exact_fit"] is True
    assert construction_identity["route_alias"] == expected_route_alias
    assert trace.generation_boundary["lowering"]["route_id"] == expected_route_id
    assert boundary["generation_boundary"]["lowering"]["route_id"] == expected_route_id
    assert boundary["generation_boundary"]["construction_identity"]["route_alias"] == expected_route_alias
    assert trace.generation_boundary["route_binding_authority"]["route_id"] == expected_route_id
    assert boundary["generation_boundary"]["route_binding_authority"]["route_id"] == expected_route_id
    assert trace.generation_boundary["route_binding_authority"]["authority_kind"] == "exact_backend_fit"
    assert (
        trace.generation_boundary["route_binding_authority"]["backend_binding"]["engine_family"]
        in {"analytical", "monte_carlo"}
    )
    assert (
        trace.generation_boundary["route_binding_authority"]["validation_bundle_id"]
        == trace.validation_contract["bundle_id"]
    )
    assert expected_module in trace.generation_boundary["approved_modules"]
    assert expected_module in boundary["generation_boundary"]["approved_modules"]
    if expected_helper is not None:
        assert expected_helper in trace.generation_boundary["lowering"]["helper_refs"]
        assert expected_helper in boundary["generation_boundary"]["lowering"]["helper_refs"]
        assert expected_helper in trace.generation_boundary["route_binding_authority"]["backend_binding"]["helper_refs"]
