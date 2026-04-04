"""Tests for the unified platform request/compiler loop."""

from __future__ import annotations

import sys
import types
from datetime import date

from trellis.agent.ask import TermSheet
from trellis.book import Book
from trellis.instruments.bond import Bond
from trellis.pipeline import Pipeline
from trellis.session import Session


SETTLE = date(2024, 11, 15)


def _bond():
    return Bond(
        face=100,
        coupon=0.05,
        maturity_date=date(2034, 11, 15),
        maturity=10,
        frequency=2,
    )


def _book():
    return Book({"10Y": _bond()})


def test_compile_term_sheet_request_for_existing_cap_route():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_term_sheet_request,
    )

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    term_sheet = TermSheet(
        instrument_type="cap",
        notional=1_000_000,
        parameters={
            "strike": 0.04,
            "end_date": "2029-11-15",
            "frequency": "quarterly",
            "rate_index": "USD-SOFR-3M",
        },
    )

    request = make_term_sheet_request(
        description="5Y cap at 4% on $1M SOFR",
        term_sheet=term_sheet,
        session=session,
        measures=["price", "vega"],
    )
    compiled = compile_platform_request(request)

    assert compiled.execution_plan.action == "price_existing_payoff"
    assert compiled.execution_plan.route_method == "direct_existing"
    assert compiled.execution_plan.measures == ("price", "vega")
    assert compiled.market_snapshot is session.market_snapshot
    assert compiled.product_ir is not None
    assert compiled.generation_plan is None
    assert compiled.review_knowledge_text
    assert compiled.knowledge_summary["instrument"] == compiled.product_ir.instrument


def test_compile_user_defined_blocked_request_to_block_action():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_user_defined_request,
    )

    request = make_user_defined_request(
        """\
name: blocked_structured_product
payoff_family: composite_option
payoff_traits:
  - american
  - asian
  - barrier
  - stochastic_vol
exercise_style: american
schedule_dependence: false
state_dependence: path_dependent
model_family: stochastic_volatility
candidate_engine_families:
  - exercise
  - monte_carlo
required_market_data:
  - discount
  - black_vol
preferred_method: monte_carlo
""",
        request_type="build",
    )

    compiled = compile_platform_request(request)

    assert compiled.execution_plan.action == "block"
    assert compiled.blocker_report is not None
    assert compiled.new_primitive_workflow is not None
    assert compiled.generation_plan is not None


def test_pipeline_compile_request_uses_book_execution_plan():
    from trellis.agent.platform_requests import compile_platform_request

    pipeline = (
        Pipeline()
        .instruments(_book())
        .market_data(source="mock", as_of="2024-11-15")
        .compute(["price", "dv01"])
    )

    request = pipeline.compile_request()
    compiled = compile_platform_request(request)

    assert compiled.execution_plan.action == "price_book"
    assert compiled.execution_plan.measures == ("price", "dv01")
    assert compiled.market_snapshot is not None


def test_session_greeks_request_compiles_to_direct_execution_plan():
    from trellis.agent.platform_requests import compile_platform_request

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    request = session.to_platform_request(_bond(), request_type="greeks", measures=["dv01"])
    compiled = compile_platform_request(request)

    assert compiled.execution_plan.action == "compute_greeks"
    assert compiled.execution_plan.measures == ("dv01",)
    assert compiled.generation_plan is None


def test_compile_build_request_carries_shared_knowledge_views():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Build a pricer for: American put option on equity",
        instrument_type="american_option",
        preferred_method="monte_carlo",
        metadata={"task_id": "T14", "task_title": "American put: PSOR vs tree vs LSM three-way"},
    )

    assert compiled.knowledge_text
    assert compiled.review_knowledge_text
    assert compiled.routing_knowledge_text
    assert compiled.knowledge_summary["instrument"] == compiled.product_ir.instrument
    assert compiled.knowledge_summary["lesson_count"] >= 1
    assert compiled.knowledge_summary["prompt_sizes"]["builder"]["compact_chars"] > 0
    metadata = dict(compiled.request.metadata)
    assert metadata["task_id"] == "T14"
    assert metadata["task_title"] == "American put: PSOR vs tree vs LSM three-way"


def test_compile_build_request_records_sensitivity_support_for_requested_measures():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "Build a pricer for: Callable bond with issuer call schedule",
        instrument_type="callable_bond",
        measures=["price", "dv01", "duration"],
    )

    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == "rate_tree"
    assert compiled.pricing_plan.sensitivity_support is not None
    assert compiled.pricing_plan.sensitivity_support.level == "bump_only"
    assert "dv01" in compiled.pricing_plan.sensitivity_support.supported_measures


def test_compile_build_request_enriches_fx_analytical_route_inputs():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "FX vanilla option: Garman-Kohlhagen vs MC on EURUSD",
        instrument_type="european_option",
    )

    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == "analytical"
    assert "fx_rates" in compiled.pricing_plan.required_market_data
    assert "forward_curve" in compiled.pricing_plan.required_market_data
    assert compiled.generation_plan is not None
    assert compiled.generation_plan.primitive_plan is not None
    assert compiled.generation_plan.primitive_plan.route == "analytical_garman_kohlhagen"


def test_platform_trace_round_trip(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import (
        load_platform_trace_events,
        load_platform_traces,
        record_platform_trace,
        summarize_platform_traces,
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
    traces = load_platform_traces(root=tmp_path)
    summary = summarize_platform_traces(traces)

    assert trace_path.exists()
    assert len(traces) == 1
    assert traces[0].request_id == compiled.request.request_id
    assert summary["compile_only"] == 1
    assert [event.event for event in load_platform_trace_events(trace_path)] == ["request_succeeded"]
    assert traces[0].knowledge_summary["instrument"] == compiled.product_ir.instrument
    assert traces[0].knowledge_summary["lesson_count"] >= 1


def test_platform_trace_persists_sensitivity_support(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import load_platform_traces, record_platform_trace

    compiled = compile_build_request(
        "Build a pricer for: Callable bond with issuer call schedule",
        instrument_type="callable_bond",
        measures=["dv01"],
    )

    record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )
    traces = load_platform_traces(root=tmp_path)

    assert len(traces) == 1
    assert traces[0].sensitivity_support["level"] == "bump_only"
    assert "dv01" in traces[0].sensitivity_support["supported_measures"]


def test_platform_trace_appends_failure_events(tmp_path):
    from trellis.agent.platform_requests import compile_platform_request
    from trellis.agent.platform_traces import (
        append_platform_trace_event,
        load_platform_trace_events,
        load_platform_traces,
        record_platform_trace,
    )

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    request = session.to_platform_request(_bond(), request_type="price")
    compiled = compile_platform_request(request)

    append_platform_trace_event(
        compiled,
        "request_compiled",
        status="ok",
        details={"action": compiled.execution_plan.action},
        root=tmp_path,
    )
    record_platform_trace(
        compiled,
        success=False,
        outcome="price_failed",
        details={"error": "boom"},
        root=tmp_path,
    )

    traces = load_platform_traces(root=tmp_path)

    assert len(traces) == 1
    assert traces[0].status == "failed"
    assert traces[0].outcome == "price_failed"
    assert [event.event for event in load_platform_trace_events(tmp_path / f"{compiled.request.request_id}.yaml")] == [
        "request_compiled",
        "request_failed",
    ]
    assert traces[0].details["error"] == "boom"


def test_platform_trace_can_persist_token_usage(tmp_path):
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.platform_traces import (
        attach_platform_trace_token_usage,
        load_platform_traces,
        record_platform_trace,
    )

    compiled = compile_build_request(
        "Build a pricer for: European call option on equity",
        instrument_type="european_option",
        preferred_method="analytical",
    )

    trace_path = record_platform_trace(
        compiled,
        success=True,
        outcome="build_completed",
        root=tmp_path,
    )
    attach_platform_trace_token_usage(
        trace_path,
        {
            "call_count": 3,
            "calls_with_usage": 3,
            "calls_without_usage": 0,
            "prompt_tokens": 210,
            "completion_tokens": 95,
            "total_tokens": 305,
            "by_stage": {
                "decomposition": {"total_tokens": 40},
                "code_generation": {"total_tokens": 200},
                "reflection": {"total_tokens": 65},
            },
            "by_provider": {"anthropic": {"total_tokens": 305}},
        },
    )

    traces = load_platform_traces(root=tmp_path)

    assert traces[0].token_usage["total_tokens"] == 305
    assert traces[0].token_usage["by_stage"]["code_generation"]["total_tokens"] == 200


def test_platform_trace_can_link_linear_issue(tmp_path, monkeypatch):
    from trellis.agent.platform_requests import compile_platform_request
    from trellis.agent.platform_traces import (
        append_platform_trace_event,
        load_platform_traces,
    )

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    request = session.to_platform_request(_bond(), request_type="price")
    compiled = compile_platform_request(request)

    monkeypatch.setenv("TRELLIS_SYNC_REQUEST_ISSUES", "1")
    monkeypatch.setenv("LINEAR_API_KEY", "linear-test-key")
    monkeypatch.setenv("LINEAR_TEAM_ID", "QUA")

    calls = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        query = json["query"]
        if "TeamByKey" in query:
            return _FakeResponse(
                {
                    "data": {
                        "teams": {
                            "nodes": [
                                {"id": "8d2ff012-4369-49bc-b2d9-f2bcdb8bc7b6"}
                            ]
                        }
                    }
                }
            )
        if "issueCreate" in query:
            return _FakeResponse(
                {
                    "data": {
                        "issueCreate": {
                            "success": True,
                            "issue": {
                                "id": "issue-1",
                                "identifier": "TRE-123",
                                "url": "https://linear.app/trellis/issue/TRE-123",
                            },
                        }
                    }
                }
            )
        return _FakeResponse(
            {
                "data": {
                    "commentCreate": {
                        "success": True,
                        "comment": {"id": "comment-1"},
                    }
                }
            }
        )

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=fake_post))

    append_platform_trace_event(
        compiled,
        "build_started",
        status="info",
        details={"module_name": "trellis.instruments._agent.swaption"},
        root=tmp_path,
    )

    traces = load_platform_traces(root=tmp_path)

    assert traces[0].linear_issue_id == "issue-1"
    assert traces[0].linear_issue_identifier == "TRE-123"
    assert len(calls) == 3


def test_platform_trace_can_link_github_issue(tmp_path, monkeypatch):
    from trellis.agent.platform_requests import compile_platform_request
    from trellis.agent.platform_traces import (
        append_platform_trace_event,
        load_platform_traces,
    )

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    request = session.to_platform_request(_bond(), request_type="price")
    compiled = compile_platform_request(request)

    monkeypatch.setenv("TRELLIS_SYNC_REQUEST_ISSUES", "1")
    monkeypatch.setenv("GITHUB_TOKEN", "github-test-token")
    monkeypatch.setenv("GITHUB_REQUEST_AUDIT_REPOSITORY", "steveya/trellis")

    calls = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_request(method, url, headers=None, json=None, timeout=None):
        calls.append((method, url, json))
        if url.endswith("/issues"):
            return _FakeResponse(
                {
                    "id": 1001,
                    "number": 42,
                    "html_url": "https://github.com/steveya/trellis/issues/42",
                }
            )
        return _FakeResponse({"id": 2002})

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(request=fake_request))

    append_platform_trace_event(
        compiled,
        "build_started",
        status="info",
        details={"module_name": "trellis.instruments._agent.swaption"},
        root=tmp_path,
    )

    traces = load_platform_traces(root=tmp_path)

    assert traces[0].github_issue_number == 42
    assert traces[0].github_issue_repository == "steveya/trellis"
    assert traces[0].github_issue_url == "https://github.com/steveya/trellis/issues/42"
    assert len(calls) == 2


def test_platform_trace_issue_sync_disabled_by_default(tmp_path, monkeypatch):
    from trellis.agent.platform_requests import compile_platform_request
    from trellis.agent.platform_traces import (
        append_platform_trace_event,
        load_platform_traces,
    )

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    request = session.to_platform_request(_bond(), request_type="price")
    compiled = compile_platform_request(request)

    monkeypatch.delenv("TRELLIS_SYNC_REQUEST_ISSUES", raising=False)
    monkeypatch.setenv("LINEAR_API_KEY", "linear-test-key")
    monkeypatch.setenv("GITHUB_TOKEN", "github-test-token")

    def _unexpected(*args, **kwargs):
        raise AssertionError("external issue sync should be disabled by default")

    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(post=_unexpected, request=_unexpected),
    )

    append_platform_trace_event(
        compiled,
        "build_started",
        status="info",
        details={"module_name": "trellis.instruments._agent.swaption"},
        root=tmp_path,
    )

    traces = load_platform_traces(root=tmp_path)

    assert traces[0].linear_issue_id is None
    assert traces[0].github_issue_number is None


def test_compile_comparison_request_creates_method_specific_plans():
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_comparison_request,
    )

    request = make_comparison_request(
        description="European equity call: 5-way (tree, PDE, MC, FFT, COS)",
        instrument_type="european_option",
        methods=["rate_tree", "pde_solver", "monte_carlo", "fft_pricing"],
        reference_method="analytical",
        validation_targets={
            "internal": ("crr_tree", "bs_pde", "mc_exact", "fft", "cos"),
            "analytical": "black_scholes",
        },
    )

    compiled = compile_platform_request(request)

    assert compiled.execution_plan.action == "compare_methods"
    assert compiled.execution_plan.route_method == "comparison"
    assert compiled.comparison_spec is not None
    assert compiled.comparison_spec.method_families == (
        "rate_tree",
        "pde_solver",
        "monte_carlo",
        "fft_pricing",
    )
    assert compiled.comparison_spec.reference_method == "analytical"
    assert len(compiled.comparison_method_plans) == 4
    assert [plan.preferred_method for plan in compiled.comparison_method_plans] == [
        "rate_tree",
        "pde_solver",
        "monte_carlo",
        "fft_pricing",
    ]
    assert [plan.pricing_plan.method for plan in compiled.comparison_method_plans] == [
        "rate_tree",
        "pde_solver",
        "monte_carlo",
        "fft_pricing",
    ]


def test_compile_build_request_respects_preferred_method():
    from trellis.agent.platform_requests import compile_build_request

    compiled = compile_build_request(
        "European call option",
        instrument_type="european_option",
        preferred_method="monte_carlo",
    )

    assert compiled.pricing_plan is not None
    assert compiled.pricing_plan.method == "monte_carlo"
    assert compiled.execution_plan.route_method == "monte_carlo"
