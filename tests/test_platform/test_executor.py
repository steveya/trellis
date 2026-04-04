"""Tests for the governed platform executor skeleton."""

from __future__ import annotations

from datetime import date

import pytest


SETTLE = date(2024, 11, 15)


def _curve():
    from trellis.curves.yield_curve import YieldCurve

    return YieldCurve.flat(0.045)


def _bond():
    from trellis.instruments.bond import Bond

    return Bond(
        face=100,
        coupon=0.045,
        maturity_date=date(2034, 11, 15),
        maturity=10,
        frequency=2,
    )


def _book():
    from trellis.book import Book

    return Book({"10Y": _bond()})


def _compiled_request_for(action: str):
    from trellis.platform.requests import CompiledPlatformRequest, ExecutionPlan, PlatformRequest

    request = PlatformRequest(
        request_id=f"request_{action}",
        request_type="price",
        entry_point="session",
        settlement=date(2026, 4, 3),
        instrument_type="bond",
    )
    return CompiledPlatformRequest(
        request=request,
        market_snapshot=None,
        execution_plan=ExecutionPlan(
            action=action,
            reason="unit_test",
            route_method="analytical",
            requires_build=False,
            measures=("price",),
        ),
    )


def test_default_execution_handlers_cover_current_action_space():
    from trellis.platform.executor import REQUIRED_EXECUTION_ACTIONS, default_execution_handlers

    assert set(default_execution_handlers()) == set(REQUIRED_EXECUTION_ACTIONS)


def test_execute_compiled_request_dispatches_registered_handler_and_returns_stable_envelope():
    from trellis.platform.context import build_execution_context
    from trellis.platform.executor import execute_compiled_request
    from trellis.platform.runs import ArtifactReference

    compiled_request = _compiled_request_for("price_existing_instrument")
    context = build_execution_context(
        session_id="sess_executor_dispatch",
        market_source="mock",
        run_mode="sandbox",
        default_output_mode="structured",
    )

    def handler(compiled_request, execution_context, run_id):
        assert compiled_request.execution_plan.action == "price_existing_instrument"
        assert execution_context.session_id == "sess_executor_dispatch"
        assert run_id
        return {
            "status": "succeeded",
            "result_payload": {"price": 101.25},
            "warnings": ("used_test_handler",),
            "artifacts": (
                ArtifactReference(
                    artifact_id="trace",
                    artifact_kind="platform_trace",
                    uri="/tmp/request_price_existing_instrument.yaml",
                ),
            ),
            "audit_summary": {"handler": "test"},
            "provenance": {"engine_id": "pricing_engine.local"},
            "trace_path": "/tmp/request_price_existing_instrument.yaml",
        }

    result = execute_compiled_request(
        compiled_request,
        context,
        handlers={"price_existing_instrument": handler},
    )

    assert result.status == "succeeded"
    assert result.action == "price_existing_instrument"
    assert result.output_mode == "structured"
    assert result.result_payload["price"] == 101.25
    assert result.warnings == ("used_test_handler",)
    assert result.provenance["run_mode"] == "sandbox"
    assert result.provenance["engine_id"] == "pricing_engine.local"
    assert result.policy_outcome["allowed"] is True
    assert result.trace_path == "/tmp/request_price_existing_instrument.yaml"
    assert result.artifacts[0].artifact_kind == "platform_trace"


def test_execute_compiled_request_returns_blocked_result_when_policy_denies_execution():
    from trellis.platform.context import ExecutionContext, ProviderBindings
    from trellis.platform.executor import execute_compiled_request

    compiled_request = _compiled_request_for("compile_only")
    context = ExecutionContext(
        session_id="sess_executor_policy_block",
        run_mode="production",
        provider_bindings=ProviderBindings(),
    )

    result = execute_compiled_request(compiled_request, context)

    assert result.status == "blocked"
    assert result.result_payload["reason"] == "policy_blocked"
    assert "provider_binding_required" in result.policy_outcome["blocker_codes"]
    assert "missing_provenance_field" in result.policy_outcome["blocker_codes"]


def test_execute_compiled_request_converts_handler_exceptions_into_failed_results():
    from trellis.platform.context import build_execution_context
    from trellis.platform.executor import execute_compiled_request

    compiled_request = _compiled_request_for("analyze_existing_instrument")
    context = build_execution_context(
        session_id="sess_executor_failure",
        market_source="mock",
        run_mode="sandbox",
    )

    def handler(compiled_request, execution_context, run_id):
        raise RuntimeError("boom")

    result = execute_compiled_request(
        compiled_request,
        context,
        handlers={"analyze_existing_instrument": handler},
    )

    assert result.status == "failed"
    assert result.result_payload["error"] == "boom"
    assert result.result_payload["error_type"] == "RuntimeError"
    assert result.policy_outcome["allowed"] is True


def test_execute_compiled_request_prices_direct_instrument_via_default_adapter():
    from trellis.core.types import PricingResult
    from trellis.agent.platform_requests import compile_platform_request
    from trellis.platform.executor import execute_compiled_request
    from trellis.session import Session

    session = Session(curve=_curve(), settlement=SETTLE)
    compiled_request = compile_platform_request(
        session.to_platform_request(
            _bond(),
            request_type="price",
            measures=["price", "dv01"],
        )
    )

    result = execute_compiled_request(
        compiled_request,
        session.to_execution_context(run_mode="sandbox"),
    )

    assert result.status == "succeeded"
    assert isinstance(result.result_payload["result"], PricingResult)
    assert result.result_payload["result"].clean_price > 0
    assert "dv01" in result.result_payload["result"].greeks


def test_execute_compiled_request_prices_book_via_default_adapter():
    from trellis.book import BookResult
    from trellis.agent.platform_requests import compile_platform_request
    from trellis.platform.executor import execute_compiled_request
    from trellis.session import Session

    session = Session(curve=_curve(), settlement=SETTLE)
    compiled_request = compile_platform_request(
        session.to_platform_request(
            book=_book(),
            request_type="price",
            measures=["price", "dv01"],
        )
    )

    result = execute_compiled_request(
        compiled_request,
        session.to_execution_context(run_mode="sandbox"),
    )

    assert result.status == "succeeded"
    assert isinstance(result.result_payload["result"], BookResult)
    assert result.result_payload["result"].book_dv01 > 0


def test_execute_compiled_request_computes_greeks_via_default_adapter():
    from trellis.agent.platform_requests import compile_platform_request
    from trellis.platform.executor import execute_compiled_request
    from trellis.session import Session

    session = Session(curve=_curve(), settlement=SETTLE)
    compiled_request = compile_platform_request(
        session.to_platform_request(
            _bond(),
            request_type="greeks",
            measures=["dv01"],
        )
    )

    result = execute_compiled_request(
        compiled_request,
        session.to_execution_context(run_mode="sandbox"),
    )

    assert result.status == "succeeded"
    assert result.result_payload["result"]["dv01"] > 0


def test_execute_compiled_request_analyzes_existing_instrument_via_default_adapter():
    from trellis.agent.platform_requests import compile_platform_request
    from trellis.analytics.result import AnalyticsResult
    from trellis.core.payoff import DeterministicCashflowPayoff
    from trellis.platform.executor import execute_compiled_request
    from trellis.session import Session

    session = Session(curve=_curve(), settlement=SETTLE)
    compiled_request = compile_platform_request(
        session.to_platform_request(
            DeterministicCashflowPayoff(_bond()),
            request_type="analytics",
            measures=["price", "duration"],
        )
    )

    result = execute_compiled_request(
        compiled_request,
        session.to_execution_context(run_mode="sandbox"),
    )

    assert result.status == "succeeded"
    assert isinstance(result.result_payload["result"], AnalyticsResult)
    assert result.result_payload["result"].price > 0
    assert result.result_payload["result"].duration > 0


def test_execute_compiled_request_prices_existing_payoff_via_default_adapter():
    from trellis.agent.ask import TermSheet
    from trellis.agent.platform_requests import compile_platform_request, make_term_sheet_request
    from trellis.platform.executor import execute_compiled_request
    from trellis.session import Session

    session = Session(curve=_curve(), settlement=SETTLE)
    request = make_term_sheet_request(
        description="10Y bond with 4.5% coupon",
        term_sheet=TermSheet(
            instrument_type="bond",
            notional=100.0,
            currency="USD",
            parameters={"coupon": 0.045, "maturity": 10},
            raw_description="10Y bond with 4.5% coupon",
        ),
        session=session,
        measures=["price"],
    )
    compiled_request = compile_platform_request(request)

    result = execute_compiled_request(
        compiled_request,
        session.to_execution_context(run_mode="sandbox"),
    )

    assert compiled_request.execution_plan.action == "price_existing_payoff"
    assert result.status == "succeeded"
    assert result.result_payload["price"] > 0
    assert result.result_payload["matched_existing"] is True
    assert result.result_payload["payoff_class"] == "DeterministicCashflowPayoff"


def test_execute_compiled_request_builds_and_prices_candidate_path_via_default_adapter():
    from types import SimpleNamespace

    from trellis.agent.term_sheet import TermSheet
    from trellis.platform.context import build_execution_context
    from trellis.platform.executor import execute_compiled_request
    from trellis.platform.requests import CompiledPlatformRequest, ExecutionPlan, PlatformRequest
    from trellis.session import Session

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    term_sheet = TermSheet(
        instrument_type="swaption",
        notional=1_000_000,
        parameters={"strike": 0.045, "maturity": 5},
        raw_description="1Y into 5Y payer swaption at 4.5%",
    )
    compiled_request = CompiledPlatformRequest(
        request=PlatformRequest(
            request_id="request_build_then_price",
            request_type="price",
            entry_point="ask",
            settlement=SETTLE,
            market_snapshot=session.market_snapshot,
            description=term_sheet.raw_description,
            instrument_type=term_sheet.instrument_type,
            requested_outputs=("price",),
            term_sheet=term_sheet,
        ),
        market_snapshot=session.market_snapshot,
        execution_plan=ExecutionPlan(
            action="build_then_price",
            reason="unit_test",
            route_method="rate_tree",
            requires_build=True,
            requested_outputs=("price",),
        ),
    )
    context = build_execution_context(
        session_id="sess_executor_build_then_price",
        market_snapshot=session.market_snapshot,
        run_mode="sandbox",
    )

    class GeneratedPayoff:
        requirements = {"discount_curve"}

        def __init__(self, spec=None):
            self.spec = spec

        def evaluate(self, market_state):
            return 42.0

    with __import__("unittest").mock.patch(
        "trellis.agent.executor.build_payoff",
        return_value=GeneratedPayoff,
    ):
        with __import__("unittest").mock.patch(
            "trellis.agent.planner.plan_build",
            return_value=SimpleNamespace(spec_schema=object()),
        ):
            with __import__("unittest").mock.patch(
                "trellis.agent.executor._make_test_payoff",
                return_value=GeneratedPayoff(),
            ):
                result = execute_compiled_request(compiled_request, context)

    assert result.status == "succeeded"
    assert result.result_payload["price"] == pytest.approx(42.0)
    assert result.result_payload["matched_existing"] is False
    assert result.result_payload["payoff_class"] == "GeneratedPayoff"
    assert "candidate_generation" in result.warnings


def test_execute_compiled_request_build_then_price_uses_config_default_model():
    from types import SimpleNamespace

    from trellis.agent.term_sheet import TermSheet
    from trellis.platform.context import build_execution_context
    from trellis.platform.executor import execute_compiled_request
    from trellis.platform.requests import CompiledPlatformRequest, ExecutionPlan, PlatformRequest
    from trellis.session import Session

    session = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
    term_sheet = TermSheet(
        instrument_type="swaption",
        notional=1_000_000,
        parameters={"strike": 0.045, "maturity": 5},
        raw_description="1Y into 5Y payer swaption at 4.5%",
    )
    compiled_request = CompiledPlatformRequest(
        request=PlatformRequest(
            request_id="request_build_then_price_default_model",
            request_type="price",
            entry_point="ask",
            settlement=SETTLE,
            market_snapshot=session.market_snapshot,
            description=term_sheet.raw_description,
            instrument_type=term_sheet.instrument_type,
            requested_outputs=("price",),
            term_sheet=term_sheet,
        ),
        market_snapshot=session.market_snapshot,
        execution_plan=ExecutionPlan(
            action="build_then_price",
            reason="unit_test",
            route_method="rate_tree",
            requires_build=True,
            requested_outputs=("price",),
        ),
    )
    context = build_execution_context(
        session_id="sess_executor_build_then_price_default_model",
        market_snapshot=session.market_snapshot,
        run_mode="sandbox",
    )

    class GeneratedPayoff:
        requirements = {"discount_curve"}

        def __init__(self, spec=None):
            self.spec = spec

        def evaluate(self, market_state):
            return 42.0

    observed = {}

    def fake_build_payoff(*args, **kwargs):
        observed["build_payoff_model"] = kwargs["model"]
        return GeneratedPayoff

    def fake_plan_build(*args, **kwargs):
        observed["plan_build_model"] = kwargs["model"]
        return SimpleNamespace(spec_schema=object())

    with __import__("unittest").mock.patch(
        "trellis.agent.config.get_default_model",
        return_value="configured-default-model",
    ):
        with __import__("unittest").mock.patch(
            "trellis.agent.executor.build_payoff",
            side_effect=fake_build_payoff,
        ):
            with __import__("unittest").mock.patch(
                "trellis.agent.planner.plan_build",
                side_effect=fake_plan_build,
            ):
                with __import__("unittest").mock.patch(
                    "trellis.agent.executor._make_test_payoff",
                    return_value=GeneratedPayoff(),
                ):
                    result = execute_compiled_request(compiled_request, context)

    assert result.status == "succeeded"
    assert observed["build_payoff_model"] == "configured-default-model"
    assert observed["plan_build_model"] == "configured-default-model"
