"""Tests for governed policy-bundle evaluation and execution guards."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot
from trellis.platform.context import (
    ProviderBinding,
    ProviderBindingSet,
    ProviderBindings,
    RunMode,
    build_execution_context,
)


SETTLE = date(2024, 11, 15)


def _context(
    *,
    run_mode: RunMode,
    provider_id: str | None,
    fallback_provider_id: str | None = None,
    allow_mock_data: bool | None = None,
    require_provider_disclosure: bool | None = None,
    policy_bundle_id: str | None = None,
):
    return build_execution_context(
        session_id="sess_policy_001",
        run_mode=run_mode,
        provider_bindings=ProviderBindings(
            market_data=ProviderBindingSet(
                primary=None if provider_id is None else ProviderBinding(provider_id),
                fallback=(
                    None
                    if fallback_provider_id is None
                    else ProviderBinding(fallback_provider_id)
                ),
            )
        ),
        allow_mock_data=(
            run_mode is RunMode.SANDBOX if allow_mock_data is None else allow_mock_data
        ),
        require_provider_disclosure=(
            run_mode is not RunMode.SANDBOX
            if require_provider_disclosure is None
            else require_provider_disclosure
        ),
        policy_bundle_id=policy_bundle_id,
    )


def _snapshot(
    *,
    provider_id: str,
    include_snapshot_id: bool = True,
) -> MarketSnapshot:
    source = "mock" if "mock" in provider_id else "treasury_gov"
    provenance = {
        "source": source,
        "source_kind": "provider_snapshot",
        "provider_id": provider_id,
    }
    if include_snapshot_id:
        provenance["snapshot_id"] = "snapshot_policy_001"
    return MarketSnapshot(
        as_of=SETTLE,
        source=source,
        discount_curves={"discount": YieldCurve.flat(0.045)},
        default_discount_curve="discount",
        provenance=provenance,
    )


def test_policy_bundle_round_trips_through_serialized_payload():
    from trellis.platform.policies import PolicyBundle

    bundle = PolicyBundle(
        policy_id="policy_bundle.production.default",
        name="Production Default",
        allowed_run_modes=("production",),
        allow_mock_data=False,
        require_provider_disclosure=True,
        allowed_model_statuses=("approved",),
        required_provenance_fields=("provider_id", "market_snapshot_id"),
    )

    restored = PolicyBundle.from_dict(bundle.to_dict())

    assert restored == bundle


def test_sandbox_policy_allows_explicit_mock_provider_when_context_allows_it():
    from trellis.platform.policies import evaluate_execution_policy

    result = evaluate_execution_policy(
        execution_context=_context(
            run_mode=RunMode.SANDBOX,
            provider_id="market_data.mock",
            allow_mock_data=True,
            require_provider_disclosure=False,
        ),
        market_snapshot=_snapshot(provider_id="market_data.mock"),
    )

    assert result.allowed is True
    assert result.blocker_codes == ()


def test_research_policy_blocks_missing_market_data_provider_disclosure():
    from trellis.platform.policies import evaluate_execution_policy

    result = evaluate_execution_policy(
        execution_context=_context(
            run_mode=RunMode.RESEARCH,
            provider_id=None,
            require_provider_disclosure=True,
        ),
    )

    assert result.allowed is False
    assert "provider_binding_required" in result.blocker_codes


def test_research_policy_blocks_mock_provider_outside_sandbox():
    from trellis.platform.policies import evaluate_execution_policy

    result = evaluate_execution_policy(
        execution_context=_context(
            run_mode=RunMode.RESEARCH,
            provider_id="market_data.mock_unit",
            allow_mock_data=True,
        ),
        market_snapshot=_snapshot(provider_id="market_data.mock_unit"),
    )

    assert result.allowed is False
    assert "mock_data_not_allowed" in result.blocker_codes


def test_production_policy_blocks_unapproved_models_and_missing_snapshot_provenance():
    from trellis.platform.policies import evaluate_execution_policy

    result = evaluate_execution_policy(
        execution_context=_context(
            run_mode=RunMode.PRODUCTION,
            provider_id="market_data.treasury_gov",
            allow_mock_data=False,
        ),
        market_snapshot=_snapshot(
            provider_id="market_data.treasury_gov",
            include_snapshot_id=False,
        ),
        selected_model={"model_id": "vanilla_option_analytical", "status": "validated"},
    )

    assert result.allowed is False
    assert "model_lifecycle_not_allowed" in result.blocker_codes
    assert "missing_provenance_field" in result.blocker_codes


def test_enforce_execution_policy_raises_structured_guard_error():
    from trellis.platform.policies import PolicyViolationError, enforce_execution_policy

    with pytest.raises(PolicyViolationError) as excinfo:
        enforce_execution_policy(
            execution_context=_context(
                run_mode=RunMode.RESEARCH,
                provider_id=None,
                require_provider_disclosure=True,
            ),
        )

    assert excinfo.value.evaluation.allowed is False
    assert excinfo.value.evaluation.blockers[0].code == "provider_binding_required"


def test_policy_outcome_can_be_attached_to_run_records():
    from trellis.platform.policies import evaluate_execution_policy
    from trellis.platform.runs import build_run_record

    context = _context(
        run_mode=RunMode.PRODUCTION,
        provider_id="market_data.treasury_gov",
        allow_mock_data=False,
    )
    evaluation = evaluate_execution_policy(
        execution_context=context,
        market_snapshot=_snapshot(provider_id="market_data.treasury_gov"),
        selected_model={"model_id": "vanilla_option_analytical", "status": "approved"},
    )

    record = build_run_record(
        run_id="run_policy_001",
        request_id="request_policy_001",
        status="blocked" if not evaluation.allowed else "ready",
        action="price_trade",
        execution_context=context,
        market_snapshot_id="snapshot_policy_001",
        policy_outcome=evaluation.to_dict(),
    )

    assert record.policy_outcome["allowed"] is True
    assert record.policy_outcome["policy_id"] == "policy_bundle.production.default"
