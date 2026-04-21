"""Tests for the model validation agent."""

from datetime import date
from types import SimpleNamespace

import pytest

from trellis.agent.validation_report import ValidationFinding, ValidationReport
from trellis.agent.validation_tests import check_sensitivity_signs, check_benchmark
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _ms(rate=0.05, vol=0.20):
    return MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(rate),
        vol_surface=FlatVol(vol),
    )


class TestValidationReport:

    def test_no_findings_approved(self):
        report = ValidationReport(instrument="bond", method="analytical")
        assert report.approved is False  # not yet approved (no explicit approval)
        assert not report.has_blockers

    def test_critical_finding_blocks(self):
        report = ValidationReport(instrument="callable_bond", method="rate_tree")
        report.findings.append(ValidationFinding(
            id="MV-001", severity="critical", category="sensitivity",
            description="Zero vega", evidence="...", remediation="...",
        ))
        assert report.has_blockers

    def test_medium_finding_does_not_block(self):
        report = ValidationReport(instrument="callable_bond", method="rate_tree")
        report.findings.append(ValidationFinding(
            id="MV-001", severity="medium", category="calibration",
            description="Minor diff", evidence="...", remediation="...",
        ))
        assert not report.has_blockers

    def test_summary(self):
        report = ValidationReport(instrument="test", method="test")
        report.findings.append(ValidationFinding(
            id="MV-001", severity="critical", category="test",
            description="x", evidence="y", remediation="z",
        ))
        assert "BLOCKED" in report.summary()
        assert "1 critical" in report.summary()

    def test_format_for_builder(self):
        report = ValidationReport(instrument="test", method="test")
        report.findings.append(ValidationFinding(
            id="MV-001", severity="high", category="calibration",
            description="Tree uncalibrated", evidence="diff=14pt",
            remediation="Solve for theta(t)",
        ))
        feedback = report.format_for_builder()
        assert "MV-001" in feedback
        assert "theta" in feedback


class TestSensitivityValidation:

    def test_vol_insensitive_is_critical(self):
        """A callable bond with zero vega gets a critical finding."""
        class FakeCallable:
            @property
            def requirements(self):
                return {"discount_curve", "black_vol_surface"}
            def evaluate(self, ms):
                return 100.0 * ms.discount.discount(5.0)

        findings = check_sensitivity_signs(
            lambda: FakeCallable(), _ms,
            instrument_type="callable_bond",
        )
        critical = [f for f in findings if f.severity == "critical"]
        assert len(critical) > 0
        assert "vega" in critical[0].description.lower()

    def test_vol_sensitive_cap_passes(self):
        """A cap (properly vol-sensitive) should have no sensitivity findings."""
        from trellis.instruments.cap import CapFloorSpec, CapPayoff
        from trellis.core.types import Frequency

        spec = CapFloorSpec(
            notional=1e6, strike=0.05,
            start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
        )
        findings = check_sensitivity_signs(
            lambda: CapPayoff(spec), _ms,
            instrument_type="cap",
        )
        critical = [f for f in findings if f.severity == "critical"]
        assert len(critical) == 0

    def test_explicit_sabr_cap_strip_skips_flat_surface_vega_contract(self):
        """Explicit SABR strips should not be flagged for zero flat-surface vega."""
        from trellis.conventions.day_count import DayCountConvention
        from trellis.core.types import Frequency
        from trellis.curves.date_aware_flat_curve import DateAwareFlatYieldCurve
        from trellis.instruments.cap import CapFloorSpec
        from trellis.models.rate_cap_floor import price_rate_cap_floor_strip_analytical

        class _SabrCapStripPayoff:
            def __init__(self, spec):
                self._spec = spec

            @property
            def spec(self):
                return self._spec

            @property
            def requirements(self):
                return {"discount_curve", "forward_curve", "black_vol_surface"}

            def evaluate(self, market_state):
                return price_rate_cap_floor_strip_analytical(
                    market_state,
                    self._spec,
                    instrument_class="cap",
                )

        curve = DateAwareFlatYieldCurve(
            value_date=SETTLE,
            flat_rate=0.0425,
            curve_day_count=DayCountConvention.ACT_ACT_ISDA,
        )
        spec = CapFloorSpec(
            notional=1_000_000.0,
            strike=0.04,
            start_date=SETTLE,
            end_date=date(2029, 11, 15),
            frequency=Frequency.QUARTERLY,
            day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
            model="sabr",
            sabr={"alpha": 0.025, "beta": 0.5, "rho": -0.2, "nu": 0.35},
        )

        def _sabr_market_state(rate=0.05, vol=0.20):
            del rate, vol
            return MarketState(
                as_of=SETTLE,
                settlement=SETTLE,
                discount=curve,
                forecast_curves={"USD-SOFR-3M": curve},
                vol_surface=FlatVol(0.20),
                model_parameters={
                    "sabr": {
                        "alpha": 0.025,
                        "beta": 0.5,
                        "rho": -0.2,
                        "nu": 0.35,
                    },
                },
            )

        findings = check_sensitivity_signs(
            lambda: _SabrCapStripPayoff(spec),
            _sabr_market_state,
            instrument_type="cap",
        )

        critical = [f for f in findings if f.severity == "critical"]
        assert critical == []


class TestBenchmarkValidation:

    def test_benchmark_callable_vs_quantlib(self):
        """Run the QuantLib benchmark for callable bonds."""
        from trellis.instruments.callable_bond import CallableBondPayoff, CallableBondSpec
        spec = CallableBondSpec(
            notional=100, coupon=0.05,
            start_date=SETTLE, end_date=date(2034, 11, 15),
            call_dates=[date(2027, 11, 15), date(2029, 11, 15), date(2031, 11, 15)],
        )
        findings = check_benchmark(
            lambda: CallableBondPayoff(spec), _ms,
            instrument_type="callable_bond",
        )
        # We expect findings because our tree isn't fully calibrated
        # The key test: the benchmark runs without error and produces findings
        assert isinstance(findings, list)
        if findings:
            assert findings[0].category == "benchmark"


def test_llm_conceptual_review_includes_shared_knowledge(monkeypatch):
    from trellis.agent.model_validator import _llm_conceptual_review

    captured = {}

    monkeypatch.setattr(
        "trellis.agent.config.get_default_model",
        lambda: "fake-model",
    )

    def fake_llm_generate_json(prompt, model=None):
        captured["prompt"] = prompt
        return []

    monkeypatch.setattr("trellis.agent.config.llm_generate_json", fake_llm_generate_json)

    findings = _llm_conceptual_review(
        code="def price():\n    return 0.0",
        instrument_type="callable_bond",
        method="rate_tree",
        knowledge_context="## Shared Review Principles\n- Check calibration first.",
        model="fake-model",
    )

    assert findings == []
    assert "Shared Review Principles" in captured["prompt"]
    assert "Check calibration first." in captured["prompt"]


def test_llm_conceptual_review_includes_compiled_route_contract(monkeypatch):
    from trellis.agent.model_validator import _llm_conceptual_review
    from trellis.agent.platform_requests import compile_build_request

    captured = {}

    monkeypatch.setattr(
        "trellis.agent.config.get_default_model",
        lambda: "fake-model",
    )

    def fake_llm_generate_json(prompt, model=None):
        captured["prompt"] = prompt
        return []

    monkeypatch.setattr("trellis.agent.config.llm_generate_json", fake_llm_generate_json)

    compiled = compile_build_request(
        "European equity call on AAPL with strike 120 and expiry 2025-11-15",
        instrument_type="european_option",
        model="claude-sonnet-4-6",
    )

    findings = _llm_conceptual_review(
        code="def price():\n    return 0.0",
        instrument_type="european_option",
        method="analytical",
        generation_plan=compiled.generation_plan,
        model="fake-model",
    )

    assert findings == []
    assert "Compiled Route Contract" in captured["prompt"]
    assert "bridge=`thin_compatibility_wrapper`" in captured["prompt"]
    assert "family_ir=`AnalyticalBlack76IR`" in captured["prompt"]
    assert "route_alias=`analytical_black76`" not in captured["prompt"]
    assert "bundle=`analytical:european_option`" in captured["prompt"]


def test_llm_conceptual_review_includes_residual_risk_focus(monkeypatch):
    from trellis.agent.model_validator import _llm_conceptual_review

    captured = {}

    monkeypatch.setattr(
        "trellis.agent.config.get_default_model",
        lambda: "fake-model",
    )

    def fake_llm_generate_json(prompt, model=None):
        captured["prompt"] = prompt
        return []

    monkeypatch.setattr("trellis.agent.config.llm_generate_json", fake_llm_generate_json)

    findings = _llm_conceptual_review(
        code="def price():\n    return 0.0",
        instrument_type="callable_bond",
        method="rate_tree",
        residual_risks=("unsupported_paths_declared", "measure_support_warnings_present"),
        review_reason="validation_contract_residual_risks_present",
        model="fake-model",
    )

    assert findings == []
    assert "Residual conceptual review only" in captured["prompt"]
    assert "unsupported_paths_declared" in captured["prompt"]
    assert "measure_support_warnings_present" in captured["prompt"]
    assert "Do not repeat deterministic checks" in captured["prompt"]


def test_determine_review_policy_skips_llm_for_low_risk_supported_vanilla():
    from trellis.agent.review_policy import determine_review_policy

    policy = determine_review_policy(
        validation="thorough",
        method="analytical",
        product_ir=SimpleNamespace(
            instrument="european_option",
            payoff_traits=(),
            exercise_style="european",
            state_dependence="terminal_markov",
            schedule_dependence=False,
            model_family="equity_diffusion",
            unresolved_primitives=(),
            supported=True,
        ),
    )

    assert policy.risk_level == "low"
    assert policy.run_critic is False
    assert policy.run_model_validator_llm is False
    assert policy.critic_reason == "low_risk_supported_vanilla_analytical"
    assert policy.critic_mode == "skip"
    assert policy.critic_allow_text_fallback is False


def test_determine_review_policy_keeps_llm_for_high_risk_route():
    from trellis.agent.review_policy import determine_review_policy

    policy = determine_review_policy(
        validation="thorough",
        method="rate_tree",
        product_ir=SimpleNamespace(
            instrument="callable_bond",
            payoff_traits=("callable",),
            exercise_style="issuer_call",
            state_dependence="schedule_dependent",
            schedule_dependence=True,
            model_family="interest_rate",
            unresolved_primitives=(),
            supported=True,
        ),
    )

    assert policy.risk_level == "high"
    assert policy.run_critic is True
    assert policy.run_model_validator_llm is True
    assert policy.critic_mode == "required"
    assert policy.critic_json_max_retries is None
    assert policy.critic_allow_text_fallback is True


def test_determine_review_policy_escalates_on_validation_contract_residual_risk():
    from trellis.agent.review_policy import determine_review_policy
    from trellis.agent.validation_contract import CompiledValidationContract

    policy = determine_review_policy(
        validation="standard",
        method="analytical",
        product_ir=SimpleNamespace(
            instrument="european_option",
            payoff_traits=(),
            exercise_style="european",
            state_dependence="terminal_markov",
            schedule_dependence=False,
            model_family="equity_diffusion",
            unresolved_primitives=(),
            supported=True,
        ),
        validation_contract=CompiledValidationContract(
            contract_id="analytical:european_option",
            instrument_type="european_option",
            method="analytical",
            bundle_id="analytical:european_option",
            residual_risks=("comparison_relations_unspecified",),
        ),
    )

    assert policy.risk_level == "high"
    assert policy.run_critic is True
    assert policy.run_model_validator_llm is False
    assert policy.critic_reason == "validation_contract_residual_risks_present"
    assert policy.critic_mode == "advisory"


def test_determine_review_policy_blocks_on_validation_contract_lowering_errors():
    from trellis.agent.review_policy import determine_review_policy
    from trellis.agent.validation_contract import CompiledValidationContract

    policy = determine_review_policy(
        validation="thorough",
        method="analytical",
        product_ir=SimpleNamespace(
            instrument="european_option",
            payoff_traits=(),
            exercise_style="european",
            state_dependence="terminal_markov",
            schedule_dependence=False,
            model_family="equity_diffusion",
            unresolved_primitives=(),
            supported=True,
        ),
        validation_contract=CompiledValidationContract(
            contract_id="analytical:european_option",
            instrument_type="european_option",
            method="analytical",
            bundle_id="analytical:european_option",
            lowering_errors=("lowering:missing_target_binding",),
        ),
    )

    assert policy.risk_level == "blocked"
    assert policy.run_critic is False
    assert policy.run_model_validator_llm is False
    assert policy.critic_reason == "validation_contract_lowering_errors_present"
    assert policy.critic_mode == "skip"


def test_determine_review_policy_bounds_standard_critic_path():
    from trellis.agent.review_policy import determine_review_policy

    policy = determine_review_policy(
        validation="standard",
        method="monte_carlo",
        product_ir=SimpleNamespace(
            instrument="credit_default_swap",
            payoff_traits=("credit_sensitive",),
            exercise_style="none",
            state_dependence="path_dependent",
            schedule_dependence=True,
            model_family="generic",
            unresolved_primitives=(),
            supported=True,
        ),
    )

    assert policy.risk_level == "high"
    assert policy.run_critic is True
    assert policy.run_model_validator_llm is False
    assert policy.critic_mode == "advisory"
    assert policy.critic_json_max_retries == 0
    assert policy.critic_allow_text_fallback is False


def test_validate_model_skips_llm_review_for_low_risk_route(monkeypatch):
    from trellis.agent.model_validator import validate_model

    monkeypatch.setattr(
        "trellis.agent.model_validator.check_sensitivity_signs",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.model_validator.check_benchmark",
        lambda *args, **kwargs: [],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM review should be skipped")

    monkeypatch.setattr(
        "trellis.agent.model_validator._llm_conceptual_review",
        fail_if_called,
    )

    report = validate_model(
        payoff_factory=lambda: object(),
        market_state_factory=lambda rate, vol: object(),
        code="def price():\n    return 1.0",
        instrument_type="european_option",
        method="analytical",
        product_ir=SimpleNamespace(
            instrument="european_option",
            payoff_traits=(),
            exercise_style="european",
            state_dependence="terminal_markov",
            schedule_dependence=False,
            model_family="equity_diffusion",
            unresolved_primitives=(),
            supported=True,
        ),
    )

    assert report.findings == []
    assert report.approved is True


def test_validate_model_runs_llm_review_for_high_risk_route(monkeypatch):
    from trellis.agent.model_validator import validate_model

    monkeypatch.setattr(
        "trellis.agent.model_validator.check_sensitivity_signs",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.model_validator.check_benchmark",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.model_validator._llm_conceptual_review",
        lambda *args, **kwargs: [
            ValidationFinding(
                id="MV-L001",
                severity="high",
                category="conceptual",
                description="Need deeper review",
                evidence="high-risk callable route",
                remediation="keep LLM review enabled",
            )
        ],
    )

    report = validate_model(
        payoff_factory=lambda: object(),
        market_state_factory=lambda rate, vol: object(),
        code="def price():\n    return 1.0",
        instrument_type="callable_bond",
        method="rate_tree",
        product_ir=SimpleNamespace(
            instrument="callable_bond",
            payoff_traits=("callable",),
            exercise_style="issuer_call",
            state_dependence="schedule_dependent",
            schedule_dependence=True,
            model_family="interest_rate",
            unresolved_primitives=(),
            supported=True,
        ),
    )

    assert len(report.findings) == 1
    assert report.findings[0].id == "MV-L001"
    assert report.approved is False


def test_validate_model_threads_validation_contract_residual_risks(monkeypatch):
    from trellis.agent.model_validator import validate_model
    from trellis.agent.validation_contract import CompiledValidationContract

    monkeypatch.setattr(
        "trellis.agent.model_validator.check_sensitivity_signs",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.model_validator.check_benchmark",
        lambda *args, **kwargs: [],
    )

    captured = {}

    def fake_llm_review(*args, residual_risks=(), review_reason=None, **kwargs):
        captured["residual_risks"] = residual_risks
        captured["review_reason"] = review_reason
        return []

    monkeypatch.setattr(
        "trellis.agent.model_validator._llm_conceptual_review",
        fake_llm_review,
    )

    report = validate_model(
        payoff_factory=lambda: object(),
        market_state_factory=lambda rate, vol: object(),
        code="def price():\n    return 1.0",
        instrument_type="callable_bond",
        method="rate_tree",
        validation_contract=CompiledValidationContract(
            contract_id="rate_tree:callable_bond",
            instrument_type="callable_bond",
            method="rate_tree",
            bundle_id="rate_tree:callable_bond",
            residual_risks=("unsupported_paths_declared",),
        ),
        run_llm_review=True,
    )

    assert report.findings == []
    assert captured["residual_risks"] == ("unsupported_paths_declared",)
    assert captured["review_reason"] is None


def test_validate_model_for_request_threads_generation_plan(monkeypatch):
    from trellis.agent.model_validator import validate_model_for_request
    from trellis.agent.platform_requests import compile_build_request

    monkeypatch.setattr(
        "trellis.agent.model_validator.check_sensitivity_signs",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.model_validator.check_benchmark",
        lambda *args, **kwargs: [],
    )

    captured = {}

    def fake_llm_review(*args, generation_plan=None, **kwargs):
        captured["generation_plan"] = generation_plan
        return []

    monkeypatch.setattr(
        "trellis.agent.model_validator._llm_conceptual_review",
        fake_llm_review,
    )

    compiled = compile_build_request(
        "European equity call on AAPL with strike 120 and expiry 2025-11-15",
        instrument_type="european_option",
        model="claude-sonnet-4-6",
    )

    report = validate_model_for_request(
        compiled,
        payoff_factory=lambda: object(),
        market_state_factory=lambda rate, vol: object(),
        code="def price():\n    return 1.0",
    )

    assert report.findings == []
    assert captured["generation_plan"] == compiled.generation_plan
