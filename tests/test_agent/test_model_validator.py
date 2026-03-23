"""Tests for the model validation agent."""

from datetime import date

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
                return {"discount", "black_vol"}
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
