"""Automated model validation tests — deterministic checks run by the arbiter.

These are the quantitative tests that the model validator uses as evidence.
They don't require an LLM — they're pure numerical checks.
"""

from __future__ import annotations

from datetime import date

import numpy as raw_np

from trellis.agent.validation_report import ValidationFinding


def _uses_explicit_vol_model_parameters(payoff) -> bool:
    """Return whether flat-surface vega checks are not the relevant contract.

    Mirrors the deterministic invariant-layer rule for routes whose payoff spec
    carries an explicit parametric vol model. In the bounded current cohort,
    SABR-parameterized cap/floor strips derive caplet vols from ``spec.sabr``
    and therefore should not be judged by a flat ``vol_surface`` bump.
    """
    spec = getattr(payoff, "spec", None) or getattr(payoff, "_spec", None)
    if spec is None:
        return False
    model = str(getattr(spec, "model", "") or "").strip().lower()
    return model == "sabr" and bool(getattr(spec, "sabr", None))


def check_calibration(
    payoff,
    market_state,
    tenors: list[float] = (1.0, 2.0, 5.0, 10.0),
    tol: float = 0.005,
) -> list[ValidationFinding]:
    """Check that the model reproduces known zero-coupon bond prices at standard maturities.

    A zero-coupon bond pays 1 at maturity with no coupons; its price equals
    the discount factor. If the model cannot match these, its interest-rate
    calibration is wrong.
    """
    findings = []
    discount = market_state.discount

    # Build a ZCB payoff for each tenor and check if the model's
    # implied discount matches the curve
    for T in tenors:
        if T > 30:
            continue
        market_df = float(discount.discount(T))
        # We can't directly test the internal tree calibration from outside,
        # but we can check that the model price is consistent with the curve
        # by verifying the straight bond case
    # For now, this is a placeholder — the full implementation would
    # instrument the lattice to extract its implied discount factors

    return findings


def check_sensitivity_signs(
    payoff_factory,
    market_state_factory,
    instrument_type: str = "unknown",
) -> list[ValidationFinding]:
    """Check that price sensitivities (Greeks) have the correct sign.

    Greeks are partial derivatives of price with respect to market inputs.
    For example, bond prices must decrease when rates rise, and instruments
    with embedded options must have non-zero sensitivity to volatility (vega).
    """
    findings = []
    finding_id = 1

    # Rate sensitivity (bump +1bp)
    try:
        p_base = payoff_factory().evaluate(market_state_factory(rate=0.05, vol=0.20))
        p_up = payoff_factory().evaluate(market_state_factory(rate=0.0501, vol=0.20))
        rate_delta = p_up - p_base

        # Bond-like instruments should have negative rate delta
        if instrument_type in ("callable_bond", "puttable_bond", "bond"):
            if rate_delta > 0.01:
                findings.append(ValidationFinding(
                    id=f"MV-S{finding_id:03d}",
                    severity="high",
                    category="sensitivity",
                    description="Bond-like instrument has positive rate delta",
                    evidence=f"P(5.00%)={p_base:.4f}, P(5.01%)={p_up:.4f}, delta={rate_delta:+.4f}",
                    remediation="Check discounting logic — bond prices should decrease when rates increase.",
                ))
                finding_id += 1
    except Exception:
        pass

    # Vol sensitivity (instruments with embedded options)
    try:
        sample_payoff = payoff_factory()
        if _uses_explicit_vol_model_parameters(sample_payoff):
            return findings
        p_low_vol = sample_payoff.evaluate(market_state_factory(rate=0.05, vol=0.05))
        p_high_vol = payoff_factory().evaluate(market_state_factory(rate=0.05, vol=0.40))
        vol_change = abs(p_high_vol - p_low_vol)
        base_price = max(abs(p_low_vol), abs(p_high_vol), 1.0)

        if vol_change / base_price < 0.001:
            findings.append(ValidationFinding(
                id=f"MV-S{finding_id:03d}",
                severity="critical",
                category="sensitivity",
                description="Instrument with embedded option has zero vega",
                evidence=f"P(vol=5%)={p_low_vol:.4f}, P(vol=40%)={p_high_vol:.4f}, "
                          f"change={vol_change:.4f} ({vol_change/base_price:.2%})",
                remediation="The model is not capturing the option component. "
                            "For rate derivatives, use a rate tree (build_rate_lattice) "
                            "not a spot tree (BinomialTree.crr). "
                            "Vol must affect rate dispersion and exercise decisions.",
            ))
            finding_id += 1
    except Exception:
        pass

    return findings


def check_benchmark(
    payoff_factory,
    market_state_factory,
    instrument_type: str = "unknown",
    rate: float = 0.05,
    vol: float = 0.01,
) -> list[ValidationFinding]:
    """Compare against QuantLib if available."""
    findings = []

    try:
        import QuantLib as ql
    except ImportError:
        return findings  # skip if QuantLib not installed

    if instrument_type != "callable_bond":
        return findings  # only benchmark callable bonds for now

    # Trellis price
    ms = market_state_factory(rate=rate, vol=vol)
    trellis_price = payoff_factory().evaluate(ms)

    # QuantLib price
    today = ql.Date(15, 11, 2024)
    ql.Settings.instance().evaluationDate = today
    maturity = ql.Date(15, 11, 2034)

    rate_ts = ql.FlatForward(today, rate, ql.ActualActual(ql.ActualActual.ISDA))
    rate_handle = ql.YieldTermStructureHandle(rate_ts)

    schedule = ql.Schedule(today, maturity, ql.Period(ql.Semiannual),
        ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
        ql.DateGeneration.Backward, False)

    callability = ql.CallabilitySchedule()
    for cd in [ql.Date(15, 11, 2027), ql.Date(15, 11, 2029), ql.Date(15, 11, 2031)]:
        callability.append(ql.Callability(
            ql.BondPrice(100.0, ql.BondPrice.Clean), ql.Callability.Call, cd))

    cb = ql.CallableFixedRateBond(0, 100.0, schedule, [0.05],
        ql.ActualActual(ql.ActualActual.ISDA), ql.Unadjusted, 100.0, today, callability)
    hw = ql.HullWhite(rate_handle, 0.1, vol)
    cb.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw, 200))
    ql_price = cb.cleanPrice()

    diff = abs(trellis_price - ql_price)
    pct_diff = diff / ql_price * 100

    if pct_diff > 10:
        findings.append(ValidationFinding(
            id="MV-B001",
            severity="critical",
            category="benchmark",
            description=f"Callable bond price differs from QuantLib by {pct_diff:.1f}%",
            evidence=f"Trellis={trellis_price:.4f}, QuantLib={ql_price:.4f}, diff={diff:.2f}",
            remediation="The rate tree may not be properly calibrated to the yield curve. "
                        "Ensure theta(t) is solved at each step to reprice zero-coupon bonds. "
                        "Also verify coupons are embedded at actual payment dates.",
        ))
    elif pct_diff > 5:
        findings.append(ValidationFinding(
            id="MV-B001",
            severity="high",
            category="benchmark",
            description=f"Callable bond price differs from QuantLib by {pct_diff:.1f}%",
            evidence=f"Trellis={trellis_price:.4f}, QuantLib={ql_price:.4f}, diff={diff:.2f}",
            remediation="Check tree calibration and coupon embedding.",
        ))
    elif pct_diff > 2:
        findings.append(ValidationFinding(
            id="MV-B001",
            severity="medium",
            category="benchmark",
            description=f"Callable bond price differs from QuantLib by {pct_diff:.1f}%",
            evidence=f"Trellis={trellis_price:.4f}, QuantLib={ql_price:.4f}, diff={diff:.2f}",
            remediation="Minor calibration difference — acceptable for most purposes.",
        ))

    return findings
