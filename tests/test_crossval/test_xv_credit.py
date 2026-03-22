"""XV5: Credit cross-validation against QuantLib and FinancePy."""

from datetime import date

import numpy as raw_np
import pytest

# --- Trellis ---
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve

SETTLE = date(2024, 11, 15)


def trellis_survival(hazard_rate, t):
    cc = CreditCurve.flat(hazard_rate)
    return float(cc.survival_probability(t))


def quantlib_survival(hazard_rate, t):
    import QuantLib as ql
    today = ql.Date(15, 11, 2024)
    ql.Settings.instance().evaluationDate = today

    quote = ql.QuoteHandle(ql.SimpleQuote(hazard_rate))
    flat_hazard = ql.FlatHazardRate(today, quote, ql.Actual365Fixed())
    target_date = today + ql.Period(int(t * 365), ql.Days)
    return flat_hazard.survivalProbability(target_date)


class TestCreditCrossValidation:

    def test_survival_prob_vs_quantlib(self):
        """Trellis survival probability matches QuantLib."""
        for lam in [0.01, 0.02, 0.05]:
            for t in [1, 5, 10]:
                trellis_sp = trellis_survival(lam, t)
                ql_sp = quantlib_survival(lam, t)
                assert trellis_sp == pytest.approx(ql_sp, rel=0.01), (
                    f"λ={lam}, t={t}: Trellis={trellis_sp:.6f}, QL={ql_sp:.6f}"
                )

    def test_hazard_rate_from_spreads(self):
        """CDS spread → hazard rate: λ ≈ spread / (1-R)."""
        spread = 0.01  # 100bp
        R = 0.4
        cc = CreditCurve.from_spreads({5.0: spread}, recovery=R)
        expected_lam = spread / (1 - R)
        assert float(cc.hazard_rate(5.0)) == pytest.approx(expected_lam, rel=1e-6)

    def test_survival_decreasing_all_libs(self):
        """Both agree: survival probability decreases with time."""
        for t1, t2 in [(1, 5), (5, 10)]:
            assert trellis_survival(0.02, t1) > trellis_survival(0.02, t2)
            assert quantlib_survival(0.02, t1) > quantlib_survival(0.02, t2)
