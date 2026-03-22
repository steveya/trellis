"""XV4: European/American option cross-validation against QuantLib and FinancePy."""

from datetime import date

import numpy as raw_np
import pytest

# --- Trellis ---
from trellis.models.black import black76_call, black76_put
from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.backward_induction import backward_induction
from trellis.models.calibration.implied_vol import implied_vol, _bs_price

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0


# ---------------------------------------------------------------------------
# QuantLib references
# ---------------------------------------------------------------------------

def ql_european_call():
    import QuantLib as ql
    today = ql.Date(15, 11, 2024)
    ql.Settings.instance().evaluationDate = today
    maturity = ql.Date(15, 11, 2025)

    spot = ql.SimpleQuote(S0)
    rate_ts = ql.FlatForward(today, r, ql.Actual365Fixed())
    vol_ts = ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
    div_ts = ql.FlatForward(today, 0.0, ql.Actual365Fixed())

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(div_ts),
        ql.YieldTermStructureHandle(rate_ts),
        ql.BlackVolTermStructureHandle(vol_ts),
    )

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.NPV()


def ql_american_put():
    import QuantLib as ql
    today = ql.Date(15, 11, 2024)
    ql.Settings.instance().evaluationDate = today
    maturity = ql.Date(15, 11, 2025)

    spot = ql.SimpleQuote(S0)
    rate_ts = ql.FlatForward(today, r, ql.Actual365Fixed())
    vol_ts = ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
    div_ts = ql.FlatForward(today, 0.0, ql.Actual365Fixed())

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(div_ts),
        ql.YieldTermStructureHandle(rate_ts),
        ql.BlackVolTermStructureHandle(vol_ts),
    )

    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    exercise = ql.AmericanExercise(today, maturity)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", 500))
    return option.NPV()


# ---------------------------------------------------------------------------
# FinancePy references
# ---------------------------------------------------------------------------

def fp_european_call():
    from financepy.products.equity.equity_vanilla_option import EquityVanillaOption
    from financepy.utils.date import Date as FPDate
    from financepy.utils.global_types import OptionTypes
    from financepy.models.black_scholes import BlackScholes
    from financepy.market.curves.discount_curve_flat import DiscountCurveFlat

    valuation_dt = FPDate(15, 11, 2024)
    expiry_dt = FPDate(15, 11, 2025)
    option = EquityVanillaOption(expiry_dt, K, OptionTypes.EUROPEAN_CALL)
    model = BlackScholes(sigma)
    discount_curve = DiscountCurveFlat(valuation_dt, r)
    dividend_curve = DiscountCurveFlat(valuation_dt, 0.0)
    return option.value(valuation_dt, S0, discount_curve, dividend_curve, model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEuropeanOptionCrossVal:

    def test_trellis_bs_vs_quantlib(self):
        """Trellis BS price matches QuantLib analytical."""
        trellis_call = _bs_price(S0, K, T, r, sigma, "call")
        ql_call = ql_european_call()
        assert trellis_call == pytest.approx(ql_call, rel=0.01), (
            f"Trellis={trellis_call:.4f}, QL={ql_call:.4f}"
        )

    def test_trellis_bs_vs_financepy(self):
        """Trellis BS matches FinancePy."""
        trellis_call = _bs_price(S0, K, T, r, sigma, "call")
        fp_call = fp_european_call()
        assert trellis_call == pytest.approx(fp_call, rel=0.01)

    def test_quantlib_vs_financepy(self):
        ql_call = ql_european_call()
        fp_call = fp_european_call()
        assert ql_call == pytest.approx(fp_call, rel=0.01)


class TestAmericanOptionCrossVal:

    def test_trellis_tree_vs_quantlib(self):
        """Trellis CRR tree American put vs QuantLib CRR."""
        tree = BinomialTree.crr(S0, T, 500, r, sigma)
        def put_payoff(step, node):
            return max(K - tree.value_at(step, node), 0)
        def exercise_val(step, node, t):
            return max(K - t.value_at(step, node), 0)
        trellis_amer = backward_induction(tree, put_payoff, r, "american",
                                           exercise_value_fn=exercise_val)
        ql_amer = ql_american_put()
        assert trellis_amer == pytest.approx(ql_amer, rel=0.02), (
            f"Trellis={trellis_amer:.4f}, QL={ql_amer:.4f}"
        )

    def test_american_put_geq_european(self):
        """American put ≥ European put (both libs agree)."""
        tree = BinomialTree.crr(S0, T, 500, r, sigma)
        def put_payoff(step, node):
            return max(K - tree.value_at(step, node), 0)
        def exercise_val(step, node, t):
            return max(K - t.value_at(step, node), 0)
        euro = backward_induction(tree, put_payoff, r, "european")
        amer = backward_induction(tree, put_payoff, r, "american",
                                   exercise_value_fn=exercise_val)
        assert amer >= euro - 0.01


class TestImpliedVolCrossVal:

    def test_round_trip_vs_quantlib(self):
        """Implied vol round-trip: Trellis and QuantLib should agree on vol."""
        price = _bs_price(S0, K, T, r, sigma, "call")
        trellis_iv = implied_vol(price, S0, K, T, r, "call")
        assert trellis_iv == pytest.approx(sigma, rel=1e-4)

        # QuantLib IV
        import QuantLib as ql
        today = ql.Date(15, 11, 2024)
        ql.Settings.instance().evaluationDate = today
        rate_ts = ql.FlatForward(today, r, ql.Actual365Fixed())
        div_ts = ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        maturity = ql.Date(15, 11, 2025)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)

        process = ql.GeneralizedBlackScholesProcess(
            ql.QuoteHandle(ql.SimpleQuote(S0)),
            ql.YieldTermStructureHandle(div_ts),
            ql.YieldTermStructureHandle(rate_ts),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), 0.01, ql.Actual365Fixed())
            ),
        )
        ql_iv = option.impliedVolatility(price, process)
        assert trellis_iv == pytest.approx(ql_iv, rel=1e-3)
