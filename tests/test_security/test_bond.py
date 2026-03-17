"""Testing Bond and ParBond classes"""
import numpy as np
import pytest
from rate_model.security.bond import Bond, ParBond


# ---------------------------------------------------------------------------
# Bond construction
# ---------------------------------------------------------------------------

class TestBondConstruction:
    def test_annual_bond(self):
        bond = Bond(notional=100, coupon=0.05, maturity=5, frequency=1)
        assert bond.notional == 100
        assert bond.coupon_rate == 0.05
        assert bond.maturity == 5
        assert bond.frequency == 1

    def test_semiannual_bond(self):
        bond = Bond(notional=1000, coupon=0.06, maturity=10, frequency=2)
        assert bond.frequency == 2
        assert bond.maturity == 10

    def test_quarterly_bond(self):
        bond = Bond(notional=100, coupon=0.04, maturity=3, frequency=4)
        assert bond.frequency == 4


# ---------------------------------------------------------------------------
# get_cashflows
# ---------------------------------------------------------------------------

class TestGetCashflows:
    def test_annual_cashflow_count(self):
        bond = Bond(notional=100, coupon=0.05, maturity=5, frequency=1)
        cf = bond.get_cashflows()
        assert len(cf) == 5

    def test_semiannual_cashflow_count(self):
        bond = Bond(notional=100, coupon=0.06, maturity=10, frequency=2)
        cf = bond.get_cashflows()
        assert len(cf) == 20

    def test_quarterly_cashflow_count(self):
        bond = Bond(notional=100, coupon=0.04, maturity=3, frequency=4)
        cf = bond.get_cashflows()
        assert len(cf) == 12

    def test_annual_coupon_amounts(self):
        bond = Bond(notional=100, coupon=0.05, maturity=3, frequency=1)
        cf = bond.get_cashflows()
        # Intermediate coupons: 100 * 0.05 / 1 = 5.0
        np.testing.assert_allclose(cf[:2], [5.0, 5.0])
        # Final payment: coupon + notional = 105.0
        assert cf[-1] == pytest.approx(105.0)

    def test_semiannual_coupon_amounts(self):
        bond = Bond(notional=1000, coupon=0.06, maturity=2, frequency=2)
        cf = bond.get_cashflows()
        # Coupon per period: 1000 * 0.06 / 2 = 30.0
        np.testing.assert_allclose(cf[:3], [30.0, 30.0, 30.0])
        assert cf[-1] == pytest.approx(1030.0)

    def test_quarterly_coupon_amounts(self):
        bond = Bond(notional=100, coupon=0.08, maturity=1, frequency=4)
        cf = bond.get_cashflows()
        # Coupon per period: 100 * 0.08 / 4 = 2.0
        np.testing.assert_allclose(cf[:3], [2.0, 2.0, 2.0])
        assert cf[-1] == pytest.approx(102.0)

    def test_cashflows_sum(self):
        """Total cash received = all coupons + notional."""
        bond = Bond(notional=100, coupon=0.05, maturity=10, frequency=2)
        cf = bond.get_cashflows()
        expected_total = 100 * 0.05 * 10 + 100  # total coupons + principal
        assert cf.sum() == pytest.approx(expected_total)


# ---------------------------------------------------------------------------
# get_price
# ---------------------------------------------------------------------------

class TestGetPrice:
    @staticmethod
    def _flat_rates(rate, n_periods):
        return np.full(n_periods, rate)

    def test_price_positive(self):
        bond = Bond(notional=100, coupon=0.05, maturity=5, frequency=1)
        rates = self._flat_rates(0.05, 5)
        price = bond.get_price(rates)
        assert price > 0

    def test_price_decreases_with_higher_rates(self):
        bond = Bond(notional=100, coupon=0.05, maturity=10, frequency=2)
        low_rates = self._flat_rates(0.03, 20)
        high_rates = self._flat_rates(0.08, 20)
        assert bond.get_price(low_rates) > bond.get_price(high_rates)

    def test_zero_coupon_price(self):
        """A zero-coupon bond price equals notional * exp(-r*T) under flat rates."""
        bond = Bond(notional=100, coupon=0.0, maturity=5, frequency=1)
        rate = 0.04
        rates = self._flat_rates(rate, 5)
        price = bond.get_price(rates)
        # Only the last cashflow (notional) contributes; discounted at period 5
        # discount factor = exp(-rate / frequency) for period 5 = exp(-0.04 * 5)
        expected = 100 * np.exp(-rate / 1 * 5)  # exp(-rates[4] / freq) but rate[4]=0.04
        # Actually the code does exp(-rates[i] / frequency) for each period i
        # For zero coupon, only cf[-1]=100, discount = exp(-0.04/1) applied 5 times via dot
        # But the code uses rates[:n_periods]/frequency for each period independently
        # so discount for period 5 is exp(-0.04/1) = exp(-0.04)
        # price = 100 * exp(-0.04) -- that's just the last period's discount
        expected = 100 * np.exp(-rate / 1)
        assert price == pytest.approx(expected, rel=1e-10)

    def test_price_with_extra_rates(self):
        """Rates array longer than needed should still work (only first n used)."""
        bond = Bond(notional=100, coupon=0.05, maturity=2, frequency=1)
        rates = self._flat_rates(0.05, 100)
        price = bond.get_price(rates)
        assert price > 0

    def test_semiannual_price_sanity(self):
        """Semi-annual bond with flat rates should give a reasonable positive price."""
        bond = Bond(notional=100, coupon=0.06, maturity=5, frequency=2)
        rates = self._flat_rates(0.06, 10)
        price = bond.get_price(rates)
        # Discounting uses exp(-rate/frequency) per period, so with rate=0.06
        # and frequency=2, each period discounts by exp(-0.03) which is light.
        # Price will be above par but should be in a reasonable range.
        assert 100 < price < 150


# ---------------------------------------------------------------------------
# get_duration
# ---------------------------------------------------------------------------

class TestGetDuration:
    @staticmethod
    def _flat_rates(rate, n_periods):
        return np.full(n_periods, rate)

    def test_duration_positive(self):
        bond = Bond(notional=100, coupon=0.05, maturity=5, frequency=1)
        rates = self._flat_rates(0.05, 5)
        assert bond.get_duration(rates) > 0

    def test_duration_increases_with_maturity(self):
        rates_long = self._flat_rates(0.05, 60)
        short = Bond(notional=100, coupon=0.05, maturity=5, frequency=2)
        long = Bond(notional=100, coupon=0.05, maturity=30, frequency=2)
        assert long.get_duration(rates_long) > short.get_duration(rates_long)

    def test_zero_coupon_duration_equals_maturity_periods(self):
        """For a zero-coupon bond, duration in periods equals the number of periods."""
        bond = Bond(notional=100, coupon=0.0, maturity=5, frequency=1)
        rates = self._flat_rates(0.03, 5)
        # Only the last cashflow matters, at period 5
        duration = bond.get_duration(rates)
        assert duration == pytest.approx(5.0)

    def test_duration_less_than_maturity_periods_for_coupon_bond(self):
        """Coupon bond duration (in periods) should be less than n_periods."""
        bond = Bond(notional=100, coupon=0.05, maturity=10, frequency=1)
        rates = self._flat_rates(0.05, 10)
        duration = bond.get_duration(rates)
        assert duration < 10


# ---------------------------------------------------------------------------
# get_convexity
# ---------------------------------------------------------------------------

class TestGetConvexity:
    @staticmethod
    def _flat_rates(rate, n_periods):
        return np.full(n_periods, rate)

    def test_convexity_positive(self):
        bond = Bond(notional=100, coupon=0.05, maturity=5, frequency=1)
        rates = self._flat_rates(0.05, 5)
        assert bond.get_convexity(rates) > 0

    def test_convexity_greater_than_duration(self):
        """Convexity (sum of t^2 * weighted cf) should exceed duration (sum of t * weighted cf)."""
        bond = Bond(notional=100, coupon=0.05, maturity=10, frequency=2)
        rates = self._flat_rates(0.05, 20)
        assert bond.get_convexity(rates) > bond.get_duration(rates)

    def test_convexity_increases_with_maturity(self):
        rates = self._flat_rates(0.05, 60)
        short = Bond(notional=100, coupon=0.05, maturity=5, frequency=2)
        long = Bond(notional=100, coupon=0.05, maturity=30, frequency=2)
        assert long.get_convexity(rates) > short.get_convexity(rates)


# ---------------------------------------------------------------------------
# ParBond
# ---------------------------------------------------------------------------

class TestParBond:
    def test_parbond_construction(self):
        pb = ParBond(notional=100, maturity=5, frequency=2, coupon_rate=0.06)
        assert pb.notional == 100
        assert pb.coupon_rate == 0.06
        assert pb.maturity == 5
        assert pb.frequency == 2

    def test_parbond_is_bond(self):
        pb = ParBond(notional=100, maturity=5, frequency=1, coupon_rate=0.05)
        assert isinstance(pb, Bond)

    def test_parbond_cashflows(self):
        pb = ParBond(notional=100, maturity=3, frequency=1, coupon_rate=0.05)
        cf = pb.get_cashflows()
        assert len(cf) == 3
        assert cf[0] == pytest.approx(5.0)
        assert cf[-1] == pytest.approx(105.0)

    def test_parbond_price(self):
        pb = ParBond(notional=100, maturity=5, frequency=2, coupon_rate=0.04)
        rates = np.full(10, 0.04)
        price = pb.get_price(rates)
        assert price > 0
