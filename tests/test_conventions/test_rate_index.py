"""Tests for rate index definitions."""

from trellis.conventions.day_count import DayCountConvention
from trellis.conventions.rate_index import (
    CDI, EURIBOR_3M, SOFR_ON, SOFR_3M, SONIA,
    Compounding, Currency, RateIndex,
)


class TestRateIndex:

    def test_sofr_conventions(self):
        assert SOFR_ON.currency == Currency.USD
        assert SOFR_ON.day_count == DayCountConvention.ACT_360
        assert SOFR_ON.compounding == Compounding.COMPOUNDED_DAILY
        assert SOFR_ON.is_overnight is True
        assert SOFR_ON.fixing_lag == 0

    def test_sonia_conventions(self):
        assert SONIA.currency == Currency.GBP
        assert SONIA.day_count == DayCountConvention.ACT_365
        assert SONIA.compounding == Compounding.COMPOUNDED_DAILY
        assert SONIA.is_overnight is True

    def test_euribor_conventions(self):
        assert EURIBOR_3M.currency == Currency.EUR
        assert EURIBOR_3M.day_count == DayCountConvention.ACT_360
        assert EURIBOR_3M.compounding == Compounding.SIMPLE
        assert EURIBOR_3M.fixing_lag == 2

    def test_cdi_bus252(self):
        assert CDI.currency == Currency.BRL
        assert CDI.day_count == DayCountConvention.BUS_252
        assert CDI.is_overnight is True

    def test_frozen(self):
        import pytest
        with pytest.raises(AttributeError):
            SOFR_ON.tenor = "3M"

    def test_sofr_3m_not_overnight(self):
        assert SOFR_3M.is_overnight is False
        assert SOFR_3M.tenor == "3M"
