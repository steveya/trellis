"""Rate index definitions — bundles of market conventions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from trellis.conventions.calendar import (
    BRAZIL, SYDNEY, TARGET, TOKYO, TORONTO, UK_SETTLEMENT, US_SETTLEMENT,
    WEEKEND_ONLY, ZURICH, Calendar,
)
from trellis.conventions.day_count import DayCountConvention


class Compounding(Enum):
    """Compounding rule used when translating index fixings into accrual factors."""
    SIMPLE = "simple"
    CONTINUOUS = "continuous"
    COMPOUNDED_DAILY = "compounded_daily"
    COMPOUNDED_ANNUAL = "compounded_annual"


class Currency(Enum):
    """Supported ISO-like currency identifiers for rate-index conventions."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    BRL = "BRL"
    CNY = "CNY"
    INR = "INR"
    MXN = "MXN"
    KRW = "KRW"


@dataclass(frozen=True)
class RateIndex:
    """Bundles all conventions needed to price against a floating rate."""

    name: str
    currency: Currency
    tenor: str
    day_count: DayCountConvention
    calendar: Calendar
    fixing_lag: int
    spot_lag: int
    compounding: Compounding
    is_overnight: bool = False


# ---------------------------------------------------------------------------
# USD
# ---------------------------------------------------------------------------

SOFR_ON = RateIndex("USD-SOFR-OIS", Currency.USD, "ON", DayCountConvention.ACT_360,
                     US_SETTLEMENT, 0, 2, Compounding.COMPOUNDED_DAILY, True)
SOFR_1M = RateIndex("USD-SOFR-1M", Currency.USD, "1M", DayCountConvention.ACT_360,
                     US_SETTLEMENT, 0, 2, Compounding.COMPOUNDED_DAILY)
SOFR_3M = RateIndex("USD-SOFR-3M", Currency.USD, "3M", DayCountConvention.ACT_360,
                     US_SETTLEMENT, 0, 2, Compounding.COMPOUNDED_DAILY)
SOFR_6M = RateIndex("USD-SOFR-6M", Currency.USD, "6M", DayCountConvention.ACT_360,
                     US_SETTLEMENT, 0, 2, Compounding.COMPOUNDED_DAILY)

TERM_SOFR_1M = RateIndex("USD-TERM-SOFR-1M", Currency.USD, "1M", DayCountConvention.ACT_360,
                          US_SETTLEMENT, 2, 2, Compounding.SIMPLE)
TERM_SOFR_3M = RateIndex("USD-TERM-SOFR-3M", Currency.USD, "3M", DayCountConvention.ACT_360,
                          US_SETTLEMENT, 2, 2, Compounding.SIMPLE)
TERM_SOFR_6M = RateIndex("USD-TERM-SOFR-6M", Currency.USD, "6M", DayCountConvention.ACT_360,
                          US_SETTLEMENT, 2, 2, Compounding.SIMPLE)

FED_FUNDS = RateIndex("USD-FED-FUNDS", Currency.USD, "ON", DayCountConvention.ACT_360,
                       US_SETTLEMENT, 0, 1, Compounding.COMPOUNDED_DAILY, True)

LIBOR_USD_1M = RateIndex("USD-LIBOR-1M", Currency.USD, "1M", DayCountConvention.ACT_360,
                          UK_SETTLEMENT, 2, 2, Compounding.SIMPLE)
LIBOR_USD_3M = RateIndex("USD-LIBOR-3M", Currency.USD, "3M", DayCountConvention.ACT_360,
                          UK_SETTLEMENT, 2, 2, Compounding.SIMPLE)
LIBOR_USD_6M = RateIndex("USD-LIBOR-6M", Currency.USD, "6M", DayCountConvention.ACT_360,
                          UK_SETTLEMENT, 2, 2, Compounding.SIMPLE)
LIBOR_USD_12M = RateIndex("USD-LIBOR-12M", Currency.USD, "12M", DayCountConvention.ACT_360,
                           UK_SETTLEMENT, 2, 2, Compounding.SIMPLE)

# ---------------------------------------------------------------------------
# EUR
# ---------------------------------------------------------------------------

ESTR = RateIndex("EUR-ESTR", Currency.EUR, "ON", DayCountConvention.ACT_360,
                  TARGET, 0, 1, Compounding.COMPOUNDED_DAILY, True)

EURIBOR_1W = RateIndex("EUR-EURIBOR-1W", Currency.EUR, "1W", DayCountConvention.ACT_360,
                        TARGET, 2, 2, Compounding.SIMPLE)
EURIBOR_1M = RateIndex("EUR-EURIBOR-1M", Currency.EUR, "1M", DayCountConvention.ACT_360,
                        TARGET, 2, 2, Compounding.SIMPLE)
EURIBOR_3M = RateIndex("EUR-EURIBOR-3M", Currency.EUR, "3M", DayCountConvention.ACT_360,
                        TARGET, 2, 2, Compounding.SIMPLE)
EURIBOR_6M = RateIndex("EUR-EURIBOR-6M", Currency.EUR, "6M", DayCountConvention.ACT_360,
                        TARGET, 2, 2, Compounding.SIMPLE)
EURIBOR_12M = RateIndex("EUR-EURIBOR-12M", Currency.EUR, "12M", DayCountConvention.ACT_360,
                         TARGET, 2, 2, Compounding.SIMPLE)

# ---------------------------------------------------------------------------
# GBP
# ---------------------------------------------------------------------------

SONIA = RateIndex("GBP-SONIA", Currency.GBP, "ON", DayCountConvention.ACT_365,
                   UK_SETTLEMENT, 0, 0, Compounding.COMPOUNDED_DAILY, True)

LIBOR_GBP_3M = RateIndex("GBP-LIBOR-3M", Currency.GBP, "3M", DayCountConvention.ACT_365,
                           UK_SETTLEMENT, 0, 0, Compounding.SIMPLE)
LIBOR_GBP_6M = RateIndex("GBP-LIBOR-6M", Currency.GBP, "6M", DayCountConvention.ACT_365,
                           UK_SETTLEMENT, 0, 0, Compounding.SIMPLE)

# ---------------------------------------------------------------------------
# JPY
# ---------------------------------------------------------------------------

TONA = RateIndex("JPY-TONA", Currency.JPY, "ON", DayCountConvention.ACT_365,
                  TOKYO, 0, 2, Compounding.COMPOUNDED_DAILY, True)

TIBOR_3M = RateIndex("JPY-TIBOR-3M", Currency.JPY, "3M", DayCountConvention.ACT_365,
                      TOKYO, 2, 2, Compounding.SIMPLE)
TIBOR_6M = RateIndex("JPY-TIBOR-6M", Currency.JPY, "6M", DayCountConvention.ACT_365,
                      TOKYO, 2, 2, Compounding.SIMPLE)

LIBOR_JPY_6M = RateIndex("JPY-LIBOR-6M", Currency.JPY, "6M", DayCountConvention.ACT_360,
                           UK_SETTLEMENT, 2, 2, Compounding.SIMPLE)

# ---------------------------------------------------------------------------
# CHF / CAD / AUD
# ---------------------------------------------------------------------------

SARON = RateIndex("CHF-SARON", Currency.CHF, "ON", DayCountConvention.ACT_360,
                   ZURICH, 0, 2, Compounding.COMPOUNDED_DAILY, True)

CORRA = RateIndex("CAD-CORRA", Currency.CAD, "ON", DayCountConvention.ACT_365,
                   TORONTO, 0, 0, Compounding.COMPOUNDED_DAILY, True)

AONIA = RateIndex("AUD-AONIA", Currency.AUD, "ON", DayCountConvention.ACT_365,
                   SYDNEY, 0, 1, Compounding.COMPOUNDED_DAILY, True)

BBSW_3M = RateIndex("AUD-BBSW-3M", Currency.AUD, "3M", DayCountConvention.ACT_365,
                      SYDNEY, 0, 2, Compounding.SIMPLE)
BBSW_6M = RateIndex("AUD-BBSW-6M", Currency.AUD, "6M", DayCountConvention.ACT_365,
                      SYDNEY, 0, 2, Compounding.SIMPLE)

# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------

CDI = RateIndex("BRL-CDI", Currency.BRL, "ON", DayCountConvention.BUS_252,
                 BRAZIL, 0, 0, Compounding.COMPOUNDED_DAILY, True)

SHIBOR_3M = RateIndex("CNY-SHIBOR-3M", Currency.CNY, "3M", DayCountConvention.ACT_360,
                        WEEKEND_ONLY, 0, 1, Compounding.SIMPLE)
LPR_1Y = RateIndex("CNY-LPR-1Y", Currency.CNY, "1Y", DayCountConvention.ACT_365,
                     WEEKEND_ONLY, 0, 1, Compounding.SIMPLE)
LPR_5Y = RateIndex("CNY-LPR-5Y", Currency.CNY, "5Y", DayCountConvention.ACT_365,
                     WEEKEND_ONLY, 0, 1, Compounding.SIMPLE)

MIBOR = RateIndex("INR-MIBOR", Currency.INR, "ON", DayCountConvention.ACT_365,
                    WEEKEND_ONLY, 0, 0, Compounding.COMPOUNDED_DAILY, True)

TIIE_28D = RateIndex("MXN-TIIE-28D", Currency.MXN, "28D", DayCountConvention.ACT_360,
                      WEEKEND_ONLY, 1, 1, Compounding.SIMPLE)

KOFR = RateIndex("KRW-KOFR", Currency.KRW, "ON", DayCountConvention.ACT_365,
                   WEEKEND_ONLY, 0, 1, Compounding.COMPOUNDED_DAILY, True)
