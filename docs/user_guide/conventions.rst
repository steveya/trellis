Conventions
===========

Day Count Conventions
---------------------

.. code-block:: python

   from trellis.conventions.day_count import DayCountConvention, year_fraction
   from datetime import date

   frac = year_fraction(date(2024, 1, 1), date(2024, 7, 1),
                         DayCountConvention.ACT_360)

Supported conventions: ACT/360, ACT/365 Fixed, ACT/ACT ISDA, ACT/ACT ICMA,
30/360 US, 30E/360, 30E/360 ISDA, ACT/365.25, BUS/252, 1/1.

Calendars
---------

.. code-block:: python

   from trellis.conventions.calendar import US_SETTLEMENT, TARGET

   US_SETTLEMENT.is_business_day(date(2024, 12, 25))  # False
   US_SETTLEMENT.adjust(date(2024, 11, 30),
                         BusinessDayAdjustment.MODIFIED_FOLLOWING)

Built-in: US, UK, TARGET, Tokyo, Sydney, Toronto, Zurich, Brazil.

Rate Indices
------------

35 pre-built indices bundling all conventions:

.. code-block:: python

   from trellis.conventions.rate_index import SOFR_3M, SONIA, CDI

   print(SOFR_3M.day_count)   # ACT/360
   print(SONIA.day_count)     # ACT/365
   print(CDI.day_count)       # BUS/252

Schedules
---------

.. code-block:: python

   from trellis.conventions.schedule import generate_schedule, StubType
   from trellis.core.types import Frequency

   dates = generate_schedule(start, end, Frequency.QUARTERLY,
                              stub=StubType.SHORT_LAST,
                              calendar=US_SETTLEMENT,
                              bda=BusinessDayAdjustment.MODIFIED_FOLLOWING)
