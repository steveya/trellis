import numpy as np
from datetime import datetime
from aenum import Enum


def _add_months(dt: datetime, months: int) -> datetime:
    """Add calendar months to a datetime, clamping the day to the new month's max."""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    import calendar
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


class Frequency(Enum):
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12


class Schedule:
    """
    A class to represent a schedule of cashflows. It provides methods to find bracketing dates of a given date, and calculate accrual fraction
    """
    def __init__(self, start_date: datetime, end_date: datetime, frequency: Frequency):
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        months_per_period = 12 // frequency.value
        self.cashflow_dates = np.array([_add_months(start_date, months_per_period * i) for i in range(1, frequency.value + 1)])

    def get_bracketing_dates(self, date):
        """
        Given a date, return the start and end date of the accrual period that contains the given date
        """
        if date < self.start_date or date > self.end_date:
            raise ValueError("Date is not in the schedule")
        else:
            bracketing_dates = []
            months_per_period = 12 // self.frequency.value
            for i in range(1, self.frequency.value + 1):
                upper = _add_months(self.start_date, months_per_period * i)
                if date <= upper:
                    lower = _add_months(self.start_date, months_per_period * (i - 1))
                    bracketing_dates.append(lower)
                    bracketing_dates.append(upper)
                    break
            return bracketing_dates

    def get_accrual_fraction(self, date):
        """
        Given a date, return the accrual fraction of the accrual period that contains the given date
        """
        bracketing_dates = self.get_bracketing_dates(date)
        accrual_fraction = (date - bracketing_dates[0]) / (bracketing_dates[1] - bracketing_dates[0])
        return accrual_fraction
    