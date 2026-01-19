import numpy as np
from datetime import datetime
from aenum import Enum

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
        self.cashflow_dates = np.array([start_date + (end_date - start_date) * i / frequency.value for i in range(1, frequency.value + 1)])

    def get_bracketing_dates(self, date):
        """
        Given a date, return the start and end date of the accrual period that contains the given date
        """
        if date < self.start_date or date > self.end_date:
            raise ValueError("Date is not in the schedule")
        else:
            bracketing_dates = []
            for i in range(1, self.frequency.value + 1):
                if date <= self.start_date + (self.end_date - self.start_date) * i / self.frequency.value:
                    bracketing_dates.append(self.start_date + (self.end_date - self.start_date) * (i - 1) / self.frequency.value)
                    bracketing_dates.append(self.start_date + (self.end_date - self.start_date) * i / self.frequency.value)
                    break
            return bracketing_dates

    def get_accrual_fraction(self, date):
        """
        Given a date, return the accrual fraction of the accrual period that contains the given date
        """
        bracketing_dates = self.get_bracketing_dates(date)
        accrual_fraction = (date - bracketing_dates[0]) / (bracketing_dates[1] - bracketing_dates[0])
        return accrual_fraction
    