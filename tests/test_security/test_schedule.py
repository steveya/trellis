"""Testing Schedule class"""
import pytest
from rate_model.security.schedule import Frequency, Schedule
from datetime import datetime

def test_get_bracketing_dates():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    frequency = Frequency.QUARTERLY
    schedule = Schedule(start_date, end_date, frequency)
    date = datetime(2020, 3, 1)
    assert schedule.get_bracketing_dates(date) == [datetime(2020, 1, 1), datetime(2020, 4, 1)]
    date = datetime(2020, 6, 1)
    assert schedule.get_bracketing_dates(date) == [datetime(2020, 4, 1), datetime(2020, 7, 1)]
    date = datetime(2020, 9, 1)
    assert schedule.get_bracketing_dates(date) == [datetime(2020, 7, 1), datetime(2020, 10, 1)]
    date = datetime(2020, 12, 31)
    assert schedule.get_bracketing_dates(date) == [datetime(2020, 10, 1), datetime(2021, 1, 1)]

def test_get_accrual_fraction():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    frequency = Frequency.QUARTERLY
    schedule = Schedule(start_date, end_date, frequency)
    date = datetime(2020, 3, 1)
    # 60 days into a 91-day quarter (Jan 1 to Apr 1)
    assert schedule.get_accrual_fraction(date) == pytest.approx(60 / 91)
    date = datetime(2020, 6, 1)
    # 61 days into a 91-day quarter (Apr 1 to Jul 1)
    assert schedule.get_accrual_fraction(date) == pytest.approx(61 / 91)
    date = datetime(2020, 9, 1)
    # 62 days into a 92-day quarter (Jul 1 to Oct 1)
    assert schedule.get_accrual_fraction(date) == pytest.approx(62 / 92)
    date = datetime(2020, 12, 31)
    # 91 days into a 92-day quarter (Oct 1 to Jan 1)
    assert schedule.get_accrual_fraction(date) == pytest.approx(91 / 92)

def test_get_bracketing_dates_exception():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    frequency = Frequency.QUARTERLY
    schedule = Schedule(start_date, end_date, frequency)
    date = datetime(2021, 3, 1)
    with pytest.raises(ValueError):
        schedule.get_bracketing_dates(date)

def test_get_accrual_fraction_exception():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    frequency = Frequency.QUARTERLY
    schedule = Schedule(start_date, end_date, frequency)
    date = datetime(2021, 3, 1)
    with pytest.raises(ValueError):
        schedule.get_accrual_fraction(date)
