import numpy as np


class Bond:
    def __init__(self, notional, coupon, maturity, frequency):
        self.notional = notional
        self.coupon_rate = coupon
        self.maturity = maturity
        self.frequency = frequency

    def get_cashflows(self):
        n_periods = self.maturity * self.frequency
        cashflows = np.zeros(n_periods)
        coupon_payment = self.notional * self.coupon_rate / self.frequency
        for i in range(n_periods):
            cashflows[i] = coupon_payment
        cashflows[-1] += self.notional
        return cashflows

    def get_price(self, rates):
        cashflows = self.get_cashflows()
        price = np.dot(cashflows, np.exp(-rates[:self.maturity * self.frequency] / self.frequency))
        return price

    def get_duration(self, rates):
        cashflows = self.get_cashflows()
        duration = np.dot(np.arange(1, self.maturity * self.frequency + 1), cashflows * np.exp(-rates[:self.maturity * self.frequency] / self.frequency)) / self.get_price(rates)
        return duration

    def get_convexity(self, rates):
        cashflows = self.get_cashflows()
        convexity = np.dot(np.arange(1, self.maturity * self.frequency + 1) ** 2, cashflows * np.exp(-rates[:self.maturity * self.frequency] / self.frequency)) / self.get_price(rates)
        return convexity


class ParBond(Bond):
    def __init__(self, notional, maturity, frequency, coupon_rate):
        super().__init__(notional, coupon_rate, maturity, frequency)
