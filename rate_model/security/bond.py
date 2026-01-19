import numpy as np


class Bond:
    def __init__(self, notional, coupon, maturity, frequency):
        self.notional = notional
        self.coupon_rate = coupon
        self.maturity = maturity
        self.frequency = frequency

    def get_cashflows(self):
        cashflows = np.zeros(self.maturity * self.frequency)
        for i in range(1, self.maturity * self.frequency + 1):
            if i % self.frequency == 0:
                cashflows[i - 1] = self.notional * self.coupon_rate / self.frequency
            else:
                cashflows[i - 1] = 0
        cashflows[-1] += self.notional
        return cashflows

    def get_price(self, rates):
        cashflows = self.get_cashflows(rates)
        price = np.dot(cashflows, np.exp(-rates[:self.maturity * self.frequency] / self.frequency))
        return price

    def get_duration(self, rates):
        cashflows = self.get_cashflows(rates)
        duration = np.dot(np.arange(1, self.maturity * self.frequency + 1), cashflows * np.exp(-rates[:self.maturity * self.frequency] / self.frequency)) / self.get_price(rates)
        return duration

    def get_convexity(self, rates):
        cashflows = self.get_cashflows(rates)
        convexity = np.dot(np.arange(1, self.maturity * self.frequency + 1) ** 2, cashflows * np.exp(-rates[:self.maturity * self.frequency] / self.frequency)) / self.get_price(rates)
        return convexity
    

class ParBond(Bond):
    def __init__(self, notional, maturity, frequency):
        super().__init__(notional, 0, maturity, frequency)