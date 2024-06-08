

class BatteryUsageTracker:
    def __init__(self, initial_charge=0, price_data=None):
        self.charge = initial_charge
        self.price_data = price_data

    def track_charge(self, charge):
        self.charge = charge

    def track_price(self, price):
        if self.price_data is None:
            self.price_data = {}
        self.price_data[self.charge] = price

    def get_charge(self):
        return self.charge

    def get_price(self, charge):
        if self.price_data is None or charge not in self.price_data:
            return None
        return self.price_data[charge]