

class TradingEnvironment:

    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.n_steps = len(data)
        self.current_step = 0
        self.balance = initial_balance
        self.portfolio_value = initial_balance
        self.holdings = 0
        self.done = False
    
    def get_state(self):
        return np.array([
            float(self.data.iloc[self.current_step]['Close']),
            float(self.balance),
            float(self.holdings),
        ], dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.portfolio_value = 10000
        self.holdings = 0
        self.done = False
        return self.get_state()