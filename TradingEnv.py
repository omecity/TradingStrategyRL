

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.n_steps = len(data)
        self.current_step = 0
        self.balance = initial_balance
        self.portfolio_value = initial_balance
        self.holdings = 0
        self.done = False