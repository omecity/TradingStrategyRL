import numpy as np


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

    def step(self, action):
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        current_price = self.data.iloc[self.current_step]['Close']
        if action == 1:
            shares_to_buy = self.balance // current_price
            self.holdings += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2: 
            self.balance += self.holdings * current_price
            self.holdings = 0

        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            self.done = True

        self.portfolio_value = self.balance + (self.holdings * current_price)
        # Reward is defined as the change in portfolio value
        reward = self.portfolio_value - 10000  
        return self.get_state(), reward, self.done