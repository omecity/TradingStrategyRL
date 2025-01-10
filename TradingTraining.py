

def train_agent(ticker="AAPL", start_date="2018-01-01", end_date="2020-01-01", episodes=50):
    data = fetch_stock_data(ticker, start_date, end_date)
    env = TradingEnvironment(data)
    agent = DQNAgent(state_dim=3, action_dim=3)

    rewards = []
    portfolio_values = []
    actions = []
    prices = data['Close'].values

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_portfolio_values = []
        episode_actions = []
