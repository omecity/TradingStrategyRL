

def train_agent(ticker="AAPL", start_date="2018-01-01", end_date="2020-01-01", episodes=50):
    data = fetch_stock_data(ticker, start_date, end_date)
