## A Stock Trading Algorithm Using Deep Q-Learning (DQN) With PyTorch

Built a stock trading algorithm that autonomously makes buy/sell decisions using Deep Q-
Learning (DQN) with PyTorch. The agent was trained on historical market data to maximize
long-term profit by learning optimal trading strategies.


### Fetch Stock Data ###

Fetch data from yahoo finance by passing in formal parameters - ticker, start date, and end date.


### Trading Environment ###

An environment class that would store the data, and initialize the initial balance, portfolio value, and holdings. It also include methods that returns the initial balance and holdings, resets the initial balance and holdings. Finally, another method that steps through each day and decides whether to hold, buy, or sell.
