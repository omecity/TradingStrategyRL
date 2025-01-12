# Stock Trading Algorithm using Deep Q-Learning (DQN)

### Install Required Libraries


import numpy as np
import pandas as pd
import random
import yfinance as yf
import matplotlib.pyplot as plt


from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


from TradingEnv import *
from TradingModel import *
from TradingAgent import *
from TradingTraining import *



### Run the training

ticker="AAPL"
start_date="2018-01-01"
end_date="2025-01-01"
episodes=20


rewards, portfolio_values, actions, prices = train_agent(ticker, start_date, end_date, episodes)


# total rewards over episodes 
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward Progression')
plt.show()


# portfolio value over time
plt.figure(figsize=(10, 6))
for episode_portfolio in portfolio_values:
    plt.plot(episode_portfolio, alpha=0.5)
plt.xlabel('Time Step')
plt.ylabel('Portfolio Value ($)')
plt.title('Portfolio Value Across Episodes')
plt.show()


# Trading Actions over time
plt.figure(figsize=(10, 6))
plt.plot(prices, label="Stock Price", color='blue')

# Loop through episodes and plot actions
for episode_actions in actions:
    buy_steps = [step for step, action in episode_actions if action == 1]
    sell_steps = [step for step, action in episode_actions if action == 2]
    plt.scatter(buy_steps, prices[buy_steps], marker="^", color="green", alpha=0.5)
    plt.scatter(sell_steps, prices[sell_steps], marker="v", color="red", alpha=0.5)

# Add single representative markers for legend
plt.scatter([], [], label="Buy", marker="^", color="green")
plt.scatter([], [], label="Sell", marker="v", color="red")

plt.xlabel('Time Step')
plt.ylabel('Price ($)')
plt.title('Trading Actions Overlaid on Stock Price')
plt.legend()
plt.show()


# Cumulative return and benchmark over episodes
plt.figure(figsize=(10, 6))
cumulative_returns = [portfolio[-1] / portfolio[0] - 1 for portfolio in portfolio_values]
benchmark_return = (prices[-1] / prices[0]) - 1
plt.plot(cumulative_returns, label="Agent Cumulative Return")
plt.axhline(benchmark_return, label="Buy-and-Hold Benchmark", linestyle="--", color="orange")
plt.xlabel('Episode')
plt.ylabel('Cumulative Return')
plt.title('Agent Return vs. Benchmark')
plt.legend()
plt.show()