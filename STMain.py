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
end_date="2020-01-01"
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