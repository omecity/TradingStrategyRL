import numpy as np
import pandas as pd
import yfinance as yf

from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


from FetchData import *
from TradingEnv import *
from TradingAgent import *


# Train the Agent

def train_agent(ticker="AAPL", start_date="2018-01-01", end_date="2020-01-01", episodes=10):
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
        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train(batch_size=32)
            state = next_state
            total_reward += reward

            # Collect data for visuals
            episode_portfolio_values.append(env.portfolio_value)
            episode_actions.append((env.current_step, action))

        rewards.append(total_reward)
        portfolio_values.append(episode_portfolio_values)
        actions.append(episode_actions)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

