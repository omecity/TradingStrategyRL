# Stock Trading Algorithm using Deep Q-Learning (DQN)

### Install Required Libraries


import numpy as np
import pandas as pd
import random
import yfinance as yf


from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


from TradingEnv import *
from TradingModel import *
from TradingAgent import *
from TradingTraining import *



### Run the training

train_agent()