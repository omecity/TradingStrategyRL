from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.fc(x)