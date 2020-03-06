import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self, hidden):
        super().__init__()

        self._fc1 = nn.Linear(4, hidden)
        self._fc2 = nn.Linear(hidden, hidden)
        self._fc3 = nn.Linear(hidden, 2)

    def forward(self, x):
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        return self._fc3(x)

