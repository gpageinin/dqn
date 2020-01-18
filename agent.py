import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
    def __init__(self, hidden):
        super().__init__()

        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

