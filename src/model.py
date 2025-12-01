import torch
import torch.nn as nn

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=82, out_features=256, bias=True)
        self.linear11 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.linear2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.linear3 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.linear4 = nn.Linear(in_features=32, out_features=5, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear11(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x

