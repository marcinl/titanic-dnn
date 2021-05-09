import torch
import torch.nn as nn

def swish(x):
    return x * torch.sigmoid(x)

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 16)
        self.b1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.b2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8,4)
        self.b3 = nn.BatchNorm1d(4)
        self.fc4 = nn.Linear(4,1)

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = torch.sigmoid(self.fc4(x))

        return x

class NetworkSequential(nn.Module):

    def __init__(self):
        super(NetworkSequential, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(8, 16), \
            nn.ReLU(), \
            nn.BatchNorm1d(16), \
            nn.Linear(16, 8), \
            nn.ReLU(), \
            nn.BatchNorm1d(8), \
            nn.Linear(8,4), \
            nn.ReLU(), \
            nn.BatchNorm1d(4), \
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = torch.sigmoid(x)

        return x