import torch.nn as nn


class AsteroidModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AsteroidModel, self).__init__()
        self.relu = nn.ReLU()
        self.fcIn = nn.Linear(input_size, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fcOut = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fcIn(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fcOut(x)
        return x
