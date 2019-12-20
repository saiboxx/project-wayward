import torch
from torch import nn, tensor


class ActorNet(nn.Module):
    def __init__(self, observation_space: int, action_space: int, layer_sizes: list):
        super().__init__()

        self.fc1 = nn.Linear(observation_space, layer_sizes[0])
        self.bn1 = nn.LayerNorm(layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.bn2 = nn.LayerNorm(layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], action_space)

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.fc1.weight.data.uniform_(-0.003, 0.003)
        self.fc1.bias.data.uniform_(-0.003, 0.003)
        self.fc2.weight.data.uniform_(-0.003, 0.003)
        self.fc2.bias.data.uniform_(-0.003, 0.003)
        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-0.003, 0.003)

    def forward(self, state: tensor) -> tensor:
        h1 = self.bn1(self.leaky_relu(self.fc1(state)))
        h2 = self.bn2(self.leaky_relu(self.fc2(h1)))
        out = self.tanh(self.fc3(h2))
        return out


class CriticNet(nn.Module):
    def __init__(self, observation_space: int, action_space: int, layer_sizes: list):
        super().__init__()

        self.fc1 = nn.Linear(observation_space, layer_sizes[0])
        self.bn1 = nn.LayerNorm(layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0] + action_space, layer_sizes[1])
        self.bn2 = nn.LayerNorm(layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], 1)

        self.leaky_relu = nn.LeakyReLU()

        self.fc1.weight.data.uniform_(-0.003, 0.003)
        self.fc1.bias.data.uniform_(-0.003, 0.003)
        self.fc2.weight.data.uniform_(-0.003, 0.003)
        self.fc2.bias.data.uniform_(-0.003, 0.003)
        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-0.003, 0.003)

    def forward(self, state: tensor, action: tensor) -> tensor:
        h1 = self.bn1(self.leaky_relu(self.fc1(state)))
        h2 = self.bn2(self.leaky_relu(self.fc2(torch.cat([h1, action], dim=1))))
        out = self.fc3(h2)
        return out



