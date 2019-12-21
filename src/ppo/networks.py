from torch import nn, tensor
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, observation_space: int, action_space: int,
                 layer_sizes: list, std: float):
        super().__init__()
        self.fc1 = nn.Linear(observation_space, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], action_space)

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.log_std = nn.Parameter(tensor([std for _ in range(action_space)]))

    def forward(self, state: tensor) -> tensor:
        h1 = self.leaky_relu(self.fc1(state))
        h2 = self.leaky_relu(self.fc2(h1))
        mu = self.tanh(self.fc3(h2))
        std = self.log_std.exp().expand_as(mu)
        act_dist = Normal(mu, std)
        return act_dist


class Critic(nn.Module):
    def __init__(self, observation_space: int, layer_sizes: list):
        super().__init__()

        self.fc1 = nn.Linear(observation_space, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], 1)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, state: tensor) -> tensor:
        h1 = self.leaky_relu(self.fc1(state))
        h2 = self.leaky_relu(self.fc2(h1))
        out = self.fc3(h2)
        return out



