from torch import nn, tensor
from torch.distributions import Normal


class Actor2Layer(nn.Module):
    def __init__(self, observation_space: int, action_space: int,
                 layer_sizes: list, std: float):
        super().__init__()
        self.fc1 = nn.Linear(observation_space, layer_sizes[0])
        self.bn1 = nn.LayerNorm(layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.bn2 = nn.LayerNorm(layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], action_space)

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        self.fc1.weight.data.uniform_(-0.003, 0.003)
        self.fc1.bias.data.uniform_(-0.003, 0.003)
        self.fc2.weight.data.uniform_(-0.003, 0.003)
        self.fc2.bias.data.uniform_(-0.003, 0.003)
        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-0.003, 0.003)

        self.log_std = nn.Parameter(tensor([std for _ in range(action_space)]))

    def forward(self, state: tensor) -> tensor:
        h1 = self.bn1(self.elu(self.fc1(state)))
        h2 = self.bn2(self.elu(self.fc2(h1)))
        mu = self.tanh(self.fc3(h2))
        std = self.log_std.exp().expand_as(mu)
        act_dist = Normal(mu, std)
        return act_dist


class Critic2Layer(nn.Module):
    def __init__(self, observation_space: int, layer_sizes: list):
        super().__init__()

        self.fc1 = nn.Linear(observation_space, layer_sizes[0])
        self.bn1 = nn.LayerNorm(layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.bn2 = nn.LayerNorm(layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], 1)

        self.elu = nn.ELU()

        self.fc1.weight.data.uniform_(-0.003, 0.003)
        self.fc1.bias.data.uniform_(-0.003, 0.003)
        self.fc2.weight.data.uniform_(-0.003, 0.003)
        self.fc2.bias.data.uniform_(-0.003, 0.003)
        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-0.003, 0.003)

    def forward(self, state: tensor) -> tensor:
        h1 = self.bn1(self.elu(self.fc1(state)))
        h2 = self.bn2(self.elu(self.fc2(h1)))
        out = self.fc3(h2)
        return out


class Actor3Layer(nn.Module):
    def __init__(self, observation_space: int, action_space: int,
                 layer_sizes: list, std: float):
        super().__init__()
        self.fc1 = nn.Linear(observation_space, layer_sizes[0])
        self.bn1 = nn.LayerNorm(layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.bn2 = nn.LayerNorm(layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.bn3 = nn.LayerNorm(layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], action_space)

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        self.fc1.weight.data.uniform_(-0.003, 0.003)
        self.fc1.bias.data.uniform_(-0.003, 0.003)
        self.fc2.weight.data.uniform_(-0.003, 0.003)
        self.fc2.bias.data.uniform_(-0.003, 0.003)
        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-0.003, 0.003)
        self.fc4.weight.data.uniform_(-0.003, 0.003)
        self.fc4.bias.data.uniform_(-0.003, 0.003)

        self.log_std = nn.Parameter(tensor([std for _ in range(action_space)]))

    def forward(self, state: tensor) -> tensor:
        h1 = self.bn1(self.elu(self.fc1(state)))
        h2 = self.bn2(self.elu(self.fc2(h1)))
        h3 = self.bn3(self.elu(self.fc3(h2)))
        mu = self.tanh(self.fc4(h3))
        std = self.log_std.exp().expand_as(mu)
        act_dist = Normal(mu, std)
        return act_dist


class Critic3Layer(nn.Module):
    def __init__(self, observation_space: int, layer_sizes: list):
        super().__init__()

        self.fc1 = nn.Linear(observation_space, layer_sizes[0])
        self.bn1 = nn.LayerNorm(layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.bn2 = nn.LayerNorm(layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.bn3 = nn.LayerNorm(layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], 1)

        self.elu = nn.ELU()

        self.fc1.weight.data.uniform_(-0.003, 0.003)
        self.fc1.bias.data.uniform_(-0.003, 0.003)
        self.fc2.weight.data.uniform_(-0.003, 0.003)
        self.fc2.bias.data.uniform_(-0.003, 0.003)
        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-0.003, 0.003)
        self.fc4.weight.data.uniform_(-0.003, 0.003)
        self.fc4.bias.data.uniform_(-0.003, 0.003)

    def forward(self, state: tensor) -> tensor:
        h1 = self.bn1(self.elu(self.fc1(state)))
        h2 = self.bn2(self.elu(self.fc2(h1)))
        h3 = self.bn3(self.elu(self.fc3(h2)))
        out = self.fc4(h3)
        return out
