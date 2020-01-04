import yaml
import numpy as np
from src.ddpg.networks import Actor2Layer, Critic2Layer, Actor3Layer, Critic3Layer
from src.ddpg.ou_noise import OUNoise
from torch import no_grad, from_numpy, tensor, empty, device
from torch.nn import MSELoss
from torch.optim import Adam


class Actor(object):
    def __init__(self, observation_space: int, action_space: int, device: device):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.device = device

        if len(cfg["LAYER_SIZES"]) == 2:
            self.network = Actor2Layer(observation_space, action_space, cfg["LAYER_SIZES"])
            self.target = Actor2Layer(observation_space, action_space, cfg["LAYER_SIZES"])
        else:
            self.network = Actor3Layer(observation_space, action_space, cfg["LAYER_SIZES"])
            self.target = Actor3Layer(observation_space, action_space, cfg["LAYER_SIZES"])

        self.network.to(self.device)
        self.target.to(self.device)

        self.optimizer = Adam(self.network.parameters(), lr=cfg["ACTOR_LEARNING_RATE"])
        self.ounoise = cfg["OUNOISE"]
        self.noise = OUNoise(mu=np.zeros(action_space))
        self.gaussian_std = cfg["GAUSSIAN_START"]
        self.gaussian_min = cfg["GAUSSIAN_MIN"]
        self.noise_steps = cfg["GAUSSIAN_START"] / (cfg["STEPS"] * cfg["GAUSSIAN_DECAY"])

    def predict(self, state: tensor, use_target: bool) -> tensor:
        with no_grad():
            if use_target:
                predictions = self.target(state)
                return predictions
            else:
                predictions = self.network(state)
                if self.ounoise:
                    return predictions + from_numpy(self.noise()).float().to(self.device)
                else:
                    if self.gaussian_std <= self.gaussian_min:
                        self.gaussian_std = self.gaussian_min
                    else:
                        self.gaussian_std -= self.noise_steps
                    noise = empty(predictions.shape).normal_(mean=0, std=self.gaussian_std).to(self.device)
                    return predictions + noise

    def update_target(self, tau: float):
        for target_weight, weight in zip(self.target.parameters(), self.network.parameters()):
            target_weight.data.copy_(
                target_weight.data * tau + weight.data * (1.0 - tau)
            )


class Critic(object):
    def __init__(self, observation_space: int, action_space: int, device: device):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.device = device

        if len(cfg["LAYER_SIZES"]) == 2:
            self.network = Critic2Layer(observation_space, action_space, cfg["LAYER_SIZES"])
            self.target = Critic2Layer(observation_space, action_space, cfg["LAYER_SIZES"])
        else:
            self.network = Critic3Layer(observation_space, action_space, cfg["LAYER_SIZES"])
            self.target = Critic3Layer(observation_space, action_space, cfg["LAYER_SIZES"])

        self.network.to(self.device)
        self.target.to(self.device)

        self.optimizer = Adam(self.network.parameters(), lr=cfg["CRITIC_LEARNING_RATE"])
        self.loss = MSELoss()

    def predict(self, state: tensor, action: tensor, use_target: bool) -> tensor:
        with no_grad():
            if use_target:
                return self.target(state, action)
            else:
                return self.network(state, action)

    def update_network(self, state: tensor, action: tensor, target: tensor):
        target_pred = self.network(state, action)
        loss = self.loss(target_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self, tau: float):
        for target_weight, weight in zip(self.target.parameters(), self.network.parameters()):
            target_weight.data.copy_(
                target_weight.data * tau + weight.data * (1.0 - tau)
            )
