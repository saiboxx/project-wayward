import random
import numpy as np
import yaml
from torch import tensor, device


class ReplayBuffer(object):

    def __init__(self, device: device):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.device = device
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []

    def add(self, state: tensor, action: tensor, reward: np.ndarray,
            done: np.ndarray, log_prob: tensor, value: tensor):
        if isinstance(reward, np.ndarray):
            self.states.append(state.to(self.device))
            self.actions.append(action.to(self.device))
            self.rewards.append(tensor(reward).unsqueeze(1).to(self.device))
            self.masks.append(tensor(1 - done).unsqueeze(1).to(self.device))
            self.log_probs.append(log_prob.to(self.device))
            self.values.append(value.to(self.device))
        else:
            self.states.append(state.to(self.device))
            self.actions.append(action.to(self.device))
            self.rewards.append(tensor(reward).to(self.device))
            self.masks.append(tensor(1 - done).to(self.device))
            self.log_probs.append(log_prob.to(self.device))
            self.values.append(value.to(self.device))

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []

