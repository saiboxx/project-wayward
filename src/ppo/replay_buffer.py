import random
import numpy as np
import yaml
from torch import tensor
from operator import itemgetter


class ReplayBuffer(object):

    def __init__(self):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []

    def add(self, state: tensor, action: tensor, reward: np.ndarray,
            done: np.ndarray, log_prob: tensor, value: tensor):
        """"

        """
        if isinstance(reward, np.ndarray):
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(tensor(reward).unsqueeze(1))
            self.masks.append(tensor(1 - done).unsqueeze(1))
            self.log_probs.append(log_prob)
            self.values.append(value)
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(tensor(reward))
            self.masks.append(tensor(1 - done))
            self.log_probs.append(log_prob)
            self.values.append(value)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.values = []

