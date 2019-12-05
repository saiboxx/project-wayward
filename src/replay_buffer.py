from _collections import deque
import random
import numpy as np
import yaml


class ReplayBuffer(object):

    def __init__(self):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.buffer_size = cfg["BUFFER_SIZE"]
        self.batch_size = cfg["BATCH_SIZE"]
        self.buffer = deque()

    def sample(self) -> list:
        """"
        Samples experiences from the buffer. The sampling size depends
        on the variable BATCH_SIZE in the config file.
        """
        if len(self.buffer) <= self.batch_size:
            return list(self.buffer)
        else:
            return random.sample(list(self.buffer), self.batch_size)

    def add(self, state: np.ndarray, action: np.ndarray, reward: list, new_state: np.ndarray):
        """"
        Adds an experience to the buffer. It pops the oldest experience if buffer_size is
        reached.
        """
        if len(self.buffer) >= self.buffer_size:
            self.buffer.popleft()
        self.buffer.append((state, action, reward, new_state))

