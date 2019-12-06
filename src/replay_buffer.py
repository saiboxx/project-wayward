import random
import numpy as np
import yaml


class ReplayBuffer(object):

    def __init__(self):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.max_buffer_size = cfg["BUFFER_SIZE"]
        self.cur_buffer_size = 0
        self.batch_size = cfg["BATCH_SIZE"]
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []

    def sample(self) -> tuple:
        """"
        Samples experiences from the buffer. The sampling size depends
        on the variable BATCH_SIZE in the config file.
        """
        if self.cur_buffer_size <= self.batch_size:
            sample_states = np.asarray(self.states)
            sample_actions = np.asarray(self.actions)
            sample_rewards = np.asarray(self.rewards)
            sample_new_states = np.asarray(self.new_states)
        else:
            batch_ind = random.sample(range(self.cur_buffer_size), self.batch_size)
            sample_states = np.asarray(self.states)[batch_ind]
            sample_actions = np.asarray(self.actions)[batch_ind]
            sample_rewards = np.asarray(self.rewards)[batch_ind]
            sample_new_states = np.asarray(self.new_states)[batch_ind]

        return sample_states, sample_actions, sample_rewards, sample_new_states

    def add(self, state: np.ndarray, action: np.ndarray, reward: list, new_state: np.ndarray):
        """"
        Adds an experience to the buffer. It pops the oldest experience if buffer_size is
        reached. A return of the env is an array of with size NUM_AGENTS X OBSERVATION_SPACE,
        so it has to be saved row by row.
        """
        for i in range(len(reward)):
            if self.cur_buffer_size >= self.max_buffer_size:
                del self.states[0]
                del self.actions[0]
                del self.rewards[0]
                del self.new_states[0]

            self.states.append(state[i, :])
            self.actions.append(action[i, :])
            self.rewards.append(reward[i])
            self.new_states.append(new_state[i, :])
            self.cur_buffer_size = len(self.states)

