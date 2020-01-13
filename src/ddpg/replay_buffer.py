import random
import numpy as np
import yaml
from operator import itemgetter

class ReplayBuffer(object):

    def __init__(self, cfg):

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
            sample_states = self.states
            sample_actions = self.actions
            sample_rewards = self.rewards
            sample_new_states = self.new_states
        else:
            batch_ind = random.sample(range(self.cur_buffer_size), self.batch_size)
            sample_states = itemgetter(*batch_ind)(self.states)
            sample_actions = itemgetter(*batch_ind)(self.actions)
            sample_rewards = itemgetter(*batch_ind)(self.rewards)
            sample_new_states = itemgetter(*batch_ind)(self.new_states)

        return sample_states, sample_actions, sample_rewards, sample_new_states

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, new_state: np.ndarray):
        """"
        Adds an experience to the buffer. It pops the oldest experience if buffer_size is
        reached. A return of the env is an array of with size NUM_AGENTS X OBSERVATION_SPACE,
        so it has to be saved row by row.
        """
        num_envs = len(state)
        if isinstance(reward, np.ndarray):
            if self.cur_buffer_size >= self.max_buffer_size:
                del self.states[:num_envs]
                del self.actions[:num_envs]
                del self.rewards[:num_envs]
                del self.new_states[:num_envs]

            self.states.extend(state)
            self.actions.extend(action)
            self.rewards.extend(reward)
            self.new_states.extend(new_state)
            self.cur_buffer_size = len(self.states)

        else:
            if self.cur_buffer_size >= self.max_buffer_size:
                del self.states[0]
                del self.actions[0]
                del self.rewards[0]
                del self.new_states[0]

            self.states.append(state.flatten())
            self.actions.append(action.flatten())
            self.rewards.append(reward)
            self.new_states.append(new_state.flatten())
            self.cur_buffer_size = len(self.states)


