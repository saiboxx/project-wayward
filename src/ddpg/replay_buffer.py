import numpy as np


class ReplayBuffer(object):

    def __init__(self, max_buffer_size, batch_size, state_size, action_size):

        self.max_buffer_size = max_buffer_size
        self.cur_buffer_size = 0
        self.batch_size = batch_size
        self.states = np.empty((self.max_buffer_size, state_size), dtype=np.float64)
        self.actions = np.empty((self.max_buffer_size, action_size), dtype=np.uint8)
        self.rewards = np.empty(self.max_buffer_size, dtype=np.integer)
        self.new_states = np.empty((self.max_buffer_size, state_size), dtype=np.float64)
        self.current_idx = 0

    def sample(self) -> tuple:
        """"
        Samples experiences from the buffer. The sampling size depends
        on the variable BATCH_SIZE in the config file.
        """
        if self.cur_buffer_size > self.batch_size:
            indices = np.random.choice(self.cur_buffer_size, self.batch_size)
        else:
            indices = range(self.cur_buffer_size)

        sample_states = self.states[indices]
        sample_actions = self.actions[indices]
        sample_rewards = self.rewards[indices]
        sample_new_states = self.new_states[indices]

        return sample_states, sample_actions, sample_rewards, sample_new_states

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, new_state: np.ndarray):
        """"
        Adds an experience to the buffer. It pops the oldest experience if buffer_size is
        reached. A return of the env is an array of with size NUM_AGENTS X OBSERVATION_SPACE,
        so it has to be saved row by row.
        """

        self.states[self.current_idx] = state
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.new_states[self.current_idx] = new_state

        self.cur_buffer_size = max(self.cur_buffer_size, self.current_idx + 1)
        self.current_idx = (self.current_idx + 1) % self.max_buffer_size

    def current_size(self):

        return self.cur_buffer_size
