from src.networks import Actor, Critic
from src.replay_buffer import ReplayBuffer


class Agent(object):
    """"
    Depicts the acting Entity.
    """

    def __init__(self, observation_space: int, action_space: int):
        self.actor = Actor(observation_space, action_space)
        self.target = Critic(observation_space, action_space)
        self.replay_buffer = ReplayBuffer()
