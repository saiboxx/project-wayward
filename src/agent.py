from src.networks import Actor, Critic
from src.replay_buffer import ReplayBuffer


class Agent(object):
    """"
    Depicts the acting Entity.
    """

    def __init__(self):
        self.actor = Actor()
        self.target = Critic()
        self.replay_buffer = ReplayBuffer()
