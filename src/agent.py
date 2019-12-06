import yaml
from src.networks import Actor, Critic
from src.replay_buffer import ReplayBuffer


class Agent(object):
    """"
    Depicts the acting Entity.
    """

    def __init__(self, observation_space: int, action_space: int):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.actor = Actor(observation_space, action_space)
        self.critic = Critic(observation_space, action_space)
        self.replay_buffer = ReplayBuffer()
        self.gamma = cfg["GAMMA"]

    def learn(self):

        # Get experiences from replay buffer
        state, action, reward, new_state = self.replay_buffer.sample()

        # Calculate targets
        target_values = self.critic.predict(new_state, self.actor.predict(new_state))
        target = reward + self.gamma * target_values

        # Train critic
        self.critic.update(state, action, target)