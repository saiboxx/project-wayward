import yaml
import numpy as np
import tensorflow as tf

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
        self.tau = cfg["TAU"]

    def learn(self):
        if self.replay_buffer.cur_buffer_size > self.replay_buffer.batch_size:
            # Get experiences from replay buffer
            state, action, reward, new_state = self.replay_buffer.sample()

            # Calculate targets
            target_values = self.critic.predict(new_state,
                                                self.actor.predict(new_state, use_target=True),
                                                use_target=True)
            target = reward + self.gamma * target_values.flatten()

            # Update critic
            self.critic.update_network(state, action, target)

            # Get actions from Actor with old states
            a_output = self.actor.predict(state, use_target=False)

            # Get Gradient from critic
            with tf.GradientTape() as tape:
                next_action = self.actor.network(state)
                actor_loss = -tf.reduce_mean(self.critic.network([state, next_action]))

            actor_grad = tape.gradient(actor_loss, self.actor.network.trainable_variables)

            print(actor_grad)
            # Apply gradient to actor network
            #self.actor.update_network(gradient_critic)

            # Update target networks
            #self.actor.update_target(self.tau)
            #self.critic.update_target(self.tau)
