import os
import yaml
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

        tf.get_logger().setLevel("ERROR")

        self.actor = Actor(observation_space, action_space)
        self.critic = Critic(observation_space, action_space)
        self.replay_buffer = ReplayBuffer()
        self.gamma = cfg["GAMMA"]
        self.tau = cfg["TAU"]
        self.output_path = os.path.join("models", cfg["EXECUTABLE"])

    def learn(self):
        if self.replay_buffer.cur_buffer_size > (self.replay_buffer.max_buffer_size * 0.25):
            # Get experiences from replay buffer
            state, action, reward, new_state = self.replay_buffer.sample()

            # Calculate targets
            target_values = self.critic.predict(new_state,
                                                self.actor.predict(new_state, use_target=True),
                                                use_target=True)
            target = reward + self.gamma * target_values.flatten()

            # Update critic
            self.critic.update_network(state, action, target)

            # Get Gradient from critic
            with tf.GradientTape() as tape:
                action = self.actor.network(state)
                loss = -tf.reduce_mean(self.critic.network([state, action]))

            actor_grad = tape.gradient(loss, self.actor.network.trainable_variables)

            # Apply gradient to actor network
            self.actor.optimizer.apply_gradients(zip(actor_grad,
                                                     self.actor.network.trainable_variables))

            # Update target networks
            self.actor.update_target(self.tau)
            self.critic.update_target(self.tau)

    def save_models(self, steps: int):
        os.makedirs(self.output_path, exist_ok=True)
        self.actor.network.save(os.path.join(self.output_path, ("actor_" + str(steps) + ".h5")))
        self.actor.target.save(os.path.join(self.output_path, ("actor_target_" + str(steps) + ".h5")))
        self.critic.network.save(os.path.join(self.output_path, ("critic_" + str(steps) + ".h5")))
        self.critic.target.save(os.path.join(self.output_path, ("critic_target" + str(steps) + ".h5")))
