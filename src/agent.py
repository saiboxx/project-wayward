import os
import yaml
import torch
from torch import from_numpy

from src.actor_critic import Actor, Critic
from src.replay_buffer import ReplayBuffer
from src.summary import Summary

class Agent(object):
    """"
    Depicts the acting Entity.
    """

    def __init__(self, observation_space: int, action_space: int, summary: Summary):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.actor = Actor(observation_space, action_space)
        self.critic = Critic(observation_space, action_space)
        self.replay_buffer = ReplayBuffer()
        self.gamma = cfg["GAMMA"]
        self.tau = cfg["TAU"]
        self.output_path = os.path.join("models", cfg["EXECUTABLE"])
        self.summary = summary

    def learn(self):
        # Get experiences from replay buffer
        state, action, reward, new_state = self.replay_buffer.sample()

        state = from_numpy(state).float()
        action = from_numpy(action).float()
        reward = from_numpy(reward).float()
        new_state = from_numpy(new_state).float()

        # Calculate targets
        target_values = self.critic.predict(new_state,
                                            self.actor.predict(new_state, use_target=True),
                                            use_target=True)

        target = reward.unsqueeze(1) + self.gamma * target_values

        # Update critic
        self.critic.update_network(state, action, target)

        # Get Gradient from critic and apply to actor
        a_pred = self.actor.network(state)
        loss = self.critic.network(state, a_pred)
        mean_loss = -1 * loss.mean()
        self.actor.optimizer.zero_grad()
        mean_loss.backward()
        self.actor.optimizer.step()
        self.summary.add_scalar("Loss/Critic", mean_loss)

        # Update target networks
        self.actor.update_target(self.tau)
        self.critic.update_target(self.tau)

    def save_models(self, steps: int):
        os.makedirs(self.output_path, exist_ok=True)
        torch.save(self.actor.network, os.path.join(self.output_path, ("actor_" + str(steps) + ".pt")))
        torch.save(self.actor.target, os.path.join(self.output_path, ("actor_target_" + str(steps) + ".pt")))
        torch.save(self.critic.network, os.path.join(self.output_path, ("critic_" + str(steps) + ".pt")))
        torch.save(self.critic.target, os.path.join(self.output_path, ("critic_target" + str(steps) + ".pt")))
