import os
import yaml
import torch
from torch import tensor

from src.ddpg.actor_critic import Actor, Critic
from src.ddpg.replay_buffer import ReplayBuffer
from src.ddpg.summary import Summary


class DDPGAgent(object):
    """"
    Depicts the acting Entity.
    """

    def __init__(self, observation_space: int, action_space: int, summary: Summary):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        if cfg["UTILIZE_CUDA"]:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print("Utilizing device {}.".format(self.device))

        self.actor = Actor(observation_space, action_space, self.device)
        self.critic = Critic(observation_space, action_space, self.device)
        self.replay_buffer = ReplayBuffer()
        self.gamma = cfg["GAMMA"]
        self.tau = cfg["TAU"]
        self.output_path = os.path.join("models", cfg["EXECUTABLE"])
        self.summary = summary

    def learn(self):
        # Get experiences from replay buffer
        state, action, reward, new_state = self.replay_buffer.sample()

        state = tensor(state).float().to(self.device)
        action = tensor(action).float().to(self.device)
        reward = tensor(reward).float().to(self.device)
        new_state = tensor(new_state).float().to(self.device)

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
