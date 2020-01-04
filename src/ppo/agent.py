import os
import yaml
import torch
from numpy.random import shuffle
from numpy import array_split, arange
from torch import tensor, no_grad
from torch.nn import MSELoss
from torch.optim import Adam

from src.ppo.networks import Actor2Layer, Critic2Layer, Actor3Layer, Critic3Layer
from src.ppo.replay_buffer import ReplayBuffer
from src.ppo.summary import Summary


class PPOAgent(object):
    """"
    Depicts the acting Entity.
    """

    def __init__(self, observation_space: int, action_space: int, cfg, summary: Summary):
        if len(cfg["LAYER_SIZES"]) == 2:
            self.actor = Actor2Layer(observation_space, action_space, cfg["LAYER_SIZES"], cfg["PPO_STD"])
            self.critic = Critic2Layer(observation_space, cfg["LAYER_SIZES"])
        else:
            self.actor = Actor3Layer(observation_space, action_space, cfg["LAYER_SIZES"], cfg["PPO_STD"])
            self.critic = Critic3Layer(observation_space, cfg["LAYER_SIZES"])

        self.actor_optimizer = Adam(self.actor.parameters(), lr=cfg["PPO_ACTOR_LEARNING_RATE"])
        self.critic_optimizer = Adam(self.critic.parameters(), lr=cfg["PPO_CRITIC_LEARNING_RATE"])
        self.critic_loss = MSELoss()

        self.replay_buffer = ReplayBuffer()
        self.gamma = cfg["PPO_GAMMA"]
        self.lam = cfg["PPO_LAMBDA"]
        self.epsilon = cfg["PPO_EPSILON"]
        self.critic_discount = cfg["CRITIC_DISCOUNT"]
        self.entropy_beta = cfg["ENTROPY_BETA"]

        self.epochs = cfg["PPO_EPOCHS"]
        self.batch_size = cfg["PPO_BATCH_SIZE"]

        self.output_path = os.path.join("models", cfg["EXECUTABLE"])
        self.summary = summary

    def learn(self, returns: tensor):
        with no_grad():
            returns = torch.cat(returns)
            log_probs = torch.cat(self.replay_buffer.log_probs)
            values = torch.cat(self.replay_buffer.values)
        states = torch.cat(self.replay_buffer.states)
        actions = torch.cat(self.replay_buffer.actions)

        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        counter = 1
        for _ in range(self.epochs):

            # Get shuffled indices
            ind_list = arange(states.size(0))
            shuffle(ind_list)
            chunks = (states.size(0) // self.batch_size)
            ind_list = array_split(ind_list, chunks)

            # Go through experiences in batches
            for ind in ind_list:
                states_batch = states[ind, :]
                actions_batch = actions[ind, :]
                log_probs_batch = log_probs[ind, :]
                returns_batch = returns[ind, :]
                advantage_batch = advantage[ind, :]

                # Get current extimates
                action_distribution = self.actor(states_batch)
                values = self.critic(states_batch)

                # Calculate entropy of distribution
                entropy = action_distribution.entropy().mean()

                # Calculate new log probs
                new_log_probs = action_distribution.log_prob(actions_batch)

                # Get ratio of new policy / old policy
                ratio = (new_log_probs - log_probs_batch).exp()

                # Calculate surrogate functions
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage_batch

                # Calculate losses
                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = self.critic_loss(returns_batch, values)
                
                # Get total loss
                loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * entropy
                
                #Log losses in tensorboard
                self.summary.add_scalar("Loss/Actor", actor_loss)
                self.summary.add_scalar("Loss/Critic", critic_loss)
                self.summary.add_scalar("Loss/Total", loss)
                self.summary.add_scalar("Entropy", entropy)

                # Backpropagate the error
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def get_returns(self, new_state: tensor) -> tensor:
        new_value = self.critic(new_state)
        rewards = self.replay_buffer.rewards
        values = self.replay_buffer.values + [new_value]
        masks = self.replay_buffer.masks

        with no_grad():
            # GAE Calculation thanks to https://github.com/colinskow
            gae = 0
            returns = []
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
                gae = delta + self.gamma * self.lam * masks[step] * gae
                # prepend to get correct order back
                returns.insert(0, gae + values[step])
        return returns

    def save_models(self, steps: int):
        os.makedirs(self.output_path, exist_ok=True)
        torch.save(self.actor, os.path.join(self.output_path, ("actor_" + str(steps) + ".pt")))
        torch.save(self.critic, os.path.join(self.output_path, ("critic" + str(steps) + ".pt")))
