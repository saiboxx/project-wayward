import os
import yaml
import time
import gym
import numpy as np
from torch import tensor, save
from src.ppo.agent import PPOAgent
from src.ppo.summary import Summary


def main():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment.")
    env = gym.make("LunarLanderContinuous-v2")
    env.reset()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    state = np.zeros((1, observation_space))
    summary = Summary(cfg)

    print("Creating Agent.")
    agent = PPOAgent(observation_space, action_space, cfg, summary)

    print("Starting training with {} steps.".format(cfg["STEPS"]))
    mean_step_reward = []
    reward_cur_episode = []
    reward_last_episode = 0
    episode = 1
    start_time = time.time()
    for steps in range(1, cfg["STEPS"]):
        state = tensor(state).float().detach()
        action_distribution = agent.actor(state)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        value = agent.critic(state)
        env.render()
        new_state, reward, done, info = env.step(np.reshape(action.cpu().numpy(), action_space))

        new_state = np.reshape(new_state, (1, observation_space))
        agent.replay_buffer.add(state, action, reward, done, log_prob, value)

        if steps % cfg["PPO_BUFFER_SIZE"] == 0:
            returns = agent.get_returns(tensor(new_state).float())
            agent.learn(returns)
            agent.replay_buffer.reset()

        mean_step_reward.append(reward)
        reward_cur_episode.append(reward)

        if steps % cfg["VERBOSE_STEPS"] == 0:
            elapsed_time = time.time() - start_time
            print("Ep. {0:>4} with {1:>7} steps total; {2:8.2f} last ep. reward; {3:+.3f} step reward; {4}h elapsed" \
                  .format(episode, steps, reward_last_episode, np.mean(mean_step_reward), format_timedelta(elapsed_time)))
            mean_step_reward = []

        if done:
            reward_last_episode = sum(reward_cur_episode)
            reward_cur_episode = []
            episode += 1
            env.reset()

        state = new_state

    os.makedirs("models/lunar_lander/", exist_ok=True)
    save(agent.actor.network, os.path.join("models/lunar_lander/", "actor.pt"))
    save(agent.actor.target, os.path.join("models/lunar_lander/", "actor_target.pt"))

    print("Closing environment.")
    env.close()


def predict_total_time(time_elapsed, episodes_elapsed, episodes_total):
    time_per_episode = time_elapsed / episodes_elapsed
    return episodes_total * time_per_episode


def format_timedelta(timedelta):
    total_seconds = int(timedelta)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)


if __name__ == '__main__':
    main()
