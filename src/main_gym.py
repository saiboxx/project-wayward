import os
import yaml
import time
import gym
import numpy as np
from torch import tensor, save
from tqdm import tqdm
from src.agent import Agent
from src.summary import Summary


def main():
    """"
    <TBD>
    """
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
    agent = Agent(observation_space, action_space, summary)

    print("Initiating with warm-up phase")
    for _ in tqdm(range(cfg["BUFFER_SIZE"]//100)):
        action = np.random.uniform(-1, 1, size=(len(state), action_space))
        env.render()
        new_state, reward, done, info = env.step(np.reshape(action, action_space))
        new_state = np.reshape(new_state, (1, observation_space))
        agent.replay_buffer.add(state, action, reward, new_state)

    print("Starting training with {} steps.".format(cfg["STEPS"]))
    acc_reward = 0
    mean_reward = 0
    reward_cur_episode = []
    reward_last_episode = 0
    episode = 1
    start_time = time.time()
    for steps in range(1, cfg["STEPS"]):
        action = agent.actor.predict(tensor(state).float(), use_target=False)
        action = action.cpu().numpy()
        env.render()
        new_state, reward, done, info = env.step(np.reshape(action, action_space))
        new_state = np.reshape(new_state, (1, observation_space))
        agent.replay_buffer.add(state, action, reward, new_state)

        agent.learn()

        acc_reward += reward
        mean_reward += reward
        reward_cur_episode.append(reward)

        if steps % cfg["VERBOSE_STEPS"] == 0:
            mean_reward = mean_reward / cfg["VERBOSE_STEPS"]
            elapsed_time = time.time() - start_time
            print("Ep. {0:>4} with {1:>7} steps total; {2:8.2f} last ep. reward; {3:+.3f} step reward; {4}h elapsed" \
                  .format(episode, steps, reward_last_episode, mean_reward, format_timedelta(elapsed_time)))
            mean_reward = 0

        if done:
            reward_last_episode = sum(reward_cur_episode)
            reward_cur_episode = []
            episode += 1
            env.reset()
            state = np.zeros((1, observation_space))

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
