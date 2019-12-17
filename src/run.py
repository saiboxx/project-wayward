import os
import yaml
import time
import numpy as np
from typing import Tuple
from src.mlagents.environment import UnityEnvironment
from src.mlagents.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel

import torch
from torch import from_numpy


def main():
    """
    Loads network specified in cfg file and runs Unity Environment
    """
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment {}.".format(cfg["RUN_EXECUTABLE"]))
    worker_id = 1
    env, config_channel = load_environment(cfg["EXECUTABLE"], cfg["NO_GRAPHICS"], worker_id)
    env.reset()
    group_name = env.get_agent_groups()[0]
    group_spec = env.get_agent_group_spec(group_name)
    step_result = env.get_step_result(group_name)
    state = step_result.obs[0]

    print("Loading Model.")
    actor = torch.load(cfg["RUN_MODEL"])

    print("Starting Run with {} steps.".format(cfg["STEPS"]))
    acc_reward = 0
    mean_reward = 0
    reward_cur_episode = []
    mean_reward_episodes = 0
    episode = 1
    reward_last_episode = 0
    start_time = time.time()
    for steps in range(1, cfg["RUN_STEPS"]):
        with torch.no_grad():
            action = actor(from_numpy(np.array(state)).float())
        action = action.cpu().numpy()
        env.set_actions(group_name, action)
        env.step()
        step_result = env.get_step_result(group_name)
        new_state = step_result.obs[0]
        reward = step_result.reward
        done = step_result.done[0]

        mean_step = sum(reward) / len(reward)
        acc_reward += mean_step
        mean_reward += mean_step
        reward_cur_episode.append(reward[0])

        if episode % cfg["RUN_VERBOSE_STEPS"] == 0:
            mean_reward = mean_reward / cfg["VERBOSE_EPISODES"]
            elapsed_time = time.time() - start_time
            print("Ep {0:>4} with {1:>7} steps total; {2:8.3f} mean ep. reward; {3:+.3f} step reward; {4}h elapsed" \
                  .format(episode, steps, mean_reward_episodes, mean_reward, format_timedelta(elapsed_time)))
            mean_reward = 0

        if done:
            mean_reward = mean_reward / cfg["VERBOSE_STEPS"]
            elapsed_time = time.time() - start_time
            print("Ep. {0:>4} with {1:>7} steps total; {2:8.2f} last ep. reward; {3:+.3f} step reward; {4}h elapsed" \
                  .format(episode, steps, reward_last_episode, mean_reward, format_timedelta(elapsed_time)))
            mean_reward = 0

        if done:
            reward_last_episode = sum(reward_cur_episode)
            reward_cur_episode = []
            episode += 1

    print("Closing environment.")
    env.close()


def load_environment(env_name: str, no_graphics: bool, worker_id: int) \
        -> Tuple[UnityEnvironment, EngineConfigurationChannel]:
    """
    Loads a Unity environment with a given key name.
    """
    env_path = os.path.join("executables", env_name)
    files_in_dir = os.listdir(env_path)
    env_file = [os.path.join(env_path, f) for f in files_in_dir
                if os.path.isfile(os.path.join(env_path, f))][0]
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_file,
                           no_graphics=no_graphics,
                           worker_id=worker_id,
                           side_channels=[engine_configuration_channel])
    return env, engine_configuration_channel


def format_timedelta(timedelta):
    total_seconds = int(timedelta)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)


if __name__ == '__main__':
    main()
