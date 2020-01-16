import os
import yaml
import time
import numpy as np
from typing import Tuple
from src.mlagents.environment import UnityEnvironment
from src.mlagents.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel

import torch
from torch import tensor


def main():
    """
    Loads network specified in cfg file and runs Unity Environment
    """
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment {}.".format(cfg["RUN_EXECUTABLE"]))
    worker_id = np.random.randint(20)
    env, config_channel = load_environment(cfg["RUN_EXECUTABLE"], cfg["RUN_NO_GRAPHICS"], worker_id)
    env.reset()
    group_name = env.get_agent_groups()[0]
    step_result = env.get_step_result(group_name)
    state = step_result.obs[0]
    num_agents = len(state)

    print("Loading Model.")
    actor = torch.load(cfg["RUN_MODEL"])
    actor.eval()

    print("Starting Run with {} steps.".format(cfg["RUN_STEPS"]))
    acc_reward = 0
    mean_reward = 0
    reward_cur_episode = np.zeros(num_agents)
    reward_last_episode = np.zeros(num_agents)
    reward_mean_episode = 0
    episode = 1
    start_time = time.time()
    for steps in range(1, cfg["RUN_STEPS"] + 1):
        with torch.no_grad():
            state = tensor(state).float().detach()
            action_distribution = actor(state)
            action = action_distribution.sample()
        env.set_actions(group_name, action.cpu().numpy())
        env.step()
        step_result = env.get_step_result(group_name)
        new_state = step_result.obs[0]
        reward = step_result.reward
        done = step_result.done

        mean_step = sum(reward) / len(reward)
        reward_cur_episode += reward

        for i, d in enumerate(done):
            if d:
                reward_last_episode[i] = reward_cur_episode[i]
                reward_cur_episode[i] = 0

        if done[0]:
            reward_mean_episode = reward_last_episode.mean()
            elapsed_time = time.time() - start_time
            print("Ep. {0:>4} with {1:>7} steps total; {2:8.2f} last ep. rewards; {3}h elapsed" \
                  .format(episode, steps, reward_mean_episode, format_timedelta(elapsed_time)))
            episode += 1

        state = new_state

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
