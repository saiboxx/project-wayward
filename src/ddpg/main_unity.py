import os
import yaml
import time
import numpy as np
from typing import Tuple
from src.mlagents.environment import UnityEnvironment
from src.mlagents.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from torch import from_numpy
from tqdm import tqdm

from src.ddpg.agent import Agent
from src.ddpg.summary import Summary


def main():
    """"
    <TBD>
    """
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment {}.".format(cfg["EXECUTABLE"]))
    worker_id = np.random.randint(20)
    env, config_channel = load_environment(cfg["EXECUTABLE"], cfg["NO_GRAPHICS"], worker_id)
    env.reset()
    group_name = env.get_agent_groups()[0]
    group_spec = env.get_agent_group_spec(group_name)
    action_space = group_spec.action_shape
    observation_space = group_spec.observation_shapes[0][0]
    step_result = env.get_step_result(group_name)
    state = step_result.obs[0]
    num_agents = len(state)
    summary = Summary(cfg)

    print("Creating Agent.")
    agent = Agent(observation_space, action_space, summary)

    print("Starting training with {} steps.".format(cfg["STEPS"]))
    acc_reward = 0
    mean_reward = 0
    reward_cur_episode = []
    reward_last_episode = 0
    start_time_episode = time.time()
    episode = 1

    print("Initiating with warm-up phase")
    for b in tqdm(range(cfg["BUFFER_SIZE"]//num_agents)):
        action = np.random.uniform(-1, 1, size=(len(state), action_space))
        env.set_actions(group_name, action)
        env.step()
        step_result = env.get_step_result(group_name)
        new_state = step_result.obs[0]
        reward = step_result.reward
        done = step_result.done[0]
        agent.replay_buffer.add(state, action, reward, new_state)

    start_time = time.time()
    for steps in range(1, cfg["STEPS"] + 1):
        action = agent.actor.predict(from_numpy(np.array(state)).float(), use_target=False)
        action = action.cpu().numpy()
        
        env.set_actions(group_name, action)
        env.step()
        step_result = env.get_step_result(group_name)
        new_state = step_result.obs[0]
        reward = step_result.reward
        done = step_result.done[0]
        agent.replay_buffer.add(state, action, reward, new_state)
    
        agent.learn()

        mean_step = sum(reward) / len(reward)
        acc_reward += mean_step
        mean_reward += mean_step
        reward_cur_episode.append(reward[0])
        
        summary.add_scalar("Reward/Step", mean_step);

        if steps % cfg["VERBOSE_STEPS"] == 0:
            mean_reward = mean_reward / cfg["VERBOSE_STEPS"]
            elapsed_time = time.time() - start_time
            print("Ep. {0:>4} with {1:>7} steps total; {2:8.2f} last ep. reward; {3:+.3f} step reward; {4}h elapsed" \
                  .format(episode, steps, reward_last_episode, mean_reward, format_timedelta(elapsed_time)))
            mean_reward = 0

        if done:
            reward_last_episode = sum(reward_cur_episode)
            reward_cur_episode = []
            duration_last_episode = time.time() - start_time_episode
            start_time_episode = time.time()
            summary.add_scalar("Reward/Episode", reward_last_episode, True);
            summary.add_scalar("Duration/Episode", duration_last_episode, True);
            summary.adv_episode()
            episode += 1

        if steps % cfg["CHECKPOINTS"] == 0:
            print("CHECKPOINT: Saving Models and Summary.")
            agent.save_models(steps)
            summary.writer.flush()

        state = new_state
        summary.adv_step()

    print("Closing environment.")
    env.close()
    summary.writer.flush()
    summary.writer.close()


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
