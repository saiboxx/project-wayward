import os
import yaml
import time
import numpy as np
from typing import Tuple
from src.mlagents.environment import UnityEnvironment
from src.mlagents.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from torch import tensor

from src.ppo.agent import PPOAgent
from src.ppo.summary import Summary


def main():
    run([])


def run(cfg=[]):
    if cfg == []:
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment {}.".format(cfg["EXECUTABLE"]))
    num_envs = cfg["ENVIRONMENTS"]
    worker_id = np.random.randint(2000)
    envs, config_channel = list(zip(*[load_environment(cfg["EXECUTABLE"],
                                                       cfg["NO_GRAPHICS"],
                                                       worker_id + x)
                                      for x in range(num_envs)]))

    [c.set_configuration_parameters(time_scale=cfg["TIME_SCALE"]) for c in config_channel]
    [e.reset() for e in envs]

    group_name = envs[0].get_agent_groups()[0]
    group_spec = envs[0].get_agent_group_spec(group_name)
    action_space = group_spec.action_shape
    observation_space = group_spec.observation_shapes[0][0]
    step_results = [e.get_step_result(group_name) for e in envs]
    state = np.vstack([step_result.obs[0] for step_result in step_results])
    num_agents = len(state)
    summary = Summary(cfg)

    print("Creating Agent.")
    agent = PPOAgent(observation_space, action_space, cfg, summary)

    print("Starting training with {} steps.".format(cfg["STEPS"]))
    reward_cur_episode = np.zeros(num_agents)
    reward_last_episode = np.zeros(num_agents)
    rolling_reward_mean_episode = []
    start_time_episode = time.time()
    episode = 1

    start_time = time.time()
    for steps in range(1, cfg["STEPS"] + 1):
        state = tensor(state).float().detach().to(agent.device)
        action_distribution = agent.actor(state)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        value = agent.critic(state)

        action_temp = np.split(action, num_envs)
        [e.set_actions(group_name, a.cpu().numpy()) for (e, a) in zip(envs, action_temp)]
        [e.step() for e in envs]
        step_results = [e.get_step_result(group_name) for e in envs]
        new_state = np.vstack([step_result.obs[0] for step_result in step_results])
        reward = np.hstack([step_result.reward for step_result in step_results])
        done = np.hstack([step_result.done for step_result in step_results])
        agent.replay_buffer.add(state, action, reward, done, log_prob, value)

        if steps % (cfg["PPO_BUFFER_SIZE"] // num_agents) == 0:
            returns = agent.get_returns(tensor(new_state).float())
            agent.learn(returns)
            agent.replay_buffer.reset()

        mean_step_reward = np.mean(reward)
        reward_cur_episode += reward

        summary.add_scalar("Reward/Step", mean_step_reward)

        if steps % cfg["VERBOSE_STEPS"] == 0:
            elapsed_time = time.time() - start_time
            print("Ep. {0:>4} with {1:>7} steps total; {2:8.2f} last ep. rewards; {3:+.3f} step reward; {4}h elapsed" \
                  .format(episode, steps, reward_mean_episode, mean_step_reward, format_timedelta(elapsed_time)))

        for i, d in enumerate(done):
            if d:
                reward_last_episode[i] = reward_cur_episode[i]
                if steps >= cfg["STEPS"] * 0.9:
                    rolling_reward_mean_episode.append(reward_cur_episode[i])
                reward_cur_episode[i] = 0

        if done[0]:
            reward_mean_episode = reward_last_episode.mean()
            duration_last_episode = time.time() - start_time_episode
            start_time_episode = time.time()
            summary.add_scalar("Reward/Episode", reward_mean_episode, True)
            summary.add_scalar("Duration/Episode", duration_last_episode, True)
            summary.adv_episode()
            episode += 1

        if steps % cfg["CHECKPOINTS"] == 0:
            print("CHECKPOINT: Saving Models and Summary.")
            agent.save_models(steps)
            summary.writer.flush()

        state = new_state
        summary.adv_step()

    print("Closing environment.")
    [e.close() for e in envs]
    max_reward_mean_episode = np.mean(rolling_reward_mean_episode)
    summary.close(max_reward_mean_episode)
    return max_reward_mean_episode


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
