import os
import yaml
import time
from mlagents.envs.environment import UnityEnvironment

from src.agent import Agent


def main():
    """"
    <TBD>
    """
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment {}.".format(cfg["EXECUTABLE"]))
    worker_id = 0
    env = load_environment(cfg["EXECUTABLE"], cfg["NO_GRAPHICS"], worker_id)
    info = env.reset()
    brain_info = info[env.external_brain_names[0]]
    observation_space = brain_info.vector_observations.shape[1]
    action_space = brain_info.action_masks.shape[1]
    state = brain_info.vector_observations

    print("Creating Agent.")
    agent = Agent(observation_space, action_space)

    print("Starting training with {} steps.".format(cfg["STEPS"]))
    acc_reward = 0
    mean_reward = 0
    reward_cur_episode = []
    episode = 1
    start_time = time.time()
    for steps in range(1, cfg["STEPS"]):
        action = agent.actor.predict(state, use_target=False)
        info = env.step(action)
        brain_info = info[env.external_brain_names[0]]
        new_state = brain_info.vector_observations
        reward = brain_info.rewards
        done = brain_info.local_done[0]
        agent.replay_buffer.add(state, action, reward, new_state)
        agent.learn()

        mean_step = sum(brain_info.rewards) / len(brain_info.rewards)
        acc_reward += mean_step
        mean_reward += mean_step
        reward_cur_episode.append(brain_info.rewards[0])

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
            info = env.reset()
            brain_info = info[env.external_brain_names[0]]
            new_state = brain_info.vector_observations

        if steps % cfg["CHECKPOINTS"] == 0:
            print("CHECKPOINT: Saving Models.")
            agent.save_models(steps)

        if steps % 10000 == 0:
            worker_id += 1
            env.close()
            env = load_environment(cfg["EXECUTABLE"], cfg["NO_GRAPHICS"], worker_id)
            info = env.reset()
            brain_info = info[env.external_brain_names[0]]
            new_state = brain_info.vector_observations

        state = new_state

    print("Closing environment.")
    env.close()


def load_environment(env_name: str, no_graphics: bool, worker_id: int) -> UnityEnvironment:
    """
    Loads a Unity environment with a given key name.
    """
    env_path = os.path.join("executables", env_name)
    files_in_dir = os.listdir(env_path)
    env_file = [os.path.join(env_path, f) for f in files_in_dir
                if os.path.isfile(os.path.join(env_path, f))][0]
    return UnityEnvironment(file_name=env_file, no_graphics=no_graphics, worker_id=worker_id)


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
