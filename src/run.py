import os
import yaml
import time
import tensorflow as tf
from mlagents.environment import UnityEnvironment


def main():
    """
    Loads network specified in cfg file and runs Unity Environment
    """
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment {}.".format(cfg["RUN_EXECUTABLE"]))
    env = load_environment(cfg["RUN_EXECUTABLE"])
    info = env.reset(train_mode=False)
    brain_info = info[env.external_brain_names[0]]
    state = brain_info.vector_observations

    print("Loading Model.")
    actor = tf.keras.models.load_model(cfg["RUN_MODEL"])

    print("Starting Run with {} steps.".format(cfg["STEPS"]))
    acc_reward = 0
    mean_reward = 0
    reward_cur_episode = []
    mean_reward_episodes = 0
    episode = 1
    steps = 1
    start_time = time.time()
    while True:
        action = actor.predict(state)
        info = env.step(action)
        brain_info = info[env.external_brain_names[0]]
        new_state = brain_info.vector_observations
        done = brain_info.local_done[0]

        mean_step = sum(brain_info.rewards) / len(brain_info.rewards)
        acc_reward += mean_step
        mean_reward += mean_step
        reward_cur_episode.append(brain_info.rewards[0])

        if episode % cfg["VERBOSE_EPISODES"] == 0:
            mean_reward = mean_reward / cfg["VERBOSE_EPISODES"]
            elapsed_time = time.time() - start_time
            print("Ep {0:>4} with {1:>7} steps total; {2:8.3f} mean ep. reward; {3:+.3f} step reward; {4}h elapsed" \
                  .format(episode, steps, mean_reward_episodes, mean_reward, format_timedelta(elapsed_time)))
            mean_reward = 0

        if done:
            mean_reward_episodes += (sum(reward_cur_episode) - mean_reward_episodes) / episode
            reward_cur_episode = []
            episode += 1
            info = env.reset(train_mode=False)
            brain_info = info[env.external_brain_names[0]]
            new_state = brain_info.vector_observations

        if episode >= cfg["RUN_EPISODES"]:
            break
        state = new_state
        steps += 1

    print("Closing environment.")
    env.close()


def load_environment(env_name: str) -> UnityEnvironment:
    """
    Loads a Unity environment with a given key name.
    """
    env_path = os.path.join("executables", env_name)
    files_in_dir = os.listdir(env_path)
    env_file = [os.path.join(env_path, f) for f in files_in_dir
                if os.path.isfile(os.path.join(env_path, f))][0]
    env = UnityEnvironment(file_name=env_file, worker_id=2)
    return env


def format_timedelta(timedelta):
    total_seconds = int(timedelta)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)


if __name__ == '__main__':
    main()
