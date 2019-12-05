import os
import yaml
from mlagents.envs.environment import UnityEnvironment
import numpy as np

from src.agent import Agent


def main():
    """"
    <TBD>
    """
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment {}.".format(cfg["EXECUTABLE"]))
    env = load_environment(cfg["EXECUTABLE"])
    env.reset()

    print("Creating Agent.")
    agent = Agent()

    print("Starting training with {} steps.".format(cfg["STEPS"]))
    acc_reward = 0
    for steps in range(cfg["STEPS"]):
        info = env.step(np.random.randn(10, 20))
        brain_info = info[env.external_brain_names[0]]

        acc_reward += sum(brain_info.rewards) / len(brain_info.rewards)

        if steps % cfg["VERBOSE_STEPS"] == 0:
            print("Total reward with {0} steps: {1:.5f}".format(steps, acc_reward))

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
    return UnityEnvironment(file_name=env_file)


if __name__ == '__main__':
    main()
