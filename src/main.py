import os
import yaml
from mlagents.envs.environment import UnityEnvironment

from src.agent import Agent


def main():
    """"
    <TBD>
    """
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment " + cfg["EXECUTABLE"] + ".")
    env = load_environment(cfg["EXECUTABLE"])

    print("Creating Agent.")
    agent = Agent()


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
