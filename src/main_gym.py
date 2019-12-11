import os
import yaml
import time
import gym
from src.agent import Agent


def main():
    """"
    <TBD>
    """
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    print("Loading environment {}.".format(cfg["EXECUTABLE"]))
    worker_id = 0
    env = gym.make("LunarLanderContinuous-v2")
    state, reward, done, info = env.reset()

    print("Creating Agent.")
    agent = Agent(env.observation_space, env.action_space)

    print("Starting training with {} steps.".format(cfg["STEPS"]))
    acc_reward = 0
    mean_reward = 0
    reward_cur_episode = []
    reward_last_episode = 0
    episode = 1
    start_time = time.time()
    for steps in range(1, cfg["STEPS"]):
        action = agent.actor.predict(state, use_target=False)
        new_state, reward, done, info = env.step(action)
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
            new_state, reward, done, info = env.step(action)

        if steps % cfg["CHECKPOINTS"] == 0:
            print("CHECKPOINT: Saving Models.")
            agent.save_models(steps)

        state = new_state

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
