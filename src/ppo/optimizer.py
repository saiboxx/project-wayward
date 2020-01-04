import os
import yaml
import time
import numpy as np
from hyperopt import pyll, hp
from src.ppo.main_unity import run
from hyperopt import fmin, tpe, hp
from datetime import datetime


def main():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
        date_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = os.path.join("runs_ppo", "opt", date_string)
    
    space = {
        "buffer": hp.choice("buffer", np.arange(cfg["OPT_PPO_BUFFER_SIZE"][0], cfg["OPT_PPO_BUFFER_SIZE"][1] + 1, dtype=int)),
        "batch": hp.choice("batch", cfg["OPT_PPO_BATCH_SIZE"]),
        "epochs": hp.choice("epochs", np.arange(cfg["OPT_PPO_EPOCHS"][0], cfg["OPT_PPO_EPOCHS"][1] + 1, dtype=int)),
        "std": hp.uniform("std", cfg["OPT_PPO_STD"][0], cfg["OPT_PPO_STD"][1]),
        "gamma": hp.uniform("gamma", cfg["OPT_PPO_GAMMA"][0], cfg["OPT_PPO_GAMMA"][1]),
        "lmd": hp.uniform("lmd", cfg["OPT_PPO_LAMBDA"][0], cfg["OPT_PPO_LAMBDA"][1]),
        "eps": hp.uniform("eps", cfg["OPT_PPO_EPSILON"][0], cfg["OPT_PPO_EPSILON"][1]),
        "discount": hp.uniform("discount", cfg["OPT_CRITIC_DISCOUNT"][0], cfg["OPT_CRITIC_DISCOUNT"][1]),
        "entropy": hp.uniform("entropy", cfg["OPT_ENTROPY_BETA"][0], cfg["OPT_ENTROPY_BETA"][1]),
        "actor_rate": hp.uniform("actor_rate", cfg["OPT_PPO_ACTOR_LEARNING_RATE"][0], cfg["OPT_PPO_ACTOR_LEARNING_RATE"][1]),
        "critic_rate": hp.uniform("critic_rate", cfg["OPT_PPO_CRITIC_LEARNING_RATE"][0], cfg["OPT_PPO_CRITIC_LEARNING_RATE"][1]),
        "log_dir": log_dir
        }
    
    best = fmin(
        fn=ppo,
        space=space,
        algo=tpe.suggest,
        max_evals=cfg["OPT_PPO_RUNS"],
        verbose=1
    )
    print("best: ", best)


def ppo(space):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    print("Starting optimization run for space:", space)
    date_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(space["log_dir"], date_string)
    cfg["SUMMARY_FOLDER"] = log_dir
    
    cfg["STEPS"] = cfg["OPT_PPO_STEPS"]
    cfg["CHECKPOINTS"] = cfg["OPT_PPO_CHECKPOINTS"]
    cfg["VERBOSE_STEPS"] = cfg["OPT_PPO_VERBOSE_STEPS"]
    
    cfg["PPO_BUFFER_SIZE"] = min(space["buffer"] * space["batch"], 409600)
    cfg["PPO_BATCH_SIZE"] = space["batch"]
    cfg["PPO_EPOCHS"] = space["epochs"]
    cfg["PPO_STD"] = space["std"]
    cfg["PPO_GAMMA"] = space["gamma"]
    cfg["PPO_LAMBDA"] = space["lmd"]
    cfg["PPO_EPSILON"] = space["eps"]
    cfg["CRITIC_DISCOUNT"] = space["discount"]
    cfg["ENTROPY_BETA:"] = space["entropy"]
    cfg["PPO_ACTOR_LEARNING_RATE"] = space["actor_rate"]
    cfg["PPO_CRITIC_LEARNING_RATE"] = space["critic_rate"]
    
    result = run(cfg)
    print("Run yielded {} max mean reward".format(result)) 
    
    return -result


if __name__ == '__main__':
    main()