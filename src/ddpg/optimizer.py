import os
import yaml
import time
import numpy as np
from hyperopt import pyll, hp
from src.ddpg.main_unity import run
from hyperopt import fmin, tpe, hp
from datetime import datetime


def main():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
        date_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = os.path.join("runs_ddpg", "opt", date_string)
    
    space = {
        "buffer": hp.choice("buffer", np.arange(cfg["OPT_DDPG_BUFFER_SIZE"][0], cfg["OPT_DDPG_BUFFER_SIZE"][1] + 1, dtype=int)),
        "batch": hp.choice("batch", cfg["OPT_DDPG_BATCH_SIZE"]),
        "actor_rate": hp.uniform("actor_rate", cfg["OPT_DDPG_ACTOR_LEARNING_RATE"][0], cfg["OPT_DDPG_ACTOR_LEARNING_RATE"][1]),
        "critic_rate": hp.uniform("critic_rate", cfg["OPT_DDPG_CRITIC_LEARNING_RATE"][0], cfg["OPT_DDPG_CRITIC_LEARNING_RATE"][1]),
        "tau": hp.uniform("tau", cfg["OPT_DDPG_TAU"][0], cfg["OPT_DDPG_TAU"][1]),
        "gamma": hp.uniform("gamma", cfg["OPT_DDPG_GAMMA"][0], cfg["OPT_DDPG_GAMMA"][1]),
        "noise": hp.choice('noise', [
            ('gauss', {
                "start": hp.uniform("start", cfg["OPT_DDPG_GAUSS_START"][0], cfg["OPT_DDPG_GAUSS_START"][1]),
                "decay": hp.uniform("decay", cfg["OPT_DDPG_GAUSS_DECAY"][0], cfg["OPT_DDPG_GAUSS_DECAY"][1]),
                "min": hp.uniform("min", cfg["OPT_DDPG_GAUSS_MIN"][0], cfg["OPT_DDPG_GAUSS_MIN"][1])
            }),
            ('ou', None)
        ]),
        "log_dir": log_dir
        }
    
    best = fmin(
        fn=ddpg,
        space=space,
        algo=tpe.suggest,
        max_evals=cfg["OPT_DDPG_RUNS"],
        verbose=1
    )
    print("best: ", best)


def ddpg(space):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    print("Starting optimization run for space:", space)
    date_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(space["log_dir"], date_string)
    cfg["SUMMARY_FOLDER"] = log_dir
    
    cfg["STEPS"] = cfg["OPT_DDPG_STEPS"]
    cfg["CHECKPOINTS"] = cfg["OPT_DDPG_CHECKPOINTS"]
    cfg["VERBOSE_STEPS"] = cfg["OPT_DDPG_VERBOSE_STEPS"]
    
    cfg["BUFFER_SIZE"] = space["buffer"]
    cfg["BATCH_SIZE"] = space["batch"]
    cfg["ACTOR_LEARNING_RATE"] = space["actor_rate"]
    cfg["CRITIC_LEARNING_RATE"] = space["critic_rate"]
    cfg["TAU"] = space["tau"]
    cfg["GAMMA"] = space["gamma"]
    
    if(space["noise"][0] == "gauss"):
        cfg["GAUSSIAN_START"] = space["noise"][1]["start"]
        cfg["GAUSSIAN_DECAY"] = space["noise"][1]["decay"]
        cfg["GAUSSIAN_MIN"] = space["noise"][1]["min"]
        cfg["OUNOISE"] = False
    else:
        cfg["OUNOISE"] = True
    
    result = run(cfg)
    print("Run yielded {} max mean reward".format(result)) 
    
    return -result


if __name__ == '__main__':
    main()