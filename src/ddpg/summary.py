from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

class Summary(object):
    def __init__(self, cfg):
        folder = cfg["SUMMARY_FOLDER"]
        
        if(folder is None):
            date_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            folder = os.path.join("runs_ddpg", date_string)
    
        self.writer = SummaryWriter(log_dir=folder)
        self.step = 1
        self.episode = 1
        self.cfg = cfg
  
    def hparams(self, cfg):
        ounoise = cfg["OUNOISE"]
        dict = {
            "STEPS": int(cfg["STEPS"]),
            "DDPG_LEARN_STEPS": int(cfg["LEARN_STEPS"]),
            "DDPG_OUNOISE": ounoise,
            "DDPG_GAUSSIAN_START": cfg["GAUSSIAN_START"] if not ounoise else "",
            "DDPG_GAUSSIAN_DECAY": cfg["GAUSSIAN_DECAY"] if not ounoise else "",
            "DDPG_GAUSSIAN_MIN": cfg["GAUSSIAN_MIN"] if not ounoise else "",
            "DDPG_BUFFER_SIZE": int(cfg["BUFFER_SIZE"]),
            "DDPG_BATCH_SIZE": int(cfg["BATCH_SIZE"]),
            "DDPG_ACTOR_LEARNING_RATE": cfg["ACTOR_LEARNING_RATE"],
            "DDPG_CRITIC_LEARNING_RATE": cfg["CRITIC_LEARNING_RATE"],
            "DDPG_TAU": cfg["TAU"],
            "DDPG_GAMMA": cfg["GAMMA"]
        }
        i = 1
        for val in cfg["LAYER_SIZES"]:
            dict["LAYER_" + str(i)] = val
            i += 1
        return dict
        
    def add_scalar(self, tag: str, value, episode: bool = False):
        step = self.step
        if episode:
            step = self.episode
            
        self.writer.add_scalar(tag, value, step)
        
    def adv_step(self):
        self.step += 1
    
    def adv_episode(self):
        self.episode += 1
        
    def close(self, max_reward):
        self.writer.add_hparams(self.hparams(self.cfg),{"MAX_REWARD": max_reward})
        self.writer.flush()
        self.writer.close()