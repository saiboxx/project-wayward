from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

class Summary(object):
    def __init__(self, cfg):
        folder = cfg["SUMMARY_FOLDER"]
        
        if(folder is None):
            date_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            folder = os.path.join("runs_ppo", date_string)
        
        self.writer = SummaryWriter(log_dir=folder)
        self.step = 1
        self.episode = 1
        self.cfg = cfg
  
    def hparams(self, cfg):
        dict = {
            "STEPS": int(cfg["STEPS"]),
            "GRAVITY": cfg["GRAVITY"],
            "PPO_BUFFER_SIZE": int(cfg["PPO_BUFFER_SIZE"]),
            "PPO_BATCH_SIZE": int(cfg["PPO_BATCH_SIZE"]),
            "PPO_EPOCHS": int(cfg["PPO_EPOCHS"]),
            "PPO_STD": cfg["PPO_STD"],
            "PPO_GAMMA": cfg["PPO_GAMMA"],
            "PPO_LAMBDA": cfg["PPO_LAMBDA"],
            "PPO_EPSILON": cfg["PPO_EPSILON"],
            "PPO_CRITIC_DISCOUNT": cfg["CRITIC_DISCOUNT"],
            "PPO_ENTROPY_BETA": cfg["ENTROPY_BETA"],
            "PPO_ACTOR_LEARNING_RATE": cfg["PPO_ACTOR_LEARNING_RATE"],
            "PPO_CRITIC_LEARNING_RATE": cfg["PPO_CRITIC_LEARNING_RATE"]
        }
        print(dict)
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