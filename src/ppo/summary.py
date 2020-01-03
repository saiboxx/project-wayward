from torch.utils.tensorboard import SummaryWriter


class Summary(object):
    def __init__(self, cfg):
        self.writer = SummaryWriter()
        #self.writer.add_text("Config/Executable", cfg["EXECUTABLE"])
        #self.writer.add_text("Config/Graphics", str(cfg["NO_GRAPHICS"]))
        self.writer.add_hparams(self.hparams(cfg),dict())
        self.step = 1
        self.episode = 1
  
    def hparams(self, cfg):
        dict = {
            "STEPS": cfg["STEPS"],
            "PPO_BUFFER_SIZE": cfg["PPO_BUFFER_SIZE"],
            "PPO_BATCH_SIZE": cfg["PPO_BATCH_SIZE"],
            "PPO_EPOCHS": cfg["PPO_EPOCHS"],
            "PPO_STD": cfg["PPO_STD"],
            "PPO_GAMMA": cfg["PPO_GAMMA"],
            "PPO_LAMBDA": cfg["PPO_LAMBDA"],
            "PPO_EPSILON": cfg["PPO_EPSILON"],
            "PPO_CRITIC_DISCOUNT": cfg["CRITIC_DISCOUNT"],
            "PPO_ENTROPY_BETA": cfg["ENTROPY_BETA"],
            "PPO_ACTOR_LEARNING_RATE": cfg["PPO_ACTOR_LEARNING_RATE"],
            "PPO_CRITIC_LEARNING_RATE": cfg["PPO_CRITIC_LEARNING_RATE"]
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