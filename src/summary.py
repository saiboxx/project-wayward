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
            "NOISE_START": cfg["NOISE_START"],
            "NOISE_DECAY": cfg["NOISE_DECAY"],
            "BUFFER_SIZE": cfg["BUFFER_SIZE"],
            "BATCH_SIZE": cfg["BATCH_SIZE"],
            "ACTOR_LEARNING_RATE": cfg["ACTOR_LEARNING_RATE"],
            "CRITIC_LEARNING_RATE": cfg["CRITIC_LEARNING_RATE"],
            "TAU": cfg["TAU"],
            "GAMMA": cfg["GAMMA"]
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