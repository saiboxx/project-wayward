import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error


class Actor(object):
    def __init__(self, observation_space: int, action_space: int):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.network = self.build(observation_space, action_space, cfg["LAYER_SIZES"])
        self.optimizer = Adam(learning_rate=cfg["ACTOR_LEARNING_RATE"])

    def build(self, observation_space: int, action_space: int, layer_sizes: list)\
            -> Sequential:
        """
        Builds Actor Network.
        """
        model = Sequential()

        model.add(Dense(layer_sizes[0], input_dim=observation_space))
        model.add(Activation("relu"))

        if len(layer_sizes) > 1:
            for layer_size in layer_sizes[1:]:
                model.add(Dense(layer_size, kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3)))
                model.add(Activation("relu"))

        model.add(Dense(action_space, kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3)))
        model.add(Activation("linear"))

        return model

    def predict(self, state: np.ndarray) -> np.ndarray:
        # <TODO>
        # ADD NOISE
        # HAVE A LOOK AT Ornstein Uhlenbeck Action Noise!
        return self.network.predict(state)

    def update(self):
        pass


class Critic(object):
    def __init__(self, observation_space: int, action_space: int):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.loss = mean_squared_error
        self.optimizer = Adam(learning_rate=cfg["CRITIC_LEARNING_RATE"])
        self.network = self.build(observation_space, action_space, cfg["LAYER_SIZES"])

    def build(self, observation_space: int, action_space: int, layer_sizes: list)\
            -> Sequential:
        """
        Builds Actor Network.
        """

        model = Sequential()

        model.add(Dense(layer_sizes[0], input_dim=(observation_space + action_space)))
        model.add(Activation("relu"))

        if len(layer_sizes) > 1:
            for layer_size in layer_sizes[1:]:
                model.add(Dense(layer_size, kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)))
                model.add(Activation("relu"))

        model.add(Dense(1, kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)))
        model.add(Activation("linear"))

        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        net_input = np.hstack([state, action])
        return self.network.predict(net_input)

    def update(self, state: np.ndarray, action: np.ndarray, target: np.ndarray):
        net_input = np.hstack([state, action])
        self.network.fit(net_input, target, batch_size=len(target), verbose=0)

    def get_gradients(self, state: np.ndarray, a_output: np.ndarray):
        pass
        # <TODO>
        # Get gradients w.r.t. a_output