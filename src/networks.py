import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, LeakyReLU, Activation
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error


class Actor(object):
    def __init__(self, observation_space: int, action_space: int):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.network = self.build(observation_space, action_space, cfg["LAYER_SIZES"])
        self.target = self.build(observation_space, action_space, cfg["LAYER_SIZES"])
        self.optimizer = Adam(learning_rate=cfg["ACTOR_LEARNING_RATE"])
        self.noise = cfg["NOISE_START"]
        self.noise_steps = cfg["NOISE_START"] / (cfg["STEPS"] * cfg["NOISE_DECAY"])

    def build(self, observation_space: int, action_space: int, layer_sizes: list)\
            -> Model:
        """
        Builds Actor Network.
        """
        input = Input(observation_space)

        dense1 = Dense(layer_sizes[0], kernel_initializer=he_normal())(input)
        activation1 = LeakyReLU()(dense1)
        dense2 = Dense(layer_sizes[1], kernel_initializer=he_normal())(activation1)
        activation2 = LeakyReLU()(dense2)
        dense3 = Dense(layer_sizes[2], kernel_initializer=he_normal())(activation2)
        activation3 = LeakyReLU()(dense3)

        output = Dense(action_space, kernel_initializer=he_normal())(activation3)
        activation_out = Activation("tanh")(output)

        model = Model(inputs=input, outputs=activation_out)

        return model

    def predict(self, state: np.ndarray, use_target: bool) -> np.ndarray:
        if use_target:
            predictions = np.array(self.target.predict_on_batch(state))
            return predictions
        else:
            predictions = np.array(self.network.predict_on_batch(state))
            if self.noise > 0:
                noise = np.random.normal(0, self.noise, predictions.shape)
                self.noise -= self.noise_steps
                return predictions + noise
        return predictions

    def update_target(self, tau: float):
        new_weights = np.array(self.target.get_weights()) * tau + np.array(self.network.get_weights()) * (1 - tau)
        self.target.set_weights(new_weights)


class Critic(object):
    def __init__(self, observation_space: int, action_space: int):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.loss = mean_squared_error
        self.optimizer = Adam(learning_rate=cfg["CRITIC_LEARNING_RATE"])
        self.network = self.build(observation_space, action_space, cfg["LAYER_SIZES"])
        self.target = self.build(observation_space, action_space, cfg["LAYER_SIZES"])

    def build(self, observation_space: int, action_space: int, layer_sizes: list)\
            -> Model:
        """
        Builds Critic Network.
        """

        input1 = Input(observation_space)
        input2 = Input(action_space)

        concat_layer = Concatenate()([input1, input2])
        dense1 = Dense(layer_sizes[0], kernel_initializer=he_normal())(concat_layer)
        activation1 = LeakyReLU()(dense1)
        dense2 = Dense(layer_sizes[1], kernel_initializer=he_normal())(activation1)
        activation2 = LeakyReLU()(dense2)
        dense3 = Dense(layer_sizes[2], kernel_initializer=he_normal())(activation2)
        activation3 = LeakyReLU()(dense3)

        output = Dense(1, kernel_initializer=he_normal())(activation3)
        activation_out = Activation("linear")(output)

        model = Model(inputs=[input1, input2], outputs=activation_out)

        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def predict(self, state: np.ndarray, action: np.ndarray, use_target: bool) -> np.ndarray:
        if use_target:
            return np.array(self.target.predict_on_batch([state, action]))
        else:
            return np.array(self.network.predict_on_batch([state, action]))

    def update_network(self, state: np.ndarray, action: np.ndarray, target: np.ndarray):
        self.network.train_on_batch([state, action], target)

    def update_target(self, tau: float):
        new_weights = np.array(self.target.get_weights()) * tau + np.array(self.network.get_weights()) * (1 - tau)
        self.target.set_weights(new_weights)
