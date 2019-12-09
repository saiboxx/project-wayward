import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Activation
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error


class Actor(object):
    def __init__(self, observation_space: int, action_space: int):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.network = self.build(observation_space, action_space, cfg["LAYER_SIZES"])
        self.target = self.build(observation_space, action_space, cfg["LAYER_SIZES"])
        self.optimizer = Adam(learning_rate=cfg["ACTOR_LEARNING_RATE"])

    def build(self, observation_space: int, action_space: int, layer_sizes: list)\
            -> Model:
        """
        Builds Actor Network.
        """
        input = Input(observation_space)

        dense1 = Dense(layer_sizes[0], kernel_initializer=RandomUniform(minval=-0.03, maxval=0.03))(input)
        activation1 = Activation("relu")(dense1)
        dense2 = Dense(layer_sizes[1], kernel_initializer=RandomUniform(minval=-0.03, maxval=0.03))(activation1)
        activation2 = Activation("relu")(dense2)
        dense3 = Dense(layer_sizes[2], kernel_initializer=RandomUniform(minval=-0.03, maxval=0.03))(activation2)
        activation3 = Activation("relu")(dense3)

        output = Dense(action_space, kernel_initializer=RandomUniform(minval=-0.03, maxval=0.03))(activation3)
        activation_out = Activation("tanh")(output)

        model = Model(inputs=input, outputs=activation_out)

        return model

    def predict(self, state: np.ndarray, use_target: bool) -> np.ndarray:
        if use_target:
            predictions = self.target.predict(state)
        else:
            predictions = self.network.predict(state)
        noise = np.random.normal(0, 0.1, predictions.shape)
        return predictions + noise

    def update_target(self, tau: float):
        new_weights = np.array(self.target.get_weights()) * tau + np.array(self.network.get_weights()) * (1 - tau)
        return self.target.set_weights(new_weights)


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
        dense1 = Dense(layer_sizes[0], kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3))(concat_layer)
        activation1 = Activation("relu")(dense1)
        dense2 = Dense(layer_sizes[1], kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3))(activation1)
        activation2 = Activation("relu")(dense2)
        dense3 = Dense(layer_sizes[2], kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3))(activation2)
        activation3 = Activation("relu")(dense3)

        output = Dense(1, kernel_initializer=RandomUniform(minval=-0.3, maxval=0.3))(activation3)
        activation_out = Activation("linear")(output)

        model = Model(inputs=[input1, input2], outputs=output)

        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def predict(self, state: np.ndarray, action: np.ndarray, use_target: bool) -> np.ndarray:
        if use_target:
            return self.target.predict([state, action])
        else:
            return self.network.predict([state, action])

    def update_network(self, state: np.ndarray, action: np.ndarray, target: np.ndarray):
        self.network.fit([state, action], target, batch_size=len(target), verbose=0)

    def update_target(self, tau: float):
        new_weights = np.array(self.target.get_weights()) * tau + np.array(self.network.get_weights()) * (1 - tau)
        return self.target.set_weights(new_weights)
