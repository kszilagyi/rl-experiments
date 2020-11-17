import random
from time import sleep

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.distributions.categorical import Categorical

from environment import Algo
from model import PolicyModel

env = gym.make('CartPole-v0').env


optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
# adam = tf.keras.optimizers.Adam(learning_rate=0.01) # todo probably this shouldn't be adam
policy_model = PolicyModel()
# value_model = ValueModel()


PRINT_FREQ = 100
GAMMA = 0.95


class PolicyGradient(Algo):
    def __init__(self, episode_length):
        self.gradients = [None] * episode_length
        self.rewards = [0.0] * episode_length

    @tf.function
    def action(self, observation, t):
        well_formed_obs = np.expand_dims(observation, axis=1).T

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            action_probabilities: tf.Tensor = policy_model(well_formed_obs, training=True)
            action_probabilities = tf.math.maximum(action_probabilities, 1e-9)
            log_action_probabilities = tf.math.log(action_probabilities)

            dist = Categorical(probs=[action_probabilities.numpy()])
            sampled_action = dist.sample().numpy()[0, 0]
            if sampled_action == 0:
                higher_prob = log_action_probabilities[0, 0]
            else:
                higher_prob = log_action_probabilities[0, 1]

        gradients = tape.gradient(higher_prob, policy_model.trainable_variables)
        self.gradients[t] = gradients
        return sampled_action

    def step(self, observation, action, reward: float, new_observation, done: bool, t: int):
        self.rewards[t] = reward

    def episode_end(self, t):
        cumulative_reward = 0
        cumulative_rewards = [None] * (t + 1)

        for time in range(t, -1, -1):
            cumulative_reward = self.rewards[time] + cumulative_reward * GAMMA
            cumulative_rewards[time] = cumulative_reward
            cumulative_rewards = np.array(cumulative_rewards)
        mean = np.mean(cumulative_rewards)
        std = np.std(cumulative_rewards)
        cumulative_rewards = (cumulative_rewards - mean) / std
        cumulative_rewards = cumulative_rewards.tolist()
        for time in range(t, -1, -1):
            adjusted_gradients = [-g * (cumulative_rewards[time]) / t for g in self.gradients[time]]
            optimizer.apply_gradients(zip(adjusted_gradients, policy_model.trainable_variables))