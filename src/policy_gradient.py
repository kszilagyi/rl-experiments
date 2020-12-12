
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.distributions.categorical import Categorical

from src.environment import Algo
from src.model import PolicyModel


optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
policy_model = PolicyModel()
GAMMA = 0.95


class PolicyGradient(Algo):
    def start_episode(self):
        pass

    def __init__(self, episode_length):
        self.gradients = [None] * episode_length
        self.rewards = [0.0] * episode_length

    @tf.function
    def _action(self, well_formed_obs, t):
        with tf.GradientTape() as tape:
            action_logits: tf.Tensor = policy_model(well_formed_obs, training=True)
            dist = Categorical(logits=action_logits)
            sampled_action = dist.sample()
            higher_prob = dist.log_prob(sampled_action)

        gradients = tape.gradient(higher_prob, policy_model.trainable_variables)
        return sampled_action, gradients

    def action(self, observation, t):
        well_formed_obs = np.expand_dims(observation, axis=1).T
        sampled_actions, gradients = self._action(tf.constant(well_formed_obs), tf.constant(t))
        self.gradients[t] = gradients
        return sampled_actions.numpy()[0]

    def step(self, observation, action, reward: float, new_observation, done: bool, t: int):
        self.rewards[t] = reward
        return 0

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
        applied_gradients = []
        for time in range(t, -1, -1): # todo is this wrong? Shouldn't we apply the gradients in one batch, try batch behind flag
            adjusted_gradients = [-g * (cumulative_rewards[time]) / t for g in self.gradients[time]]
            optimizer.apply_gradients(zip(adjusted_gradients, policy_model.trainable_variables))
            applied_gradients.append(np.concatenate([a.numpy().flatten() for a in adjusted_gradients]))
        return t + 1, applied_gradients, policy_model.get_weights()
