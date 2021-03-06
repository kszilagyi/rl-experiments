from typing import Dict

import numpy as np
import tensorflow as tf
from gym.spaces import Box
from tensorflow.python.ops.distributions.categorical import Categorical
from tensorflow.python.ops.distributions.normal import Normal

from src.environment import Algo
from src.model import CategoricalPolicyModel, GaussianPolicyModel


class CategoricalPolicy:
    def __init__(self, outputs):
        self.model = CategoricalPolicyModel(outputs)

    def action(self, well_formed_obs):
        action_logits: tf.Tensor = self.model(well_formed_obs, training=True)
        dist = Categorical(logits=action_logits)
        sampled_action = dist.sample()
        sampled_prob = dist.log_prob(sampled_action)
        return sampled_action, sampled_prob


class GaussianPolicy:
    def __init__(self, outputs, scale):
        self.model = GaussianPolicyModel(outputs, scale)

    def action(self, well_formed_obs):
        mu: tf.Tensor = self.model(well_formed_obs, training=True)
        dist = Normal(loc=mu, scale=0.5)
        sampled_action = dist.sample()
        sampled_prob = dist.log_prob(tf.stop_gradient(sampled_action))
        return sampled_action, tf.reduce_mean(sampled_prob) # reduce mean?


class PolicyGradient(Algo):
    def start_episode(self):
        pass

    def __init__(self, episode_length: int, max_returns: np.ndarray, hyperparams: Dict):
        self.grads = [None] * episode_length
        self.rewards = [0.0] * episode_length
        self.hyperparams = hyperparams
        self.max_returns = max_returns
        optimizer_name = hyperparams['optimizer']
        if optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD
        elif optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam
        else:
            assert(False)
        self.optimizer = optimizer(learning_rate=hyperparams['lr'])

    def init_model(self, env):
        action_space = env.action_space
        if not isinstance(action_space, Box):
            self.policy = CategoricalPolicy(action_space.n)
        else:
            low = action_space.low
            high = action_space.high
            assert np.isscalar(low) or (low[0] == low).all()
            assert np.isscalar(high) or (high[0] == high).all()
            if not np.isscalar(low):
                low = low[0]
            if not np.isscalar(high):
                high = high[0]
            assert abs(low) == high
            self.policy = GaussianPolicy(action_space.shape[0], scale=(high - low)/2)


    def save_model(self, path: str):
        self.policy.model.save_weights(path)

    def load_model(self, path: str):
        self.policy.model.load_weights(path)

    @tf.function
    def _action(self, well_formed_obs, t):
        with tf.GradientTape() as tape:
            sampled_action, sampled_prob = self.policy.action(well_formed_obs)

        grads = tape.gradient(sampled_prob, self.policy.model.trainable_variables)
        return sampled_action, grads

    def action(self, observation, t):
        well_formed_obs = np.expand_dims(observation, axis=1).T
        sampled_actions, gradients = self._action(tf.constant(well_formed_obs), tf.constant(t))
        self.grads[t] = gradients
        return sampled_actions.numpy()[0]

    def step(self, observation, action, reward: float, new_observation, done: bool, t: int):
        self.rewards[t] = reward
        return 0

    def episode_end(self, t):
        cum_reward = 0
        returns = [None] * (t + 1)

        for time in range(t, -1, -1):
            cum_reward = self.rewards[time] + cum_reward * self.hyperparams['gamma']
            returns[time] = cum_reward
        returns = np.array(returns)
        if self.hyperparams['normalise_with_max_returns']:
            returns -= self.max_returns[:len(returns)]

        if self.hyperparams['center_returns']:
            mean = np.mean(returns)
        else:
            mean = 0
        if self.hyperparams['normalise_returns']:
            std = np.std(returns)
            if std < 1e-6:
                std = 1
        else:
            std = 1
        returns = (returns - mean) / std
        returns = returns.tolist()
        grads_for_debug = []
        grad_acc = None

        if self.hyperparams['normalise_returns_with_episode_length']:
            episode_length_divider = t
        else:
            episode_length_divider = 1
        for time in range(t, -1, -1):
            adjusted_grads = [-g * (returns[time]) / episode_length_divider for g in self.grads[time]]
            if grad_acc is None:
                grad_acc = adjusted_grads
            else:
                grad_acc = [g + acc for g, acc in zip(adjusted_grads, grad_acc)]
            grads_for_debug.append(np.concatenate([a.numpy().flatten() for a in adjusted_grads]))

        self.optimizer.apply_gradients(zip(grad_acc, self.policy.model.trainable_variables))
        return t + 1, grads_for_debug, self.policy.model.get_weights()
