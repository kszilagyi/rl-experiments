
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
        self.grads = [None] * episode_length
        self.rewards = [0.0] * episode_length

    @tf.function
    def _action(self, well_formed_obs, t):
        with tf.GradientTape() as tape:
            action_logits: tf.Tensor = policy_model(well_formed_obs, training=True)
            dist = Categorical(logits=action_logits)
            sampled_action = dist.sample()
            higher_prob = dist.log_prob(sampled_action)

        grads = tape.gradient(higher_prob, policy_model.trainable_variables)
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
        cum_rewards = [None] * (t + 1)

        for time in range(t, -1, -1):
            cum_reward = self.rewards[time] + cum_reward * GAMMA
            cum_rewards[time] = cum_reward
            cum_rewards = np.array(cum_rewards)
        mean = np.mean(cum_rewards)
        std = np.std(cum_rewards)
        cum_rewards = (cum_rewards - mean) / std
        cum_rewards = cum_rewards.tolist()
        grads_for_debug = []
        grad_acc = None
        for time in range(t, -1, -1):
            adjusted_grads = [-g * (cum_rewards[time]) / t for g in self.grads[time]]
            if grad_acc is None:
                grad_acc = adjusted_grads
            else:
                grad_acc = [g + acc for g, acc in zip(adjusted_grads, grad_acc)]
            grads_for_debug.append(np.concatenate([a.numpy().flatten() for a in adjusted_grads]))

        optimizer.apply_gradients(zip(grad_acc, policy_model.trainable_variables))
        return t + 1, grads_for_debug, policy_model.get_weights()
