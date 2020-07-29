from time import sleep

import gym
import numpy as np
import tensorflow as tf
from tensorflow import metrics
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.ops.distributions.categorical import Categorical

from model import PolicyModel, ValueModel

env = gym.make('CartPole-v0')

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
adam = tf.keras.optimizers.Adam()
policy_model = PolicyModel()
value_model = ValueModel()


PRINT_FREQ = 100
EPISODE_LEN = 200


# @tf.function
def forward(state, episode):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        action_probabilities: tf.Tensor = policy_model(state, training=True)
        log_action_probabilities = tf.math.log(action_probabilities)

        dist = Categorical(probs=[action_probabilities.numpy()])
        sample = dist.sample().numpy()[0, 0]
        if sample == 0:
            higher_prob = log_action_probabilities[0, 0]
        else:
            higher_prob = log_action_probabilities[0, 1]

    if episode % PRINT_FREQ == 0:
        print(action_probabilities.numpy())

    gradients = tape.gradient(higher_prob, policy_model.trainable_variables)
    return gradients


def update_value(observations, cumulative_rewards):
    for t in range(len(cumulative_rewards)):
        with tf.GradientTape() as tape:
            value_estimate = value_model(observations[t], training=True)
            loss = tf.losses.MSE(cumulative_rewards[t], value_estimate)
        grad = tape.gradient(loss, value_model.trainable_variables)
        adam.apply_gradients(zip(grad, value_model.trainable_variables))


def main():
    for i_episode in range(20000):
        observation = env.reset()
        gradients = [None] * EPISODE_LEN
        rewards = [0] * EPISODE_LEN
        observations = []
        predictions = []
        for t in range(EPISODE_LEN):
            # env.render()
            well_formed_obs = np.expand_dims(observation, axis=1).T
            observations.append(well_formed_obs)
            gradients[t] = forward(well_formed_obs, i_episode)
            predictions.append(value_model(well_formed_obs, training=False).numpy()[0, 0])

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            rewards[t] = reward
            if done:

                cumulative_reward = 0
                cumulative_rewards = [None] * (t + 1)

                for time in range(t, -1, -1):
                    cumulative_reward += rewards[time]
                    cumulative_rewards[time] = cumulative_reward
                    adjusted_gradients = [-g * (cumulative_reward - predictions[time]) for g in gradients[time]]
                    optimizer.apply_gradients(zip(adjusted_gradients, policy_model.trainable_variables))

                update_value(observations, cumulative_rewards)

                if i_episode % PRINT_FREQ == 0:
                    print('values, lr=0.001')
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                    print(f'Cumulative reward {sum([r for r in rewards])}')
                    print(f'predictions: {predictions}')
                    print(f'cumulative rewards: {cumulative_rewards}')
                    print(f'Value MSE: {tf.keras.metrics.MSE(cumulative_rewards, predictions)}')

                    mae = tf.keras.metrics.MeanAbsoluteError()
                    mae.update_state(cumulative_rewards, predictions)
                    print(f'Value MAE: {mae.result()}')


                break
        policy_model.save('model')
    env.close()

if __name__ == '__main__':
    main()