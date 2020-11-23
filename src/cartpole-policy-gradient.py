import random
from time import sleep

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.distributions.categorical import Categorical

from src.model import PolicyModel

env = gym.make('CartPole-v0').env


optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
# adam = tf.keras.optimizers.Adam(learning_rate=0.01) # todo probably this shouldn't be adam
policy_model = PolicyModel()
# value_model = ValueModel()


PRINT_FREQ = 100
EPISODE_LEN = 1000
GAMMA = 0.95


# @tf.function
def forward(state, episode):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        action_probabilities: tf.Tensor = policy_model(state, training=True)
        action_probabilities = tf.math.maximum(action_probabilities, 1e-9)
        log_action_probabilities = tf.math.log(action_probabilities)

        dist = Categorical(probs=[action_probabilities.numpy()])
        sampled_action = dist.sample().numpy()[0, 0]
        # print(action_probabilities)
        if sampled_action == 0:
            higher_prob = log_action_probabilities[0, 0]
        else:
            higher_prob = log_action_probabilities[0, 1]

    if episode % PRINT_FREQ == 0:
        print(action_probabilities.numpy())
    # print(f'higher: {higher_prob}')
    # print(f'both: {log_action_probabilities}')
    gradients = tape.gradient(higher_prob, policy_model.trainable_variables)
    # print(gradients)
    # print('trainables:' + str(policy_model.trainable_variables))
    return gradients, sampled_action


# def update_value(observations, cumulative_rewards):
#     for t in range(len(cumulative_rewards)):
#         with tf.GradientTape() as tape:
#             value_estimate = value_model(observations[t], training=True)
#             loss = tf.losses.MSE(cumulative_rewards[t], value_estimate)
#         grad = tape.gradient(loss, value_model.trainable_variables)
        # adam.apply_gradients(zip(grad, value_model.trainable_variables))

np.random.seed(0)
random.seed(0)

def main():
    max_return = 0
    cum_return = 0
    for i_episode in range(100000):
        observation = env.reset()
        gradients = [None] * EPISODE_LEN
        rewards = [0] * EPISODE_LEN
        observations = []
        predictions = []
        for t in range(EPISODE_LEN):
            if i_episode % PRINT_FREQ == 0:
                env.render()
                sleep(0.1)
            well_formed_obs = np.expand_dims(observation, axis=1).T
            observations.append(well_formed_obs)
            gradients[t], sampled_action = forward(well_formed_obs, i_episode)
            optimal_value = 200 - t
            predictions.append(optimal_value)
            # predictions.append(value_model(well_formed_obs, training=False).numpy()[0, 0])

            observation, reward, done, info = env.step(sampled_action)
            # if sampled_action == 0:
            #     reward = 1
            #     done = False
            # else:
            #     reward = 0
            #     done = True

            rewards[t] = reward
            if done or t + 1 == EPISODE_LEN:
                cumulative_reward = 0
                cumulative_rewards = [None] * (t + 1)

                for time in range(t, -1, -1):
                    cumulative_reward = rewards[time] + cumulative_reward * GAMMA
                    cumulative_rewards[time] = cumulative_reward
                    cumulative_rewards = np.array(cumulative_rewards)
                mean = np.mean(cumulative_rewards)
                std = np.std(cumulative_rewards)
                cumulative_rewards = (cumulative_rewards - mean) / std
                cumulative_rewards = cumulative_rewards.tolist()
                for time in range(t, -1, -1):
                    adjusted_gradients = [-g * (cumulative_rewards[time]) / t for g in gradients[time]]
                    optimizer.apply_gradients(zip(adjusted_gradients, policy_model.trainable_variables))

                # update_value(observations, cumulative_rewards)
                max_return = max(sum([r for r in rewards]), max_return)
                cum_return += sum([r for r in rewards])
                if i_episode % PRINT_FREQ == 0:
                    print('trying deterministc value function, lr=0.0001 ')
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                    print(f'Cumulative reward {sum([r for r in rewards])}')
                    print(f'predictions: {predictions}')
                    print(f'cumulative rewards: {cumulative_rewards}')
                    print(f'Value MSE: {tf.keras.metrics.MSE(cumulative_rewards, predictions)}')

                    mae = tf.keras.metrics.MeanAbsoluteError()
                    mae.update_state(cumulative_rewards, predictions)
                    print(f'Value MAE: {mae.result()}')
                if i_episode % 10 == 0:
                    print(f'max return: {max_return}')
                    print(f'avg return: {cum_return / 10}')
                    max_return = 0
                    cum_return = 0

                break
        policy_model.save('model')
    env.close()

if __name__ == '__main__':
    main()