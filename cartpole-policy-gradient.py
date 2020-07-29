from time import sleep

import gym
import numpy as np
import tensorflow as tf
from tensorflow import metrics
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.ops.distributions.categorical import Categorical

from model import MyModel

env = gym.make('CartPole-v0')

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model = MyModel()


# @tf.function
def forward(state, episode):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        action_probabilities: tf.Tensor = model(state, training=True)
        log_action_probabilities = tf.math.log(action_probabilities)

        dist = Categorical(probs=[action_probabilities.numpy()])
        sample = dist.sample().numpy()[0, 0]
        if sample == 0:
            higher_prob = log_action_probabilities[0, 0]
        else:
            higher_prob = log_action_probabilities[0, 1]

    if episode % 100 == 0:
        print(action_probabilities.numpy())

    gradients = tape.gradient(higher_prob, model.trainable_variables)

    return gradients

    # train_loss(loss)
    # train_accuracy(labels, action_probabilities)

EPISODE_LEN = 100

def main():
    for i_episode in range(20000):
        observation = env.reset()
        gradients = [None] * EPISODE_LEN
        rewards = [0] * EPISODE_LEN

        for t in range(EPISODE_LEN):
            # env.render()
            gradients[t] = forward(np.expand_dims(observation, axis=1).T, i_episode)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            rewards[t] = reward
            if done:
                if i_episode % 100 == 0:
                    print('lr=0.01')
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                    print(f'Cumulative reward {sum([r for r in rewards])}')
                cumulative_reward = 0
                for time in range(t, -1, -1):
                    cumulative_reward += rewards[time]
                    adjusted_gradients = [-g * cumulative_reward for g in gradients[time]]
                    optimizer.apply_gradients(zip(adjusted_gradients, model.trainable_variables))


                break
        model.save('model')
    env.close()

if __name__ == '__main__':
    main()