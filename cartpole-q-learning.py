import random
from time import sleep
from typing import List, Tuple

import gym
import numpy as np
import tensorflow as tf

from model import QModel

env = gym.make('CartPole-v0').env
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

qmodel = QModel()
qmodel_old = QModel()
State = Tuple[float, float, float, float]
Action = int
buffer: List[Tuple[State, Action, float, State]] = []
PRINT_FREQ = 100
EPISODE_LEN = 200
COPY_FREQ = 100

np.random.seed(0)
random.seed(0)
env.seed(0)

BATCH_SIZE = 256
all_actions = [0, 1]
def train():
    np.random.shuffle(buffer)
    for i in range(0, len(buffer), BATCH_SIZE):
        states, actions, rewards, new_states = list(zip(*buffer[i:i+BATCH_SIZE]))

        best_q_values_from_new_state = np.max(qmodel_old.predict(flattened))
        with tf.GradientTape() as tape:
            q_values = qmodel.predict(zip(states, actions))
            loss = tf.square(q_values - (rewards + best_q_values_from_new_state))
        gradients = tape.gradient(loss, qmodel.trainable_variables)
        optimizer.apply_gradients(zip(gradients, qmodel.trainable_variables))

def main():
    global qmodel_old
    max_return = 0
    cum_return = 0
    for i_episode in range(100000):
        state = env.reset()
        episode_return = 0
        for t in range(EPISODE_LEN):
            if i_episode % PRINT_FREQ == 0 and i_episode != 0:
                env.render()
                sleep(0.1)

            action = 0
            new_state, reward, done, info = env.step(action)
            episode_return += reward
            buffer.append((state, action, reward, new_state))
            state = new_state

            if done or t + 1 == EPISODE_LEN:
                max_return = max(max_return, episode_return)
                cum_return += episode_return
                if len(buffer) >= BATCH_SIZE:
                    train()
                if i_episode % PRINT_FREQ == 0:
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                if i_episode % COPY_FREQ == 0:
                    print('COPY')
                    for a, b in zip(qmodel_old.variables, qmodel.variables):
                        a.assign(b)
                    # qmodel_old = tf.keras.models.clone_model(qmodel)
                if i_episode % 10 == 0:
                    print(f'max return: {max_return}')
                    print(f'avg return: {cum_return / 10}')
                    max_return = 0
                    cum_return = 0

                break
    env.close()


if __name__ == '__main__':
    main()