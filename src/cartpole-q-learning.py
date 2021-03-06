import random
from time import sleep
from typing import List, Tuple
import matplotlib
import matplotlib.pyplot as plt

import gym
import numpy as np
import tensorflow as tf

from src.model import create_q_model

env = gym.make('CartPole-v0').env
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

qmodel = create_q_model()
qmodel_old = tf.keras.models.clone_model(qmodel)
qmodel_old.set_weights(qmodel.get_weights())
State = Tuple[float, float, float, float]
Action = int
IsTerminal = float
buffer: List[Tuple[State, Action, float, State, IsTerminal]] = []
PRINT_FREQ = 1
EPISODE_LEN = 5000
EPSILON = 0.1
GAMMA = 0.95
np.random.seed(0)
random.seed(0)
env.seed(0)

BATCH_SIZE = 256
all_actions = [0, 1]
def train(i_episode):
    previous_last_loss = 100
    for _ in range(20):
        last_loss = 100
        while last_loss > 0.1:
            states, actions, rewards, new_states, is_terminal = list(zip(*random.sample(buffer, BATCH_SIZE)))
            left = np.array([[*s, 0] for s in new_states])
            right = np.array([[*s, 1] for s in new_states])
            q_values_from_new_state = np.array(list(zip(qmodel_old(left, training=False), qmodel_old(right, training=False))))
            best_q_values_from_new_state = np.max(q_values_from_new_state, axis=1)
            with tf.GradientTape() as tape:
                q_queries = np.array([(*s, a) for s, a in list(zip(states, actions))])
                q_values = qmodel(q_queries, training=False)
                diff = tf.squeeze(q_values) - (rewards + GAMMA * tf.squeeze(best_q_values_from_new_state)) * (1.0 - np.array(is_terminal))
                loss = tf.reduce_mean(tf.square(diff))
            # if i_episode %10 == 0:
            # print('mean: ' + str(np.mean(q_values.numpy())))
            # print(loss.numpy())
            if last_loss != 100:
                previous_last_loss = last_loss
            last_loss = loss.numpy()
            gradients = tape.gradient(loss, qmodel.trainable_variables)
            optimizer.apply_gradients(zip(gradients, qmodel.trainable_variables))
        print('COPY')
        print('mean: ' + str(np.mean(q_values.numpy())))
        qmodel_old.set_weights(qmodel.get_weights())
        # if previous_last_loss < 0.1 and last_loss < 0.1:
        #     print('BREAK')
        #     break
        previous_last_loss = last_loss
    # print(f'mean loss: {total_loss/total_cnt}, total cnt: {total_cnt}')


log = False
def epsilon_greedy_action(state):
    if np.random.uniform() < EPSILON:
        return np.random.randint(0, 2)
    a0 = np.append(state, 0)
    a1 = np.append(state, 1)
    q_values = qmodel(np.array([a0, a1]), training=False)
    if log:
        print(q_values.numpy())
    return np.argmax(q_values)


def main():
    global qmodel_old
    global buffer
    global EPSILON
    max_return = 0
    cum_return = 0
    for i_episode in range(100000):
        state = env.reset()
        episode_return = 0
        pre_buffer = []
        for t in range(EPISODE_LEN):
            # if i_episode % PRINT_FREQ == 0 and i_episode != 0:
            #     env.render()
            #     sleep(0.1)

            action = epsilon_greedy_action(state)
            new_state, reward, done, info = env.step(action)
            episode_return += reward
            pre_buffer.append((state, action, reward, new_state, float(done)))
            # buffer = buffer[-MAX_SIZE:]
            # if done:
            #     for i in range(int(t/2)):
            #         buffer.append((state, action, reward, new_state, float(done))) # balancing classes todo do i need this, maybe it was the lr rate after all

            state = new_state

            if done or t + 1 == EPISODE_LEN:
                max_return = max(max_return, episode_return)
                cum_return += episode_return
                if i_episode % PRINT_FREQ == 0:
                    print("500 - Episode {} finished after {} timesteps, buffer size: {}".format(i_episode, t+1, len(buffer)))
                if t < 500:
                    buffer.extend(pre_buffer)
                if len(buffer) >= BATCH_SIZE*10 and t < 500:
                    EPSILON = 0
                    train(i_episode)


                if i_episode % 10 == 0:
                    print(f'max return: {max_return}')
                    print(f'avg return: {cum_return / 10}')
                    max_return = 0
                    cum_return = 0

                break
    env.close()


if __name__ == '__main__':
    main()