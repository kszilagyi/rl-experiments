import numpy as np


def max_possible_returns(env, max_episode_length, gamma):
    max_return = 0
    max_returns = [None] * max_episode_length
    if env == 'CartPole-v1' or env == 'CartPole-v0':
        for time in range(max_episode_length - 1, -1, -1):
            max_return = 1 + max_return * gamma
            max_returns[time] = max_return
        return np.array(max_returns)


