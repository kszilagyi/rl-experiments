import random
from abc import ABC
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf


class Algo(ABC):
    def start_episode(self):
        pass

    def action(self, observation, t: int):
        pass

    def step(self, observation, action, reward: float, new_observation, done: bool, t: int):
        pass

    def episode_end(self, t: int):
        pass


@dataclass(frozen=True)
class Environment:
    num_episodes: int
    episode_length: int
    env_creator: Any
    algo: Algo

    def train(self, seed: int):
        algo = self.algo
        env = self.env_creator()
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)
        tf.random.set_seed(seed)
        episode_returns = []
        for i_episode in range(self.num_episodes):
            observation = env.reset()
            algo.start_episode()
            episode_return = 0
            for t in range(self.episode_length):
                action = algo.action(observation, t)
                new_observation, reward, done, info = env.step(action)
                episode_return += reward
                algo.step(observation, action, reward, new_observation, done, t)

                observation = new_observation

                if done or t + 1 == self.episode_length:
                    algo.episode_end(t)
                    break
            episode_returns.append(episode_return)
        env.close()
        return episode_returns
