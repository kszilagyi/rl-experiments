import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, final, List

import numpy as np
import tensorflow as tf


class Algo(ABC):
    @abstractmethod
    def start_episode(self):
        pass

    @abstractmethod
    def action(self, observation, t: int):
        pass

    @abstractmethod
    def step(self, observation, action, reward: float, new_observation, done: bool, t: int) -> int:
        """:returns number of training steps taken(number of samples processed)"""
        pass

    @abstractmethod
    def episode_end(self, t: int):
        pass


class LoggerBackend(ABC):
    def log(self, data: Dict[str, float]):
        pass

class Logger:
    def __init__(self, columns: List[str], backend: LoggerBackend):
        self.backend = backend
        self.columns = columns

    @final
    def log(self, data: Dict, episode_num: int, sample_cnt: int, elapsed_time: float, training_steps: int):
        assert set(self.columns) == set(data.keys())
        augmented_data = dict(data)
        augmented_data['episode_num'] = episode_num
        augmented_data['sample_cnt'] = sample_cnt
        augmented_data['elapsed_time'] = elapsed_time
        augmented_data['training_steps'] = training_steps
        self.backend.log(augmented_data)



@dataclass(frozen=True)
class Environment:
    num_episodes: int
    episode_length: int
    env_creator: Any
    algo: Algo

    def train(self, seed: int, logger: Logger):
        algo = self.algo
        env = self.env_creator()
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)
        tf.random.set_seed(seed)
        episode_returns = []
        sample_cnt = 0
        start_time = time.time()
        training_steps = 0
        for i_episode in range(self.num_episodes):
            observation = env.reset()
            algo.start_episode()
            episode_return = 0
            for t in range(self.episode_length):
                action = algo.action(observation, t)
                new_observation, reward, done, info = env.step(action)
                episode_return += reward
                sample_cnt += 1
                training_steps += algo.step(observation, action, reward, new_observation, done, t)

                observation = new_observation

                if done or t + 1 == self.episode_length:
                    training_steps += algo.episode_end(t)
                    break
            logger.log({'episode_return': episode_return}, episode_num=i_episode, sample_cnt=sample_cnt,
                       elapsed_time=time.time() - start_time, training_steps=training_steps)
            episode_returns.append(episode_return)
        env.close()
        return episode_returns

