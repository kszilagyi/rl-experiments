import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import animation

from src.logg import OUTPUT_DIR


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
    def episode_end(self, t: int) -> Tuple[int, List, List]:
        pass

    @abstractmethod
    def init_model(self, env):
        pass

    @abstractmethod
    def save_model(self, path: str):
        pass

    @abstractmethod
    def load_model(self, path: str):
        pass


class LoggerBackend(ABC):
    @abstractmethod
    def log(self, data: Dict[str, Any]):
        pass

    @abstractmethod
    def close(self):
        pass


MANDATORY_COLUMNS = ['episode_num', 'sample_cnt', 'elapsed_time', 'training_steps', 'job_id', 'batch_name']


class Logger:
    def __init__(self, backends: List[LoggerBackend], params: Dict):
        self.backends = backends
        self.params = params

    def log(self, data: Dict, episode_num: int, sample_cnt: int, elapsed_time: float, training_steps: int):
        augmented_data = {**self.params, **data}
        augmented_data['episode_num'] = episode_num
        augmented_data['sample_cnt'] = sample_cnt
        augmented_data['elapsed_time'] = elapsed_time
        augmented_data['training_steps'] = training_steps
        for backend in self.backends:
            backend.log(augmented_data)

    def close(self):
        for b in self.backends:
            b.close()

# copied from: https://nbviewer.jupyter.org/github/patrickmineault/xcorr-notebooks/blob/master/Render%20OpenAI%20gym%20as%20GIF.ipynb
def display_frames_as_gif(path, frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path)


@dataclass(frozen=True)
class Environment:
    max_sample_cnt: int
    episode_length: int
    model_save_freq: int
    env_creator: Any
    algo: Algo

    def train(self, seed: int, logger: Logger):
        algo = self.algo
        env = self.env_creator()
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)
        env.action_space.seed(seed)
        tf.random.set_seed(seed)
        algo.init_model(env)
        episode_returns = []
        sample_cnt = 0
        start_time = time.time()
        training_steps = 0
        episode_num = 0
        try:
            one_last_model_save = True
            while sample_cnt < self.max_sample_cnt or one_last_model_save:
                if sample_cnt >= self.max_sample_cnt:
                    one_last_model_save = False
                observation = env.reset()
                algo.start_episode()
                episode_return = 0
                save_model = (episode_num + 1) % self.model_save_freq == 0 or sample_cnt >= self.max_sample_cnt
                for t in range(self.episode_length):
                    action = algo.action(observation, t)
                    new_observation, reward, done, info = env.step(action)
                    episode_return += reward
                    sample_cnt += 1
                    training_steps += algo.step(observation, action, reward, new_observation, done, t)

                    observation = new_observation

                    if done or t + 1 == self.episode_length:
                        current_training_steps, gradients, weights = algo.episode_end(t)
                        training_steps += current_training_steps
                        break
                gradients = np.concatenate(gradients)
                weights = np.concatenate([w.flatten() for w in weights])
                if np.isnan(gradients).any():
                    abs_mean_gradient = max_gradient = min_gradient = mean_gradient = np.nan
                else:
                    max_gradient = max(gradients, key=abs)
                    mean_gradient = np.mean(gradients)
                    abs_mean_gradient = np.mean(np.abs(gradients))
                    min_gradient = min(gradients, key=abs)
                if np.isnan(weights).any():
                    abs_mean_weight = min_weight = max_weight = mean_weight = np.nan
                else:
                    max_weight = max(weights, key=abs)
                    mean_weight = np.mean(weights)
                    abs_mean_weight = np.mean(np.abs(weights))
                    min_weight = min(weights, key=abs)
                logger.log({'episode_return': episode_return,
                            'abs_max_gradient': max_gradient, 'abs_min_gradient': min_gradient,
                            'mean_gradient': mean_gradient, 'abs_mean_gradient': abs_mean_gradient,
                            'abs_max_weight': max_weight, 'abs_min_weight': min_weight,
                            'mean_weight': mean_weight, 'abs_mean_weight': abs_mean_weight},
                           episode_num=episode_num,
                           sample_cnt=sample_cnt,
                           elapsed_time=time.time() - start_time, training_steps=training_steps)
                episode_returns.append(episode_return)
                if save_model:
                    save_dir = OUTPUT_DIR / f'model_save_episode_{episode_num}'
                    save_dir.mkdir(parents=True)
                    self.algo.save_model(save_dir / 'ckpt')
                episode_num += 1
            return episode_returns
        finally:
            env.close()
            logger.close()

    def render(self, seed):
        algo = self.algo
        env = self.env_creator()
        algo.init_model(env)
        algo.load_model('render/model/ckpt')
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)
        env.action_space.seed(seed)
        tf.random.set_seed(seed)
        for i_episode in range(20):
            observation = env.reset()
            ret = 0
            t = 0
            while True:
                env.render()
                action = algo.action(observation, t)
                observation, reward, done, info = env.step(action)
                ret += reward
                if done or t+1 >= self.episode_length:
                    print(f'Episode finished after {t+1} timesteps, with return {ret}')
                    break
                t += 1
        env.close()

