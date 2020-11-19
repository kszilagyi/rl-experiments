from typing import Dict

from environment import LoggerBackend
import matplotlib.pyplot as plt

class LiveGraphLogger(LoggerBackend):
    def __init__(self, key, values):
        self.key = key

    def log(self, data: Dict[str, float]):
        plt.scatter(data[self.key], data['episode_return'])
        plt.pause(0.05)
