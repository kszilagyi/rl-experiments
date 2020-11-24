from collections import deque
from typing import Dict, Any

from src.environment import LoggerBackend
import matplotlib.pyplot as plt


class LiveGraphLogger(LoggerBackend):
    def __init__(self, x_key, y_key):
        self.x_key = x_key
        self.y_key = y_key
        self.data_x = deque(maxlen=2)
        self.data_y = deque(maxlen=2)

    def log(self, data: Dict[str, Any]):
        self.data_x.append(data[self.x_key])
        self.data_y.append(data[self.y_key])
        plt.plot(self.data_x, self.data_y, color='blue', linewidth=1)
        plt.pause(0.05)

    def close(self):
        pass