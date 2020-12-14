import numpy as np

from src.max_returns import max_possible_returns


def test_max_returns():
    assert np.array_equal(max_possible_returns('CartPole-v0', 5, 1), [5, 4, 3, 2, 1])


if __name__ == '__main__':
    test_max_returns()