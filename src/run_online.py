import argparse
import importlib
import json

import gym

from src.environment import Environment, Logger, MANDATORY_COLUMNS, Algo
from src.filelogger import FileLogger
from src.logg import logg

logger = logg(__name__)


def main():
    with open('params.json') as f:
        params = json.load(f)
    logger.error(f'Params: {params}')
    algo_creator = getattr(importlib.import_module(params['algo_path']), params['algo_name'])

    episode_length = params['episode_length']
    algo = algo_creator(episode_length=episode_length)
    env = Environment(num_episodes=params['num_episodes'], episode_length=episode_length,
                      env_creator=lambda: gym.make(params['environment']), algo=algo)
    env.train(params['seed'], Logger([FileLogger('seed', 'seed', MANDATORY_COLUMNS + ['episode_return'])]))


# todo unify it with local by giving it a live logger
if __name__ == '__main__':
    main()
