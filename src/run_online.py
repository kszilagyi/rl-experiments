import argparse
import importlib
import json
import traceback
from typing import Dict, List

import gym

from src.environment import Environment, Logger, MANDATORY_COLUMNS, Algo, LoggerBackend
from src.filelogger import FileLogger
from src.logg import logg

logger = logg(__name__)


def run(params: Dict, extra_logging_backends: List[LoggerBackend]):
    try:
        logger.error(f'Params: {params}')
        algo_creator = getattr(importlib.import_module(params['algo_path']), params['algo_name'])

        episode_length = params['episode_length']
        algo = algo_creator(episode_length=episode_length)
        env = Environment(num_episodes=params['num_episodes'], episode_length=episode_length,
                          env_creator=lambda: gym.make(params['environment']), algo=algo)
        env.train(params['seed'], Logger([FileLogger('seed', 'seed', MANDATORY_COLUMNS + ['episode_return'])]
                                         + extra_logging_backends))
    except BaseException as e:
        logger.error(traceback.format_exc())
        raise e

def main():
    with open('params.json') as f:
        params = json.load(f)
    run(params, [])


if __name__ == '__main__':
    main()
