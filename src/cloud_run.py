import importlib
import json
import os
import traceback
from typing import Dict, List

import gym
from google.cloud import storage

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
        env.train(params['seed'], Logger([FileLogger(list(params.keys()) + MANDATORY_COLUMNS + ['episode_return'])]
                                         + extra_logging_backends, params))
    except BaseException as e:
        logger.error(traceback.format_exc())
        raise e

def main():
    storage_client = storage.Client(project='rl-experiments-296208')
    bucket = storage_client.get_bucket('rl-experiments')
    batch_name = os.environ['BATCH_NAME']
    job_id = os.environ['JOB_ID']
    blob = bucket.blob(batch_name + '/' + job_id + '/' + 'params.json')

    params = json.loads(blob.download_as_text())
    run(params, [])


if __name__ == '__main__':
    main()
