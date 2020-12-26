import importlib
import json
import os
import traceback
import random
from typing import Dict, List

import gym
from google.cloud import storage

from src.environment import Environment, Logger, MANDATORY_COLUMNS, Algo, LoggerBackend
from src.filelogger import FileLogger
from src.logg import logg, OUTPUT_DIR
from src.max_returns import max_possible_returns

logger = logg(__name__)


def run(params: Dict, extra_logging_backends: List[LoggerBackend]):
    try:
        logger.error(f'Params: {params}')
        algo_creator = getattr(importlib.import_module(params['algo_path']), params['algo_name'])

        episode_length = params['episode_length']
        env_name = params['environment']
        gamma = params['gamma']
        algo = algo_creator(episode_length=episode_length, max_returns=max_possible_returns(env_name, episode_length, gamma),
                            hyperparams=params)
        env = Environment(max_sample_cnt=params['max_sample_cnt'], episode_length=episode_length,
                          env_creator=lambda: gym.make(env_name), algo=algo)
        env.train(params['seed'], Logger([FileLogger(list(params.keys()) + MANDATORY_COLUMNS + ['episode_return'])]
                                         + extra_logging_backends, params))
    except BaseException as e:
        logger.error(traceback.format_exc())
        raise e


def main():
    storage_client = storage.Client(project='rl-experiments-296208')
    bucket = storage_client.bucket('rl-experiments')
    batch_name = os.environ['BATCH_NAME']
    job_id = os.environ['JOB_ID']
    cloud_root = batch_name + '/' + job_id + '/'
    blob = bucket.blob(cloud_root + 'params.json')

    params = json.loads(blob.download_as_text())
    params['job_id'] = job_id
    status = 'SUCCESS'
    try:
        run(params, [])
    except BaseException as e:
        status = 'FAILURE'
        raise e
    finally:
        with open(OUTPUT_DIR / 'result.json', 'w') as f:
            json.dump({'status': status}, f)
        for name in os.listdir(OUTPUT_DIR):
            full_path = OUTPUT_DIR / name
            if full_path.is_file():
                print(cloud_root + 'job_output/' + name)
                blob = bucket.blob(cloud_root + 'job_output/' + name)
                blob.upload_from_filename(str(full_path))



if __name__ == '__main__':
    main()
