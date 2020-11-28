import importlib
import json
import os
import time
import traceback
from typing import Dict, List

import gym
from google.cloud import storage

from src.environment import Environment, Logger, MANDATORY_COLUMNS, Algo, LoggerBackend
from src.filelogger import FileLogger
from src.logg import logg, OUTPUT_DIR

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
    print(storage_client.get_service_account_email())
    bucket = storage_client.get_bucket('rl-experiments')
    batch_name = os.environ['BATCH_NAME']
    job_id = os.environ['JOB_ID']
    cloud_root = batch_name + '/' + job_id + '/'
    blob = bucket.blob(cloud_root + 'params.json')

    params = json.loads(blob.download_as_text())
    status = 'SUCCESS'
    try:
        run(params, [])
    except BaseException as e:
        status = 'FAILURE'
        raise e
    finally:
        time.sleep(100000)
        with open(OUTPUT_DIR / 'result.json', 'w') as f:
            json.dump({'status': status}, f)
        for name in os.listdir(OUTPUT_DIR):
            full_path = OUTPUT_DIR / name
            if full_path.is_file():
                print(cloud_root + 'job_output/' + name)
                blob = bucket.blob(cloud_root + 'job_output/' + name)
                print(str(full_path))
                blob.upload_from_filename(str(full_path))



if __name__ == '__main__':
    main()
