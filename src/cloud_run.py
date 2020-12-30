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
from src.max_returns import max_possible_returns

logger = logg(__name__)


def run(params: Dict, extra_logging_backends: List[LoggerBackend], render_only: bool):
    try:
        logger.error(f'Params: {params}')
        algo_creator = getattr(importlib.import_module(params['algo_path']), params['algo_name'])

        episode_length = params['episode_length']
        env_name = params['environment']
        gamma = params['gamma']
        algo = algo_creator(episode_length=episode_length, max_returns=max_possible_returns(env_name, episode_length, gamma),
                            hyperparams=params)
        env = Environment(max_sample_cnt=params['max_sample_cnt'], episode_length=episode_length,
                          env_creator=lambda: gym.make(env_name), algo=algo, model_save_freq=params['model_save_freq'])
        if not render_only:
            env.train(params['seed'], Logger([FileLogger(list(params.keys()) + MANDATORY_COLUMNS + ['episode_return'])]
                                             + extra_logging_backends, params))
        else:
            env.render(int(time.time()))
    except BaseException as e:
        logger.error(traceback.format_exc())
        raise e


def upload_files(cloud_root, local_root, subdir, bucket):
    local_current = local_root if subdir is None else local_root / subdir
    for name in os.listdir(local_current):
        full_path = local_current / name
        if full_path.is_file():
            print(cloud_root + 'job_output/' + name)
            blob = bucket.blob(cloud_root + (f'job_output/{subdir}/' if subdir is not None else 'job_output/') + name)
            blob.upload_from_filename(str(full_path))
        else:
            upload_files(cloud_root, local_root, (subdir + "/" + name) if subdir is not None else name, bucket) # recurse into directories

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
        upload_files(cloud_root, OUTPUT_DIR, None, bucket)


if __name__ == '__main__':
    main()
