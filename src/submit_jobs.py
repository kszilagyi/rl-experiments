import argparse
import importlib
import itertools
import json
import subprocess
import time
from typing import Dict, Any
from uuid import uuid4
from google.cloud import storage
from kubernetes import client, config
import datetime
import yaml

from src.logg import logg

logger = logg(__name__)

def main():
    parser = argparse.ArgumentParser('Run online')
    parser.add_argument('--job_spec_path', type=str)
    parser.add_argument('--docker_image', type=str)
    parser.add_argument('--test', dest='test', action='store_const', default=False, const=True,
                        help='Only submits the first job, easier for testing')
    args = parser.parse_args()
    logger.info(args)
    job_spec_module: Any = importlib.import_module(args.job_spec_path)
    static_params= job_spec_module.static
    grid_search: Dict = job_spec_module.grid
    jobs = []
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')[:12]

    batch_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + job_spec_module.name + '_' + git_hash

    all_combinations = list(itertools.product(*list(grid_search.values())))
    for combo in all_combinations:
        job = static_params.copy()
        for idx, key in enumerate(grid_search.keys()):
            job[key] = combo[idx]
        job['id'] = str(uuid4())
        job['batch_name'] = batch_name
        jobs.append(job)

    if args.test:
        jobs = jobs[:1]

    storage_client = storage.Client()
    bucket = storage_client.bucket('rl-experiments')
    logger.info('Creating job params files')
    for job in jobs:
        blob = bucket.blob(batch_name + '/' + job['id'] + '/' + 'params.json')
        blob.upload_from_string(json.dumps(job, indent=4))


    # Configs can be set in Configuration class directly or using helper utility
    config.load_kube_config()

    v1_core = client.CoreV1Api()
    with open('src/pod_template.yaml') as f:
        pod_template = f.read()
    logger.info(f'Submitting {len(jobs)} jobs. ({batch_name})')
    for idx, job in enumerate(jobs):
        job_desc = pod_template.replace('$NAME', job['id']).replace('$IMAGE', args.docker_image)\
            .replace('$JOB_ID', job['id']).replace('$BATCH_NAME', batch_name)
        v1_core.create_namespaced_pod('default', yaml.safe_load(job_desc))
        if (idx + 1) % 50 == 0:
            logger.info(f'{idx + 1} jobs have been submitted')

    logger.info('All jobs have been submitted')
    finished = 0
    while finished < len(jobs):
        cluster_pods = v1_core.list_namespaced_pod('default', label_selector=f'batch={batch_name}').items
        finished = sum([1 for p in cluster_pods if p.status.phase in ['Succeeded', 'Failed']])
        pending = sum([1 for p in cluster_pods if p.status.phase in ['Pending']])
        succeeded = sum([1 for p in cluster_pods if p.status.phase in ['Succeeded']])
        failed = sum([1 for p in cluster_pods if p.status.phase in ['Failed']])
        running = sum([1 for p in cluster_pods if p.status.phase in ['Running']])

        logger.info(f'Pod statuses: pending: {pending}, running: {running}, succeeded: {succeeded}, failed: {failed}')
        time.sleep(10)

    logger.info(f'Finished, batch name was: {batch_name}')


if __name__ == '__main__':
    main()