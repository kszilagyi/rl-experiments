import argparse
import importlib
import itertools
import json
from typing import Dict, Any
from uuid import uuid4
from google.cloud import storage
from kubernetes import client, config
import datetime

def main():
    parser = argparse.ArgumentParser('Run online')
    parser.add_argument('job_spec_path', type=str)
    parser.add_argument('docker_image', type=str)
    args = parser.parse_args()
    job_spec_module: Any = importlib.import_module(args.job_spec_path)
    static_params= job_spec_module.static
    grid_search: Dict = job_spec_module.grid
    jobs = []
    batch_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + job_spec_module.name

    all_combinations = list(itertools.product(*list(grid_search.values())))
    for combo in all_combinations:
        job = static_params.copy()
        for idx, key in enumerate(grid_search.keys()):
            job[key] = combo[idx]
        job['id'] = str(uuid4())
        job['batch_name'] = batch_name
        jobs.append(job)


    storage_client = storage.Client()
    bucket = storage_client.get_bucket('rl-experiments')
    for job in jobs:
        blob = bucket.blob(batch_name + '/' + job['id'] + '/' + 'config.json')
        blob.upload_from_string(json.dumps(job, indent=4))


    # Configs can be set in Configuration class directly or using helper utility
    config.load_kube_config()

    v1 = client.BatchV1Api()
    with open('src/job_template.yaml') as f:
        job_template = f.read()
    for job in jobs:
        job_template = job_template.replace('$NAME', job['id']).replace('$IMAGE', args.docker_image)\
            .replace('$JOB_ID', job['id']).replace('$BATCH_NAME', batch_name)
        ret = v1.create_namespaced_job('default', job_template)
        print(ret)


if __name__ == '__main__':
    main()