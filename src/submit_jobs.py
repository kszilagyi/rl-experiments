import argparse
import importlib
import itertools
from typing import Dict


def main():
    parser = argparse.ArgumentParser('Run online')
    parser.add_argument('job_spec_path', type=str)
    args = parser.parse_args()
    job_spec_module = importlib.import_module(args.job_spec_path)
    static_params= job_spec_module.static
    grid_search: Dict = job_spec_module.grid
    jobs = []
    all_combinations = list(itertools.product(*list(grid_search.values())))
    for combo in all_combinations:
        job = static_params.copy()
        for idx, key in enumerate(grid_search.keys()):
            job[key] = combo[idx]
        jobs.append(job)

    for j in jobs:
        print(j)



if __name__ == '__main__':
    main()