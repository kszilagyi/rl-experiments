from dataclasses import dataclass


@dataclass(frozen=True)
class JobSpec:
    algo: type
    episode_length: int
    environment: str
    num_episodes: int



job_specs = {}

def register_job_spec(name: str, job_spec: JobSpec):
    global job_specs
    assert job_spec not in job_specs
    job_specs[name] = job_spec