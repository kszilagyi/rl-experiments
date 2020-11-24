parser = argparse.ArgumentParser('Run online')
parser.add_argument('job_spec_path', type=str)
args = parser.parse_args()
job_spec_module = importlib.import_module(args['job_spec_path'])
job_spec: JobSpec = job_spec_module.job_spec
algo = job_spec.algo(episode_length=job_spec.episode_length)
env = Environment(num_episodes=job_spec.num_episodes, episode_length=job_spec.episode_length,
                  env_creator=lambda: gym.make(job_spec.environment), algo=algo)
env.train(1, Logger([FileLogger('seed', 'seed', MANDATORY_COLUMNS + ['episode_return'])]))