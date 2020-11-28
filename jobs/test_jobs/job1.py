name = 'test-job'

static = {
    'algo_path': 'src.policy_gradient',
    'algo_name': 'PolicyGradient',
    'episode_length': 200,
    'environment': 'CartPole-v1',
    'num_episodes': 10
}

grid = {
    'seed': list(range(2)),
    'size': ['small', 'large']
}