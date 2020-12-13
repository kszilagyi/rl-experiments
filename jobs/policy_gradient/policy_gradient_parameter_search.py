name = 'policy-gradient-parameter-search'

static = {
    'algo_path': 'src.policy_gradient',
    'algo_name': 'PolicyGradient',
    'episode_length': 200,
    'environment': 'CartPole-v1',
    'num_episodes': 1000
}

grid = {
    'seed': list(range(100)),
}