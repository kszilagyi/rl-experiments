name = 'policy-gradient-v0'

static = {
    'algo_path': 'src.policy_gradient',
    'algo_name': 'PolicyGradient',
    'episode_length': 200,
    'environment': 'CartPole-v1',
    'num_episodes': 2000
}

grid = {
    'seed': list(range(100)),
}