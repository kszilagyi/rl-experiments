import os

name = os.path.basename(__file__).replace('_', '-')

static = {
    'algo_path': 'src.policy_gradient',
    'algo_name': 'PolicyGradient',
    'episode_length': 200,
    'environment': 'CartPole-v1',
    'max_sample_cnt': 500*200,
    'gamma': 0.95,
    'normalise_with_max_returns': False,
    'normalise_returns': True,
    'center_returns': True,
}

seeds = list(range(100))

search = [
    {
        'grid': {
            'seed': seeds,
            'lr': [0.0025, 0.005, 0.01, 0.02, 0.04],
            'optimizer': ['adam'],
            'normalise_returns_with_episode_length': [True],
        }
    },
    {
        'grid': {
            'seed': seeds,
            'lr': [0.025, 0.05, 0.1, 0.2, 0.4],
            'optimizer': ['sgd'],
            'normalise_returns_with_episode_length': [True],
        }
    },
    {
        'grid': {
            'seed': seeds,
            'lr': [0.01],
            'optimizer': ['adam'],
            'normalise_returns_with_episode_length': [False],
        }
    },
]