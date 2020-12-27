import os

name = os.path.basename(__file__).replace('_', '-')

static = {
    'algo_path': 'src.policy_gradient',
    'algo_name': 'PolicyGradient',

    'max_sample_cnt': 1000*200,
    'gamma': 0.95,
    'normalise_with_max_returns': False,
    'normalise_returns': True,
    'center_returns': True,
    'optimizer': 'adam',
    'render_freq': 100000000,
}

seeds = list(range(100))

search = [
    {
        'grid': {
            'seed': seeds,
            'normalise_returns_with_episode_length': [True, False],
            'environment': ['CartPole-v1', 'MountainCar-v0'],
            'episode_length': 200,
            'lr': [0.0125, 0.0025, 0.005]
        }
    },
    {
        'grid': {
            'seed': seeds,
            'normalise_returns_with_episode_length': [True, False],
            'environment': ['LunarLander-v2'],
            'episode_length': 1000,
            'lr': [0.0125, 0.0025, 0.005]
        }
    },
]