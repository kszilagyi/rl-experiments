import os

name = os.path.basename(__file__).replace('_', '-')

static = {
    'algo_path': 'src.policy_gradient',
    'algo_name': 'PolicyGradient',

    'gamma': 0.95,
    'normalise_with_max_returns': False,
    'normalise_returns': True,
    'center_returns': True,
    'optimizer': 'adam',
    'model_save_freq': 100000000,
    'normalise_returns_with_episode_length': True,

}

seeds = list(range(20))
search = [
    {
        'grid': {
            'max_sample_cnt': 1000*1000,
            'seed': list(range(10)),
            'environment': ['BipedalWalker-v3'],
            'episode_length': [1000],
            'lr': [0.0125, 0.0025, 0.005]
        }
    },
    {
        'grid': {
            'max_sample_cnt': 1000*500,
            'seed': seeds,
            'environment': ['CartPole-v1'],
            'episode_length': [200],
            'lr': [0.0025]
        }
    },
    {
        'grid': {
            'max_sample_cnt': 1000*500,
            'seed': seeds,
            'environment': ['LunarLander-v2'],
            'episode_length': [1000],
            'lr': [0.0025, 0.005]
        }
    },
    {
        'grid': {
            'max_sample_cnt': 1000*500,
            'seed': seeds,
            'environment': ['Pendulum-v0'],
            'episode_length': [200],
            'lr': [0.0125, 0.0025, 0.005]
        }
    },
]