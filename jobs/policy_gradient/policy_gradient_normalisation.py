name = 'policy-gradient-normalisation'

static = {
    'algo_path': 'src.policy_gradient',
    'algo_name': 'PolicyGradient',
    'episode_length': 200,
    'environment': 'CartPole-v1',
    'num_episodes': 100,
    'normalise_returns_with_episode_length': True,
    'gamma': 0.95,
}

seeds = list(range(1)) + [1]
# random_search_n = 8
# random_search = {
#     'normalise_returns': [False, True],
#     'center_returns': [False, True],
#
#     'lr': [1e-4, 1e-3, 1e-2, 1e-1]
# }

search = [
    {
        'grid': {
            'seed': seeds,
            'normalise_with_max_returns': [False, True],
            'normalise_returns': [True],
            'center_returns': [True],
            'lr': [1e-1]
        }
    },
    {
        'grid': {
            'seed': seeds,
            'normalise_with_max_returns': [True],
            'normalise_returns': [False],
            'center_returns': [False, True],
            'lr': [1e-4, 1e-3, 1e-2, 1e-1]
        }
    },

]