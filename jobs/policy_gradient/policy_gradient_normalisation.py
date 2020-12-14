name = 'policy-gradient-parameter-search'

static = {
    'algo_path': 'src.policy_gradient',
    'algo_name': 'PolicyGradient',
    'episode_length': 200,
    'environment': 'CartPole-v1',
    'num_episodes': 1000,
    'normalise_returns_with_episode_length': True,
}

grid = {
    'seed': list(range(100)),

    'normalise_with_max_returns': [False, True],

}

# random_search_n = 8
# random_search = {
#     'normalise_returns': [False, True],
#     'center_returns': [False, True],
#
#     'lr': [1e-4, 1e-3, 1e-2, 1e-1]
# }

list_search = [
    {
        'grid': {
            'normalise_returns': [True],
            'center_returns': [True],
            'lr': [1e-1]
        }
    },
    {
        'grid': {
            'normalise_returns': [False],
            'center_returns': [False, True],
            'lr': [1e-4, 1e-3, 1e-2, 1e-1]
        }
    },

]