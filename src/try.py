import os

from src.cloud_run import run
from src.live_graph_logger import LiveGraphLogger

print(os.path.basename(__file__))
def main():
    hyperparams = { 'algo_path': 'src.policy_gradient',
                    'algo_name': 'PolicyGradient',
                    'episode_length': 200,
                    'environment': 'CartPole-v1',
                    'max_sample_cnt': 500*200,
                    'normalise_returns_with_episode_length': True,
                    'seed': 2,
                    'normalise_with_max_returns': True,
                    'normalise_returns': False,
                    'center_returns': False,
                    'lr': 1e-1,
                    'gamma': 0.95,
                    'optimizer': 'sgd'}

    run(hyperparams, [
        LiveGraphLogger('episode_num', 'episode_return'),
                            LiveGraphLogger('episode_num', 'abs_max_gradient'),
                            LiveGraphLogger('episode_num', 'abs_min_gradient'),
                            # LiveGraphLogger('episode_num', 'abs_min_weight'),
                            # LiveGraphLogger('episode_num', 'abs_max_weight'),
                            # LiveGraphLogger('episode_num', 'abs_mean_weight'),
                            # LiveGraphLogger('episode_num', 'abs_mean_gradient'),

                          ])


if __name__ == '__main__':
    import cProfile
    # cProfile.run('main()', 'main_stats')
    main()
