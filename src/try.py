import os

from src.cloud_run import run
from src.live_graph_logger import LiveGraphLogger

print(os.path.basename(__file__))
def main():
    hyperparams = { 'algo_path': 'src.policy_gradient',
                    'algo_name': 'PolicyGradient',
                    'episode_length': 200,
                    # 'environment': 'LunarLander-v2',
                    'environment': 'Pendulum-v0',
                    'max_sample_cnt': 1000*1000,
                    'normalise_returns_with_episode_length': True,
                    'seed': 4,
                    'normalise_with_max_returns': False,
                    'normalise_returns': True,
                    'center_returns': True,
                    'lr': 0.0025,
                    'gamma': 0.95,
                    'optimizer': 'adam',
                    'model_save_freq': 100}

    run(hyperparams, [
                            LiveGraphLogger('episode_num', 'episode_return'),
                            # LiveGraphLogger('sample_cnt', 'episode_return'),
                            # LiveGraphLogger('episode_num', 'abs_max_gradient'),
                            # LiveGraphLogger('episode_num', 'abs_min_gradient'),
                            # LiveGraphLogger('episode_num', 'abs_min_weight'),
                            # LiveGraphLogger('episode_num', 'abs_max_weight'),
                            # LiveGraphLogger('episode_num', 'abs_mean_weight'),
                            # LiveGraphLogger('episode_num', 'abs_mean_gradient'),

                          ], render_only=False)


if __name__ == '__main__':
    import cProfile
    # cProfile.run('main()', 'main_stats')
    main()
