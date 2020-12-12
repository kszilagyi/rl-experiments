import importlib
import gym

from src.environment import Environment, Logger, MANDATORY_COLUMNS
from src.filelogger import FileLogger
from src.jobspec import job_specs
from src.live_graph_logger import LiveGraphLogger
from src.policy_gradient import PolicyGradient


def main():
    episode_length = 100
    algo = PolicyGradient(episode_length=episode_length)
    env = Environment(num_episodes=1000, episode_length=episode_length, env_creator=lambda: gym.make('CartPole-v1'), algo=algo)
    print(env.train(0,
                    Logger([LiveGraphLogger('episode_num', 'episode_return'),
                            LiveGraphLogger('episode_num', 'abs_max_gradient'),
                            LiveGraphLogger('episode_num', 'abs_min_gradient'),
                            LiveGraphLogger('episode_num', 'abs_min_weight'),
                            LiveGraphLogger('episode_num', 'abs_max_weight'),
                            LiveGraphLogger('episode_num', 'abs_mean_weight'),
                            LiveGraphLogger('episode_num', 'abs_mean_gradient'),

                            FileLogger(MANDATORY_COLUMNS + ['episode_return'])], {})))


if __name__ == '__main__':
    main()
