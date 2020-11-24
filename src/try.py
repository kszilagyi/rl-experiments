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
    print(env.train(1,
                    Logger([LiveGraphLogger('episode_num', 'episode_return'),
                            FileLogger('seed', 'seed', MANDATORY_COLUMNS + ['episode_return'])])))


if __name__ == '__main__':
    main()
