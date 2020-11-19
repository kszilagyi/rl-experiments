import gym

from environment import Environment, Logger
from live_graph_logger import LiveGraphLogger
from policy_gradient import PolicyGradient


def main():
    episode_length = 100
    algo = PolicyGradient(episode_length=episode_length)
    env = Environment(num_episodes=1000, episode_length=episode_length, env_creator=lambda: gym.make('CartPole-v1'), algo=algo)
    print(env.train(1,
                    Logger(['episode_return'], [LiveGraphLogger('episode_num', 'episode_return'),
                                                ])))


if __name__ == '__main__':
    main()
