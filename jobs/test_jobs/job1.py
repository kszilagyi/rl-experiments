
from src.jobspec import JobSpec
from src.policy_gradient import PolicyGradient

job_spec = JobSpec(PolicyGradient, 200, 'CartPole-v1', num_episodes=1000)