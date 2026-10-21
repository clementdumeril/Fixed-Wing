"""Check a reward before a long training run: how often is it nearly flat while the agent explores?

python examples/reward_saturation.py [--noise 0.1] [--episodes 4]

The PID flies with Gaussian noise added to its commands, as SAC explores during training, under the initial and the
final weights of the curriculum. The reward is nearly flat where the cost passes twice the cap: the tanh slope is
then below 7 % of its maximum, and the agent learns little about tracking. A share close to 100 % means the reward has
to be changed before training (this is what stopped the first curriculum run).
"""
import argparse

import numpy as np

from fixed_wing.airplane.environment import EnvConfig, FixedWingEnv, RewardCurriculum
from fixed_wing.airplane.pid import PIDPolicy


def flat_share(reward_config, noise, episodes):
    env = FixedWingEnv(EnvConfig(reward=reward_config))
    pid = PIDPolicy()
    rng = np.random.default_rng(0)
    flat = total = 0
    for seed in range(episodes):
        observation, _ = env.reset(seed=seed)
        pid.reset(env)
        done = False
        while not done:
            action = np.clip(pid(env, observation) + noise * rng.standard_normal(4), -1, 1)
            observation, _, term, trunc, info = env.step(action)
            done = term or trunc
            total += 1
            flat += info["cost"] >= 2 * reward_config.cost_cap * reward_config.survival
    return flat / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.1, 0.3], help="std of the command noise")
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()
    curriculum = RewardCurriculum()
    for noise in args.noise:
        for name, weights in (("initial", curriculum.initial), ("final", curriculum.target)):
            print(f"noise {noise}: {name:7s} weights, reward nearly flat on "
                  f"{100 * flat_share(weights, noise, args.episodes):5.1f} % of steps")
