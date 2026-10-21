"""PID baseline on the environment task: the score an RL policy has to beat, on the same 100 test scenarios.

python examples/pid_baseline.py [--wind-max 5 --turbulence light --noise 1]   (same flags as examples/train_sac.py)
"""
import argparse
import json

from fixed_wing.airplane.environment import add_condition_arguments, config_from_arguments, evaluate
from fixed_wing.airplane.pid import PIDPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    add_condition_arguments(parser)
    args = parser.parse_args()
    print(json.dumps(evaluate(PIDPolicy(), config_from_arguments(args), seeds=range(args.episodes)), indent=2))
