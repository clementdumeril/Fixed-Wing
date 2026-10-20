"""PID baseline on the environment task: the score an RL policy has to beat, on the same 100 scenarios."""
import json

from fixed_wing.airplane.environment import EnvConfig, evaluate
from fixed_wing.airplane.pid import PIDPolicy

if __name__ == "__main__":
    print(json.dumps(evaluate(PIDPolicy(), EnvConfig(), seeds=range(100)), indent=2))
