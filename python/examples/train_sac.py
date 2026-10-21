"""Train SAC + MLP on the low-level task, then evaluate on the test scenarios (seeds 0 to 99).

python examples/train_sac.py --steps 1000000 --seed 0 --n-envs 8 --out runs/sac_seed0

Writes to --out: config.json, progress.csv (learning curves), checkpoints/, model.zip, evaluation.json.
Training scenarios use seeds from 10 000 up, so they never coincide with the test scenarios.
Flight conditions: --wind-max, --turbulence, --noise, --randomization (default: calm air, exact sensors, nominal
airplane). The reward weights follow the L2F curriculum (RewardCurriculum) unless --no-curriculum. The evaluation always uses
the initial weights, the same reward as examples/pid_baseline.py, so the returns stay comparable.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from fixed_wing.airplane.asymmetric_sac import AsymmetricSACPolicy, CurriculumCallback, SB3Policy
from fixed_wing.airplane.environment import (
    FixedWingEnv, RewardCurriculum, add_condition_arguments, config_from_arguments, evaluate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000, help="environment steps, all environments together")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1, help="environments simulated in parallel processes")
    parser.add_argument("--threads", type=int, default=1, help="torch threads")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--no-curriculum", action="store_true", help="constant reward weights (the first run)")
    parser.add_argument("--out", type=Path, required=True)
    add_condition_arguments(parser)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    config = config_from_arguments(args, privileged_observation=True)
    (args.out / "config.json").write_text(json.dumps({"args": vars(args), "env": asdict(config)}, indent=2, default=str))

    env = make_vec_env(
        FixedWingEnv,
        n_envs=args.n_envs,
        seed=10_000 + 1_000 * args.seed,
        env_kwargs=dict(config=config),
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else DummyVecEnv,
    )
    model = SAC(
        AsymmetricSACPolicy,
        env,
        gamma=args.gamma,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=-1,  # one gradient step per environment step collected
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[256, 256])),  # small actor, to run on board
        seed=args.seed,
        verbose=1,
    )
    model.set_logger(configure(str(args.out), ["stdout", "csv"]))

    start = time.time()
    callbacks = [CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1), save_path=str(args.out / "checkpoints"))]
    if not args.no_curriculum:
        callbacks.append(CurriculumCallback(RewardCurriculum(), config))
    model.learn(total_timesteps=args.steps, callback=callbacks, log_interval=10)
    model.save(args.out / "model")
    training_seconds = time.time() - start

    # same flight conditions as the training, initial reward weights (those of examples/pid_baseline.py)
    results = evaluate(SB3Policy(model), config_from_arguments(args, privileged_observation=True), seeds=range(args.eval_episodes))
    results["training_seconds"] = training_seconds
    (args.out / "evaluation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
