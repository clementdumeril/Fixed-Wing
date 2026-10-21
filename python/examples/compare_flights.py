"""RL policy and PID on the same test scenarios: airspeed, altitude, course error and commands over time.

python examples/compare_flights.py --model runs/sac_123_seed0/model.zip --seeds 0 1 2 --out flights.png
Without --model, only the PID flies.
"""
import argparse
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fixed_wing.airplane.airplane import AILERON, ELEVATOR, THROTTLE
from fixed_wing.airplane.environment import EnvConfig, FixedWingEnv, measurements, wrap
from fixed_wing.airplane.pid import PIDPolicy


def fly(env, policy, seed):
    observation, _ = env.reset(seed=seed)
    policy.reset(env)
    log = defaultdict(list)
    done = False
    while not done:
        action = np.asarray(policy(env, observation))
        observation, _, term, trunc, info = env.step(action)
        m = measurements(env.parameters, env.state, env.config)
        log["t"].append(env.steps / env.config.frequency)
        log["airspeed"].append(m["airspeed"])
        log["altitude"].append(m["altitude"])
        log["course_error"].append(np.rad2deg(wrap(m["course"] - env.setpoint.course)))
        log["action"].append(action)
        done = term or trunc
    return {k: np.array(v) for k, v in log.items()}, env.setpoint, info["termination"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model.zip of examples/train_sac.py")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="test scenarios")
    parser.add_argument("--out", default="flights.png")
    args = parser.parse_args()

    env = FixedWingEnv(EnvConfig(privileged_observation=True))
    pilots = {"PID": PIDPolicy()}
    if args.model:
        from stable_baselines3 import SAC

        from fixed_wing.airplane.asymmetric_sac import SB3Policy

        pilots["RL"] = SB3Policy(SAC.load(args.model, device="cpu"))
    styles = {"PID": dict(color="tab:blue"), "RL": dict(color="tab:orange")}

    fig, axes = plt.subplots(len(args.seeds), 4, figsize=(16, 3.2 * len(args.seeds)), squeeze=False)
    for row, seed in zip(axes, args.seeds):
        for name, policy in pilots.items():
            log, setpoint, end = fly(env, policy, seed)
            label = name if end is None else f"{name} ({end})"
            row[0].plot(log["t"], log["airspeed"], label=label, **styles[name])
            row[1].plot(log["t"], log["altitude"], **styles[name])
            row[2].plot(log["t"], log["course_error"], **styles[name])
            row[3].plot(log["t"], log["action"][:, ELEVATOR], **styles[name])
            row[3].plot(log["t"], log["action"][:, AILERON], linestyle=":", **styles[name])
            row[3].plot(log["t"], log["action"][:, THROTTLE], linestyle="--", linewidth=0.8, **styles[name])
        row[0].axhline(setpoint.airspeed, color="gray", linestyle="--", linewidth=0.8)
        row[1].axhline(setpoint.altitude, color="gray", linestyle="--", linewidth=0.8)
        row[2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        row[0].set(ylabel=f"scenario {seed}\nairspeed (m/s)")
        row[1].set(ylabel="altitude (m)")
        row[2].set(ylabel="course error (deg)")
        row[3].set(ylabel="normalized commands")
        row[0].legend(fontsize=8)
    for ax, title in zip(axes[0], ["Airspeed (dashed: setpoint)", "Altitude", "Course error",
                                   "Elevator, aileron (dotted), throttle (dashed)"]):
        ax.set_title(title, fontsize=10)
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
