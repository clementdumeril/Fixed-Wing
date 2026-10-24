"""How a pilot flies the 100 test scenarios beyond the RMS errors: attitude in straight flight and how flights end.

python examples/diagnose_flights.py runs/sac_123_seed0 --out diagnosis.json   (conditions of that training run)
python examples/diagnose_flights.py PID --wind-max 5 --turbulence light --noise 1 --out pid_windy.json

Straight flight is the last 10 s of each complete flight, once the setpoint is reached. There a pilot that flies
level has no bank, no sideslip and commands close to trim; the first SAC run (results/sac_run1) flew 25 deg banked in
a 9 deg sideslip. For each flight that ended early: the reason, the time, and the state at the end.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fixed_wing.airplane.airplane import AILERON, ELEVATOR, RUDDER, THROTTLE
from fixed_wing.airplane.environment import (FixedWingEnv, add_condition_arguments, config_from_arguments,
                                             config_from_run, measurements, wrap)
from fixed_wing.airplane.operations import air_data
from fixed_wing.airplane.pid import PIDPolicy

STRAIGHT_SECONDS = 10.0


def fly(env, policy, seed):
    observation, _ = env.reset(seed=seed)
    policy.reset(env)
    log, done = [], False
    while not done:
        action = np.asarray(policy(env, observation))
        observation, _, term, trunc, info = env.step(action)
        m = measurements(env.parameters, env.state, env.config)
        _, airspeed, alpha, beta = air_data(env.parameters, env.state)
        trim = env.setpoint.trim_action
        log.append(dict(t=env.steps / env.config.frequency, roll=np.rad2deg(m["roll"]), beta=np.rad2deg(beta),
                        alpha=np.rad2deg(alpha), aileron=action[AILERON] - trim[AILERON],
                        rudder=action[RUDDER] - trim[RUDDER], elevator=action[ELEVATOR], throttle=action[THROTTLE],
                        airspeed=airspeed, altitude_error=m["altitude"] - env.setpoint.altitude,
                        course_error=np.rad2deg(wrap(m["course"] - env.setpoint.course))))
        done = term or trunc
    return log, info["termination"] if term else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot", help="a run directory of examples/train_sac.py, or PID")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=Path)
    add_condition_arguments(parser)
    args = parser.parse_args()

    if args.pilot == "PID":
        config, policy = config_from_arguments(args, privileged_observation=True), PIDPolicy()
    else:
        from stable_baselines3 import SAC

        from fixed_wing.airplane.asymmetric_sac import SB3Policy

        config = config_from_run(args.pilot, privileged_observation=True)
        policy = SB3Policy(SAC.load(Path(args.pilot) / "model.zip", device="cpu"))
    env = FixedWingEnv(config)
    straight, failures = [], []
    for seed in range(args.episodes):
        log, reason = fly(env, policy, seed)
        if reason is None:
            last = log[-round(STRAIGHT_SECONDS * config.frequency):]
            straight.append([np.mean([abs(s[k]) for s in last]) for k in ("roll", "beta", "aileron", "rudder")])
        else:
            end = log[-1]
            failures.append(dict(seed=seed, reason=reason, t=end["t"], max_abs_roll_deg=max(abs(s["roll"]) for s in log),
                                 **{f"{k}_end": end[k] for k in ("alpha", "roll", "elevator", "throttle", "airspeed",
                                                                  "altitude_error", "course_error")}))
    straight = np.array(straight)
    result = dict(pilot=args.pilot, complete_flights=len(straight),
                  straight_flight_mean_abs=dict(zip(("roll_deg", "sideslip_deg", "aileron_from_trim", "rudder_from_trim"),
                                                    straight.mean(axis=0).round(3).tolist())),
                  straight_flight_max_abs_roll_deg=round(float(straight[:, 0].max()), 2),
                  failures_within_5_s=sum(f["t"] <= 5 for f in failures),
                  failures=[{k: round(float(v), 2) if isinstance(v, (float, np.floating)) else v for k, v in f.items()}
                            for f in failures])
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
