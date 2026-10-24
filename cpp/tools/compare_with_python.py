"""Checks the C++ port against the Python environment, step by step.

python tools/compare_with_python.py <path to the check executable>   (from cpp/, with the Python package installed)

Records flights with FixedWingEnv (calm air, seeds 0 to 9, random commands around trim, plus one full nose-up flight
that stalls), has the C++ port replay the same commands from the same start, and compares the states, rewards,
terminations and observations at every step.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from fixed_wing.airplane.environment import EnvConfig, FixedWingEnv
from fixed_wing.airplane.trim import trim

fmt = "{:.17g}".format
REASONS = {None: 0, "diverged": 1, "low_altitude": 2, "alpha": 3, "airspeed": 4, "altitude": 5}
TRIM_AIRSPEEDS = [20.0, 25.0, 30.0]


def record(env, seed, n_steps, stall=False):
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(1000 + seed)
    s, sp = env.state, env.setpoint
    start = np.concatenate([s.position, s.orientation, s.linear_velocity, s.angular_velocity, s.actuators,
                            [sp.airspeed, sp.altitude, sp.course], sp.trim_action])
    actions, rows = [], []
    action = env.previous_action.copy()
    trim = sp.trim_action
    for _ in range(n_steps):
        if stall:
            action = action.copy()
            action[1] = -1.0  # full nose up
        else:
            # random commands that wander around the trim commands, so that most flights last the whole episode
            action = np.clip(trim + 0.95 * (action - trim) + 0.03 * rng.standard_normal(4), -1, 1)
        observation, r, term, trunc, info = env.step(action)
        b = env.state
        rows.append(np.concatenate([b.position, b.orientation, b.linear_velocity, b.angular_velocity, b.actuators,
                                    [r, REASONS[info["termination"]]], observation["actor"], observation["critic"]]))
        actions.append(action)
        if term or trunc:
            break
    return start, np.array(actions), np.array(rows)


def main():
    executable = sys.argv[1]
    env = FixedWingEnv(EnvConfig(privileged_observation=True))
    cases = [record(env, seed, 1500) for seed in range(10)] + [record(env, 5, 500, stall=True)]
    parameters = env.parameters
    work = Path(executable).resolve().parent / "compare"  # kept next to the executable, to rerun it by hand
    work.mkdir(exist_ok=True)
    if True:
        cases_path, out_path = work / "cases.txt", work / "out.txt"
        with open(cases_path, "w") as f:
            f.write(f"{len(TRIM_AIRSPEEDS)} {' '.join(map(repr, TRIM_AIRSPEEDS))}\n{len(cases)}\n")
            for start, actions, _ in cases:
                f.write(" ".join(map(fmt, start[:20])) + " " + " ".join(map(fmt, start[20:])) + f" {len(actions)}\n")
                for a in actions:
                    f.write(" ".join(map(fmt, a)) + "\n")
        subprocess.run([executable, str(cases_path), str(out_path)], check=True)
        lines = out_path.read_text().split("\n")

    worst = {}

    def compare(name, cpp, python, tolerance):
        error = float(np.max(np.abs(np.asarray(cpp) - np.asarray(python))))
        worst[name] = max(worst.get(name, 0.0), error)
        return error <= tolerance

    ok = True
    for k, airspeed in enumerate(TRIM_AIRSPEEDS):
        x = trim(parameters, airspeed)
        pitch = 2 * np.arctan2(-x.orientation[2], x.orientation[0])
        cpp = np.array(lines[k].split(), dtype=float)
        ok &= compare("trim", cpp, [pitch, x.actuators[1], x.actuators[3]], 1e-8)
    line = len(TRIM_AIRSPEEDS)
    for start, actions, rows in cases:
        ok &= compare("trim commands", np.array(lines[line].split(), dtype=float), np.zeros(4), 1e-8)
        line += 1
        for row in rows:
            cpp = np.array(lines[line].split(), dtype=float)
            line += 1
            ok &= compare("state", cpp[:17], row[:17], 1e-7 * max(1.0, np.max(np.abs(row[:17]))))
            ok &= compare("reward", cpp[17], row[17], 1e-6)
            ok &= compare("termination", cpp[18], row[18], 0)
            ok &= compare("observations", cpp[19:], row[19:], 1e-5)  # Python stores them as float32
    for name, error in worst.items():
        print(f"{name:14s} max |C++ - Python| = {error:.2e}")
    n_steps = sum(len(rows) for _, _, rows in cases)
    print(f"{len(cases)} flights, {n_steps} steps: {'OK' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
