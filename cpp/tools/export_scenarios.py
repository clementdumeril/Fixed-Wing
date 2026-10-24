"""Writes the 100 test scenarios of the Python environment (seeds 0 to 99, calm air) for the C++ evaluation, so that
both implementations, the PID and later the NMPC fly the same starts and setpoints.

python tools/export_scenarios.py scenarios_calm.txt   (from cpp/, with the Python package installed)

One line per scenario: position (3), orientation (4), linear velocity (3), angular velocity (3), actuators (4),
setpoint airspeed, altitude, course.
"""
import sys

import numpy as np

from fixed_wing.airplane.environment import EnvConfig, FixedWingEnv


fmt = "{:.17g}".format


def main():
    env = FixedWingEnv(EnvConfig())
    with open(sys.argv[1], "w") as f:
        f.write("100\n")
        for seed in range(100):
            env.reset(seed=seed)
            s, sp = env.state, env.setpoint
            row = np.concatenate([s.position, s.orientation, s.linear_velocity, s.angular_velocity, s.actuators,
                                  [sp.airspeed, sp.altitude, sp.course]])
            f.write(" ".join(map(fmt, row)) + "\n")
    print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
