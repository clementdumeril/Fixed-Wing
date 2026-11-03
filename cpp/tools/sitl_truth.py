"""Gazebo's ground truth of an identification flight (IDENTIFY=1 deploy/ros2/fly_sitl.sh) as a flight log for compare_sitl.

python tools/sitl_truth.py <flight directory> [start, s of simulation time]

joints.jsonl holds Gazebo's joint states at every physics step (250 Hz): the model pose, the servo angles and the rotor
speed. This writes <flight directory>/truth.csv, one row per control period (20 ms) from the start of the policy's
flight: position, velocity and body rates from the pose (central differences), and the measured surface and rotor
positions, in the columns of flight.csv plus sim_t and measured_*. compare_sitl truth.csv then compares this simulator
with Gazebo's physics alone, without PX4's state estimate and its timing.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

directory = Path(sys.argv[1])
times, positions, quaternions, servos, rotor = [], [], [], [], []
for line in open(directory / "joints.jsonl"):
    message = json.loads(line)
    stamp = message["header"]["stamp"]
    times.append(int(stamp.get("sec", 0)) + int(stamp.get("nsec", 0)) * 1e-9)
    p, o = message["pose"]["position"], message["pose"]["orientation"]  # world ENU, body FLU, as in the simulator
    positions.append([p.get("x", 0), p.get("y", 0), p.get("z", 0)])
    quaternions.append([o.get("w", 1), o.get("x", 0), o.get("y", 0), o.get("z", 0)])
    joints = {j["name"]: j["axis1"] for j in message["joint"] if "axis1" in j}
    servos.append([joints[f"servo_{i}"].get("position", 0) for i in range(4)])
    rotor.append(abs(joints["rotor_puller_joint"].get("velocity", 0)) * 10)  # rotorVelocitySlowdownSim
t, P, Q, S, W = map(np.array, (times, positions, quaternions, servos, rotor))


def multiply(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


span = (t[2:] - t[:-2])[:, None]
velocity = (P[2:] - P[:-2]) / span
turn = multiply(Q[:-2] * np.array([1, -1, -1, -1]), Q[2:])
turn *= np.sign(turn[:, :1])
rates = 2 * turn[:, 1:] / span  # body frame, small rotation over 8 ms

# the policy's flight starts at the switch to offboard: from flight.csv aligned on the pose, or given
if len(sys.argv) > 2:
    start = float(sys.argv[2])
else:
    first = next(csv.DictReader(open(directory / "flight.csv")))
    # Gazebo's world and PX4's local frame share their origin and axes, to a few decimetres
    here = np.array([float(first["x"]), float(first["y"]), float(first["altitude"])])
    start = t[np.argmin(((P - here) ** 2).sum(1))]

quarter = np.pi / 4  # servo command +-1 = +-45 deg
rows = []
for i in range(0, len(t) - 2, 5):  # every 5 physics steps: the 20 ms control period
    k = i + 1
    if t[k] < start:
        continue
    aileron, elevator, rudder = (S[k, 0] - S[k, 1]) / 2, S[k, 2], S[k, 3]
    throttle = (W[k] - 10) / 1590
    rows.append([t[k] - start, *P[k], *velocity[i], *Q[k], *rates[i], 0, 0, 0, aileron / quarter, elevator / quarter,
                 rudder / quarter, 2 * throttle - 1, 14, t[k], aileron, elevator, rudder, throttle])
header = ("t,x,y,altitude,vx,vy,vz,qw,qx,qy,qz,p,q,r,reference_x,reference_y,reference_altitude,aileron,elevator,rudder,"
          "throttle,nav_state,sim_t,measured_aileron,measured_elevator,measured_rudder,measured_throttle").split(",")
with open(directory / "truth.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
print(f"wrote {directory / 'truth.csv'}: {len(rows)} rows from {start:.2f} s of simulation time")
