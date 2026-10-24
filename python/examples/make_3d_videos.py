"""3D GIFs of a training run.

python examples/make_3d_videos.py runs/sac_123_seed0 --out runs/videos

swarm_3d.gif  the 100 test scenarios flown at the same time, one dot and a short trail per airplane, by the PID and by
              the policy at the start, middle and end of training. A red cross marks a flight that ended early.
              The vertical axis is stretched: the altitudes span tens of meters, the flights hundreds.
flight_3d.gif one test scenario, RL and PID side by side, with an airplane model that banks and yaws; the camera
              follows it. The airplane is drawn 15 times larger than the Aerosonde so that its attitude is visible.
"""
import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from fixed_wing.airplane.environment import FixedWingEnv, config_from_run, measurements
from fixed_wing.airplane.operations import air_data
from fixed_wing.airplane.pid import PIDPolicy
from fixed_wing.quaternion_helper import quaternion_to_rotation_matrix

SAMPLE = 0.25  # s of flight between frames
FPS = 15
TRAIL = 8  # samples of trail behind each dot

# airplane outline in the body frame (x forward, y left, z up), meters, Aerosonde-like proportions
WING = np.array([[0.2, 1.45, 0], [0.2, -1.45, 0], [-0.15, -1.45, 0], [-0.15, 1.45, 0]])
TAIL = np.array([[-1.1, 0.5, 0], [-1.1, -0.5, 0], [-1.35, -0.5, 0], [-1.35, 0.5, 0]])
FIN = np.array([[-1.05, 0, 0], [-1.35, 0, 0], [-1.35, 0, 0.45], [-1.2, 0, 0.45]])
BODY = np.array([[0.9, 0.08, 0], [0.9, -0.08, 0], [-1.35, -0.05, 0], [-1.35, 0.05, 0]])


def fly(env, policy, seed, sample=SAMPLE):
    observation, _ = env.reset(seed=seed)
    policy.reset(env)
    every = round(sample * env.config.frequency)
    log = dict(position=[env.state.position.copy()], R=[quaternion_to_rotation_matrix(env.state.orientation)],
               roll=[0.0], beta=[0.0])
    done = False
    while not done:
        observation, _, term, trunc, info = env.step(policy(env, observation))
        done = term or trunc
        if env.steps % every == 0 or done:
            m = measurements(env.parameters, env.state, env.config)
            log["position"].append(env.state.position.copy())
            log["R"].append(m["rotation"])
            log["roll"].append(np.rad2deg(m["roll"]))
            log["beta"].append(np.rad2deg(air_data(env.parameters, env.state)[3]))
    return {k: np.array(v) for k, v in log.items()}, info["termination"], env.setpoint


def load_policy(path):
    from stable_baselines3 import SAC

    from fixed_wing.airplane.asymmetric_sac import SB3Policy

    return SB3Policy(SAC.load(path, device="cpu"))


def swarm_video(env, pilots, seeds, path):
    """pilots: list of (title, policy, color)."""
    flown = []
    for title, policy, color in pilots:
        flights = [fly(env, policy, seed) for seed in seeds]
        ended = sum(reason is not None for _, reason, _ in flights)
        flown.append((f"{title}\n{len(seeds) - ended}/{len(seeds)} flights complete", flights, color))
        print(f"  {title}: {ended} ended early")
    n = 1 + round(30 / SAMPLE)
    allpos = np.concatenate([f[0]["position"] for _, flights, _ in flown for f in flights])
    lim = np.abs(allpos[:, :2]).max() + 50

    columns = 2 if len(flown) > 2 else len(flown)
    rows = -(-len(flown) // columns)
    fig = plt.figure(figsize=(6.5 * columns, 5.5 * rows))
    axes = [fig.add_subplot(rows, columns, i + 1, projection="3d") for i in range(len(flown))]
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0, hspace=0.05)
    title = fig.suptitle("", fontsize=13)

    def draw(k):
        for ax, (name, flights, color) in zip(axes, flown):
            ax.cla()
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(20, 140))
            ax.set_title(name, fontsize=11, pad=0)
            ax.set_box_aspect((1, 1, 0.6))
            ax.set_zlabel("altitude (m)", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.view_init(elev=28, azim=-60 + 0.6 * k)
            for log, reason, _ in flights:
                p = log["position"]
                i = min(k, len(p) - 1)
                ax.plot(p[:i + 1, 0], p[:i + 1, 1], p[:i + 1, 2], color=color, linewidth=0.5, alpha=0.35)
                if reason is not None and k >= len(p) - 1:
                    ax.scatter(*p[-1], color="red", marker="x", s=40, linewidths=1.5)
                    continue
                trail = p[max(0, i - TRAIL):i + 1]
                ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=color, linewidth=1.5)
                ax.scatter(*p[i], color=color, s=10)
        title.set_text(f"The same {len(seeds)} test scenarios flown at once   t = {k * SAMPLE:4.1f} s   (red cross: flight ended)")
        return axes

    frames = list(range(n)) + [n - 1] * FPS
    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)


def airplane_faces(position, R, scale):
    return [position + (R @ (shape * scale).T).T for shape in (WING, TAIL, FIN, BODY)]


def flight_video(env, pilots, seed, path, scale=15.0, sample=0.2):
    flights = [(title, fly(env, policy, seed, sample), color) for title, policy, color in pilots]
    setpoint = flights[0][1][2]
    n = max(len(log["position"]) for _, (log, _, _), _ in flights)
    fig = plt.figure(figsize=(6 * len(flights), 5.4))
    axes = [fig.add_subplot(1, len(flights), i + 1, projection="3d") for i in range(len(flights))]
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.88, wspace=0)
    title = fig.suptitle("")
    half = 40.0  # m around the airplane

    def draw(k):
        lines = []
        for ax, (name, (log, reason, _), color) in zip(axes, flights):
            ax.cla()
            i = min(k, len(log["position"]) - 1)
            p = log["position"][i]
            ax.set(xlim=(p[0] - half, p[0] + half), ylim=(p[1] - half, p[1] + half), zlim=(p[2] - half / 2, p[2] + half / 2))
            ax.set_box_aspect((1, 1, 0.5))
            ax.tick_params(labelsize=6)
            ax.view_init(elev=18, azim=np.rad2deg(np.arctan2(log["R"][i][1, 0], log["R"][i][0, 0])) - 150)
            inside = lambda q: np.where((np.abs(q[:, :2] - p[:2]).max(axis=1) <= half)[:, None], q, np.nan)  # 3D axes do not clip
            path_so_far = inside(log["position"][:i + 1])
            ax.plot(path_so_far[:, 0], path_so_far[:, 1], path_so_far[:, 2], color=color, linewidth=1)
            ground = path_so_far.copy()
            ground[:, 2] = p[2] - half / 2
            ax.plot(ground[:, 0], ground[:, 1], ground[:, 2], color="0.75", linewidth=0.8)  # shadow
            ahead = inside(p + np.linspace(0, half, 20)[:, None] * [np.cos(setpoint.course), np.sin(setpoint.course), 0])
            ax.plot(ahead[:, 0], ahead[:, 1], np.full(len(ahead), setpoint.altitude), "k--", linewidth=0.8)
            ax.add_collection3d(Poly3DCollection(airplane_faces(p, log["R"][i], scale), facecolor=color,
                                                 edgecolor="k", linewidth=0.4, alpha=0.95))
            status = "" if i < len(log["position"]) - 1 or reason is None else f"  ended: {reason}"
            ax.set_title(f"{name}: roll {log['roll'][i]:5.1f} deg, sideslip {log['beta'][i]:4.1f} deg{status}", fontsize=9)
            lines.append(ax)
        title.set_text(f"Test scenario {seed}   t = {min(k * sample, 30):4.1f} s   dashed: setpoint course at setpoint altitude")
        return lines

    frames = list(range(n)) + [n - 1] * FPS
    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="directory written by examples/train_sac.py")
    parser.add_argument("--stages", type=int, nargs="+", default=[100_000, 500_000, 1_000_000],
                        help="checkpoints shown in the swarm (training steps)")
    parser.add_argument("--episodes", type=int, default=100, help="test scenarios in the swarm")
    parser.add_argument("--seed", type=int, default=1, help="test scenario of flight_3d.gif")
    parser.add_argument("--out", type=Path, default=Path("runs/videos"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = FixedWingEnv(config_from_run(args.run, privileged_observation=True))  # wind etc. of the training
    checkpoints = {int(re.search(r"(\d+)_steps", p.name).group(1)): p for p in (args.run / "checkpoints").glob("*_steps.zip")}
    pilots = [("PID", PIDPolicy(), "tab:blue")]
    pilots += [(f"RL after {s:,} steps", load_policy(checkpoints[s]), "tab:orange") for s in args.stages]
    swarm_video(env, pilots, range(args.episodes), args.out / f"{args.run.name}_swarm_3d.gif")
    print("saved swarm video")
    final = [("RL", load_policy(args.run / "model.zip"), "tab:orange"), ("PID", PIDPolicy(), "tab:blue")]
    flight_video(env, final, args.seed, args.out / f"{args.run.name}_flight_3d_scenario{args.seed}.gif")
    print("saved flight video")


if __name__ == "__main__":
    main()
