"""Animated GIFs of a training run, seen from above and over time.

python examples/make_videos.py runs/sac_123_seed0 --seed 1 --out runs/videos

learning.gif  the policy saved every 100 000 steps flying the same test scenario, one after the other: how it learned
rl_vs_pid.gif the final policy and the PID on that scenario at the same time. The airplane symbol points where the
              nose points, so flying sideways (sideslip) shows as a nose off the track.
"""
import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from fixed_wing.airplane.environment import EnvConfig, FixedWingEnv, measurements
from fixed_wing.airplane.operations import air_data
from fixed_wing.airplane.pid import PIDPolicy

SAMPLE = 0.5  # s of flight between two frames in learning.gif (7.5x real time), 0.2 s in rl_vs_pid.gif (3x)
FPS = 15  # frames per second of the GIFs


def fly(env, policy, seed, sample=SAMPLE):
    """Samples of one flight every sample seconds, and how it ended."""
    observation, _ = env.reset(seed=seed)
    policy.reset(env)
    every = round(sample * env.config.frequency)
    samples = []
    done, reason = False, None
    while not done:
        observation, _, term, trunc, info = env.step(policy(env, observation))
        done, reason = term or trunc, info["termination"]
        if env.steps % every == 0 or done:
            m = measurements(env.parameters, env.state, env.config)
            _, _, _, beta = air_data(env.parameters, env.state)
            R = m["rotation"]
            samples.append(dict(
                t=env.steps / env.config.frequency, x=env.state.position[0], y=env.state.position[1],
                altitude=m["altitude"], airspeed=m["airspeed"], yaw=np.arctan2(R[1, 0], R[0, 0]),
                roll=np.rad2deg(m["roll"]), beta=np.rad2deg(beta),
            ))
    return samples, reason


def airplane(ax, color):
    return ax.fill([], [], color=color, zorder=5)[0]


def place(symbol, x, y, yaw, size):
    shape = np.array([[1.0, 0], [-0.6, 0.5], [-0.3, 0], [-0.6, -0.5]]) * size
    c, s = np.cos(yaw), np.sin(yaw)
    symbol.set_xy(shape @ np.array([[c, s], [-s, c]]) + [x, y])


def layout(setpoint, extent):
    fig = plt.figure(figsize=(10, 5.2))
    top = fig.add_axes([0.05, 0.1, 0.45, 0.78])
    altitude = fig.add_axes([0.58, 0.55, 0.38, 0.33])
    airspeed = fig.add_axes([0.58, 0.1, 0.38, 0.33])
    (xmin, xmax), (ymin, ymax) = extent
    top.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Seen from above")
    reach = 2 * max(xmax - xmin, ymax - ymin)
    top.plot([0, reach * np.cos(setpoint.course)], [0, reach * np.sin(setpoint.course)], "k--", linewidth=0.8,
             label="setpoint course")
    altitude.axhline(setpoint.altitude, color="gray", linestyle="--", linewidth=0.8)
    airspeed.axhline(setpoint.airspeed, color="gray", linestyle="--", linewidth=0.8)
    altitude.set(xlim=(0, 30), ylabel="altitude (m)")
    airspeed.set(xlim=(0, 30), ylabel="airspeed (m/s)", xlabel="time (s)")
    for ax in (top, altitude, airspeed):
        ax.grid(alpha=0.3)
    return fig, top, altitude, airspeed


def extent(flights):
    xs = np.concatenate([[s["x"] for s in f] for f in flights] + [[0.0]])
    ys = np.concatenate([[s["y"] for s in f] for f in flights] + [[0.0]])
    pad = 60
    return (xs.min() - pad, xs.max() + pad), (ys.min() - pad, ys.max() + pad)


def limits(axis, flights, key, setpoint_value):
    values = np.concatenate([[s[key] for s in f] for f in flights] + [[setpoint_value]])
    margin = max(1.0, 0.1 * (values.max() - values.min()))
    axis.set_ylim(values.min() - margin, values.max() + margin)


def learning_video(env, checkpoints, seed, path):
    from stable_baselines3 import SAC

    from fixed_wing.airplane.asymmetric_sac import SB3Policy

    flights = []
    for steps, file in checkpoints:
        samples, reason = fly(env, SB3Policy(SAC.load(file, device="cpu")), seed)
        flights.append((steps, samples, reason))
    setpoint = env.setpoint
    fig, top, altitude, airspeed = layout(setpoint, extent([f for _, f, _ in flights]))
    limits(altitude, [f for _, f, _ in flights], "altitude", setpoint.altitude)
    limits(airspeed, [f for _, f, _ in flights], "airspeed", setpoint.airspeed)
    size = 0.03 * np.diff(top.get_xlim())[0]
    track, = top.plot([], [], color="tab:orange")
    alt_line, = altitude.plot([], [], color="tab:orange")
    speed_line, = airspeed.plot([], [], color="tab:orange")
    symbol = airplane(top, "tab:orange")
    title = fig.suptitle("")
    frames = [(i, k) for i, (_, samples, _) in enumerate(flights) for k in range(len(samples))]
    frames += [frames[-1]] * FPS  # hold the last image one second

    def draw(frame):
        i, k = frame
        steps, samples, reason = flights[i]
        if k == 0:  # previous flights stay as gray ghosts
            for _, previous, _ in flights[:i][-1:]:
                top.plot([s["x"] for s in previous], [s["y"] for s in previous], color="0.8", linewidth=0.8, zorder=1)
                altitude.plot([s["t"] for s in previous], [s["altitude"] for s in previous], color="0.8", linewidth=0.8)
                airspeed.plot([s["t"] for s in previous], [s["airspeed"] for s in previous], color="0.8", linewidth=0.8)
        part = samples[:k + 1]
        track.set_data([s["x"] for s in part], [s["y"] for s in part])
        alt_line.set_data([s["t"] for s in part], [s["altitude"] for s in part])
        speed_line.set_data([s["t"] for s in part], [s["airspeed"] for s in part])
        s = part[-1]
        place(symbol, s["x"], s["y"], s["yaw"], size)
        end = "" if k < len(samples) - 1 else (f"  ->  ended: {reason}" if reason else "  ->  flew the whole 30 s")
        title.set_text(f"After {steps:,} training steps   t = {s['t']:4.1f} s   roll {s['roll']:5.1f} deg   "
                       f"sideslip {s['beta']:4.1f} deg{end}")
        return track, alt_line, speed_line, symbol, title

    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)


def comparison_video(env, model_file, seed, path):
    from stable_baselines3 import SAC

    from fixed_wing.airplane.asymmetric_sac import SB3Policy

    pilots = {"PID": (PIDPolicy(), "tab:blue"), "RL": (SB3Policy(SAC.load(model_file, device="cpu")), "tab:orange")}
    flights = {name: fly(env, policy, seed, sample=0.2) for name, (policy, _) in pilots.items()}
    setpoint = env.setpoint
    samples = [f for f, _ in flights.values()]
    fig, top, altitude, airspeed = layout(setpoint, extent(samples))
    limits(altitude, samples, "altitude", setpoint.altitude)
    limits(airspeed, samples, "airspeed", setpoint.airspeed)
    size = 0.03 * np.diff(top.get_xlim())[0]
    artists = {}
    for name, (_, color) in pilots.items():
        artists[name] = (top.plot([], [], color=color, label=name)[0], altitude.plot([], [], color=color)[0],
                         airspeed.plot([], [], color=color)[0], airplane(top, color))
    top.legend(loc="best", fontsize=8)
    title = fig.suptitle("")
    n = max(len(f) for f in samples)
    frames = list(range(n)) + [n - 1] * FPS

    def draw(k):
        text = []
        for name, (track, alt_line, speed_line, symbol) in artists.items():
            part = flights[name][0][:k + 1]
            track.set_data([s["x"] for s in part], [s["y"] for s in part])
            alt_line.set_data([s["t"] for s in part], [s["altitude"] for s in part])
            speed_line.set_data([s["t"] for s in part], [s["airspeed"] for s in part])
            s = part[-1]
            place(symbol, s["x"], s["y"], s["yaw"], size)
            text.append(f"{name}: roll {s['roll']:5.1f} deg, sideslip {s['beta']:4.1f} deg")
        t = max(flights[name][0][min(k, len(flights[name][0]) - 1)]["t"] for name in flights)
        title.set_text(f"t = {t:4.1f} s     " + "     ".join(text))
        return title,

    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="directory written by examples/train_sac.py")
    parser.add_argument("--seed", type=int, default=1, help="test scenario")
    parser.add_argument("--out", type=Path, default=Path("runs/videos"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = FixedWingEnv(EnvConfig(privileged_observation=True))
    checkpoints = sorted((int(re.search(r"(\d+)_steps", p.name).group(1)), p)
                         for p in (args.run / "checkpoints").glob("*_steps.zip"))
    name = args.run.name
    learning_video(env, checkpoints, args.seed, args.out / f"{name}_learning_scenario{args.seed}.gif")
    print("saved learning video")
    comparison_video(env, args.run / "model.zip", args.seed, args.out / f"{name}_rl_vs_pid_scenario{args.seed}.gif")
    print("saved comparison video")


if __name__ == "__main__":
    main()
