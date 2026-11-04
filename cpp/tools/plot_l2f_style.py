"""Training curves and test flights in the format of Learning to Fly in Seconds (Eschmann et al., RA-L 2024).

python tools/plot_l2f_style.py <runs> <out_dir> [--compare name=<runs> ...] [--flights <flights.csv>]
    [--sitl <sitl flight.csv>]

<runs>: a glob of run directories (one per seed) with progress.csv and checkpoints.csv, e.g.
"results/long_px4plane/seed*". As in the paper (Fig. 3 and 4): the episode length and the episode return against the
steps, with the wall-clock time under the step axis, mean and standard deviation over the seeds, the maximum episode
length dashed and the worst runs (smallest area under the curve) in grey. Also the validation flights of the kept
checkpoints (success rate and position error), and, with --flights / --sitl, test flights in the frame of the moving point, colored by
time, the point as a black dot (Fig. 5 of the paper, in the frame of the point).
"""
import argparse
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEAL = "#7DB9B6"  # the color of the L2F curves
DARK = "#3C3C3C"
MAX_EPISODE = 1500  # 30 s at 50 Hz
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def progress(runs, key):
    """steps, values (seeds x steps) and the mean wall-clock time at each step"""
    rows = [read(Path(d) / "progress.csv") for d in runs]
    n = min(len(r) for r in rows)
    steps = np.array([float(r["time/total_timesteps"]) for r in rows[0][:n]])
    values = np.array([[float(r[key]) for r in rr[:n]] for rr in rows])
    seconds = np.array([[float(r["time/time_elapsed"]) for r in rr[:n]] for rr in rows]).mean(0)
    return steps, values, seconds


def checkpoints(runs):
    rows = [read(Path(d) / "checkpoints.csv") for d in runs]
    n = min(len(r) for r in rows)
    steps = np.array([float(r["step"]) for r in rows[0][:n]])
    success = np.array([[100 * float(r["success_rate"]) for r in rr[:n]] for rr in rows])
    position = np.array([[float(r["position_rmse"]) for r in rr[:n]] for rr in rows])
    return steps, success, position


def time_axis(ax, steps, seconds):
    """the wall-clock time under the step ticks, as in the paper"""
    ticks = [t for t in ax.get_xticks() if steps[0] <= t <= steps[-1] or t == 0]
    minutes = np.interp(ticks, np.r_[0, steps], np.r_[0, seconds]) / 60
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / 1e6:.1f}M\n{m:.0f} min" if t >= 1e6 else f"{t / 1e3:.0f}k\n{m:.0f} min"
                        for t, m in zip(ticks, minutes)])
    ax.set_xlabel("Step\nWall-clock time, one Savio core")


def band(ax, steps, values, label="Mean & standard deviation", color=TEAL):
    m, s = values.mean(0), values.std(0)
    ax.fill_between(steps, m - s, np.minimum(m + s, 100) if values.max() <= 100 else m + s, color=color, alpha=0.3, lw=0)
    ax.plot(steps, m, color=color, lw=2.5, label=label)


def worst(ax, steps, values, k):
    area = values.sum(1)
    for i, j in enumerate(np.argsort(area)[:k]):
        ax.plot(steps, values[j], color="#B0B0B0", lw=0.9, label=f"Worst {k} runs" if i == 0 else None, zorder=1)


def episode_length(runs, out, title, k_worst, xmax=None):
    steps, values, seconds = progress(runs, "rollout/ep_len_mean")
    keep = steps <= (xmax or steps[-1])
    steps, values, seconds = steps[keep], values[:, keep], seconds[keep]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(MAX_EPISODE, color=DARK, ls="--", lw=1.2, label="Max episode length (30 s)")
    worst(ax, steps, values, k_worst)
    band(ax, steps, values)
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, MAX_EPISODE * 1.06)
    ax.set_ylabel("Episode length [steps of 20 ms]")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    time_axis(ax, steps, seconds)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def episode_return(runs, out, title):
    steps, values, seconds = progress(runs, "rollout/ep_rew_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    band(ax, steps, values)
    ax.set_xlim(0, steps[-1])
    ax.set_ylabel("Episode return")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    time_axis(ax, steps, seconds)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def validation(runs, out, title):
    steps, success, position = checkpoints(runs)
    _, _, seconds = progress(runs, "rollout/ep_len_mean")
    p_steps, _, _ = progress(runs, "rollout/ep_len_mean")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, v, label in ((a, success, "Validation flights completed [%]"), (b, position, "Position RMSE [m]")):
        band(ax, steps, np.clip(v, 0, 100) if v is success else v)
        ax.set_xlim(0, steps[-1])
        ax.set_ylabel(label)
        time_axis(ax, p_steps, seconds)
    a.set_ylim(0, 102)
    b.set_ylim(0, None)
    a.legend(loc="lower right", frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def compare(sets, key, out, ylabel, title, xmax):
    fig, ax = plt.subplots(figsize=(7, 4))
    if key == "rollout/ep_len_mean":
        ax.axhline(MAX_EPISODE, color=DARK, ls="--", lw=1.2, label="Max episode length")
    for (name, runs), color in zip(sets, (TEAL, "#E8A33D", "#4C6EB1", "#9B59B6")):
        steps, values, _ = progress(runs, key)
        keep = steps <= xmax
        band(ax, steps[keep], values[:, keep], label=f"{name} ({len(runs)} seeds)", color=color)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Step")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f"{t / 1e3:.0f}k"))
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def colored_path(ax, x, y, c, vmax):
    from matplotlib.collections import LineCollection
    points = np.c_[x, y].reshape(-1, 1, 2)
    lines = LineCollection(np.concatenate([points[:-1], points[1:]], 1), cmap="inferno", lw=2.0)
    lines.set_array(c[:-1])
    lines.set_clim(0, vmax)
    ax.add_collection(lines)
    return lines


def relative(t, x, y, z, rx, ry, rz):
    """the airplane seen from the point: along its velocity (ahead positive), to its right, above"""
    vx, vy = np.gradient(rx, t), np.gradient(ry, t)
    n = np.hypot(vx, vy)
    ux, uy = vx / n, vy / n
    dx, dy = x - rx, y - ry
    return dx * ux + dy * uy, -dx * uy + dy * ux, z - rz


def rmse_second_half(along, side, up):
    e = np.sqrt(along ** 2 + side ** 2 + up ** 2)
    return np.sqrt((e[len(e) // 2:] ** 2).mean())


def relative_panel(ax, tracks, title, vmax=30):
    ax.axhline(0, color="#DDDDDD", lw=0.8, zorder=0)
    ax.axvline(0, color="#DDDDDD", lw=0.8, zorder=0)
    for t, along, side in tracks:
        lines = colored_path(ax, side, along, t, vmax)
        ax.plot(side[0], along[0], "o", color="#888888", ms=4)
    ax.plot(0, 0, "o", color="k", ms=11, label="Reference point", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_xlabel("To the right of the point [m]")
    ax.set_ylabel("Ahead of the point [m]")
    ax.set_title(title, fontsize=10)
    return lines


def flights(path, sitl_path, out, scenarios, label):
    """as Fig. 5 of the paper, but in the frame of the moving point: the airplane joins the black dot; color: time"""
    panels = 2 if sitl_path else 1
    fig, axes = plt.subplots(1, panels, figsize=(5.4 * panels, 5.2))
    axes = np.atleast_1d(axes)
    rows = read(path)
    tracks, errors = [], []
    for k in scenarios:
        f = [r for r in rows if int(r["scenario"]) == k]
        g = lambda c: np.array([float(r[c]) for r in f])
        t = g("t")
        along, side, up = relative(t, g("x"), g("y"), g("altitude"), g("reference_x"), g("reference_y"),
                                   g("reference_altitude"))
        tracks.append((t, along, side))
        errors.append(rmse_second_half(along, side, up))
    lines = relative_panel(axes[0], tracks, f"Simulation, {label}: {len(scenarios)} test flights,\n"
                                            f"{np.mean(errors):.1f} m RMSE over the 2nd half (mean)")
    axes[0].legend(loc="lower left", frameon=False)
    if sitl_path:
        rows = read(sitl_path)
        g = lambda c: np.array([float(r[c]) for r in rows])
        t = g("t")
        along, side, up = relative(t, g("x"), g("y"), g("altitude"), g("reference_x"), g("reference_y"),
                                   g("reference_altitude"))
        relative_panel(axes[1], [(t, along, side)], "PX4 SITL in Gazebo, policy S9 (400k steps):\n"
                                                    f"{rmse_second_half(along, side, up):.1f} m RMSE over the 2nd half")
    fig.colorbar(lines, ax=axes, label="time [s]", shrink=0.8)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs")
    p.add_argument("out")
    p.add_argument("--name", default="")
    p.add_argument("--compare", nargs="*", default=[], help="name=glob, curves against the steps")
    p.add_argument("--compare-xmax", type=float, default=4e5)
    p.add_argument("--flights")
    p.add_argument("--flights-label", default="best checkpoint")
    p.add_argument("--scenarios", default="0,1,2,3,4")
    p.add_argument("--sitl")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(glob.glob(a.runs))
    n = len(runs)
    k = max(1, n // 5)
    episode_length(runs, out / "episode_length.png", f"Episode length, {a.name}, {n} seeds", k)
    episode_length(runs, out / "episode_length_early.png", f"Episode length, first 400k steps, {a.name}, {n} seeds", k,
                   xmax=4e5)
    episode_return(runs, out / "episode_return.png", f"Episode return, {a.name}, {n} seeds")
    if all((Path(d) / "checkpoints.csv").exists() for d in runs):
        validation(runs, out / "validation.png", f"100 validation flights from random starts every 50k steps, {a.name}, "
                                                 f"{n} seeds")
    if a.compare:
        sets = [(s.split("=", 1)[0], sorted(glob.glob(s.split("=", 1)[1]))) for s in a.compare]
        compare(sets, "rollout/ep_len_mean", out / "compare_episode_length.png", "Episode length [steps of 20 ms]",
                "Episode length against the steps", a.compare_xmax)
        compare(sets, "train/position_error_m", out / "compare_position_error.png", "Position error in training [m]",
                "Position error against the steps", a.compare_xmax)
    if a.flights:
        flights(a.flights, a.sitl, out / "test_flights.png", [int(s) for s in a.scenarios.split(",")], a.flights_label)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
