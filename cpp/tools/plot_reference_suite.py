"""Summary and figures of evaluate_reference over several runs (seeds).

python tools/plot_reference_suite.py <out_dir> <suite_dir> [<suite_dir> ...] [--trajectory-run K] [--trajectory-flight F]

Each suite_dir is the out_dir of one evaluate_reference run (reference_suite.json and flights/). Writes into out_dir:
  summary.csv: per start, motion and level, the mean and standard deviation over the runs of the success rate, the
    tracked rate and the errors, and the failures summed over the runs
  per_seed/<suite_dir name>.json: the reference_suite.json of each run
  success_rmse.png: success rate and position RMSE against the difficulty of each motion, level and random starts
  trajectories.png: top-down flights (point in black, airplane coloured by its speed) on circles and figure-eights at
    an easy, a medium and a breaking level, from the run K (default: the one of median RMSE on the straight line)
  circle_limits.png: on the circles, the bank and angle of attack flown against those of a steady turn, and how the
    flights end
"""
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

G = 9.8
STARTS = {"level": ("Level start (test A)", "#2a78d6", "-"), "random": ("Random start (test B, training)", "#eb6834", "--")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# airplane speed: one hue, light to dark (kept dark enough to read on white)
SPEED_MAP = LinearSegmentedColormap.from_list("speed", ["#9cc4f0", "#2a78d6", "#0b2e5c"])
# panels of the summary figure: motion, x quantity, x label
PANELS = [("circle", "bank", "Circle"), ("helix", "bank", "Helix, climbing or descending 2 m/s"),
          ("figure_eight", "bank", "Figure-eight (peak)"), ("speed", "acceleration", "Speed oscillation 12-18 m/s"),
          ("speed_range", "min_speed", "Speed oscillation at 2 m/s$^2$")]
X_LABELS = {"bank": "equivalent bank atan(a/g) [deg]", "acceleration": "peak acceleration along the path [m/s$^2$]",
            "min_speed": "lowest speed of the point [m/s]"}
TRAJECTORY_LEVELS = [4, 7, 11]  # circle and figure-eight levels: R = 60, 25 and 6.5 m at 15 m/s


def x_of(level, quantity):
    if quantity == "bank":
        return level["equivalent_bank_deg"]
    if quantity == "min_speed":
        return level["min_speed_at_15"]
    return level["acceleration"]


def main():
    args = sys.argv[1:]
    options = {}
    for name in ("--trajectory-run", "--trajectory-flight"):
        if name in args:
            i = args.index(name)
            options[name] = int(args[i + 1])
            del args[i:i + 2]
    out = Path(args[0])
    runs = [Path(a) for a in args[1:]]
    (out / "per_seed").mkdir(parents=True, exist_ok=True)
    suites = []
    for run in runs:
        shutil.copy(run / "reference_suite.json", out / "per_seed" / (run.name + ".json"))
        suites.append(json.loads((run / "reference_suite.json").read_text())["levels"])
    keys = [(l["start"], l["motion"], i) for i, l in enumerate(suites[0])]

    # ----- summary over the runs ---------------------------------------------------------------------------------
    rows = []
    for j, (start, motion, _) in enumerate(keys):
        levels = [s[j] for s in suites]
        first = levels[0]

        def stat(name, completed_only=True):
            values = [l[name] for l in levels if l["success_rate"] > 0 or not completed_only]
            return (float(np.mean(values)), float(np.std(values))) if values else (math.nan, math.nan)

        failures = {}
        for l in levels:
            for cause, count in l["failures"].items():
                failures[cause] = failures.get(cause, 0) + count
        row = {"start": start, "motion": motion, "acceleration": first["acceleration"],
               "equivalent_bank_deg": first["equivalent_bank_deg"], "climb": first["climb"],
               "speed_amplitude": first["speed_amplitude"], "radius_at_15": first["radius_at_15"],
               "period_at_15": first["period_at_15"], "peak_speed_at_15": first["peak_speed_at_15"],
               "min_speed_at_15": first["min_speed_at_15"], "runs": len(levels), "flights_per_run": first["flights"]}
        for name in ("success_rate", "tracked_rate"):
            row[name], row[name + "_std"] = stat(name, False)
        for name in ("position_rmse", "velocity_rmse", "along_track_mean", "cross_track_inside_mean", "vertical_mean",
                     "mean_action_change", "second_half_mean_bank_deg", "second_half_max_bank_deg",
                     "second_half_max_alpha_deg"):
            row[name], row[name + "_std"] = stat(name)
        row["failures_within_5_s"] = sum(l["failures_within_5_s"] for l in levels)
        row["failures"] = " ".join(f"{k}:{v}" for k, v in sorted(failures.items()))
        rows.append(row)
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in row.items()})

    # ----- success and RMSE against the difficulty ---------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, len(PANELS), figsize=(17, 6.6), sharey="row")
    straight = {r["start"]: r for r in rows if r["motion"] == "straight"}
    for c, (motion, quantity, title) in enumerate(PANELS):
        for start, (label, color, style) in STARTS.items():
            selected = [(j, r) for j, r in enumerate(rows) if r["motion"] == motion and r["start"] == start]
            x = np.array([x_of(suites[0][j], quantity) for j, _ in selected])
            if quantity != "min_speed":  # the straight line is the zero-acceleration point
                x = np.concatenate([[0.0], x])
                selected = [(None, straight[start])] + selected
            for r_i, metric in enumerate(("success_rate", "position_rmse")):
                ax = axes[r_i, c]
                y = np.array([r[metric] for _, r in selected])
                s = np.array([r[metric + "_std"] for _, r in selected])
                if metric == "success_rate":
                    y, s = 100 * y, 100 * s
                else:  # the RMSE of a level where few flights complete is not shown
                    keep = np.array([r["success_rate"] >= 0.1 for _, r in selected])
                    y, s = np.where(keep, y, np.nan), np.where(keep, s, np.nan)
                ax.fill_between(x, y - s, y + s, color=color, alpha=0.15, linewidth=0)
                ax.plot(x, y, style, color=color, linewidth=2, marker="o", markersize=4, label=label)
                ax.axhline(100 * straight[start]["success_rate"] if metric == "success_rate" else straight[start][metric],
                           color=color, linewidth=0.8, linestyle=":")
        axes[0, c].set_title(title, color=INK, fontsize=10)
        axes[1, c].set_xlabel(X_LABELS[quantity])
        if quantity == "bank":  # the acceleration on a second axis
            top = axes[0, c].secondary_xaxis("top", functions=(lambda b: G * np.tan(np.radians(np.clip(b, 0, 85))),
                                                               lambda a: np.degrees(np.arctan(np.asarray(a) / G))))
            top.set_xticks([2, 5, 10, 20, 50])
            top.set_xlabel("acceleration [m/s$^2$]", color=MUTED)
            top.tick_params(colors=MUTED)
        for ax in axes[:, c]:
            ax.grid(True, color=GRID, linewidth=0.6)
    axes[0, 0].set_ylabel("flights completed (30 s) [%]")
    axes[1, 0].set_ylabel("position RMSE, second half [m]")
    axes[0, 0].set_ylim(0, 102)
    axes[1, 0].set_ylim(0, 30)
    axes[0, 0].legend(loc="lower left", frameon=False)
    fig.text(0.5, 0.005, f"Mean and standard deviation over {len(runs)} seeds, {rows[0]['flights_per_run']} flights per level "
             "and start. Dotted: the straight line at constant velocity. RMSE over the completed flights, not shown below "
             "10 % completed.", ha="center", color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "success_rmse.png", dpi=150)
    plt.close(fig)

    # ----- trajectories -------------------------------------------------------------------------------------------
    k = options.get("--trajectory-run")
    if k is None:
        straight_rmse = [s[0]["position_rmse"] for s in suites]  # the first entry: level start, straight line
        k = int(np.argsort(straight_rmse)[len(straight_rmse) // 2])
    flight = options.get("--trajectory-flight", 0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    norm = plt.Normalize(8, 24)
    for r_i, motion in enumerate(("circle", "figure_eight")):
        for c, index in enumerate(TRAJECTORY_LEVELS):
            ax = axes[r_i, c]
            level = [l for l in suites[k] if l["start"] == "level" and l["motion"] == motion][index]
            data = np.genfromtxt(runs[k] / "flights" / f"level_{motion}_{index}.csv", delimiter=",", names=True)
            data = data[data["flight"] == flight]
            ax.plot(data["reference_x"], data["reference_y"], color=INK, linewidth=1.5, label="point")
            points = np.stack([data["x"], data["y"]], axis=1)
            segments = np.stack([points[:-1], points[1:]], axis=1)
            lines = LineCollection(segments, cmap=SPEED_MAP, norm=norm, linewidth=2)
            lines.set_array(data["ground_speed"][1:])
            ax.add_collection(lines)
            ax.plot(*points[0], "o", color=MUTED, markersize=6)
            ax.plot(data["reference_x"][0], data["reference_y"][0], "s", color=INK, markersize=5)
            end = int(data["termination"][-1])
            if end != 0:
                ax.plot(*points[-1], "x", color="#e34948", markersize=9, markeredgewidth=2)
            names = {0: "completed", 3: "angle of attack", 4: "airspeed", 6: "300 m away"}
            if motion == "circle":
                shape = f"Circle R {level['radius_at_15']:.0f} m"
            else:
                shape = f"Eight, period {level['period_at_15']:.0f} s"
            outcome = names[0] if end == 0 else f"ended at {data['t'][-1]:.1f} s ({names.get(end, 'limit')})"
            ax.set_title(f"{shape}: {level['acceleration']:.1f} m/s$^2$, {level['equivalent_bank_deg']:.0f} deg\n"
                         f"{outcome}; seed: {100 * level['success_rate']:.0f} % completed", fontsize=9, color=INK)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
    fig.subplots_adjust(hspace=0.3, wspace=0.25, right=0.88, bottom=0.08)
    fig.colorbar(lines, cax=fig.add_axes([0.9, 0.3, 0.012, 0.4]), label="airplane ground speed [m/s]")
    fig.text(0.47, 0.02, f"Run {runs[k].name}, flight {flight} of the level starts (same start in every panel). Black: the "
             "point, black square: its start; grey dot: the airplane's start; red cross: the flight ended.\nRadius and period "
             "at 15 m/s; the acceleration is the peak lateral one for the eight.",
             ha="center", color=MUTED)
    fig.savefig(out / "trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- why the circles break: bank, angle of attack and how the flights end ----------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    circles = {start: [r for r in rows if r["motion"] == "circle" and r["start"] == start] for start in STARTS}
    for start, (label, color, style) in STARTS.items():
        x = np.array([r["equivalent_bank_deg"] for r in circles[start]])
        for ax, name in ((axes[0], "second_half_mean_bank_deg"), (axes[1], "second_half_max_alpha_deg")):
            y = np.array([r[name] for r in circles[start]])
            keep = np.array([r["success_rate"] >= 0.1 for r in circles[start]])
            ax.plot(x[keep], y[keep], style, color=color, linewidth=2, marker="o", markersize=4, label=label)
        n = np.array([r["runs"] * r["flights_per_run"] for r in circles[start]])
        def ended(cause):
            return np.array([sum(int(v) for c, v in (f.split(":") for f in r["failures"].split()) if c == cause)
                             for r in circles[start]]) / n * 100
        axes[2].plot(x, ended("alpha"), style, color=color, linewidth=2, marker="o", markersize=4,
                     label=f"{label.split(' (')[0]}: angle of attack")
        axes[2].plot(x, ended("airspeed"), style, color=color, linewidth=1, marker="^", markersize=4,
                     label=f"{label.split(' (')[0]}: airspeed")
    bank = np.linspace(0, 80, 81)
    axes[0].plot(bank, bank, color=MUTED, linewidth=1, linestyle=":", label="bank of a coordinated turn")
    for V in (12, 15, 18):
        a = G * np.tan(np.radians(bank))
        axes[1].plot(bank, [steady_alpha(V, ai) for ai in a], color=MUTED, linewidth=1, linestyle=":")
        i = np.where(np.isfinite([steady_alpha(V, ai) for ai in a]))[0][-1]
        axes[1].annotate(f"{V} m/s", (bank[i], steady_alpha(V, G * np.tan(np.radians(bank[i])))), color=MUTED,
                         fontsize=8, xytext=(3, -3), textcoords="offset points")
    axes[1].axhline(18, color="#e34948", linewidth=1)
    axes[1].annotate("limit: the flight ends at 18 deg", (2, 18.4), color=MUTED, fontsize=8)
    axes[0].set_ylabel("mean bank, second half [deg]")
    axes[1].set_ylabel("largest angle of attack, second half [deg]")
    axes[2].set_ylabel("flights ended by [%]")
    axes[0].set_title("Bank flown (completed flights)", color=INK, fontsize=10)
    axes[1].set_title("Angle of attack (completed flights); dotted: steady turn", color=INK, fontsize=10)
    axes[2].set_title("How the flights end", color=INK, fontsize=10)
    for ax in axes:
        ax.set_xlabel("equivalent bank of the circle atan(a/g) [deg]")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.legend(frameon=False, fontsize=8, loc="center left" if ax is axes[1] else "upper left")
    fig.tight_layout()
    fig.savefig(out / "circle_limits.png", dpi=150)
    plt.close(fig)
    print("wrote", out)


def steady_alpha(V, a):
    """Angle of attack [deg] of a level coordinated turn at V [m/s] and lateral acceleration a [m/s^2], from the lift
    curve of the advanced_plane model (dynamics.h, namespace advanced: attached flow blended into a flat plate past
    19.4 deg), surfaces neutral; NaN where the lift needed is above its maximum (about 1.15, near 15 deg)."""
    M, stall, CL0, CLa = 15, 0.3391428111, 0.15188, 5.015
    alpha = np.radians(np.linspace(0, 18, 1801))
    up, down = np.exp(-M * (alpha - stall)), np.exp(M * (alpha + stall))
    sigma = (1 + up + down) / ((1 + up) * (1 + down))
    CL = (1 - sigma) * (CL0 + CLa * alpha) + sigma * 2 * np.sin(alpha) ** 2 * np.cos(alpha)
    needed = 1.005 * np.hypot(G, a) / (0.5 * 1.2041 * V * V * 0.34)
    i = np.argmax(CL >= needed)
    return float(np.degrees(alpha[i])) if CL[i] >= needed else math.nan


if __name__ == "__main__":
    main()
