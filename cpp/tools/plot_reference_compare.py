"""Success rate and position RMSE of several reference-suite summaries side by side (level starts).

python tools/plot_reference_compare.py <out.png> <label>=<summary.csv> [<label>=<summary.csv> ...]

Each summary.csv is written by tools/plot_reference_suite.py. Panels: circles and figure-eights against the equivalent
bank, speed oscillations at 2 m/s^2 against the lowest speed of the point; mean over the seeds, band one standard
deviation.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#8a8984", "#2a78d6", "#eb6834", "#1f9d6b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PANELS = [("circle", "equivalent_bank_deg", "Circle", "equivalent bank atan(a/g) [deg]"),
          ("figure_eight", "equivalent_bank_deg", "Figure-eight (peak)", "equivalent bank atan(a/g) [deg]"),
          ("speed_range", "min_speed_at_15", "Speed oscillation at 2 m/s$^2$", "lowest speed of the point [m/s]")]


def rows(path, motion, x):
    out = []
    for r in csv.DictReader(open(path)):
        if r["start"] == "level" and (r["motion"] == motion or (r["motion"] == "straight" and motion != "speed_range")):
            out.append((float(r[x]) if r["motion"] != "straight" else 0.0, float(r["success_rate"]) * 100,
                        float(r["success_rate_std"]) * 100, float(r["position_rmse"]), float(r["position_rmse_std"])))
    return np.array(sorted(out))


def main():
    out = sys.argv[1]
    series = [a.split("=", 1) for a in sys.argv[2:]]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    for j, (motion, x, title, xlabel) in enumerate(PANELS):
        for i, (label, path) in enumerate(series):
            d = rows(path, motion, x)
            for k, (m, s) in enumerate([(1, 2), (3, 4)]):
                a = axes[k, j]
                a.plot(d[:, 0], d[:, m], color=COLORS[i], marker="o", markersize=3.5, linewidth=1.6, label=label)
                low, high = d[:, m] - d[:, s], d[:, m] + d[:, s]
                if k == 0:
                    low, high = np.clip(low, 0, 100), np.clip(high, 0, 100)
                a.fill_between(d[:, 0], low, high, color=COLORS[i], alpha=0.12, linewidth=0)
        axes[0, j].set_title(title, fontsize=11, loc="left")
        axes[1, j].set_xlabel(xlabel)
        if motion == "speed_range":
            for k in range(2):
                axes[k, j].invert_xaxis()
        for k in range(2):
            axes[k, j].grid(True, color=GRID, linewidth=0.8)
            for side in ("top", "right"):
                axes[k, j].spines[side].set_visible(False)
    axes[0, 0].set_ylabel("flights completed [%]")
    axes[1, 0].set_ylabel("position RMSE [m]")
    axes[0, 0].set_ylim(0, 102)
    axes[1, 0].set_ylim(0, 30)
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
