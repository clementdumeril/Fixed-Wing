"""Straight-line tracking error on the advanced_plane after each change of the training, from the committed results.

python tools/plot_error_reduction.py <out.png>

Run from cpp/. Each bar: mean over 10 seeds of position_rmse in <run>/seed*/evaluation_best.json (test A, best
checkpoint), dots: the seeds. Grey: L2F reward and recipe as published; blue: changes of this repository.
"""
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEPS = [  # label, folder, changed by this repository
    ("L2F reward\nK = 0.1 /s", "results/l2f_reward_px4plane/savio_sac_c128_l2freward_4M", False),
    ("velocity gain\nK = 0.3 /s", "results/precision_px4plane/guidance03", True),
    ("+ fine position input\n+ closeness bonus", "results/fine_px4plane/fine2m_bonus_adaptive", True),
    ("+ precise\ncheckpoint choice", "results/fine_px4plane/second_run/fine2m_bonus1", True),
    ("bonus\nweight 2", "results/fine_px4plane/second_run/fine2m_bonus2", True),
    ("bonus\nweight 4", "results/fine_px4plane/third_run/fine2m_bonus4", True),
    ("8M steps\ninstead of 4M", "results/speed_px4plane/c128_8M", True),
]
INK, MUTED, GRID, BLUE, GREY = "#1a1a1a", "#5a5955", "#ebeae6", "#2a78d6", "#b9b7b0"


def main():
    out = sys.argv[1]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": MUTED,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (label, folder, ours) in enumerate(STEPS):
        runs = sorted(glob.glob(folder + "/seed*"))
        e = np.array([json.load(open(r + "/evaluation_best.json"))["position_rmse"] for r in runs])
        done = np.mean([json.load(open(r + "/evaluation_best.json"))["success_rate"] for r in runs]) * 100
        ax.bar(i, e.mean(), width=0.6, color=BLUE if ours else GREY, zorder=2)
        ax.scatter(i + np.linspace(-0.17, 0.17, len(e)), e, s=9, color=INK, alpha=0.5, linewidth=0, zorder=3)
        ax.text(i, e.max() + 0.35, f"{e.mean():.1f} ± {e.std():.1f} m\n{done:.0f} % done", ha="center", fontsize=9)
    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([s[0] for s in STEPS], fontsize=9)
    ax.set_ylabel("position RMSE, test A [m]")
    ax.set_ylim(0, 16)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Following a point flying straight, PX4 advanced_plane (simulation): each change adds to the previous",
                 fontsize=10, loc="left")
    fig.text(0.01, 0.01, "Bar: mean over 10 seeds, best checkpoint; dots: seeds; ± is the std over seeds. "
             "100 test flights per seed, RMSE over the last 15 s of 30 s.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
