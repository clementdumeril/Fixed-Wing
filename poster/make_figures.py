"""Figures of the poster, from the committed results (run from the repository root).

python poster/make_figures.py [--flights DIR]

  error.pdf     straight-line tracking error after each change of the training (cpp/results/*/seed*/evaluation_best.json)
  turning.pdf   success and error on circles against the bank they need (cpp/results/reference_suite*/summary.csv)
  compute.pdf   test error and run time of smaller critics and longer training (cpp/results/speed_px4plane)
  flights.pdf   top-down flights of the final policy (needs the recorded flights of evaluate_reference --record 3 for
                fine2m_bonus4 seed 4, not committed: --flights <runs>/reference_suite_fine/fine2m_bonus4_seed4/flights)
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "cpp" / "results"
OUT = Path(__file__).resolve().parent / "figures"
BLUE, GOLD, GREY, INK, LIGHT = "#003262", "#C4820E", "#9A9A9A", "#1A1A1A", "#DADADA"
plt.rcParams.update({"font.size": 28, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8A8A8A", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 1.2})


def rmse(folder):
    runs = sorted(glob.glob(str(folder / "seed*")))
    return np.array([json.load(open(Path(r) / "evaluation_best.json"))["position_rmse"] for r in runs])


def error():
    steps = [("L2F\nreward", R / "l2f_reward_px4plane" / "savio_sac_c128_l2freward_4M", GREY),
             ("pull toward\nthe point\n×3", R / "precision_px4plane" / "guidance03", GOLD),
             ("closeness\nbonus\n+ 2 m input", R / "fine_px4plane" / "fine2m_bonus_adaptive", GOLD),
             ("precise\ncheckpoint", R / "fine_px4plane" / "second_run" / "fine2m_bonus1", BLUE),
             ("bonus\nweight 2", R / "fine_px4plane" / "second_run" / "fine2m_bonus2", GOLD),
             ("bonus\nweight 4", R / "fine_px4plane" / "third_run" / "fine2m_bonus4", GOLD),
             ("8M steps", R / "speed_px4plane" / "c128_8M", BLUE)]
    fig, ax = plt.subplots(figsize=(15, 6.6))
    for i, (label, folder, color) in enumerate(steps):
        e = rmse(folder)
        ax.bar(i, e.mean(), width=0.64, color=color, zorder=2)
        ax.scatter(i + np.linspace(-0.2, 0.2, len(e)), e, s=22, color=INK, alpha=0.45, lw=0, zorder=3)
        ax.text(i, e.max() + 0.45, f"{e.mean():.1f} m", ha="center", fontsize=26, fontweight="bold", color=INK)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=21)
    ax.set_ylabel("position error, RMS (m)")
    ax.set_ylim(0, 16)
    ax.yaxis.grid(True, color="#EDEDED", zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "error.pdf")
    plt.close(fig)


def summary(path):
    rows = [r for r in csv.DictReader(open(path)) if r["start"] == "level" and r["motion"] in ("straight", "circle")]
    rows.sort(key=lambda r: float(r["equivalent_bank_deg"]))
    get = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    return get("equivalent_bank_deg"), get("success_rate") * 100, get("position_rmse"), get("position_rmse_std")


def turning():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.0))
    for path, color, label in ((R / "reference_suite" / "summary.csv", GREY, "L2F reward (9.8 m on a line)"),
                               (R / "reference_suite_fine" / "fine2m_bonus4" / "summary.csv", BLUE,
                                "with the precision changes (1.8 m on a line)")):
        bank, done, err, std = summary(path)
        axes[0].plot(bank, done, "-o", color=color, lw=3, ms=7, label=label)
        axes[1].plot(bank, err, "-o", color=color, lw=3, ms=7, label=label)
        axes[1].fill_between(bank, err - std, err + std, color=color, alpha=0.15, lw=0)
    axes[0].set_ylabel("flights completed (%)")
    axes[0].set_ylim(0, 103)
    axes[1].set_ylabel("position error, RMS (m)")
    axes[1].set_ylim(0, 28)
    for ax in axes:
        ax.set_xlabel("bank the circle needs (deg)")
        ax.axvspan(60, 80, color="#F6E7CF", zorder=0)
        ax.grid(True, color="#EDEDED")
        ax.set_xlim(-2, 80)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=25, loc="upper center", ncol=1)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(OUT / "turning.pdf")
    plt.close(fig)


def compute():
    rows = [("critic 128 (ours), 4M", "c128_8M", 16.1, None), ("critic 64, as L2F", "c64", 9.2, None),
            ("critic 32", "c32", 5.9, None), ("critic 128, 8M steps", "c128_8M", 31.9, None)]
    ref = rmse(R / "fine_px4plane" / "third_run" / "fine2m_bonus4")
    values = [ref.mean(), rmse(R / "speed_px4plane" / "c64").mean(), rmse(R / "speed_px4plane" / "c32").mean(),
              rmse(R / "speed_px4plane" / "c128_8M").mean()]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    y = np.arange(len(rows))[::-1]
    colors = [BLUE, GREY, GREY, BLUE]
    ax.barh(y, values, color=colors, height=0.62, zorder=2)
    for yi, v, (label, _, minutes, _) in zip(y, values, rows):
        ax.text(v + 0.3, yi, f"{v:.1f} m  ·  {minutes:.0f} min", va="center", fontsize=27, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=27)
    ax.set_xlabel("position error, RMS (m)")
    ax.set_xlim(0, 27)
    ax.xaxis.grid(True, color="#EDEDED", zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "compute.pdf")
    plt.close(fig)


def flights(folder):
    panels = [("straight", 0, "line"), ("circle", 4, "circle, 21°"),
              ("figure_eight", 7, "eight, 43° at most"), ("circle", 10, "circle, 69°: fails")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 7.0))
    for ax, (motion, level, title) in zip(axes, panels):
        rows = [r for r in csv.DictReader(open(Path(folder) / f"level_{motion}_{level}.csv")) if r["flight"] == "0"]
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["y"]) for r in rows])
        px = np.array([float(r["reference_x"]) for r in rows])
        py = np.array([float(r["reference_y"]) for r in rows])
        ax.plot(py, px, color=INK, lw=2.2, ls=(0, (4, 3)), label="point")
        ax.plot(y, x, color=BLUE, lw=3.2, label="airplane")
        ax.plot(y[0], x[0], "o", color=GREY, ms=12)
        if rows[-1]["termination"] != "0":
            ax.plot(y[-1], x[-1], "X", color="#C0392B", ms=18)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title, fontsize=30)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=28, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "flights.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    error()
    turning()
    compute()
    if args.flights:
        flights(args.flights)
    print("wrote", OUT)
