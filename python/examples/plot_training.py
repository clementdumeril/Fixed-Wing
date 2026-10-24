"""Learning curves of one or more training runs, from their progress.csv.

python examples/plot_training.py runs/sac_123_seed0 [runs/sac_124_seed1 ...] --out training.png

Episode length is the clearest measure (L2F, p. 5): 1500 steps = the airplane flew the whole 30 s. With the curriculum
the training return is computed with weights that grow, so it is not comparable to the PID: compare evaluation.json.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(run):
    with open(Path(run) / "progress.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("rollout/ep_len_mean")]
    column = lambda name: [float(r[name]) if r.get(name) else float("nan") for r in rows]
    names = {"steps": "time/total_timesteps", "length": "rollout/ep_len_mean", "return": "rollout/ep_rew_mean",
             "course_error": "train/course_error_deg", "stage": "curriculum/stage"}  # the last two since the gated curriculum
    return {key: column(name) for key, name in names.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="training.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    (length, ret), (course, stage) = axes
    for run in args.runs:
        c = read(run)
        for ax, key in ((length, "length"), (ret, "return"), (course, "course_error"), (stage, "stage")):
            ax.plot(c["steps"], c[key], label=Path(run).name)
    length.axhline(1500, color="gray", linestyle="--", linewidth=0.8)
    course.axhline(15, color="gray", linestyle="--", linewidth=0.8)
    length.set(ylabel="episode length (steps, max 1500)", title="How long the airplane flies")
    ret.set(ylabel="training return", title="Return (with exploration and curriculum)")
    course.set(ylabel="mean |course error| (deg)", title="Course error (dashed: curriculum threshold)", yscale="log")
    stage.set(ylabel="stage", title="Reward curriculum stage (0: initial weights, 5: final)")
    for ax in axes.ravel():
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
