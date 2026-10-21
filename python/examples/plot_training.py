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
    return column("time/total_timesteps"), column("rollout/ep_len_mean"), column("rollout/ep_rew_mean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="training.png")
    args = parser.parse_args()

    fig, (length, ret) = plt.subplots(1, 2, figsize=(11, 4))
    for run in args.runs:
        steps, lengths, returns = read(run)
        length.plot(steps, lengths, label=Path(run).name)
        ret.plot(steps, returns, label=Path(run).name)
    length.axhline(1500, color="gray", linestyle="--", linewidth=0.8)
    length.set(xlabel="environment steps", ylabel="episode length (steps, max 1500)", title="How long the airplane flies")
    ret.set(xlabel="environment steps", ylabel="training return", title="Return (with exploration and curriculum)")
    for ax in (length, ret):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
