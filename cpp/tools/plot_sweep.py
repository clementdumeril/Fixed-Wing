"""Figure and table of the network and batch size sweep (savio/sweep.slurm, savio/train.slurm).

python tools/plot_sweep.py results/sweep   (from cpp/)

Each run directory is <variant>_seed<k> with progress.csv and evaluation.json. Writes sweep.png and prints a Markdown
table (mean over seeds) with the PID and the Python SAC of python/results/sac_gated_curriculum for reference.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = {  # directory prefix: label
    "c256_b256": "critic 256, batch 256",
    "c256_b64": "critic 256, batch 64",
    "c128_b128": "critic 128, batch 128",
    "c128_b64": "critic 128, batch 64",
    "c128_b64_3M": "critic 128, batch 64, 3 M steps",
    "c64_b256": "critic 64, batch 256",
    "c64_b64": "critic 64, batch 64 (L2F)",
}


def read(run):
    with open(run / "progress.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    column = lambda name: np.array([float(r[name]) for r in rows])
    return dict(steps=column("time/total_timesteps"), course=column("train/course_error_deg"),
                stage=column("curriculum/stage"), seconds=column("time/time_elapsed"),
                evaluation=json.loads((run / "evaluation.json").read_text()))


def main():
    folder = Path(sys.argv[1])
    runs = defaultdict(list)
    for run in sorted(folder.iterdir()):
        if run.is_dir():
            runs[run.name.rsplit("_seed", 1)[0]].append(read(run))
    python = Path(__file__).resolve().parents[2] / "python" / "results" / "sac_gated_curriculum"
    pid = json.loads((python / "pid" / "evaluation_calm.json").read_text())
    sac = [json.loads((python / f"calm_seed{s}" / "evaluation.json").read_text()) for s in range(3)]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(VARIANTS)))
    for (prefix, label), color in zip(VARIANTS.items(), colors):
        if prefix not in runs:
            continue
        rs = runs[prefix]
        n = min(len(r["steps"]) for r in rs)
        left.plot(rs[0]["steps"][:n] / 1e6, np.mean([r["course"][:n] for r in rs], axis=0), color=color, label=label)
        right.scatter([r["seconds"][-1] / 60 for r in rs], [r["evaluation"]["mean_return"] for r in rs],
                      color=color, s=36, label=label)
    left.axhline(15, color="gray", linestyle="--", linewidth=0.8)
    left.set(xlabel="environment steps (M)", ylabel="mean |course error| in training (deg)", yscale="log",
             title="Course error over training (mean of 3 seeds; dashed: curriculum gate)")
    right.axhline(pid["mean_return"], color="black", linestyle="--", linewidth=1, label="PID")
    right.axhline(np.mean([s["mean_return"] for s in sac]), color="gray", linestyle=":", linewidth=1.5,
                  label="Python SAC (3.5 h on 8 cores)")
    right.set(xlabel="training time on one Savio core (min)", ylabel="mean return on the 100 test scenarios",
              title="Result against training cost (one dot per seed)", xscale="log")
    for ax in (left, right):
        ax.grid(alpha=0.3)
    right.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / "sweep.png", dpi=150)

    print("| Critic, batch | Time (1 core) | Curriculum stage | Complete flights | Mean return | Airspeed RMS | "
          "Altitude RMS | Course RMS | Command change |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for prefix, label in VARIANTS.items():
        if prefix not in runs:
            continue
        rs = runs[prefix]
        m = lambda k: np.mean([r["evaluation"][k] for r in rs])
        stages = ", ".join(str(int(r["stage"][-1])) for r in rs)
        print(f"| {label} | {np.mean([r['seconds'][-1] for r in rs]) / 60:.0f} min | {stages} | "
              f"{m('success_rate') * 100:.0f} % | {m('mean_return'):.0f} | {m('airspeed_rmse'):.2f} m/s | "
              f"{m('altitude_rmse'):.2f} m | {m('course_rmse_deg'):.1f}° | {m('mean_action_change'):.4f} |")
    ms = lambda k: np.mean([s[k] for s in sac])
    print(f"| Python SAC, critic 256, batch 256 | 3.5 h on 8 cores | 5, 5, 5 | {ms('success_rate') * 100:.0f} % | "
          f"{ms('mean_return'):.0f} | {ms('airspeed_rmse'):.2f} m/s | {ms('altitude_rmse'):.2f} m | "
          f"{ms('course_rmse_deg'):.1f}° | {ms('mean_action_change'):.4f} |")
    print(f"| PID | | | {pid['success_rate'] * 100:.0f} % | {pid['mean_return']:.0f} | {pid['airspeed_rmse']:.2f} m/s | "
          f"{pid['altitude_rmse']:.2f} m | {pid['course_rmse_deg']:.1f}° | {pid['mean_action_change']:.4f} |")


if __name__ == "__main__":
    main()
