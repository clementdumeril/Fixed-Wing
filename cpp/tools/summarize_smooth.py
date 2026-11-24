"""python tools/summarize_smooth.py <results folder with <variant>/seed<k>/>

Markdown tables of the smoothness evaluations: mean over the seeds of each variant."""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

COLUMNS = [
    ("Success", "success_rate", 100, "{:.0f}%"),
    ("RMSE [m]", "position_rmse", 1, "{:.2f}"),
    ("Mean sideslip", "sideslip_abs_mean_deg", 1, "{:.1f}°"),
    ("Max sideslip", "sideslip_abs_max_deg", 1, "{:.0f}°"),
    ("Body rates", "rate_norm_rms_dps", 1, "{:.0f}°/s"),
    ("Command rate", "action_rate_rms", 1, "{:.3f}"),
    ("Roll std", "roll_std_deg", 1, "{:.1f}°"),
]

sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1])
variants = sorted(p for p in root.iterdir() if p.is_dir())
for file, point in (("smoothness_best.json", "ground-fixed point"), ("smoothness_best_air.json", "air-relative point")):
    for condition in ("disturbed", "calm"):
        print(f"\n{condition.capitalize()} test flights, {point}, mean of the seeds:\n")
        print("| Variant | " + " | ".join(c[0] for c in COLUMNS) + " | Failures |")
        print("| --- " * (len(COLUMNS) + 2) + "|")
        for variant in variants:
            data = [json.loads(f.read_text())[condition]["all"] for f in sorted(variant.glob(f"seed*/{file}"))]
            if not data:
                continue
            cells = []
            for _, key, scale, fmt in COLUMNS:
                values = [d[key] * scale for d in data if d.get(key) is not None]
                cells.append(fmt.format(statistics.mean(values)) if values else "-")
            failures = Counter()
            for d in data:
                failures.update(d.get("failures") or {})
            text = ", ".join(f"{k} {v}" for k, v in failures.most_common()) or "-"
            print(f"| {variant.name} | " + " | ".join(cells) + f" | {text} |")
