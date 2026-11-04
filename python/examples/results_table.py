"""Print the results table of the top-level README from the committed evaluation files.

python examples/results_table.py   (from python/, no simulation: reads results/ and ../cpp/results/)

SAC rows: mean and sample standard deviation over the seeds. Every pilot flies the same 100 test scenarios.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON_RESULTS = HERE.parent / "results" / "sac_gated_curriculum"
CPP_RESULTS = HERE.parent.parent / "cpp" / "results" / "sweep"

COLUMNS = [  # (header, key, scale, format)
    ("Complete flights / 100", "success_rate", 100, "{:.0f}"),
    ("Mean return (max 1500)", "mean_return", 1, "{:.0f}"),
    ("Airspeed RMS (m/s)", "airspeed_rmse", 1, "{:.2f}"),
    ("Altitude RMS (m)", "altitude_rmse", 1, "{:.2f}"),
    ("Course RMS (deg)", "course_rmse_deg", 1, "{:.1f}"),
]


def load(paths):
    return [json.loads(Path(p).read_text()) for p in paths]


def cell(runs, key, scale, fmt):
    values = [scale * r[key] for r in runs]
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.mean(values))} ± {fmt.format(statistics.stdev(values))}"


ROWS = [  # (pilot, condition, evaluation files)
    ("PID", "calm", [PYTHON_RESULTS / "pid" / "evaluation_calm.json"]),
    ("SAC, Python (3 seeds)", "calm", sorted(PYTHON_RESULTS.glob("calm_seed*/evaluation.json"))),
    ("SAC, C++ critic 256 (3 seeds)", "calm", sorted(CPP_RESULTS.glob("c256_b256_seed*/evaluation.json"))),
    ("SAC, C++ critic 64 (3 seeds)", "calm", sorted(CPP_RESULTS.glob("c64_b64_seed*/evaluation.json"))),
    ("PID", "windy", [PYTHON_RESULTS / "pid" / "evaluation_windy.json"]),
    ("SAC, Python (3 seeds)", "windy", sorted(PYTHON_RESULTS.glob("windy_seed*/evaluation.json"))),
]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("| Pilot | Air | " + " | ".join(c[0] for c in COLUMNS) + " |")
    print("| --- | --- |" + " --- |" * len(COLUMNS))
    for pilot, condition, paths in ROWS:
        runs = load(paths)
        print(f"| {pilot} | {condition} | " + " | ".join(cell(runs, *c[1:]) for c in COLUMNS) + " |")
