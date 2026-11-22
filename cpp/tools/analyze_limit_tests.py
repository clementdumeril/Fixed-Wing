"""Tables and figures of the scenario tests of evaluate_limits.

python tools/analyze_limit_tests.py <out_dir>

Reads <out_dir>/raw/summary.csv and <out_dir>/raw/flights/*.csv, writes <out_dir>/levels.csv, breaking_points.csv,
tables.md and figures/*.png.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
REASONS = {0: "none", 1: "diverged", 2: "low altitude", 3: "angle of attack", 4: "airspeed", 5: "altitude error",
           6: "distance"}
EVENT_KINDS = {"motor", "wind", "gust", "jump", "heading"}
THRESHOLD = 0.9

plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def failures(series):
    counts = series[series > 0].map(REASONS).value_counts()
    return ", ".join(f"{name} {count}" for name, count in counts.items())


def aggregate(d):
    rows = []
    for (scenario, index), g in d.groupby(["scenario", "level_index"], sort=False):
        completed = g[g.completed == 1]
        recovered = g[g.recovered == 1]
        row = {"scenario": scenario, "kind": g.kind.iloc[0], "unit": g.unit.iloc[0], "background": g.background.iloc[0],
               "level_index": index, "level": g.level.iloc[0], "flights": len(g),
               "completed": g.completed.mean(), "survived": g.survived.mean(), "recovered": g.recovered.mean(),
               "rmse_completed": completed.rmse_strict.mean() if len(completed) else np.nan,
               "rmse_tail_survived": g[g.rmse_tail >= 0].rmse_tail.median() if (g.rmse_tail >= 0).any() else np.nan,
               "max_distance_median": g.max_distance_event.median() if g.kind.iloc[0] in EVENT_KINDS else g.max_distance.median(),
               "max_distance_p90": (g.max_distance_event if g.kind.iloc[0] in EVENT_KINDS else g.max_distance).quantile(0.9),
               "recover_time_median": recovered.recover_time.median() if len(recovered) else np.nan,
               "recover_time_p90": recovered.recover_time.quantile(0.9) if len(recovered) else np.nan,
               "min_airspeed_median": g.min_airspeed.median(), "max_alpha_median": g.max_alpha.median(),
               "strict_failures": failures(g.strict_reason), "hard_failures": failures(g.hard_reason)}
        if g.kind.iloc[0] == "motor":
            row.update({"alt_lost_total_median": g.alt_lost_total.median(), "alt_lost_total_max": g.alt_lost_total.max(),
                        "alt_lost_cut_median": g.alt_lost_event.median(), "glide_airspeed": g.glide_airspeed.median(),
                        "glide_alpha": g.glide_alpha.median(), "glide_alpha_max": g.glide_alpha_max.median(),
                        "glide_airspeed_min": g.glide_airspeed_min.median(), "glide_sink": g.glide_sink.median(),
                        "glide_pitch": g.glide_pitch.median(),
                        "glide_ratio": g[g.glide_ratio > 0].glide_ratio.median() if (g.glide_ratio > 0).any() else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def breaking_points(levels):
    rows = []
    for scenario, g in levels.groupby("scenario", sort=False):
        if g.kind.iloc[0] in ("baseline", "reproduce") or len(g) < 2:
            continue
        g = g.sort_values("level_index")
        event = g.kind.iloc[0] in EVENT_KINDS
        measure = g.recovered if event else g.survived

        def first_below(values, threshold):
            below = g[values < threshold]
            return below.level.iloc[0] if len(below) else np.nan

        def last_safe(values):
            ok = []
            for level, value in zip(g.level, values):
                if value < THRESHOLD:
                    break
                ok.append(level)
            return ok[-1] if ok else np.nan

        b = first_below(g.completed, THRESHOLD)
        at = g[g.level == b]
        rows.append({"scenario": scenario, "unit": g.unit.iloc[0], "background": g.background.iloc[0],
                     "safe_up_to": last_safe(g.completed), "break_completed_below_90": b,
                     "completed_at_break": at.completed.iloc[0] if len(at) else np.nan,
                     "break_completed_below_50": first_below(g.completed, 0.5),
                     "recovery_measure": "recovered" if event else "survived",
                     "break_recovery_below_90": first_below(measure, THRESHOLD),
                     "break_recovery_below_50": first_below(measure, 0.5),
                     "failure_at_break": at.strict_failures.iloc[0] if len(at) else "",
                     "levels_tested": " ".join(f"{v:g}" for v in g.level)})
    return pd.DataFrame(rows)


def fmt(v, digits=1):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{digits}f}"


def pct(v):
    return "-" if np.isnan(v) else f"{100 * v:.0f} %"


def markdown_tables(levels, breaks):
    out = ["## Breaking points\n",
           "| scenario | unit | safe up to (completed >= 90 %) | first level < 90 % completed | completed there | first level < 50 % | recovery < 90 % | failures at the break |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in breaks.iterrows():
        out.append(f"| {r.scenario} | {r.unit} | {fmt(r.safe_up_to, 2).rstrip('0').rstrip('.')} | "
                   f"{fmt(r.break_completed_below_90, 2).rstrip('0').rstrip('.') if not np.isnan(r.break_completed_below_90) else 'not reached'} | "
                   f"{pct(r.completed_at_break)} | "
                   f"{fmt(r.break_completed_below_50, 2).rstrip('0').rstrip('.') if not np.isnan(r.break_completed_below_50) else 'not reached'} | "
                   f"{fmt(r.break_recovery_below_90, 2).rstrip('0').rstrip('.') if not np.isnan(r.break_recovery_below_90) else 'not reached'} | "
                   f"{r.failure_at_break} |")
    out.append("")
    for scenario, g in levels.groupby("scenario", sort=False):
        motor = g.kind.iloc[0] == "motor"
        out.append(f"### {scenario} ({g.background.iloc[0]}, {int(g.flights.iloc[0])} flights per level)\n")
        head = "| level [" + g.unit.iloc[0] + "] | completed | survived | recovered | RMSE completed [m] | max distance median / p90 [m] | recovery time median / p90 [s] |"
        sep = "| --- | --- | --- | --- | --- | --- | --- |"
        if motor:
            head += " altitude lost median / max [m] | glide airspeed [m/s] | glide alpha mean / max [deg] | sink [m/s] | glide ratio |"
            sep += " --- | --- | --- | --- | --- |"
        head += " failures (real rules) |"
        sep += " --- |"
        out += [head, sep]
        for _, r in g.iterrows():
            line = (f"| {r.level:g} | {pct(r.completed)} | {pct(r.survived)} | {pct(r.recovered)} | {fmt(r.rmse_completed)} | "
                    f"{fmt(r.max_distance_median)} / {fmt(r.max_distance_p90)} | {fmt(r.recover_time_median)} / {fmt(r.recover_time_p90)} |")
            if motor:
                line += (f" {fmt(r.alt_lost_total_median)} / {fmt(r.alt_lost_total_max)} | {fmt(r.glide_airspeed) if r.level > 0 else '-'} | "
                         f"{(fmt(r.glide_alpha) + ' / ' + fmt(r.glide_alpha_max)) if r.level > 0 else '-'} | "
                         f"{fmt(r.glide_sink, 2) if r.level > 0 else '-'} | {fmt(r.glide_ratio)} |")
            line += f" {r.strict_failures or '-'} |"
            out.append(line)
        out.append("")
    return "\n".join(out)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


GROUPS = [
    ("Motor cut", ["motor_cut", "motor_cut_disturbed", "motor_cut_soft_restart"], "cut duration [s]"),
    ("Wind step", ["wind_head", "wind_tail", "wind_cross"], "wind [m/s]"),
    ("Wind ramp (10 s)", ["wind_head_ramp", "wind_tail_ramp"], "wind [m/s]"),
    ("Gust (1-cos, 2 s)", ["gust_head", "gust_tail", "gust_side", "gust_down", "gust_up"], "gust peak [m/s]"),
    ("Turbulence", ["turbulence"], "sigma / moderate"),
    ("Thermals", ["thermal"], "core updraft [m/s]"),
    ("Mass", ["mass_light", "mass_heavy"], "mass factor"),
    ("Thrust", ["thrust"], "thrust factor"),
    ("Control effectiveness", ["control_weak", "control_strong"], "control factor"),
    ("Latency", ["latency"], "latency [steps of 20 ms]"),
    ("Sensor noise", ["noise"], "noise / base"),
    ("Reference jump", ["jump_ahead", "jump_behind", "jump_side", "jump_down", "jump_up"], "jump [m]"),
    ("Reference heading jump", ["heading_jump"], "heading change [deg]"),
]
COLORS = [BLUE, ORANGE, AQUA, YELLOW, VIOLET]
TRAINING = {"Motor cut": None, "Wind step": 6, "Wind ramp (10 s)": 6, "Gust (1-cos, 2 s)": 3, "Turbulence": 1,
            "Thermals": 3, "Mass": (0.9, 1.1), "Thrust": (0.85, 1.15), "Control effectiveness": (0.9, 1.1),
            "Latency": (0, 2), "Sensor noise": 0, "Reference jump": None, "Reference heading jump": None}


def overview(levels, path):
    fig, axes = plt.subplots(4, 4, figsize=(15, 13))
    axes = axes.ravel()
    for ax, (title, names, xlabel) in zip(axes, GROUPS):
        for color, name in zip(COLORS, names):
            g = levels[levels.scenario == name].sort_values("level")
            if g.empty:
                continue
            event = g.kind.iloc[0] in EVENT_KINDS
            label = name.replace("_", " ")
            ax.plot(g.level, 100 * g.completed, color=color, linewidth=2, marker="o", markersize=4, label=label + " (completed)")
            ax.plot(g.level, 100 * (g.recovered if event else g.survived), color=color, linewidth=1.2, linestyle=(0, (3, 2)),
                    label=label + (" (recovered)" if event else " (survived)"))
        band = TRAINING.get(title)
        if isinstance(band, tuple):
            ax.axvspan(*band, color=GRID, alpha=0.7, linewidth=0)
        elif band:
            ax.axvspan(0, band, color=GRID, alpha=0.7, linewidth=0)
        ax.axhline(90, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_ylim(-3, 103)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("flights [%]")
        style(ax)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    for ax in axes[len(GROUPS):]:
        ax.axis("off")
    axes[len(GROUPS)].text(0, 0.8, "solid: completed under the real termination rules\n"
                                   "dashed: recovered (event scenarios: alive and back within 12 m\n"
                                   "for the last 5 s) or survived (persistent changes: no stall,\n"
                                   "overspeed or crash, the 30 m rule ignored)\n"
                                   "grey band: range seen in training\n"
                                   "dotted: 90 %", fontsize=9, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def rmse_figure(levels, path):
    fig, axes = plt.subplots(4, 4, figsize=(15, 13))
    axes = axes.ravel()
    for ax, (title, names, xlabel) in zip(axes, GROUPS):
        for color, name in zip(COLORS, names):
            g = levels[levels.scenario == name].sort_values("level")
            g = g[g.completed > 0]
            if g.empty:
                continue
            ax.plot(g.level, g.rmse_completed, color=color, linewidth=2, marker="o", markersize=4, label=name.replace("_", " "))
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RMSE of completed flights [m]")
        ax.set_ylim(bottom=0)
        style(ax)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes[len(GROUPS):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def motor_figure(levels, d, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = {"motor_cut": "calm", "motor_cut_disturbed": "training disturbances", "motor_cut_soft_restart": "calm, 2 s soft restart"}
    for color, name in zip([BLUE, ORANGE, AQUA], labels):
        g = levels[levels.scenario == name].sort_values("level")
        g = g[g.level <= 45]
        if g.empty:
            continue
        label = labels[name]
        axes[0].plot(g.level, 100 * g.completed, color=color, linewidth=2, marker="o", label=label + ", completed")
        axes[0].plot(g.level, 100 * g.recovered, color=color, linewidth=1.2, linestyle=(0, (3, 2)), label=label + ", recovered")
        f = d[(d.scenario == name) & (d.level > 0) & (d.level <= 45)]
        axes[1].plot(g.level, g.alt_lost_total_median, color=color, linewidth=2, marker="o", label=label + ", median")
        axes[1].plot(g.level, g.alt_lost_total_max, color=color, linewidth=1.2, linestyle=(0, (3, 2)), label=label + ", worst")
        r = f[f.recovered == 1].groupby("level").recover_time
        axes[2].plot(r.median().index, r.median().values, color=color, linewidth=2, marker="o", label=label + ", median")
        axes[2].plot(r.quantile(0.9).index, r.quantile(0.9).values, color=color, linewidth=1.2, linestyle=(0, (3, 2)), label=label + ", p90")
    axes[0].axhline(90, color=MUTED, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("flights [%]")
    axes[1].set_ylabel("altitude lost after the cut [m]")
    axes[2].set_ylabel("time to recover after restart [s]")
    for ax, title in zip(axes, ["Completed and recovered", "Altitude lost", "Recovery time (recovered flights)"]):
        ax.set_xlabel("motor cut duration [s]")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        style(ax)
        ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def timeline(csv_path, flight, title, path, event_label):
    f = pd.read_csv(csv_path)
    f = f[f.flight == flight]
    panels = [("distance", "distance to point [m]", ORANGE), ("altitude", "altitude [m]", BLUE),
              ("airspeed", "airspeed [m/s]", BLUE), ("alpha_deg", "angle of attack [deg]", BLUE),
              ("pitch_deg", "pitch [deg]", BLUE), ("roll_deg", "roll [deg]", BLUE),
              ("vertical_speed", "vertical speed [m/s]", BLUE), ("throttle_applied", "throttle (applied)", BLUE)]
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True)
    on = f.event.values > 0
    spans = []
    start = None
    for t, flag in zip(f.t.values, on):
        if flag and start is None:
            start = t
        if not flag and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, f.t.values[-1]))
    for ax, (column, label, color) in zip(axes.T.ravel(), panels):
        ax.plot(f.t, f[column], color=color, linewidth=1.3)
        if column == "altitude":
            ax.plot(f.t, f.reference_altitude, color=INK, linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.6, label="point")
            ax.legend(fontsize=7, frameon=False)
        if column == "throttle_applied":
            ax.plot(f.t, (f.throttle + 1) / 2, color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)), label="commanded")
            ax.legend(fontsize=7, frameon=False)
        if column == "distance":
            ax.axhline(30, color=MUTED, linewidth=0.8, linestyle=":")
        if column == "airspeed":
            ax.axhline(7, color=MUTED, linewidth=0.8, linestyle=":")
        if column == "alpha_deg":
            ax.axhline(18, color=MUTED, linewidth=0.8, linestyle=":")
        for a, b in spans:
            ax.axvspan(a, b, color=GRID, alpha=0.8, linewidth=0)
        ax.set_ylabel(label)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle(title + (f"   (grey: {event_label})" if spans else ""), x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


WIND_SIGN = {"wind_head": 1, "wind_head_ramp": 1, "wind_tail": -1, "wind_tail_ramp": -1}


def required_airspeed(d):
    rows = []
    for name, sign in WIND_SIGN.items():
        g = d[d.scenario == name].copy()
        g["required"] = g.reference_speed + sign * g.level
        g["family"] = "ramp" if name.endswith("ramp") else "step"
        rows.append(g)
    g = pd.concat(rows)
    bins = np.arange(-6, 41, 2)
    g["bin"] = pd.cut(g.required, bins, right=False)
    table = g.groupby(["family", "bin"], observed=True).agg(flights=("completed", "size"), completed=("completed", "mean"),
                                                            survived=("survived", "mean")).reset_index()
    table["required_low"] = [b.left for b in table.bin]
    return table.drop(columns="bin")


def required_figure(table, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for color, family, label in [(BLUE, "step", "wind step"), (ORANGE, "ramp", "wind ramp over 10 s")]:
        t = table[(table.family == family) & (table.flights >= 5)]
        ax.plot(t.required_low + 1, 100 * t.completed, color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.axvspan(7, 30, color=GRID, alpha=0.6, linewidth=0)
    ax.text(18.5, 50, "airspeed band allowed\nby the termination rules\n(7 to 30 m/s)", ha="center", fontsize=8, color=MUTED)
    ax.set_xlabel("airspeed needed to hold the point = point speed + headwind - tailwind [m/s]")
    ax.set_ylabel("flights completed [%]")
    ax.set_title("Head and tail wind: success against the airspeed the task requires", loc="left", fontsize=10, color=INK)
    ax.set_ylim(-3, 103)
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def motor_by_speed(d):
    rows = []
    for name in ("motor_cut", "motor_cut_disturbed", "motor_cut_soft_restart"):
        g = d[(d.scenario == name) & (d.level <= 45)].copy()
        g["point_speed"] = pd.cut(g.reference_speed, [12, 14, 16, 18.01], right=False, labels=["12-14", "14-16", "16-18"])
        t = g.groupby(["level", "point_speed"], observed=True).agg(flights=("completed", "size"), completed=("completed", "mean"),
                                                                   recovered=("recovered", "mean")).reset_index()
        t.insert(0, "scenario", name)
        rows.append(t)
    return pd.concat(rows)


def main():
    out = sys.argv[1]
    d = pd.read_csv(os.path.join(out, "raw", "summary.csv"))
    levels = aggregate(d)
    levels.to_csv(os.path.join(out, "levels.csv"), index=False)
    breaks = breaking_points(levels)
    breaks.to_csv(os.path.join(out, "breaking_points.csv"), index=False)
    with open(os.path.join(out, "tables.md"), "w", encoding="utf-8") as f:
        f.write(markdown_tables(levels, breaks))
    figures = os.path.join(out, "figures")
    os.makedirs(figures, exist_ok=True)
    overview(levels, os.path.join(figures, "success_vs_severity.png"))
    rmse_figure(levels, os.path.join(figures, "rmse_vs_severity.png"))
    motor_figure(levels, d, os.path.join(figures, "motor_cut_summary.png"))
    required = required_airspeed(d)
    required.to_csv(os.path.join(out, "wind_required_airspeed.csv"), index=False)
    required_figure(required, os.path.join(figures, "wind_required_airspeed.png"))
    by_speed = motor_by_speed(d)
    by_speed.to_csv(os.path.join(out, "motor_cut_by_point_speed.csv"), index=False)
    with open(os.path.join(out, "tables.md"), "a", encoding="utf-8") as f:
        f.write("\n## Wind: completed flights against the required airspeed\n\n| family | required airspeed [m/s] | flights | completed | survived |\n| --- | --- | --- | --- | --- |\n")
        for _, r in required.iterrows():
            f.write(f"| {r.family} | {r.required_low:g} to {r.required_low + 2:g} | {r.flights} | {pct(r.completed)} | {pct(r.survived)} |\n")
        f.write("\n## Motor cut: completed / recovered against the speed of the point\n\n| scenario | cut [s] | point speed [m/s] | flights | completed | recovered |\n| --- | --- | --- | --- | --- | --- |\n")
        for _, r in by_speed.iterrows():
            f.write(f"| {r.scenario} | {r.level:g} | {r.point_speed} | {r.flights} | {pct(r.completed)} | {pct(r.recovered)} |\n")
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--timeline":
        timeline(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], "event")
    elif len(sys.argv) == 7 and sys.argv[1] == "--timeline":
        timeline(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        main()
