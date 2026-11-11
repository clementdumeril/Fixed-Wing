"""3D animation of recorded tracking flights (the flights/*.csv of evaluate_reference, or flights*.csv of train_sac).

python tools/render_flight_video.py <flights.csv> <out.gif|out.mp4> [--flight F] [--title TEXT] [--speed X] [--fps N] [--dpi N]

Left: the airplane (drawn 6 times larger than its span so that its attitude reads) and its trail, the point and its
whole path, at true scale on all three axes. Right: the distance to the point and the airspeed against time. The
airplane is placed from x, y, altitude and oriented from roll, pitch and heading. --speed: playback
speed (default 2, real time 1); --fps: frames per second (default 20). A .gif needs only Pillow; .mp4 needs ffmpeg (on the PATH, or
pip install imageio-ffmpeg).
"""
import csv
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PLANE, POINT = "#2a78d6", "#0b0b0b"
SPAN = 2.0  # m, drawn at SCALE times
SCALE = 8.0  # at least; larger on wide paths, a sixth of the half width of the view
REASONS = {"1": "diverged", "2": "too low", "3": "angle of attack limit", "4": "airspeed limit",
           "5": "altitude limit", "6": "too far from the point"}  # evaluate_reference.cpp, names of termination()


def body_polygons(scale):
    """Fuselage, wing, tailplane and fin in the body frame (forward, left, up), meters before scaling."""
    s = SPAN / 2
    wing = [(0.15, -s, 0), (0.15, s, 0), (-0.25, s * 0.9, 0), (-0.25, -s * 0.9, 0)]
    tail = [(-0.95, -0.4, 0), (-0.95, 0.4, 0), (-1.15, 0.4, 0), (-1.15, -0.4, 0)]
    fin = [(-0.9, 0, 0), (-1.15, 0, 0), (-1.15, 0, 0.4), (-1.0, 0, 0.4)]
    fuselage = [(0.6, 0, -0.05), (0.6, 0, 0.05), (-1.15, 0, 0.03), (-1.15, 0, -0.03)]
    fuselage_top = [(0.6, -0.06, 0), (0.6, 0.06, 0), (-1.15, 0.03, 0), (-1.15, -0.03, 0)]
    return [np.array(p) * scale for p in (wing, tail, fin, fuselage, fuselage_top)]


def rotation(roll, pitch, yaw):
    """Body (forward, left, up) to world (x, y, up), as the recorded angles are read from the state's rotation matrix
    (evaluate_reference.cpp: roll atan2(R21, R22), pitch asin(R20), heading atan2(R10, R00))."""
    def rx(a): return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    def ry(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    def rz(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return rz(yaw) @ ry(-pitch) @ rx(roll)


def load(path, flight):
    rows = [r for r in csv.DictReader(open(path)) if r["flight"] == str(flight)]
    f = lambda k: np.array([float(r[k]) for r in rows])
    return {k: f(k) for k in ("t", "x", "y", "altitude", "roll_deg", "pitch_deg", "heading_deg", "airspeed",
                                "reference_x", "reference_y", "reference_altitude")}, rows[-1]["termination"]


def main():
    args = sys.argv[1:]
    opts = {"--flight": "0", "--title": "", "--speed": "2", "--fps": "20", "--dpi": "100"}
    for k in list(opts):
        if k in args:
            i = args.index(k)
            opts[k] = args[i + 1]
            del args[i:i + 2]
    src, out = args
    d, termination = load(src, opts["--flight"])
    plane = np.stack([d["x"], d["y"], d["altitude"]], 1)  # the simulator's frame, right-handed, z up
    point = np.stack([d["reference_x"], d["reference_y"], d["reference_altitude"]], 1)
    error = np.linalg.norm(plane - point, axis=1)
    dt = d["t"][1] - d["t"][0]
    fps, speed = int(opts["--fps"]), float(opts["--speed"])
    stride = max(1, round(speed / (fps * dt)))
    frames = list(range(0, len(d["t"]), stride))
    if frames[-1] != len(d["t"]) - 1:
        frames.append(len(d["t"]) - 1)
    frames += [frames[-1]] * fps  # hold the last frame one second

    both = np.concatenate([plane, point])
    low, high = both.min(0), both.max(0)
    centre, half = (low + high) / 2, max((high - low).max() / 2, 15) + 8
    zhalf = max((high[2] - low[2]) / 2 + 8, 0.35 * half)

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(9.6, 5.4), dpi=int(opts["--dpi"]))
    ax = fig.add_axes([-0.03, 0.0, 0.66, 0.88], projection="3d")
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - zhalf, centre[2] + zhalf)
    ax.set_box_aspect((1, 1, zhalf / half), zoom=1.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("altitude [m]", labelpad=6)
    ax.view_init(elev=28, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = GRID
    ax.plot(*point.T, color=POINT, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.45)
    trail, = ax.plot([], [], [], color=PLANE, linewidth=1.4)
    shadow, = ax.plot([], [], [], color=PLANE, linewidth=0.8, alpha=0.25)
    dot, = ax.plot([], [], [], "o", color=POINT, markersize=5)
    link, = ax.plot([], [], [], color="#eb6834", linewidth=1.0)
    scale = max(SCALE, half / 6)
    body = body_polygons(scale)
    mesh = Poly3DCollection([], facecolor=PLANE, edgecolor="#0b2e5c", linewidth=0.4, alpha=0.95)
    ax.add_collection3d(mesh)
    floor = centre[2] - zhalf

    e_ax = fig.add_axes([0.75, 0.56, 0.235, 0.3])
    e_ax.plot(d["t"], error, color="#eb6834", linewidth=1.2)
    e_ax.set_ylabel("distance to point [m]")
    e_ax.set_xlim(0, max(d["t"][-1], 1))
    e_ax.set_ylim(0, min(error.max() * 1.1 + 0.5, 60))
    s_ax = fig.add_axes([0.75, 0.14, 0.235, 0.3], sharex=e_ax)
    s_ax.plot(d["t"], d["airspeed"], color=PLANE, linewidth=1.2)
    s_ax.axhline(7, color=MUTED, linewidth=0.8, linestyle=":")
    s_ax.set_ylabel("airspeed [m/s]")
    s_ax.set_xlabel("time [s]")
    for a in (e_ax, s_ax):
        a.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    cursors = [a.axvline(0, color=MUTED, linewidth=0.8) for a in (e_ax, s_ax)]
    fig.text(0.02, 0.95, opts["--title"], fontsize=12)
    info = fig.text(0.02, 0.9, "", fontsize=10, color=MUTED)
    second_half = d["t"] >= d["t"][-1] / 2
    rmse = math.sqrt((error[second_half] ** 2).mean())
    end = REASONS.get(termination, "ends: " + termination)
    summary = (f"flight ended at {d['t'][-1]:.1f} s: {end}" if termination != "0" else
               f"completed, RMSE second half {rmse:.1f} m")
    fig.text(0.985, 0.92, summary, fontsize=10, color=MUTED, ha="right")

    def draw(i):
        trail.set_data_3d(plane[:i + 1, 0], plane[:i + 1, 1], plane[:i + 1, 2])
        shadow.set_data_3d(plane[:i + 1, 0], plane[:i + 1, 1], np.full(i + 1, floor))
        dot.set_data_3d([point[i, 0]], [point[i, 1]], [point[i, 2]])
        link.set_data_3d([plane[i, 0], point[i, 0]], [plane[i, 1], point[i, 1]], [plane[i, 2], point[i, 2]])
        R = rotation(math.radians(d["roll_deg"][i]), math.radians(d["pitch_deg"][i]),
                     math.radians(d["heading_deg"][i]))
        polys = []
        for p in body:
            polys.append(p @ R.T + plane[i])
        mesh.set_verts(polys)
        for c in cursors:
            c.set_xdata([d["t"][i]] * 2)
        info.set_text(f"t {d['t'][i]:4.1f} s   distance {error[i]:5.1f} m   bank {d['roll_deg'][i]:5.0f}°   "
                      f"airspeed {d['airspeed'][i]:4.1f} m/s   (x{speed:g} speed, airplane drawn x{scale:.0f})")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    if out.endswith(".gif"):
        writer = animation.PillowWriter(fps=fps)
    else:
        try:  # the ffmpeg of the imageio-ffmpeg package when none is on the PATH
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1800,
                                        extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out, writer=writer)
    print("wrote", out)


if __name__ == "__main__":
    main()
