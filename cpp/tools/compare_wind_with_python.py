"""python tools/compare_wind_with_python.py <path to the check_wind executable>"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from fixed_wing.airplane import operations
from fixed_wing.airplane.airplane import Disturbances, State, Thermals, Turbulence
from fixed_wing.airplane.parameters import aerosonde, dryden_low_altitude_moderate
from fixed_wing.quaternion_helper import rotate_vector_by_quaternion

fmt = "{:.17g}".format


class RecordingRng:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.draws = []

    def standard_normal(self, size=None):
        value = self.rng.standard_normal(size)
        self.draws.extend(np.atleast_1d(value).tolist())
        return value


def random_unit_quaternion(rng):
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def wind_case(rng, k):
    wind = np.zeros(3)
    speed, direction = rng.uniform(0, 6), rng.uniform(-np.pi, np.pi)
    wind[:2] = speed * np.cos(direction), speed * np.sin(direction)
    sigma = rng.uniform(0, 1) * dryden_low_altitude_moderate().sigma
    length_scale = dryden_low_altitude_moderate().length_scale
    gusts = []
    for _ in range(rng.integers(0, 5)):
        vertical, azimuth = rng.uniform(-1, 1), rng.uniform(-np.pi, np.pi)
        peak = rng.uniform(0, 3) * np.array([np.sqrt(1 - vertical ** 2) * np.cos(azimuth),
                                             np.sqrt(1 - vertical ** 2) * np.sin(azimuth), vertical])
        gusts.append((rng.uniform(1, 25), rng.uniform(1, 3), peak))
    size = rng.uniform(600, 3500)
    thermals = operations.sample_thermals(rng, size, size, rng.uniform(40, 140)) if k % 4 != 3 else Thermals()
    n = len(thermals.centers)
    thermals.w_gain = rng.uniform(0.3, 2.0, n)
    thermals.r_gain = rng.uniform(0.5, 1.5, n)
    thermals.sink = bool(rng.integers(0, 2))
    thermals.downdraft_ring = [0.0, 1.0, rng.uniform(0, 1)][k % 3]
    if k % 5 == 4:
        thermals.z_i = rng.uniform(800, 2000)
        thermals.w_star = rng.uniform(1, 4)
    queries = []
    for q in range(60):
        t = rng.uniform(0, 30)
        if n > 0 and q % 3 != 2:
            centre = thermals.centers[rng.integers(0, n)]
            offset = rng.uniform(0, 250) * np.array([np.cos(a := rng.uniform(-np.pi, np.pi)), np.sin(a)])
            horizontal = centre + wind[:2] * t + offset
        else:
            horizontal = rng.uniform(-size / 2, size / 2, 2)
        height = rng.uniform(-10, 160) if q % 4 != 3 else rng.uniform(160, 2200)
        if q % 10 == 0 and gusts:
            start, duration, _ = gusts[rng.integers(0, len(gusts))]
            t = start + rng.uniform(-0.2, 1.2) * duration
        queries.append((np.array([*horizontal, height]), random_unit_quaternion(rng), t, rng.standard_normal(5)))
    return wind, sigma, length_scale, gusts, thermals, queries


def expected_wind(wind, sigma, length_scale, gusts, thermals, position, orientation, t, turbulence):
    drifted = position - np.array([wind[0] * t, wind[1] * t, 0.0])
    base = Disturbances(wind=wind, thermals=thermals)
    parameters = aerosonde()
    parameters.disturbances = base
    total = operations.wind(parameters, drifted)
    for start, duration, peak in gusts:
        parameters.disturbances = Disturbances(wind=np.zeros(3), gust=peak, gust_direction=np.array([1.0, 0, 0]),
                                               gust_start=start, gust_length=duration)
        total = total + operations.wind(parameters, np.array([t, 0.0, 0.0]))
    parameters.disturbances = Disturbances(wind=np.zeros(3), turbulence=Turbulence(sigma=sigma, length_scale=length_scale))
    state = State(position=position, orientation=orientation, linear_velocity=np.zeros(3), angular_velocity=np.zeros(3),
                  actuators=np.zeros(4), turbulence=turbulence)
    total = total + rotate_vector_by_quaternion(orientation, operations.turbulence_velocity(parameters, state))
    return total, operations.thermal_updraft(thermals, drifted)


def symmetric_eigen_2x2(a, b, c):
    sm, df, tb = a + c, a - c, b + b
    adf, ab = abs(df), abs(tb)
    acmx, acmn = (a, c) if abs(a) > abs(c) else (c, a)
    if adf > ab:
        rt = adf * math.sqrt(1 + (ab / adf) * (ab / adf))
    elif adf < ab:
        rt = ab * math.sqrt(1 + (adf / ab) * (adf / ab))
    else:
        rt = ab * math.sqrt(2.0)
    if sm < 0:
        rt1, sgn1 = 0.5 * (sm - rt), -1
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    elif sm > 0:
        rt1, sgn1 = 0.5 * (sm + rt), 1
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    else:
        rt1, rt2, sgn1 = 0.5 * rt, -0.5 * rt, 1
    cs, sgn2 = (df + rt, 1) if df >= 0 else (df - rt, -1)
    if abs(cs) > ab:
        ct = -tb / cs
        sn1 = 1 / math.sqrt(1 + ct * ct)
        cs1 = ct * sn1
    elif ab == 0:
        cs1, sn1 = 1.0, 0.0
    else:
        tn = -cs / tb
        cs1 = 1 / math.sqrt(1 + tn * tn)
        sn1 = tn * cs1
    if sgn1 == sgn2:
        cs1, sn1 = -sn1, cs1
    return rt1, rt2, cs1, sn1


def noise_gain_as_cpp(covariance):
    d0, d1, e = float(covariance[0, 0]), float(covariance[1, 1]), float(covariance[1, 0])
    eps, safe_min = np.finfo(float).eps / 2, np.finfo(float).tiny
    if e == 0 or abs(e) <= math.sqrt(abs(d0)) * math.sqrt(abs(d1)) * eps or e * e <= (eps * eps * abs(d0)) * abs(d1) + safe_min:
        values, vectors = [d0, d1], np.eye(2)
    else:
        rt1, rt2, cs, sn = symmetric_eigen_2x2(d0, e, d1)
        values, vectors = [rt1, rt2], np.array([[cs, -sn], [sn, cs]])
    if values[1] < values[0]:
        values, vectors = values[::-1], vectors[:, ::-1]
    gain = vectors * np.sqrt(np.clip(values, 0, None))
    lapack_covariance = lapack_noise_gain(covariance) @ lapack_noise_gain(covariance).T
    factor_error[0] = max(factor_error[0], float(np.max(np.abs(gain @ gain.T - lapack_covariance))))
    factor_error[1] = max(factor_error[1], float(np.max(np.abs(lapack_covariance))))
    return gain


lapack_noise_gain = operations._noise_gain
factor_error = [0.0, 0.0]
operations._noise_gain = noise_gain_as_cpp


def turbulence_sequence(rng, k):
    length_scale = [dryden_low_altitude_moderate().length_scale, np.array([533.0, 533.0, 533.0]),
                    rng.uniform(20, 600, 3)][k % 3]
    parameters = aerosonde()
    parameters.disturbances = Disturbances(wind=np.zeros(3), turbulence=Turbulence(sigma=np.ones(3), length_scale=length_scale))
    recorder = RecordingRng(100 + k)
    state = operations.sample_turbulence(recorder)
    initial = state.copy()
    steps = []
    for j in range(300):
        distance = [0.0, rng.uniform(0, 0.5), rng.uniform(0, 30), rng.uniform(100, 3000)][0 if j == 5 else (1 if j % 7 else (2 if j % 2 else 3))]
        recorder.draws = []
        state = operations.advance_turbulence(parameters, state, distance, recorder)
        steps.append((distance, list(recorder.draws), state.copy()))
    return length_scale, initial, steps


def main():
    executable = sys.argv[1]
    rng = np.random.default_rng(2026)
    wind_cases = [wind_case(rng, k) for k in range(24)]
    sequences = [turbulence_sequence(rng, k) for k in range(9)]
    work = Path(executable).resolve().parent / "compare_wind"
    work.mkdir(exist_ok=True)
    cases_path, out_path = work / "cases.txt", work / "out.txt"
    with open(cases_path, "w") as f:
        f.write(f"{len(wind_cases)}\n")
        for wind, sigma, length_scale, gusts, thermals, queries in wind_cases:
            f.write(" ".join(map(fmt, [*wind, *sigma, *length_scale])) + f"\n{len(gusts)}\n")
            for start, duration, peak in gusts:
                f.write(" ".join(map(fmt, [start, duration, *peak])) + "\n")
            n = len(thermals.centers)
            f.write(f"{n} {fmt(thermals.w_star)} {fmt(thermals.z_i)} {fmt(thermals.area)} {int(thermals.sink)} {fmt(thermals.downdraft_ring)}\n")
            f.write(" ".join(map(fmt, np.ravel(thermals.centers))) + "\n")
            f.write(" ".join(map(fmt, thermals.w_gain)) + "\n" + " ".join(map(fmt, thermals.r_gain)) + "\n")
            f.write(f"{len(queries)}\n")
            for position, orientation, t, turbulence in queries:
                f.write(" ".join(map(fmt, [*position, *orientation, t, *turbulence])) + "\n")
        f.write(f"{len(sequences)}\n")
        for length_scale, initial, steps in sequences:
            f.write(" ".join(map(fmt, [*length_scale, *initial])) + f"\n{len(steps)}\n")
            for distance, draws, _ in steps:
                f.write(" ".join(map(fmt, [distance, *draws])) + "\n")
    subprocess.run([executable, str(cases_path), str(out_path)], check=True)
    lines = out_path.read_text().split("\n")

    worst, scale, counts = {}, {}, {}
    line = 0
    for wind, sigma, length_scale, gusts, thermals, queries in wind_cases:
        for position, orientation, t, turbulence in queries:
            cpp = np.array(lines[line].split(), dtype=float)
            line += 1
            total, updraft = expected_wind(wind, sigma, length_scale, gusts, thermals, position, orientation, t, turbulence)
            for name, a, b in (("local wind", cpp[:3], total), ("thermal updraft", cpp[3:4], [updraft])):
                worst[name] = max(worst.get(name, 0.0), float(np.max(np.abs(a - b))))
                scale[name] = max(scale.get(name, 0.0), float(np.max(np.abs(b))))
                counts[name] = counts.get(name, 0) + 1
    for length_scale, initial, steps in sequences:
        for distance, draws, state in steps:
            cpp = np.array(lines[line].split(), dtype=float)
            line += 1
            worst["Dryden states"] = max(worst.get("Dryden states", 0.0), float(np.max(np.abs(cpp - state))))
            scale["Dryden states"] = max(scale.get("Dryden states", 0.0), float(np.max(np.abs(state))))
            counts["Dryden states"] = counts.get("Dryden states", 0) + 1
    worst["Dryden noise gain"] = factor_error[0]
    scale["Dryden noise gain"] = factor_error[1]
    counts["Dryden noise gain"] = 8 * sum(len(steps) for _, _, steps in sequences)
    ok = True
    for name, error in worst.items():
        ok &= error <= 1e-9 * max(1.0, scale[name])
        print(f"{name:16s} {counts[name]:5d} values, max |C++ - Python| = {error:.2e} (largest value {scale[name]:.3g})")
    updrafts = [expected_wind(*case[:5], *query)[1] for case in wind_cases for query in case[5]]
    print(f"updrafts checked from {min(updrafts):.3f} to {max(updrafts):.3f} m/s, "
          f"{sum(u < -1e-6 for u in updrafts)} sinking, {sum(u > 1e-6 for u in updrafts)} rising")
    in_gust = sum(any(0 <= (query[2] - start) / duration < 1 for start, duration, _ in case[3]) for case in wind_cases for query in case[5])
    turbulent = sum(np.any(case[1] > 0) for case in wind_cases)
    print(f"{in_gust} queries inside a gust, {turbulent} of {len(wind_cases)} wind cases with turbulence")
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
