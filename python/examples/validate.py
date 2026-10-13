"""Compares the flight performance of a parameter set with the Strix numbers published by the lab.

Today the simulator still uses the Aerosonde placeholder, so the differences mostly show how far the placeholder is
from the Strix. Rerun it once the Strix model (AVL coefficients, mass, inertia, wing area) is available.
"""
import numpy as np

from fixed_wing.airplane.parameters import aerosonde
from fixed_wing.airplane.validation import (
    STRIX_REFERENCE,
    best_glide,
    calibrate_drag_polar,
    level_flight_throttle,
    vertical_speed,
)

parameters = aerosonde()
airspeed, sink, ratio = best_glide(parameters)
simulated = {
    "best-glide airspeed": airspeed,
    "sink rate at best glide": sink,
    "level-flight throttle at 12 m/s": level_flight_throttle(parameters, 12.0),
    "climb rate at 15 m/s": vertical_speed(parameters, 15.0, throttle=1.0),
    "descent rate at 15 m/s": -vertical_speed(parameters, 15.0, throttle=0.0),
}

print(f"{'quantity':40s} {'Strix':>8s} {'simulator':>10s}  source")
for r in STRIX_REFERENCE:
    value = simulated.get(r.name)
    shown = f"{value:10.3f}" if value is not None else f"{'-':>10s}"
    print(f"{r.name:40s} {r.value:8.3f} {shown}  {r.source}")
print(f"\nsimulated best glide ratio: {ratio:.1f} (Strix flight data: 12 / 1.51 = {12 / 1.51:.1f})")

# What the Strix glide data implies for the drag polar, for a few plausible wing areas (unknown for the modified Strix)
print("\nwing area [m^2]   C_D_p    oswald e   (from best glide 12 m/s, sink 1.51 m/s, 4 kg, 1.7 m span)")
for wing_area in [0.30, 0.35, 0.40, 0.45]:
    C_D_p, e = calibrate_drag_polar(4.0, wing_area, 1.7, 12.0, 1.51)
    print(f"{wing_area:15.2f} {C_D_p:8.4f} {e:10.2f}")
