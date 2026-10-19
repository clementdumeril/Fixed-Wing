"""Flight-performance figures of a parameter set, to compare the simulator with flight data.

All figures are steady, wings-level flight relative to the air mass, computed with the trim solver.
STRIX_REFERENCE lists the Strix numbers published by the lab, with their source.
"""
from dataclasses import dataclass

import numpy as np

from .airplane import THROTTLE, Parameters
from .operations import local_wind
from .trim import trim


@dataclass
class Reference:
    name: str
    value: float
    unit: str
    source: str


# Lab Strix StratoSurfer (modified: the stock airframe has a 1.5 m span). For comparison only, not used by the simulator.
STRIX_REFERENCE = [
    Reference("mass", 4.0, "kg", "Lim & Loianno 2026, arXiv 2607.15863"),
    Reference("wing span", 1.7, "m", "Lim & Loianno 2026, arXiv 2607.15863"),
    Reference("best-glide airspeed", 12.0, "m/s", "Morando et al. 2026, arXiv 2605.15619 (CFD and flight)"),
    Reference("sink rate at best glide", 1.51, "m/s", "Morando et al. 2026, arXiv 2605.15619 (CFD and flight)"),
    Reference("level-flight throttle at 12 m/s", 0.376, "-", "Morando et al. 2026, arXiv 2605.15619"),
    Reference("trim throttle (airspeed not given)", 0.60, "-", "Bobbili et al. 2026, arXiv 2608.20655 (AVL model)"),
    Reference("trim elevator (airspeed not given)", 8.0, "deg", "Bobbili et al. 2026, arXiv 2608.20655 (AVL model)"),
    Reference("climb rate at 15 m/s", 4.6, "m/s", "Lim & Loianno 2026, flight (6 m/s used in their simulation)"),
    Reference("descent rate at 15 m/s", 2.26, "m/s", "Lim & Loianno 2026, flight (3 m/s used in their simulation)"),
]


def _air_velocity(parameters: Parameters, state):
    return state.linear_velocity - local_wind(parameters, state)


def level_flight_throttle(parameters: Parameters, airspeed) -> float:
    return float(trim(parameters, airspeed).actuators[THROTTLE])


def vertical_speed(parameters: Parameters, airspeed, throttle) -> float:
    """Steady climb (> 0) or sink (< 0) rate at a fixed throttle, relative to the air mass."""
    return float(_air_velocity(parameters, trim(parameters, airspeed, throttle=throttle))[2])


def glide_ratio(parameters: Parameters, airspeed) -> float:
    v = _air_velocity(parameters, trim(parameters, airspeed, glide=True))
    return float(np.hypot(v[0], v[1]) / -v[2])


def best_glide(parameters: Parameters, airspeed_range=(6.0, 40.0), tolerance=1e-3):
    """Airspeed of maximum glide ratio (golden-section search), with its sink rate and glide ratio."""
    a, b = airspeed_range
    grid = np.linspace(a, b, 69)
    ratios = [glide_ratio(parameters, v) for v in grid]
    i = int(np.argmax(ratios))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    k = (np.sqrt(5) - 1) / 2
    while b - a > tolerance:
        c, d = b - k * (b - a), a + k * (b - a)
        if glide_ratio(parameters, c) > glide_ratio(parameters, d):
            b = d
        else:
            a = c
    airspeed = (a + b) / 2
    return airspeed, -vertical_speed(parameters, airspeed, throttle=0.0), glide_ratio(parameters, airspeed)


def calibrate_drag_polar(mass, wing_area, wing_span, best_glide_airspeed, best_glide_sink_rate, air_density=1.225, g=9.81):
    """Parabolic polar C_D = C_D_p + C_L^2 / (pi e AR) implied by a measured best glide.

    At the best glide the induced drag equals the parasitic drag, so C_D = 2 C_D_p and L/D = C_L / (2 C_D_p).
    Returns (C_D_p, oswald_efficiency). Needs the wing area, which the lab papers do not give.
    """
    gamma = np.arcsin(best_glide_sink_rate / best_glide_airspeed)
    lift_to_drag = 1 / np.tan(gamma)
    C_L = 2 * mass * g * np.cos(gamma) / (air_density * wing_area * best_glide_airspeed ** 2)
    C_D_p = C_L / (2 * lift_to_drag)
    aspect_ratio = wing_span ** 2 / wing_area
    oswald = C_L ** 2 / (np.pi * aspect_ratio * C_D_p)
    return float(C_D_p), float(oswald)
