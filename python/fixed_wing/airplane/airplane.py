"""Fixed-wing counterpart of multirotor/multirotor.py

World frame: z up. Body frame: x forward, y left, z up (same as L2F).
The aerodynamic coefficients keep the usual aerospace convention (body x forward, y right, z down),
the conversion happens in one place: airplane_aerodynamics in operations.py.
"""
from dataclasses import dataclass

import numpy as np

# order of the actions and of State.actuators
AILERON, ELEVATOR, RUDDER, THROTTLE = range(4)


@dataclass
class Aerodynamics:
    # lift
    C_L_0: float
    C_L_alpha: float
    C_L_q: float
    C_L_delta_e: float
    # drag: C_D = C_D_p + C_L^2 / (pi * e * AR)
    C_D_p: float
    oswald_efficiency: float
    # side force
    C_Y_0: float
    C_Y_beta: float
    C_Y_p: float
    C_Y_r: float
    C_Y_delta_a: float
    C_Y_delta_r: float
    # roll moment
    C_l_0: float
    C_l_beta: float
    C_l_p: float
    C_l_r: float
    C_l_delta_a: float
    C_l_delta_r: float
    # pitch moment
    C_m_0: float
    C_m_alpha: float
    C_m_q: float
    C_m_delta_e: float
    # yaw moment
    C_n_0: float
    C_n_beta: float
    C_n_p: float
    C_n_r: float
    C_n_delta_a: float
    C_n_delta_r: float


@dataclass
class Dynamics:
    mass: float
    gravity: np.ndarray  # (3,), world frame
    J: np.ndarray  # body frame (x forward, y left, z up)
    J_inv: np.ndarray
    air_density: float
    wing_area: float
    wing_span: float
    mean_chord: float
    aerodynamics: Aerodynamics
    thrust_max: float  # thrust = throttle * thrust_max, along body x
    actuator_time_constants: np.ndarray  # (4,), [aileron, elevator, rudder, throttle]
    actuator_limits: np.ndarray  # (4, 2), [min, max], rad for the surfaces, [0, 1] for the throttle


@dataclass
class Integration:
    dt: float


@dataclass
class Disturbances:
    wind: np.ndarray  # (3,), world frame, constant over an episode


@dataclass
class Parameters:
    dynamics: Dynamics
    integration: Integration
    disturbances: Disturbances


@dataclass
class State:
    position: np.ndarray  # world frame
    orientation: np.ndarray  # quaternion [w, x, y, z], body to world
    linear_velocity: np.ndarray  # world frame, ground-relative
    angular_velocity: np.ndarray  # body frame
    actuators: np.ndarray  # (4,), actual deflections [rad] and throttle [0, 1]
