"""Port of rl_tools/rl/environments/l2f/multirotor.h (ParametersBase and StateBase + StateRotors)

World frame: z up. Body frame: x forward, y left, z up.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Dynamics:
    rotor_positions: np.ndarray  # (N, 3), body frame
    rotor_thrust_directions: np.ndarray  # (N, 3)
    rotor_torque_directions: np.ndarray  # (N, 3)
    rotor_thrust_coefficients: np.ndarray  # (N, 3), thrust = c0 + c1 * rpm + c2 * rpm^2
    rotor_torque_constants: np.ndarray  # (N,)
    rotor_time_constants_rising: np.ndarray  # (N,)
    rotor_time_constants_falling: np.ndarray  # (N,)
    mass: float
    gravity: np.ndarray  # (3,), world frame
    J: np.ndarray
    J_inv: np.ndarray
    action_limit: tuple  # (min, max)


@dataclass
class Integration:
    dt: float


@dataclass
class Parameters:
    dynamics: Dynamics
    integration: Integration


@dataclass
class State:
    position: np.ndarray  # world frame
    orientation: np.ndarray  # quaternion [w, x, y, z], body to world
    linear_velocity: np.ndarray  # world frame
    angular_velocity: np.ndarray  # body frame
    rpm: np.ndarray  # (N,), same unit as the action limit
