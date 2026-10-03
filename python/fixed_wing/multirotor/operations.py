"""Port of rl_tools/rl/environments/l2f/operations_generic.h and operations_generic/*.h

The sections follow the file names of the original.
"""
import numpy as np

from ..integrators import rk4
from ..quaternion_helper import quaternion_derivative, rotate_vector_by_quaternion
from .multirotor import Parameters, State


# 20_initial_state.h
def initial_state(parameters: Parameters) -> State:
    n_rotors = len(parameters.dynamics.rotor_positions)
    return State(
        position=np.zeros(3),
        orientation=np.array([1.0, 0, 0, 0]),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        rpm=np.full(n_rotors, hovering_throttle(parameters)),
    )


def hovering_throttle(parameters: Parameters) -> float:
    d = parameters.dynamics
    c0, c1, c2 = d.rotor_thrust_coefficients[0]
    n_rotors = len(d.rotor_positions)
    target = d.mass * np.linalg.norm(d.gravity) / n_rotors
    return (-c1 + np.sqrt(c1 ** 2 - 4 * c2 * (c0 - target))) / (2 * c2)


# 60_dynamics.h
def multirotor_dynamics(parameters: Parameters, state: State, action) -> State:
    d = parameters.dynamics
    c = d.rotor_thrust_coefficients
    thrust_magnitude = c[:, 0] + c[:, 1] * state.rpm + c[:, 2] * state.rpm ** 2
    rotor_thrust = d.rotor_thrust_directions * thrust_magnitude[:, None]
    thrust = rotor_thrust.sum(axis=0)
    torque = (d.rotor_torque_directions * (thrust_magnitude * d.rotor_torque_constants)[:, None]).sum(axis=0)
    torque += np.cross(d.rotor_positions, rotor_thrust).sum(axis=0)

    tau = np.where(action >= state.rpm, d.rotor_time_constants_rising, d.rotor_time_constants_falling)
    return State(
        position=state.linear_velocity,
        orientation=quaternion_derivative(state.orientation, state.angular_velocity),
        linear_velocity=rotate_vector_by_quaternion(state.orientation, thrust) / d.mass + d.gravity,
        angular_velocity=d.J_inv @ (torque - np.cross(state.angular_velocity, d.J @ state.angular_velocity)),
        rpm=(action - state.rpm) / tau,
    )


# 70_post_integration.h
def post_integration(parameters: Parameters, next_state: State) -> State:
    next_state.orientation = next_state.orientation / np.linalg.norm(next_state.orientation)
    next_state.rpm = np.clip(next_state.rpm, *parameters.dynamics.action_limit)
    return next_state


# operations_generic.h
def scale_action(parameters: Parameters, action):
    # action in [-1, 1] -> [action_limit.min, action_limit.max]
    low, high = parameters.dynamics.action_limit
    return low + (np.clip(action, -1, 1) + 1) / 2 * (high - low)


def step(parameters: Parameters, state: State, action) -> State:
    action_scaled = scale_action(parameters, np.asarray(action, dtype=float))
    next_state = rk4(multirotor_dynamics, parameters, state, action_scaled, parameters.integration.dt)
    return post_integration(parameters, next_state)
