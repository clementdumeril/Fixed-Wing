"""Fixed-wing counterpart of multirotor/operations.py, same sections as the L2F original."""
import numpy as np

from ..integrators import rk4
from ..quaternion_helper import (
    quaternion_derivative,
    rotate_vector_by_quaternion,
    rotate_vector_by_quaternion_inverse,
)
from .airplane import Parameters, State
from .parameters import FRD_FLU

MIN_AIRSPEED = 1e-3  # below this the aerodynamic angles are undefined, no aerodynamic forces


# 20_initial_state.h
def initial_state(parameters: Parameters) -> State:
    return State(
        position=np.zeros(3),
        orientation=np.array([1.0, 0, 0, 0]),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        actuators=np.zeros(4),
    )


# 60_dynamics.h
def wind(parameters: Parameters, position):
    # world frame: constant wind + "1 - cos" gust where the airplane is
    # the gust depends on the position, not on the time, so the dynamics stay a function of (parameters, state, action)
    d = parameters.disturbances
    s = (d.gust_direction @ position - d.gust_start) / d.gust_length  # 0 -> 1 across the gust
    if 0 <= s < 1:
        return d.wind + d.gust * 0.5 * (1 - np.cos(2 * np.pi * s))
    return d.wind


def air_data(parameters: Parameters, state: State):
    """Air-relative velocity in the aerospace body frame (x forward, y right, z down), airspeed, alpha, beta"""
    air_relative_velocity = state.linear_velocity - wind(parameters, state.position)
    v = FRD_FLU @ rotate_vector_by_quaternion_inverse(state.orientation, air_relative_velocity)
    airspeed = np.linalg.norm(v)
    if airspeed < MIN_AIRSPEED:
        return v, airspeed, 0.0, 0.0
    alpha = np.arctan2(v[2], v[0])
    beta = np.arcsin(np.clip(v[1] / airspeed, -1, 1))
    return v, airspeed, alpha, beta


def airplane_aerodynamics(parameters: Parameters, state: State):
    """Aerodynamic + propulsion force and moment in the body frame (x forward, y left, z up)"""
    d = parameters.dynamics
    a = d.aerodynamics
    delta_a, delta_e, delta_r, delta_t = state.actuators
    force = np.array([delta_t * d.thrust_max, 0.0, 0.0])
    moment = np.zeros(3)

    _, airspeed, alpha, beta = air_data(parameters, state)
    if airspeed >= MIN_AIRSPEED:
        p, q, r = FRD_FLU @ state.angular_velocity
        p_hat = p * d.wing_span / (2 * airspeed)
        q_hat = q * d.mean_chord / (2 * airspeed)
        r_hat = r * d.wing_span / (2 * airspeed)

        C_L = a.C_L_0 + a.C_L_alpha * alpha + a.C_L_q * q_hat + a.C_L_delta_e * delta_e
        aspect_ratio = d.wing_span ** 2 / d.wing_area
        C_D = a.C_D_p + C_L ** 2 / (np.pi * a.oswald_efficiency * aspect_ratio)
        C_Y = a.C_Y_0 + a.C_Y_beta * beta + a.C_Y_p * p_hat + a.C_Y_r * r_hat + a.C_Y_delta_a * delta_a + a.C_Y_delta_r * delta_r
        C_l = a.C_l_0 + a.C_l_beta * beta + a.C_l_p * p_hat + a.C_l_r * r_hat + a.C_l_delta_a * delta_a + a.C_l_delta_r * delta_r
        C_m = a.C_m_0 + a.C_m_alpha * alpha + a.C_m_q * q_hat + a.C_m_delta_e * delta_e
        C_n = a.C_n_0 + a.C_n_beta * beta + a.C_n_p * p_hat + a.C_n_r * r_hat + a.C_n_delta_a * delta_a + a.C_n_delta_r * delta_r

        qS = 0.5 * d.air_density * airspeed ** 2 * d.wing_area
        sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
        force += qS * np.array([
            -C_D * cos_alpha + C_L * sin_alpha,
            C_Y,
            -C_D * sin_alpha - C_L * cos_alpha,
        ])
        moment += qS * np.array([d.wing_span * C_l, d.mean_chord * C_m, d.wing_span * C_n])

    return FRD_FLU @ force, FRD_FLU @ moment


def airplane_dynamics(parameters: Parameters, state: State, action) -> State:
    # action: commanded actuator positions, same units as State.actuators
    d = parameters.dynamics
    force, moment = airplane_aerodynamics(parameters, state)
    return State(
        position=state.linear_velocity,
        orientation=quaternion_derivative(state.orientation, state.angular_velocity),
        linear_velocity=rotate_vector_by_quaternion(state.orientation, force) / d.mass + d.gravity,
        angular_velocity=d.J_inv @ (moment - np.cross(state.angular_velocity, d.J @ state.angular_velocity)),
        actuators=(action - state.actuators) / d.actuator_time_constants,
    )


# 70_post_integration.h
def post_integration(parameters: Parameters, next_state: State) -> State:
    next_state.orientation = next_state.orientation / np.linalg.norm(next_state.orientation)
    limits = parameters.dynamics.actuator_limits
    next_state.actuators = np.clip(next_state.actuators, limits[:, 0], limits[:, 1])
    return next_state


# operations_generic.h
def scale_action(parameters: Parameters, action):
    # action in [-1, 1] -> [actuator_limits.min, actuator_limits.max]
    limits = parameters.dynamics.actuator_limits
    return limits[:, 0] + (np.clip(action, -1, 1) + 1) / 2 * (limits[:, 1] - limits[:, 0])


def normalize_action(parameters: Parameters, action_scaled):
    limits = parameters.dynamics.actuator_limits
    return (action_scaled - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) * 2 - 1


def step(parameters: Parameters, state: State, action) -> State:
    action_scaled = scale_action(parameters, np.asarray(action, dtype=float))
    next_state = rk4(airplane_dynamics, parameters, state, action_scaled, parameters.integration.dt)
    return post_integration(parameters, next_state)


def specific_energy(parameters: Parameters, state: State) -> float:
    # potential + kinetic energy per unit mass, ground-relative
    g = np.linalg.norm(parameters.dynamics.gravity)
    return g * state.position[2] + 0.5 * state.linear_velocity @ state.linear_velocity
