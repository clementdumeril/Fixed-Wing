"""Fixed-wing counterpart of multirotor/operations.py, same sections as the L2F original."""
import numpy as np

from ..integrators import rk4
from ..quaternion_helper import (
    quaternion_derivative,
    rotate_vector_by_quaternion,
    rotate_vector_by_quaternion_inverse,
)
from .airplane import Parameters, State, Thermals
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


# Allen 2006, appendix B (run_model2_3): updraft shapes, one row per ratio r1/r2 of the inner to the outer radius.
# k1, k2, k3, k4 of Eq. (16). k4 is the column the reference code uses; table 3 of the paper prints a near-zero k4
# instead (0.0001 to 0.0008), which leaves a large updraft at r = r2 rather than the trapezoid it is fitted to.
ALLEN_R1_R2 = np.array([0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80])
ALLEN_SHAPES = np.array([
    [1.5352, 2.5826, -0.0113, -0.1950],
    [1.5265, 3.6054, -0.0176, -0.1265],
    [1.4866, 4.8356, -0.0320, -0.0818],
    [1.2042, 7.7904, 0.0848, -0.0445],
    [0.8816, 13.9720, 0.3404, -0.0216],
    [0.7067, 23.9940, 0.5689, -0.0099],
    [0.6189, 42.7965, 0.7157, -0.0033],
])


def thermal_updraft(thermals: Thermals, position) -> float:
    """Vertical air velocity (m/s, up) of the Allen model at a world position, nearest updraft only, as in the paper."""
    t = thermals
    height = position[2]
    if len(t.centers) == 0 or height <= 0:
        return 0.0
    zzi = height / t.z_i
    r_bar = 0.102 * zzi ** (1 / 3) * (1 - 0.25 * zzi) * t.z_i  # Eq. (12), average outer radius
    w_bar = zzi ** (1 / 3) * (1 - 1.1 * zzi) * t.w_star  # Eq. (11), average updraft velocity
    distances = np.linalg.norm(position[:2] - t.centers, axis=1)
    k = np.argmin(distances)
    r = distances[k]
    r2 = max(r_bar * t.r_gain[k], 10.0)
    r1_r2 = 0.0011 * r2 + 0.14 if r2 < 600 else 0.8  # Eq. (13)
    r1 = r1_r2 * r2
    w_t = w_bar * t.w_gain[k]
    w_c = 3 * w_t * (r2 ** 3 - r2 ** 2 * r1) / (r2 ** 3 - r1 ** 3)  # Eq. (15), velocity at the center
    rr2 = r / r2

    w_s = 0.0
    if height < t.z_i:  # Eq. (16), bell shape closest to r1/r2
        k1, k2, k3, k4 = ALLEN_SHAPES[np.searchsorted((ALLEN_R1_R2[1:] + ALLEN_R1_R2[:-1]) / 2, r1_r2, side="right")]
        w_s = max(1 / (1 + (k1 * abs(rr2 + k3)) ** k2) + k4 * rr2, 0.0)
    # Eqs. (17), (18): downdraft ring around the updraft, high in the mixing layer
    w_1 = np.pi / 6 * np.sin(np.pi * rr2) if r > r1 and rr2 < 2 else 0.0
    s_wd = 2.5 * (zzi - 0.5) if 0.5 < zzi <= 0.9 else 0.0
    w_d = min(s_wd * w_1, 0.0)
    w = w_s * w_c + w_d * w_t

    if t.sink:  # Eq. (21): the sinking air between the updrafts carries as much air as they lift
        updraft_area = len(t.centers) * np.pi * r_bar ** 2
        if updraft_area >= t.area:
            raise ValueError("the updrafts cover more than thermals.area")
        w_e = min(-updraft_area * w_bar * (1 - s_wd) / (t.area - updraft_area), 0.0)
        if r > r1 and w_c != 0:  # Eq. (23): stretched to blend into the sink at the edge
            w = w * (1 - w_e / w_c) + w_e
    return w


def sample_thermals(rng, size_x, size_y, height, w_star=2.56, z_i=1401.0) -> Thermals:
    """Updrafts placed uniformly at random in [-size_x/2, size_x/2] x [-size_y/2, size_y/2], as many as Eq. (20)
    gives for an airplane flying at the given height above ground."""
    zzi = height / z_i
    r2 = max(0.102 * zzi ** (1 / 3) * (1 - 0.25 * zzi) * z_i, 10.0)
    n = max(round(0.6 * size_x * size_y / (z_i * r2)), 1)
    centers = rng.uniform([-size_x / 2, -size_y / 2], [size_x / 2, size_y / 2], size=(n, 2))
    return Thermals(centers=centers, w_star=w_star, z_i=z_i, w_gain=np.ones(n), r_gain=np.ones(n), area=size_x * size_y)


# 60_dynamics.h
def wind(parameters: Parameters, position):
    # world frame: constant wind + "1 - cos" gust + thermal updraft where the airplane is
    # these depend on the position, not on the time, so the dynamics stay a function of (parameters, state, action)
    d = parameters.disturbances
    w = d.wind + np.array([0.0, 0.0, thermal_updraft(d.thermals, position)])
    s = (d.gust_direction @ position - d.gust_start) / d.gust_length  # 0 -> 1 across the gust
    if 0 <= s < 1:
        return w + d.gust * 0.5 * (1 - np.cos(2 * np.pi * s))
    return w


def turbulence_velocity(parameters: Parameters, state: State):
    # body frame, from the Dryden filter states; the lateral and vertical outputs keep the variance sigma^2
    sigma = parameters.disturbances.turbulence.sigma
    u, v1, v2, w1, w2 = state.turbulence
    return sigma * np.array([u, 0.5 * v1 + np.sqrt(3) / 2 * v2, 0.5 * w1 + np.sqrt(3) / 2 * w2])


def local_wind(parameters: Parameters, state: State):
    # world frame: everything the airplane feels, turbulence included
    if not np.any(parameters.disturbances.turbulence.sigma):
        return wind(parameters, state.position)
    return wind(parameters, state.position) + rotate_vector_by_quaternion(state.orientation, turbulence_velocity(parameters, state))


def _noise_gain(covariance):
    # square root of a 2x2 covariance that can be singular (zero distance flown)
    values, vectors = np.linalg.eigh(covariance)
    return vectors * np.sqrt(np.clip(values, 0, None))


def advance_turbulence(parameters: Parameters, turbulence, distance, rng):
    """Dryden filter states after flying the given distance through the air.

    Exact discretization of the Dryden shaping filters (Beard & McLain, chapter 4), with the states scaled by their
    stationary covariance: the output keeps the variance sigma^2 and the Dryden correlations for any step,
    exp(-d / L) along the flight path, (1 - d / (2 L)) exp(-d / L) for the lateral and vertical components.
    """
    length_scale = parameters.disturbances.turbulence.length_scale
    u, v, w = turbulence[0], turbulence[1:3], turbulence[3:5]
    s = distance / length_scale
    e = np.exp(-s)
    u = e[0] * u + np.sqrt(1 - e[0] ** 2) * rng.standard_normal()
    out = [u]
    for x, si, ei in ((v, s[1], e[1]), (w, s[2], e[2])):
        transition = ei * np.array([[1 + si, si], [-si, 1 - si]])
        out.extend(transition @ x + _noise_gain(np.eye(2) - transition @ transition.T) @ rng.standard_normal(2))
    return np.array(out)


def sample_turbulence(rng):
    # stationary distribution of the scaled filter states, to start an episode in established turbulence
    return rng.standard_normal(5)


def air_data(parameters: Parameters, state: State):
    """Air-relative velocity in the aerospace body frame (x forward, y right, z down), airspeed, alpha, beta"""
    air_relative_velocity = state.linear_velocity - local_wind(parameters, state)
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


def _cross(a, b):
    # np.cross is slow on 3-vectors
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def airplane_dynamics(parameters: Parameters, state: State, action) -> State:
    # action: commanded actuator positions, same units as State.actuators
    d = parameters.dynamics
    force, moment = airplane_aerodynamics(parameters, state)
    return State(
        position=state.linear_velocity,
        orientation=quaternion_derivative(state.orientation, state.angular_velocity),
        linear_velocity=rotate_vector_by_quaternion(state.orientation, force) / d.mass + d.gravity,
        angular_velocity=d.J_inv @ (moment - _cross(state.angular_velocity, d.J @ state.angular_velocity)),
        actuators=(action - state.actuators) / d.actuator_time_constants,
        action_buffer=np.zeros_like(state.action_buffer),  # discrete, updated in step
        turbulence=np.zeros_like(state.turbulence),  # discrete, updated in step
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


def delay_action(parameters: Parameters, state: State, command):
    """Returns the command applied during this control period and the new buffer of pending commands."""
    n = parameters.latency.action_steps
    if n == 0:
        return command, state.action_buffer
    buffer = state.action_buffer
    if len(buffer) != n:  # first call: the link is assumed to hold the current actuator positions
        buffer = np.tile(state.actuators, (n, 1))
    return buffer[0], np.vstack([buffer[1:], command])


def step(parameters: Parameters, state: State, action, rng=None) -> State:
    # rng (numpy Generator): needed only with turbulence, as the rng argument of the L2F step
    command = scale_action(parameters, np.asarray(action, dtype=float))
    applied, action_buffer = delay_action(parameters, state, command)
    n = parameters.integration.n_substeps
    next_state = state
    for _ in range(n):  # as in L2F, post_integration runs once after the substeps
        next_state = rk4(airplane_dynamics, parameters, next_state, applied, parameters.integration.dt / n)
    next_state.action_buffer = action_buffer
    if np.any(parameters.disturbances.turbulence.sigma):
        if rng is None:
            raise ValueError("turbulence is on: pass rng=np.random.default_rng(seed) to step")
        _, airspeed, _, _ = air_data(parameters, state)
        next_state.turbulence = advance_turbulence(parameters, state.turbulence, airspeed * parameters.integration.dt, rng)
    return post_integration(parameters, next_state)


def specific_energy(parameters: Parameters, state: State) -> float:
    # potential + kinetic energy per unit mass, ground-relative
    g = np.linalg.norm(parameters.dynamics.gravity)
    return g * state.position[2] + 0.5 * state.linear_velocity @ state.linear_velocity
