"""Fixed-wing counterpart of multirotor/multirotor.py

World frame: z up. Body frame: x forward, y left, z up (same as L2F).
The aerodynamic coefficients keep the usual aerospace convention (body x forward, y right, z down),
the conversion happens in one place: airplane_aerodynamics in operations.py.
"""
from dataclasses import dataclass, field

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
    dt: float  # control period: one call to step
    n_substeps: int = 1  # RK4 steps per control period (N_SUBSTEPS in L2F), the command is held over all of them


@dataclass
class Latency:
    # control periods between sending a command and its application, e.g. the Orin NX -> PX4 link
    action_steps: int = 0


@dataclass
class Turbulence:
    # Dryden turbulence (MIL-F-8785C) in the body frame, frozen in the air mass: it changes with the distance flown
    # through the air, not with the time. Off by default; presets in parameters.py.
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s, [longitudinal, lateral, vertical]
    length_scale: np.ndarray = field(default_factory=lambda: np.array([200.0, 200.0, 50.0]))  # m, same order


@dataclass
class Thermals:
    # Allen updraft model (NASA DFRC, AIAA 2006-1510), thermals fixed in space, height above ground = position z.
    # No thermals by default; sample_thermals in operations.py places them as in the paper.
    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (N, 2), world x, y of each updraft
    w_star: float = 2.56  # m/s, convective velocity scale (Allen table 1, mean)
    z_i: float = 1401.0  # m, convective mixing-layer thickness, the height the updrafts reach (Allen table 1, mean)
    w_gain: np.ndarray = field(default_factory=lambda: np.ones(0))  # (N,), strength of each updraft / average
    r_gain: np.ndarray = field(default_factory=lambda: np.ones(0))  # (N,), radius of each updraft / average
    area: float = 1e6  # m², region holding the updrafts, for the environment sink
    sink: bool = True  # sinking air outside the updrafts, which balances the rising air (mass conservation)


@dataclass
class Disturbances:
    wind: np.ndarray  # (3,), world frame, constant over an episode
    # "1 - cos" discrete gust, fixed in space: the airplane meets it when it flies through the slab
    # gust_start <= gust_direction . position < gust_start + gust_length. No gust by default.
    gust: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,), peak gust velocity, world frame
    gust_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0]))  # unit, normal to the gust front
    gust_start: float = 0.0  # m, along gust_direction
    gust_length: float = 1.0  # m, the gust peaks in the middle
    turbulence: Turbulence = field(default_factory=Turbulence)
    thermals: Thermals = field(default_factory=Thermals)


@dataclass
class Parameters:
    dynamics: Dynamics
    integration: Integration
    disturbances: Disturbances
    latency: Latency = field(default_factory=Latency)


@dataclass
class State:
    position: np.ndarray  # world frame
    orientation: np.ndarray  # quaternion [w, x, y, z], body to world
    linear_velocity: np.ndarray  # world frame, ground-relative
    angular_velocity: np.ndarray  # body frame
    actuators: np.ndarray  # (4,), actual deflections [rad] and throttle [0, 1]
    # commands sent but not applied yet, oldest first, (latency.action_steps, 4); not integrated (zero derivative)
    action_buffer: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    # Dryden filter states [u, v1, v2, w1, w2], scaled to unit variance; not integrated (zero derivative), updated in step
    turbulence: np.ndarray = field(default_factory=lambda: np.zeros(5))
