"""Low-level RL task on top of the simulator: hold an airspeed, an altitude and a course.

Same pieces as the L2F environment (sample_initial_parameters, sample_initial_state, observe, reward, terminated),
plus a Gymnasium wrapper, FixedWingEnv. Every design choice is a field of EnvConfig.
"""
import copy
from dataclasses import dataclass, field, fields, replace

import numpy as np

from ..quaternion_helper import quaternion_multiply, quaternion_to_rotation_matrix, rotate_vector_by_quaternion_inverse
from .airplane import Parameters, State, Turbulence
from .operations import air_data, local_wind, normalize_action, sample_thermals, sample_turbulence, step, wind
from .parameters import aerosonde, dryden_low_altitude_light, dryden_low_altitude_moderate
from .trim import trim

try:
    import gymnasium
    from gymnasium import spaces
except ImportError:  # optional: pip install -e ".[rl]"
    gymnasium = None


@dataclass
class RewardConfig:
    # reward = survival - cost_cap * survival * tanh(cost / (cost_cap * survival)), with cost = sum of
    # weight * (error / scale)^2. Every step in flight earns at least (1 - cost_cap) * survival, even far from the
    # setpoint, so ending the episode never pays (L2F's survival bonus against "learning to terminate"). The cap is
    # smooth: with a hard min(cost, cap) the reward went flat on 98 % of the training steps once the curriculum raised
    # the weights, and the policy stopped tracking the setpoint (Savio job 40282709).
    survival: float = 1.0
    cost_cap: float = 0.5
    airspeed_scale: float = 2.0  # m/s
    altitude_scale: float = 10.0  # m
    course_scale: float = np.deg2rad(30)
    airspeed_weight: float = 0.3
    altitude_weight: float = 0.3
    course_weight: float = 0.3
    action_change_weight: float = 0.1  # on |a_t - a_t-1|^2, normalized actions: smooth commands, servo wear
    # coordinated flight: without these two terms the first SAC run flew straight in a 9 deg sideslip, banked 25 deg
    # with crossed aileron and rudder (results/sac_run1)
    sideslip_scale: float = np.deg2rad(5)
    sideslip_weight: float = 0.1
    trim_weight: float = 0.05  # on |a_t - a_trim|^2, a_trim = steady flight at the setpoint airspeed (L2F: |a - C_rab|^2)


@dataclass
class EnvConfig:
    frequency: float = 50.0  # Hz, control rate (external actuator commands to PX4 are limited to < ~50 Hz)
    n_substeps: int = 2  # RK4 steps per control period
    latency_steps: int = 1  # control periods between a command and its application
    episode_seconds: float = 30.0
    # m above ground, where the lab flies the Strix: glides from 120 m (Morando et al. 2026), 50 to 120 m above the
    # terrain (Lim & Loianno 2026); 120 m is also about the FAA Part 107 limit (400 ft). Start and setpoint stay inside.
    altitude_range: tuple = (50.0, 120.0)

    # setpoint, constant over an episode
    airspeed_range: tuple = (20.0, 30.0)  # m/s, Aerosonde scale (trim alpha 10 deg at 20 m/s)
    altitude_change_max: float = 20.0  # m, setpoint = initial altitude +- this, kept inside altitude_range
    course_change_max: float = np.deg2rad(90)  # setpoint = initial course +- this

    # initial state: trim at a random airspeed, then attitude and rate perturbations
    roll_max: float = np.deg2rad(20)
    pitch_max: float = np.deg2rad(10)
    rate_max: float = 0.3  # rad/s, each axis

    # actor observation
    airspeed_sensor: bool = True  # Pitot tube; without it the airspeed setpoint is tracked on the ground speed
    action_history: int = 10  # last actions in the observation (0 for a recurrent policy)
    observation_noise: float = 0.0  # multiplier on NOISE_STD
    privileged_observation: bool = False  # True: dict observation {"actor", "critic"} for an asymmetric actor-critic

    # termination
    alpha_range: tuple = (np.deg2rad(-10), np.deg2rad(15))  # the linear aerodynamics have no stall
    airspeed_limits: tuple = (15.0, 40.0)  # m/s
    altitude_error_max: float = 50.0  # m
    altitude_floor: float = 20.0  # m, below it the safety pilot takes over

    # domain randomization: every factor uniform in [1 - r, 1 + r]
    randomization: float = 0.0

    # disturbances (off by default)
    wind_speed_max: float = 0.0  # m/s, constant horizontal wind, random direction
    turbulence: Turbulence = None  # e.g. dryden_low_altitude_light()
    thermals: bool = False  # Allen thermals over a 2 km x 2 km square around the start

    reward: RewardConfig = field(default_factory=RewardConfig)


# standard deviation of the measurement noise, before EnvConfig.observation_noise
NOISE_STD = dict(airspeed=0.5, altitude=1.0, course=np.deg2rad(2), rotation=0.01, rates=0.02, velocity=0.2)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Setpoint:
    airspeed: float
    altitude: float
    course: float  # rad, direction of the ground velocity, counterclockwise from world x
    trim_action: np.ndarray = None  # normalized commands of steady level flight at the setpoint airspeed


@dataclass
class Randomization:
    # factors applied to the nominal airplane, kept for the critic
    mass: float = 1.0
    inertia: float = 1.0
    thrust: float = 1.0
    actuator_time: float = 1.0
    aerodynamics: np.ndarray = field(default_factory=lambda: np.ones(28))

    def vector(self):
        return np.concatenate(([self.mass, self.inertia, self.thrust, self.actuator_time], self.aerodynamics)) - 1


def sample_initial_parameters(rng, config: EnvConfig, nominal: Parameters = None):
    parameters = copy.deepcopy(nominal or aerosonde())
    parameters.integration.dt = 1.0 / config.frequency
    parameters.integration.n_substeps = config.n_substeps
    parameters.latency.action_steps = config.latency_steps

    r = config.randomization
    factor = lambda size=None: rng.uniform(1 - r, 1 + r, size)
    randomization = Randomization(mass=factor(), inertia=factor(), thrust=factor(), actuator_time=factor(), aerodynamics=factor(28))
    d = parameters.dynamics
    d.mass *= randomization.mass
    d.J = d.J * randomization.inertia
    d.J_inv = np.linalg.inv(d.J)
    d.thrust_max *= randomization.thrust
    d.actuator_time_constants = d.actuator_time_constants * randomization.actuator_time
    for f, k in zip(fields(d.aerodynamics), randomization.aerodynamics):
        setattr(d.aerodynamics, f.name, getattr(d.aerodynamics, f.name) * k)

    disturbances = parameters.disturbances
    direction = rng.uniform(-np.pi, np.pi)
    disturbances.wind = rng.uniform(0, config.wind_speed_max) * np.array([np.cos(direction), np.sin(direction), 0.0])
    if config.turbulence is not None:
        disturbances.turbulence = copy.deepcopy(config.turbulence)
    if config.thermals:
        disturbances.thermals = sample_thermals(rng, 2000.0, 2000.0, np.mean(config.altitude_range))
    return parameters, randomization


def euler_quaternion(roll, pitch, yaw):
    # body to world, world z up, body x forward / y left / z up: roll > 0 right wing down, pitch > 0 nose up,
    # yaw > 0 nose to the left
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    qy = np.array([np.cos(pitch / 2), 0, -np.sin(pitch / 2), 0])
    qx = np.array([np.cos(roll / 2), np.sin(roll / 2), 0, 0])
    return quaternion_multiply(quaternion_multiply(qz, qy), qx)


def sample_initial_state(rng, parameters: Parameters, config: EnvConfig):
    """Trimmed flight at a random airspeed, perturbed in attitude and rates, and a setpoint to reach.

    The attitude perturbation keeps the air-relative velocity fixed in the body frame, so alpha and beta start at
    their trim values. Returns the state and the setpoint.
    """
    trimmed = trim(parameters, rng.uniform(*config.airspeed_range))
    air_velocity_body = rotate_vector_by_quaternion_inverse(trimmed.orientation, trimmed.linear_velocity - wind(parameters, trimmed.position))
    pitch = np.arcsin(quaternion_to_rotation_matrix(trimmed.orientation)[2, 0])
    course = rng.uniform(-np.pi, np.pi)
    orientation = euler_quaternion(
        rng.uniform(-config.roll_max, config.roll_max),
        pitch + rng.uniform(-config.pitch_max, config.pitch_max),
        course,
    )
    position = np.array([0.0, 0.0, rng.uniform(*config.altitude_range)])
    state = State(
        position=position,
        orientation=orientation,
        linear_velocity=quaternion_to_rotation_matrix(orientation) @ air_velocity_body + wind(parameters, position),
        angular_velocity=rng.uniform(-config.rate_max, config.rate_max, 3),
        actuators=trimmed.actuators.copy(),
    )
    if np.any(parameters.disturbances.turbulence.sigma):
        state.turbulence = sample_turbulence(rng)
    setpoint = Setpoint(
        airspeed=rng.uniform(*config.airspeed_range),
        altitude=np.clip(position[2] + rng.uniform(-config.altitude_change_max, config.altitude_change_max), *config.altitude_range),
        course=wrap(course + rng.uniform(-config.course_change_max, config.course_change_max)),
    )
    setpoint.trim_action = normalize_action(parameters, trim(parameters, setpoint.airspeed).actuators)  # of this airplane
    return state, setpoint


def measurements(parameters: Parameters, state: State, config: EnvConfig):
    """What the airplane can measure (PX4 estimates), shared by the actor observation and the PID baseline."""
    R = quaternion_to_rotation_matrix(state.orientation)
    _, airspeed, _, _ = air_data(parameters, state)
    v = state.linear_velocity
    return dict(
        airspeed=airspeed if config.airspeed_sensor else np.linalg.norm(v),  # ground speed without a Pitot tube
        altitude=state.position[2],
        course=np.arctan2(v[1], v[0]),
        rotation=R,
        roll=np.arctan2(R[2, 1], R[2, 2]),
        pitch=np.arcsin(np.clip(R[2, 0], -1, 1)),
        rates=state.angular_velocity,
        velocity_body=R.T @ v,
    )


def errors(m, setpoint: Setpoint):
    return m["airspeed"] - setpoint.airspeed, m["altitude"] - setpoint.altitude, wrap(m["course"] - setpoint.course)


def observe_actor(parameters, state, setpoint, action_history, config: EnvConfig, rng, noise=None):
    m = measurements(parameters, state, config)
    s = config.observation_noise if noise is None else noise  # the critic gets the noise-free version
    if s > 0:
        m["airspeed"] += s * NOISE_STD["airspeed"] * rng.standard_normal()
        m["altitude"] += s * NOISE_STD["altitude"] * rng.standard_normal()
        m["course"] += s * NOISE_STD["course"] * rng.standard_normal()
        m["rotation"] = m["rotation"] + s * NOISE_STD["rotation"] * rng.standard_normal((3, 3))
        m["rates"] = m["rates"] + s * NOISE_STD["rates"] * rng.standard_normal(3)
        m["velocity_body"] = m["velocity_body"] + s * NOISE_STD["velocity"] * rng.standard_normal(3)
    e_airspeed, e_altitude, e_course = errors(m, setpoint)
    parts = [
        [e_airspeed / 5, e_altitude / 20, np.sin(e_course), np.cos(e_course)],
        m["rotation"].ravel(),
        m["rates"] / np.pi,
        m["velocity_body"] / 25,
    ]
    if config.airspeed_sensor:
        parts.append([m["airspeed"] / 25])
    parts.append(np.ravel(action_history))
    return np.concatenate(parts).astype(np.float32)


def observe_critic(parameters, state, setpoint, action_history, randomization: Randomization, config: EnvConfig):
    # privileged: noise-free actor observation + air data, true wind, actuator positions, randomization factors
    actor = observe_actor(parameters, state, setpoint, action_history, config, rng=None, noise=0.0)
    _, airspeed, alpha, beta = air_data(parameters, state)
    privileged = np.concatenate((
        [alpha / 0.3, beta / 0.3, airspeed / 25],
        local_wind(parameters, state) / 10,
        normalize_action(parameters, state.actuators),
        randomization.vector() / 0.3,
        setpoint.trim_action,  # last: reward_from_observations reads it there
    ))
    return np.concatenate((actor, privileged)).astype(np.float32)


def reward_from_errors(e_airspeed, e_altitude, e_course, beta, action_change_squared, trim_deviation_squared, c: RewardConfig):
    # works on scalars and on arrays (replay buffer recomputation, see reward_from_observations)
    cost = (
        c.airspeed_weight * (e_airspeed / c.airspeed_scale) ** 2
        + c.altitude_weight * (e_altitude / c.altitude_scale) ** 2
        + c.course_weight * (e_course / c.course_scale) ** 2
        + c.sideslip_weight * (beta / c.sideslip_scale) ** 2
        + c.action_change_weight * action_change_squared
        + c.trim_weight * trim_deviation_squared
    )
    cap = c.cost_cap * c.survival
    return c.survival - cap * np.tanh(cost / cap), cost


def reward(parameters, state, setpoint, action, previous_action, config: EnvConfig):
    e_airspeed, e_altitude, e_course = errors(measurements(parameters, state, config), setpoint)
    _, _, _, beta = air_data(parameters, state)  # the reward may use what the actor does not observe
    change = np.sum((np.asarray(action) - previous_action) ** 2)
    deviation = np.sum((np.asarray(action) - setpoint.trim_action) ** 2)
    return reward_from_errors(e_airspeed, e_altitude, e_course, beta, change, deviation, config.reward)


def actor_observation_size(config: EnvConfig):
    return 19 + config.airspeed_sensor + 4 * config.action_history


def reward_from_observations(observation, next_observation, action, terminated_, config: EnvConfig, reward_config=None):
    """The reward of stored transitions, recomputed from the observations with new weights (L2F recomputes the
    replay buffer after each curriculum change, p. 6).

    Observations are critic observations (the noise-free actor observation, then the privileged part with the
    sideslip and the trim commands), batched on the first axis. Needs action_history >= 1: the previous action is the
    last one in the observation history.
    """
    if config.action_history < 1:
        raise ValueError("the previous action is read from the action history")
    n = actor_observation_size(config)
    if observation.shape[1] <= n:
        raise ValueError("needs the critic observation (privileged_observation=True): sideslip and trim commands")
    previous_action = observation[:, n - 4:n]
    trim_action = observation[:, -4:]
    e_airspeed = next_observation[:, 0] * 5
    e_altitude = next_observation[:, 1] * 20
    e_course = np.arctan2(next_observation[:, 2], next_observation[:, 3])
    beta = next_observation[:, n + 1] * 0.3
    change = np.sum((action - previous_action) ** 2, axis=1)
    deviation = np.sum((action - trim_action) ** 2, axis=1)
    r, _ = reward_from_errors(e_airspeed, e_altitude, e_course, beta, change, deviation, reward_config or config.reward)
    return np.where(terminated_, 0.0, r)


WEIGHTS = ("airspeed_weight", "altitude_weight", "course_weight", "sideslip_weight", "action_change_weight", "trim_weight")


@dataclass
class RewardCurriculum:
    """L2F curriculum (p. 4): the reward weights grow from initial to target every period steps, then stay.

    Here the growth is geometric over n_stages stages, so each weight is multiplied by the same factor at each stage.
    """
    initial: RewardConfig = field(default_factory=RewardConfig)
    target: RewardConfig = field(default_factory=lambda: RewardConfig(
        airspeed_weight=1.2,  # = scale 1 m/s
        altitude_weight=1.2,  # = scale 5 m
        course_weight=2.7,  # = scale 10 deg
        action_change_weight=1.0,  # SAC explores with noisy actions: a large weight saturates the cost on every step
        sideslip_weight=1.0,  # 5 deg of sideslip costs 1
        trim_weight=0.3,
    ))
    period: int = 100_000
    n_stages: int = 5

    def at(self, step):
        x = min(step // self.period, self.n_stages) / self.n_stages
        weights = {}
        for name in WEIGHTS:
            a, b = getattr(self.initial, name), getattr(self.target, name)
            weights[name] = a * (b / a) ** x
        return replace(self.initial, **weights)


def terminated(parameters, state, setpoint, config: EnvConfig):
    """Returns the reason the flight is over, or None. The time limit is a truncation, handled by the caller."""
    if not np.all(np.isfinite(state.linear_velocity)):
        return "diverged"
    if state.position[2] < config.altitude_floor:
        return "low_altitude"
    _, airspeed, alpha, _ = air_data(parameters, state)
    if not config.alpha_range[0] <= alpha <= config.alpha_range[1]:
        return "alpha"
    if not config.airspeed_limits[0] <= airspeed <= config.airspeed_limits[1]:
        return "airspeed"
    if abs(state.position[2] - setpoint.altitude) > config.altitude_error_max:
        return "altitude"
    return None


class FixedWingEnv(gymnasium.Env if gymnasium else object):
    """Gymnasium environment (it also runs without gymnasium, then without the spaces).

    Actions: [aileron, elevator, rudder, throttle] in [-1, 1]. The rendered state is in self.state, the setpoint in
    self.setpoint, the sampled airplane in self.parameters.
    """

    def __init__(self, config: EnvConfig = None, nominal: Parameters = None):
        self.config = config or EnvConfig()
        self.nominal = nominal or aerosonde()
        self.max_steps = round(self.config.episode_seconds * self.config.frequency)
        self.rng = np.random.default_rng()
        self.reset(seed=0)
        if gymnasium is None:
            return
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        actor = spaces.Box(-np.inf, np.inf, self._actor().shape, np.float32)
        if self.config.privileged_observation:
            critic = spaces.Box(-np.inf, np.inf, self._critic().shape, np.float32)
            self.observation_space = spaces.Dict({"actor": actor, "critic": critic})
        else:
            self.observation_space = actor

    def set_reward(self, reward_config: RewardConfig):
        # for the curriculum, called through VecEnv.env_method in the training processes
        self.config.reward = reward_config

    def _actor(self):
        return observe_actor(self.parameters, self.state, self.setpoint, self.action_history, self.config, self.rng)

    def _critic(self):
        return observe_critic(self.parameters, self.state, self.setpoint, self.action_history, self.randomization, self.config)

    def _observation(self):
        if self.config.privileged_observation:
            return {"actor": self._actor(), "critic": self._critic()}
        return self._actor()

    def _info(self, **extra):
        e_airspeed, e_altitude, e_course = errors(measurements(self.parameters, self.state, self.config), self.setpoint)
        return dict(airspeed_error=e_airspeed, altitude_error=e_altitude, course_error=e_course, **extra)

    def reset(self, seed=None, options=None):
        if gymnasium is not None:
            super().reset(seed=seed)  # seeds self.np_random, as Gymnasium expects
            self.rng = self.np_random
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        self.parameters, self.randomization = sample_initial_parameters(self.rng, self.config, self.nominal)
        self.state, self.setpoint = sample_initial_state(self.rng, self.parameters, self.config)
        self.previous_action = normalize_action(self.parameters, self.state.actuators)
        self.action_history = np.tile(self.previous_action, (self.config.action_history, 1))
        self.steps = 0
        return self._observation(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.state = step(self.parameters, self.state, action, self.rng)
        self.steps += 1
        r, cost = reward(self.parameters, self.state, self.setpoint, action, self.previous_action, self.config)
        self.previous_action = action
        if self.config.action_history:
            self.action_history = np.vstack([self.action_history[1:], action])
        reason = terminated(self.parameters, self.state, self.setpoint, self.config)
        truncated = reason is None and self.steps >= self.max_steps
        return self._observation(), float(r) if reason is None else 0.0, reason is not None, truncated, \
            self._info(cost=cost, termination=reason)


def evaluate(policy, config: EnvConfig = None, seeds=range(100)):
    """Same scenarios for every controller (PID, RL, later the NMPC): one episode per seed.

    policy.reset(env) is called after each reset, policy(env, observation) returns the action.
    Tracking errors are root mean squares over the second half of the episode, once the setpoint is reached.
    """
    env = FixedWingEnv(config)
    returns, failures, reasons, tracking, smoothness = [], 0, {}, [], []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        policy.reset(env)
        total, done, errors_, changes, previous = 0.0, False, [], [], env.previous_action
        while not done:
            action = policy(env, observation)
            observation, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
            errors_.append([info["airspeed_error"], info["altitude_error"], info["course_error"]])
            changes.append(np.abs(np.asarray(action) - previous).mean())
            previous = np.asarray(action)
        returns.append(total)
        if term:
            failures += 1
            reasons[info["termination"]] = reasons.get(info["termination"], 0) + 1
        else:
            settled = np.array(errors_[len(errors_) // 2:])
            tracking.append(np.sqrt(np.mean(settled ** 2, axis=0)))
            smoothness.append(np.mean(changes))
    tracking = np.array(tracking) if tracking else np.full((1, 3), np.nan)
    return dict(
        episodes=len(returns),
        success_rate=1 - failures / len(returns),
        mean_return=float(np.mean(returns)),
        max_return=env.max_steps * env.config.reward.survival,
        failures=reasons,
        airspeed_rmse=float(np.mean(tracking[:, 0])),
        altitude_rmse=float(np.mean(tracking[:, 1])),
        course_rmse_deg=float(np.rad2deg(np.mean(tracking[:, 2]))),
        mean_action_change=float(np.mean(smoothness)) if smoothness else np.nan,
    )


TURBULENCE = {"none": None, "light": dryden_low_altitude_light, "moderate": dryden_low_altitude_moderate}


def add_condition_arguments(parser):
    """Flight conditions shared by training, evaluation and the PID baseline, so that they are compared alike."""
    parser.add_argument("--wind-max", type=float, default=0.0, help="m/s, constant horizontal wind, random speed and direction")
    parser.add_argument("--turbulence", choices=TURBULENCE, default="none", help="Dryden, low altitude")
    parser.add_argument("--noise", type=float, default=0.0, help="sensor noise, multiplier on NOISE_STD")
    parser.add_argument("--randomization", type=float, default=0.0, help="airplane parameters within +- this fraction")


def config_from_arguments(args, **kwargs):
    turbulence = TURBULENCE[args.turbulence]
    return EnvConfig(wind_speed_max=args.wind_max, turbulence=turbulence() if turbulence else None,
                     observation_noise=args.noise, randomization=args.randomization, **kwargs)
