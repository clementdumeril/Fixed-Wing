import numpy as np
import pytest

from fixed_wing.airplane.airplane import ELEVATOR
from fixed_wing.airplane.environment import (
    EnvConfig,
    FixedWingEnv,
    euler_quaternion,
    measurements,
    reward,
    sample_initial_parameters,
    sample_initial_state,
    terminated,
)
from fixed_wing.airplane.operations import air_data, normalize_action
from fixed_wing.airplane.parameters import aerosonde, dryden_low_altitude_light
from fixed_wing.airplane.pid import PIDController


def test_euler_angles_round_trip():
    config = EnvConfig()
    parameters, _ = sample_initial_parameters(np.random.default_rng(0), config)
    state, _ = sample_initial_state(np.random.default_rng(0), parameters, config)
    state.orientation = euler_quaternion(0.2, -0.1, 1.0)
    m = measurements(parameters, state, config)
    assert np.isclose(m["roll"], 0.2) and np.isclose(m["pitch"], -0.1)
    assert m["rotation"][2, 1] > 0  # roll > 0: the left wing goes up, the right wing down


def test_initial_state_keeps_trim_air_data():
    # the attitude perturbation rotates the air velocity with the airplane: alpha and beta stay at trim
    config = EnvConfig()
    rng = np.random.default_rng(1)
    parameters, _ = sample_initial_parameters(rng, config)
    state, setpoint = sample_initial_state(rng, parameters, config)
    _, airspeed, alpha, beta = air_data(parameters, state)
    assert config.airspeed_range[0] <= airspeed <= config.airspeed_range[1]
    assert 0 < alpha < config.alpha_range[1] and np.isclose(beta, 0, atol=1e-9)
    assert abs(setpoint.altitude - config.initial_altitude) <= config.altitude_change_max
    assert terminated(parameters, state, setpoint, config) is None


def test_reward_is_highest_on_the_setpoint():
    config = EnvConfig()
    rng = np.random.default_rng(2)
    parameters, _ = sample_initial_parameters(rng, config)
    state, setpoint = sample_initial_state(rng, parameters, config)
    m = measurements(parameters, state, config)
    setpoint.airspeed, setpoint.altitude, setpoint.course = m["airspeed"], m["altitude"], m["course"]
    action = normalize_action(parameters, state.actuators)
    best, _ = reward(parameters, state, setpoint, action, action, config)
    assert best == config.reward.survival
    setpoint.altitude += 5
    worse, _ = reward(parameters, state, setpoint, action, action, config)
    setpoint.altitude += 1000
    worst, _ = reward(parameters, state, setpoint, action, action, config)
    assert best > worse > worst == (1 - config.reward.cost_cap) * config.reward.survival  # flying always pays


def test_randomization_scales_the_airplane():
    nominal = aerosonde()
    parameters, randomization = sample_initial_parameters(np.random.default_rng(3), EnvConfig(randomization=0.2))
    assert np.isclose(parameters.dynamics.mass, nominal.dynamics.mass * randomization.mass)
    assert 0.8 <= randomization.mass <= 1.2 and randomization.mass != 1
    assert np.allclose(parameters.dynamics.J @ parameters.dynamics.J_inv, np.eye(3))
    same, none = sample_initial_parameters(np.random.default_rng(3), EnvConfig())
    assert same.dynamics.mass == nominal.dynamics.mass and not np.any(none.vector())


def test_episode_api_and_reproducibility():
    env = FixedWingEnv(EnvConfig(episode_seconds=1.0))
    action = np.zeros(4)

    def run(seed):
        observation, _ = env.reset(seed=seed)
        observations = [observation]
        done = False
        while not done:
            observation, r, term, trunc, info = env.step(action)
            observations.append(observation)
            done = term or trunc
        return np.array(observations), term, trunc

    a, _, _ = run(0)
    b, _, _ = run(0)
    c, _, _ = run(1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.dtype == np.float32 and np.all(np.isfinite(a))


def test_time_limit_is_a_truncation():
    config = EnvConfig(episode_seconds=1.0)
    env = FixedWingEnv(config)
    env.reset(seed=4)
    pid = PIDController(1 / config.frequency)
    pid.reset(env.setpoint)
    for _ in range(env.max_steps):
        _, r, term, trunc, info = env.step(pid(measurements(env.parameters, env.state, config)))
        assert not term and r >= (1 - config.reward.cost_cap) * config.reward.survival
    assert trunc and info["termination"] is None


def test_stall_ends_the_episode():
    env = FixedWingEnv(EnvConfig())
    env.reset(seed=5)
    action = normalize_action(env.parameters, env.state.actuators)
    action[ELEVATOR] = -1  # full nose up
    for _ in range(500):
        _, r, term, _, info = env.step(action)
        if term:
            break
    assert term and info["termination"] == "alpha" and r == 0


def test_observations():
    env = FixedWingEnv(EnvConfig(privileged_observation=True, turbulence=dryden_low_altitude_light(), wind_speed_max=5.0))
    observation, _ = env.reset(seed=6)
    assert observation["actor"].shape == (60,)  # 20 measurements + 10 past actions
    assert observation["critic"].shape[0] > observation["actor"].shape[0]
    observation, *_ = env.step(np.zeros(4))
    assert np.all(np.isfinite(observation["critic"]))
    no_pitot = FixedWingEnv(EnvConfig(airspeed_sensor=False, action_history=0))
    assert no_pitot.reset(seed=6)[0].shape == (19,)


@pytest.mark.parametrize("seed", range(5))
def test_pid_holds_the_setpoint(seed):
    config = EnvConfig()
    env = FixedWingEnv(config)
    env.reset(seed=seed)
    pid = PIDController(1 / config.frequency)
    pid.reset(env.setpoint)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(pid(measurements(env.parameters, env.state, config)))
        done = term or trunc
    assert not term
    assert abs(info["airspeed_error"]) < 1.0 and abs(info["altitude_error"]) < 3.0 and abs(info["course_error"]) < np.deg2rad(5)
