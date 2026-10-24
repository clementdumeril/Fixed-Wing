import warnings

import numpy as np
import pytest

from fixed_wing.airplane.airplane import ELEVATOR
from fixed_wing.airplane.environment import (
    EnvConfig,
    FixedWingEnv,
    RewardConfig,
    RewardCurriculum,
    euler_quaternion,
    measurements,
    reward,
    actor_observation_size,
    reward_from_observations,
    sample_initial_parameters,
    sample_initial_state,
    terminated,
)
from fixed_wing.airplane.operations import air_data, normalize_action
from fixed_wing.airplane.parameters import aerosonde, dryden_low_altitude_light
from fixed_wing.airplane.pid import PIDController
from fixed_wing.quaternion_helper import rotate_vector_by_quaternion


def rotate(state, v):
    return rotate_vector_by_quaternion(state.orientation, v)


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
    assert abs(setpoint.altitude - state.position[2]) <= config.altitude_change_max
    assert config.altitude_range[0] <= state.position[2] <= config.altitude_range[1]
    assert config.altitude_range[0] <= setpoint.altitude <= config.altitude_range[1]
    assert terminated(parameters, state, setpoint, config) is None


def test_reward_is_highest_on_the_setpoint():
    config = EnvConfig()
    rng = np.random.default_rng(2)
    parameters, _ = sample_initial_parameters(rng, config)
    state, setpoint = sample_initial_state(rng, parameters, config)
    m = measurements(parameters, state, config)
    setpoint.airspeed, setpoint.altitude, setpoint.course = m["airspeed"], m["altitude"], m["course"]
    action = setpoint.trim_action  # the initial state has no sideslip
    best, _ = reward(parameters, state, setpoint, action, action, config)
    assert best == config.reward.survival
    setpoint.altitude += 5
    worse, _ = reward(parameters, state, setpoint, action, action, config)
    setpoint.altitude += 1000
    worst, _ = reward(parameters, state, setpoint, action, action, config)
    c = config.reward
    assert best > worse > worst >= (1 - c.cost_cap) * c.survival - 3 * c.far_weight > 0  # flying always pays


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


@pytest.mark.parametrize("config", [EnvConfig(), EnvConfig(privileged_observation=True), EnvConfig(airspeed_sensor=False, action_history=0)])
def test_gymnasium_checker(config):
    env_checker = pytest.importorskip("gymnasium.utils.env_checker")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Box observation space (minimum|maximum) value is")  # unbounded on purpose
        env_checker.check_env(FixedWingEnv(config), skip_render_check=True)


def test_reward_recomputed_from_observations():
    # the replay buffer recomputation gives the reward the environment returned, with the same weights or new ones
    privileged = True
    config = EnvConfig(privileged_observation=privileged, wind_speed_max=5.0, turbulence=dryden_low_altitude_light())
    env = FixedWingEnv(config)
    observation, _ = env.reset(seed=7)
    rng = np.random.default_rng(0)
    stored = []
    for _ in range(200):
        action = np.clip(env.previous_action + 0.05 * rng.standard_normal(4), -1, 1)
        next_observation, r, term, trunc, info = env.step(action)
        stored.append((observation, next_observation, action, term, r))
        observation = next_observation
        if term or trunc:
            break
    part = (lambda o: o["critic"]) if privileged else (lambda o: o)
    o, o2 = np.array([part(t[0]) for t in stored]), np.array([part(t[1]) for t in stored])
    actions, terms, rewards = np.array([t[2] for t in stored]), np.array([t[3] for t in stored]), np.array([t[4] for t in stored])
    assert np.allclose(reward_from_observations(o, o2, actions, terms, config), rewards, atol=1e-4)
    harder = RewardCurriculum().target
    assert np.all(reward_from_observations(o, o2, actions, terms, config, harder) <= rewards + 1e-4)  # float32 observations


def test_curriculum():
    curriculum = RewardCurriculum()
    assert curriculum.at(0) == curriculum.initial
    assert curriculum.at(curriculum.period * curriculum.n_stages) == curriculum.at(10 ** 9)
    assert curriculum.at(10 ** 9).course_weight == pytest.approx(curriculum.target.course_weight)
    weights = [curriculum.at(k * curriculum.period).action_change_weight for k in range(curriculum.n_stages + 1)]
    assert np.all(np.diff(weights) > 0)  # grows at every stage
    assert curriculum.at(curriculum.period - 1) == curriculum.initial  # changes only every period
    env = FixedWingEnv(EnvConfig())
    env.set_reward(curriculum.at(10 ** 9))
    assert env.config.reward.course_weight == pytest.approx(2.7)


def test_reward_needs_the_critic_observation():
    config = EnvConfig()
    o = np.zeros((3, actor_observation_size(config)))
    with pytest.raises(ValueError):
        reward_from_observations(o, o, np.zeros((3, 4)), np.zeros(3, bool), config)


def test_sideslip_and_crossed_controls_cost():
    # the crab of the first SAC run (results/sac_run1) is now penalized
    config = EnvConfig()
    rng = np.random.default_rng(2)
    parameters, _ = sample_initial_parameters(rng, config)
    state, setpoint = sample_initial_state(rng, parameters, config)
    m = measurements(parameters, state, config)
    setpoint.airspeed, setpoint.altitude, setpoint.course = m["airspeed"], m["altitude"], m["course"]
    trimmed, _ = reward(parameters, state, setpoint, setpoint.trim_action, setpoint.trim_action, config)
    crossed = setpoint.trim_action + np.array([-0.77, 0, 0.94, 0])
    crab, _ = reward(parameters, state, setpoint, crossed, crossed, config)
    assert crab < trimmed
    state.linear_velocity = state.linear_velocity + 5 * rotate(state, np.array([0, 1.0, 0]))  # side velocity
    slipping, _ = reward(parameters, state, setpoint, setpoint.trim_action, setpoint.trim_action, config)
    assert slipping < trimmed


def test_far_field_term_always_rewards_getting_closer():
    # where the quadratic cost is saturated, a smaller error still earns more (the first runs had a flat reward there)
    from fixed_wing.airplane.environment import reward_from_errors
    c = RewardConfig()
    rewards = [reward_from_errors(0.0, 0.0, np.deg2rad(e), 0.0, 0.0, 0.0, c)[0] for e in (90, 70, 50, 30)]
    assert np.all(np.diff(rewards) > 0.005)
    assert reward_from_errors(20.0, 200.0, np.pi, 0.5, 4.0, 4.0, c)[0] > 0  # still positive at the worst


def test_curriculum_waits_for_tracking():
    from fixed_wing.airplane.asymmetric_sac import CurriculumCallback
    pytest.importorskip("stable_baselines3")
    curriculum = RewardCurriculum(period=10)
    callback = CurriculumCallback(curriculum, EnvConfig())

    class Logger:
        def record(self, *args):
            pass

    from types import SimpleNamespace as NS
    env = NS(env_method=lambda *a: None)
    callback.model = NS(logger=Logger(), get_env=lambda: env, replay_buffer=NS(full=False, pos=0))

    def run(course_error, steps):
        for _ in range(steps):
            callback.num_timesteps += 1
            callback.locals = {"infos": [{"course_error": course_error}]}
            callback._on_step()

    callback.num_timesteps = 0
    run(np.deg2rad(60), 30)  # not tracking: no stage
    assert callback.stage == 0
    run(np.deg2rad(5), 30)  # tracking: one stage per period
    assert callback.stage >= 2


def test_config_from_run(tmp_path):
    # the videos and the diagnosis fly a policy in the conditions it was trained in
    import json
    from fixed_wing.airplane.environment import config_from_run
    (tmp_path / "config.json").write_text(json.dumps({"args": {"wind_max": 5.0, "turbulence": "light", "noise": 1.0}}))
    config = config_from_run(tmp_path, privileged_observation=True)
    assert config.wind_speed_max == 5.0 and config.turbulence is not None and config.observation_noise == 1.0
    assert config.randomization == 0.0 and config.privileged_observation
