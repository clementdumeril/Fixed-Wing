import numpy as np
import pytest

from fixed_wing.airplane.airplane import AILERON, ELEVATOR, RUDDER, THROTTLE
from fixed_wing.airplane.operations import (
    air_data,
    airplane_dynamics,
    normalize_action,
    specific_energy,
    step,
)
from fixed_wing.airplane.parameters import aerosonde
from fixed_wing.airplane.trim import trim
from fixed_wing.quaternion_helper import rotate_vector_by_quaternion

AIRSPEED = 25.0


def simulate(parameters, state, action_scaled, seconds):
    action = normalize_action(parameters, action_scaled)
    for _ in range(round(seconds / parameters.integration.dt)):
        state = step(parameters, state, action)
    return state


def nose_height(state):
    return rotate_vector_by_quaternion(state.orientation, np.array([1.0, 0, 0]))[2]


def test_trim_level_flight():
    parameters = aerosonde()
    state = trim(parameters, AIRSPEED)
    change = airplane_dynamics(parameters, state, state.actuators)
    assert np.allclose(change.linear_velocity, 0, atol=1e-8)
    assert np.allclose(change.angular_velocity, 0, atol=1e-8)
    _, airspeed, alpha, beta = air_data(parameters, state)
    assert np.isclose(airspeed, AIRSPEED)
    assert 0 < np.rad2deg(alpha) < 10
    assert beta == 0
    assert 0 < state.actuators[THROTTLE] < 1


def test_trim_holds_over_time():
    parameters = aerosonde()
    trimmed = trim(parameters, AIRSPEED)
    state = simulate(parameters, trimmed, trimmed.actuators, seconds=30)
    assert np.isclose(state.position[0], AIRSPEED * 30, atol=1e-3)
    assert np.allclose(state.position[1:], 0, atol=1e-3)
    assert np.allclose(state.linear_velocity, trimmed.linear_velocity, atol=1e-4)


def test_perturbed_trim_stays_bounded():
    # longitudinal static stability: a pitch-rate kick does not diverge
    parameters = aerosonde()
    trimmed = trim(parameters, AIRSPEED)
    trimmed.angular_velocity[1] = -0.3
    state = trimmed
    for _ in range(3000):
        state = step(parameters, state, normalize_action(parameters, trimmed.actuators))
        _, airspeed, alpha, _ = air_data(parameters, state)
        assert 15 < airspeed < 35
        assert abs(np.rad2deg(alpha)) < 15


def test_trim_glide():
    parameters = aerosonde()
    state = trim(parameters, AIRSPEED, glide=True)
    assert state.actuators[THROTTLE] == 0
    sink_rate = -state.linear_velocity[2]
    glide_ratio = state.linear_velocity[0] / sink_rate
    assert sink_rate > 0
    assert 5 < glide_ratio < 30
    # gliding at constant speed: energy is lost at the rate g * sink_rate
    after = simulate(parameters, state, state.actuators, seconds=5)
    assert np.isclose(specific_energy(parameters, after) - specific_energy(parameters, state), -9.81 * sink_rate * 5, rtol=1e-6)


def test_elevator_pitches():
    parameters = aerosonde()
    trimmed = trim(parameters, AIRSPEED)
    action = trimmed.actuators.copy()
    action[ELEVATOR] -= np.deg2rad(2)  # trailing edge up
    state = simulate(parameters, trimmed, action, seconds=0.5)
    assert nose_height(state) > nose_height(trimmed)
    action[ELEVATOR] += np.deg2rad(4)
    state = simulate(parameters, trimmed, action, seconds=0.5)
    assert nose_height(state) < nose_height(trimmed)


def test_aileron_rolls():
    parameters = aerosonde()
    trimmed = trim(parameters, AIRSPEED)
    action = trimmed.actuators.copy()
    action[AILERON] = np.deg2rad(5)
    state = simulate(parameters, trimmed, action, seconds=0.5)
    assert state.angular_velocity[0] > 0  # roll to the right
    left_wing = rotate_vector_by_quaternion(state.orientation, np.array([0, 1.0, 0]))
    assert left_wing[2] > 0


def test_rudder_yaws():
    parameters = aerosonde()
    trimmed = trim(parameters, AIRSPEED)
    action = trimmed.actuators.copy()
    action[RUDDER] = np.deg2rad(5)
    state = simulate(parameters, trimmed, action, seconds=0.2)
    # C_n_delta_r < 0: nose to the left, which is a positive rotation around z (up)
    assert state.angular_velocity[2] > 0


def test_throttle_adds_energy():
    parameters = aerosonde()
    trimmed = trim(parameters, AIRSPEED)
    action = trimmed.actuators.copy()
    action[THROTTLE] = 1.0
    full = simulate(parameters, trimmed, action, seconds=5)
    action[THROTTLE] = 0.0
    idle = simulate(parameters, trimmed, action, seconds=5)
    reference = specific_energy(parameters, trimmed)  # level trim: energy is constant
    assert specific_energy(parameters, full) > reference > specific_energy(parameters, idle)
    # right after the throttle change the airplane accelerates along its path
    short = simulate(parameters, trimmed, np.array([*trimmed.actuators[:3], 1.0]), seconds=1)
    assert air_data(parameters, short)[1] > AIRSPEED


def test_actuator_lag():
    parameters = aerosonde()
    trimmed = trim(parameters, AIRSPEED)
    action = trimmed.actuators.copy()
    action[AILERON] = np.deg2rad(10)
    tau = parameters.dynamics.actuator_time_constants[AILERON]
    state = simulate(parameters, trimmed, action, seconds=tau)
    assert np.isclose(state.actuators[AILERON], np.deg2rad(10) * (1 - np.exp(-1)), rtol=1e-3)
    # commands are saturated
    state = simulate(parameters, trimmed, np.array([10.0, 0, 0, 5.0]), seconds=3)
    assert np.isclose(state.actuators[AILERON], np.deg2rad(30), rtol=1e-3)
    assert np.isclose(state.actuators[THROTTLE], 1.0, rtol=1e-3)


@pytest.mark.parametrize("wind, expected", [
    # ground velocity is 25 m/s along x, level attitude
    ([-5.0, 0, 0], dict(airspeed=30.0, alpha=0.0, beta=0.0)),  # headwind
    ([5.0, 0, 0], dict(airspeed=20.0, alpha=0.0, beta=0.0)),  # tailwind
    ([0, 0, 5.0], dict(airspeed=np.hypot(25, 5), alpha=np.arctan2(5, 25), beta=0.0)),  # updraft
    # wind blowing to the left (+y): the relative wind comes from the right, positive sideslip
    ([0, 5.0, 0], dict(airspeed=np.hypot(25, 5), alpha=0.0, beta=np.arcsin(5 / np.hypot(25, 5)))),
])
def test_wind_changes_air_data(wind, expected):
    parameters = aerosonde()
    state = trim(parameters, AIRSPEED)
    state.orientation = np.array([1.0, 0, 0, 0])
    parameters.disturbances.wind = np.array(wind)
    _, airspeed, alpha, beta = air_data(parameters, state)
    assert np.isclose(airspeed, expected["airspeed"])
    assert np.isclose(alpha, expected["alpha"])
    assert np.isclose(beta, expected["beta"])


def test_constant_wind_only_shifts_the_trajectory():
    # A uniform constant wind is a change of inertial frame: no energy can be extracted from it.
    wind = np.array([3.0, -4.0, 1.0])
    calm = aerosonde()
    windy = aerosonde()
    windy.disturbances.wind = wind
    state_calm = trim(calm, AIRSPEED)
    state_windy = trim(windy, AIRSPEED)
    assert np.allclose(state_windy.linear_velocity, state_calm.linear_velocity + wind)

    action = state_calm.actuators.copy()
    action[AILERON] = np.deg2rad(3)
    action[ELEVATOR] -= np.deg2rad(1)
    seconds = 5
    state_calm = simulate(calm, state_calm, action, seconds)
    state_windy = simulate(windy, state_windy, action, seconds)
    assert np.allclose(state_windy.position, state_calm.position + wind * seconds, atol=1e-6)
    assert np.allclose(state_windy.linear_velocity, state_calm.linear_velocity + wind, atol=1e-6)
    assert np.allclose(state_windy.orientation, state_calm.orientation, atol=1e-9)
