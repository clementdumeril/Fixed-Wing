from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fixed_wing.airplane.airplane import AILERON, ELEVATOR, RUDDER, THROTTLE, Thermals
from fixed_wing.airplane.operations import (
    advance_turbulence,
    air_data,
    airplane_dynamics,
    normalize_action,
    sample_thermals,
    sample_turbulence,
    specific_energy,
    step,
    thermal_updraft,
    turbulence_velocity,
    wind,
)
from fixed_wing.airplane.parameters import aerosonde, dryden_low_altitude_light
from fixed_wing.airplane.trim import trim
from fixed_wing.airplane.validation import best_glide, calibrate_drag_polar
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


def test_gust_profile():
    parameters = aerosonde()
    d = parameters.disturbances
    d.wind = np.array([0, 4.0, 0])
    d.gust = np.array([0, 0, 6.0])
    d.gust_start, d.gust_length = 100.0, 50.0
    at = lambda x: wind(parameters, np.array([x, 0.0, 0.0]))
    assert np.allclose(at(99.9), d.wind)  # before the gust
    assert np.allclose(at(100.0), d.wind)  # smooth start
    assert np.allclose(at(125.0), d.wind + d.gust)  # peak in the middle
    assert np.allclose(at(150.0), d.wind)  # after the gust
    # the gust front is a slab: only the position along gust_direction matters
    assert np.allclose(wind(parameters, np.array([125.0, -30.0, 80.0])), d.wind + d.gust)


def test_updraft_gust_raises_alpha_and_pitches_down():
    parameters = aerosonde()
    trimmed = trim(parameters, AIRSPEED)
    parameters.disturbances.gust = np.array([0, 0, 6.0])
    parameters.disturbances.gust_start, parameters.disturbances.gust_length = 25.0, 50.0  # crossed from t = 1 s to 3 s
    state = trimmed
    alphas, pitch_rates = [], []
    for _ in range(300):
        state = step(parameters, state, normalize_action(parameters, trimmed.actuators))
        alphas.append(air_data(parameters, state)[2])
        pitch_rates.append(-state.angular_velocity[1])  # nose up positive
    _, _, alpha_trim, _ = air_data(parameters, trimmed)
    assert max(alphas) > alpha_trim + np.deg2rad(3)  # the air comes from below
    assert min(pitch_rates) < 0  # static stability: the nose turns into the relative wind
    # the updraft gives energy to the airplane (soaring)
    assert specific_energy(parameters, state) > specific_energy(parameters, trimmed)


def test_substeps_match_finer_control_steps():
    # one call with dt = 0.02 and 2 substeps integrates like two calls with dt = 0.01 (same held command)
    fine, coarse = aerosonde(), aerosonde()
    coarse.integration.dt, coarse.integration.n_substeps = 0.02, 2
    trimmed = trim(fine, AIRSPEED)
    action = trimmed.actuators.copy()
    action[AILERON] = np.deg2rad(3)
    action = normalize_action(fine, action)
    a, b = trimmed, trimmed
    for _ in range(100):
        a = step(fine, step(fine, a, action), action)
        b = step(coarse, b, action)
    assert np.allclose(a.position, b.position, atol=1e-6)
    assert np.allclose(a.orientation, b.orientation, atol=1e-8)


def test_action_delay():
    parameters = aerosonde()
    parameters.latency.action_steps = 3
    trimmed = trim(parameters, AIRSPEED)
    command = trimmed.actuators.copy()
    command[AILERON] = np.deg2rad(10)
    action = normalize_action(parameters, command)
    state = trimmed
    for _ in range(3):  # the link still delivers the trim command
        state = step(parameters, state, action)
        assert state.actuators[AILERON] == trimmed.actuators[AILERON]
    state = step(parameters, state, action)  # the first new command arrives
    assert state.actuators[AILERON] > trimmed.actuators[AILERON]
    assert state.action_buffer.shape == (3, 4)


def test_trim_with_fixed_throttle():
    parameters = aerosonde()
    level = trim(parameters, AIRSPEED)
    same = trim(parameters, AIRSPEED, throttle=level.actuators[THROTTLE])
    assert np.isclose(same.linear_velocity[2], 0, atol=1e-6)  # the level-flight throttle gives level flight
    full = trim(parameters, AIRSPEED, throttle=1.0)
    assert full.linear_velocity[2] > 0  # full throttle climbs
    assert np.allclose(trim(parameters, AIRSPEED, throttle=0.0).linear_velocity, trim(parameters, AIRSPEED, glide=True).linear_velocity)


def test_calibration_recovers_the_drag_polar():
    # the best glide of the Aerosonde model, fed back into the calibration, gives back its own polar
    parameters = aerosonde()
    d = parameters.dynamics
    airspeed, sink, _ = best_glide(parameters)
    C_D_p, oswald = calibrate_drag_polar(d.mass, d.wing_area, d.wing_span, airspeed, sink, d.air_density, 9.81)
    assert np.isclose(C_D_p, d.aerodynamics.C_D_p, rtol=1e-3)
    assert np.isclose(oswald, d.aerodynamics.oswald_efficiency, rtol=1e-3)


def single_thermal(sink=False):
    return Thermals(centers=np.zeros((1, 2)), w_gain=np.ones(1), r_gain=np.ones(1), sink=sink)


def test_allen_check_case():
    # Allen 2006, section VI: w* = 2.56 m/s, z_i = 1401 m, z = 280 m, 1 km x 1 km -> r2 = 79.4 m, 5 updrafts
    rng = np.random.default_rng(0)
    assert len(sample_thermals(rng, 1000, 1000, height=280).centers) == 5
    thermals = single_thermal()
    at = lambda r, z=280.0: thermal_updraft(thermals, np.array([r, 0.0, z]))
    r2 = 79.4
    w_bar = (0.2 ** (1 / 3)) * (1 - 1.1 * 0.2) * 2.56  # Eq. (11)
    radii = np.linspace(0, r2, 400)
    disk_mean = np.trapezoid([at(r) * 2 * r for r in radii], radii) / r2 ** 2
    assert np.isclose(disk_mean, w_bar, rtol=0.05)  # the bell shape keeps the average of Eq. (11)
    assert at(0) > at(40) > at(60) > 0  # strongest at the center
    assert 0 <= at(r2) < 0.1 * at(0) and at(2 * r2) == 0  # fades out at r2, nothing beyond without the sink
    assert at(0, z=1500.0) == 0  # above the mixing layer


def test_thermal_sink_and_downdraft():
    thermals = sample_thermals(np.random.default_rng(0), 1000, 1000, height=280)
    far = thermals.centers[0] + [0, 300.0]
    if np.min(np.linalg.norm(thermals.centers - far, axis=1)) > 250:  # between the updrafts
        assert thermal_updraft(thermals, np.array([*far, 280.0])) < 0
    # high in the mixing layer a downdraft ring surrounds the updraft
    ring = [thermal_updraft(single_thermal(), np.array([r, 0.0, 1100.0])) for r in np.linspace(60, 200, 30)]
    assert min(ring) < 0


def test_thermal_lifts_a_glider():
    # gliding through the core of a thermal: the airplane ends higher than in still air
    calm, lifted = aerosonde(), aerosonde()
    lifted.disturbances.thermals = single_thermal()
    lifted.disturbances.thermals.centers = np.array([[100.0, 0.0]])
    glide = trim(calm, AIRSPEED, glide=True)
    glide.position[2] = 300.0
    action = normalize_action(calm, glide.actuators)
    a, b = glide, glide
    for _ in range(800):
        a, b = step(calm, a, action), step(lifted, b, action)
    assert b.position[2] > a.position[2] + 5
    assert specific_energy(lifted, b) > specific_energy(calm, a)


def test_dryden_statistics():
    # one step from the stationary distribution keeps the variance and gives the Dryden correlations
    parameters = aerosonde()
    parameters.disturbances.turbulence = dryden_low_altitude_light()
    sigma = parameters.disturbances.turbulence.sigma
    length_scale = parameters.disturbances.turbulence.length_scale
    rng = np.random.default_rng(0)
    distance = 50.0
    gust = lambda x: turbulence_velocity(parameters, SimpleNamespace(turbulence=x))
    before, after = [], []
    for _ in range(20000):
        x = sample_turbulence(rng)
        before.append(gust(x))
        after.append(gust(advance_turbulence(parameters, x, distance, rng)))
    before, after = np.array(before), np.array(after)
    assert np.allclose(after.std(axis=0), sigma, rtol=0.03)
    d = distance / length_scale
    expected = np.array([np.exp(-d[0]), (1 - d[1] / 2) * np.exp(-d[1]), (1 - d[2] / 2) * np.exp(-d[2])])
    correlation = np.mean(before * after, axis=0) / sigma ** 2
    assert np.allclose(correlation, expected, atol=0.03)


def test_turbulence_in_flight():
    parameters = aerosonde()
    parameters.disturbances.turbulence = dryden_low_altitude_light()
    trimmed = trim(parameters, AIRSPEED)
    action = normalize_action(parameters, trimmed.actuators)
    with pytest.raises(ValueError):
        step(parameters, trimmed, action)  # turbulence needs a random generator

    def fly(seed):
        rng = np.random.default_rng(seed)
        state = replace(trimmed, turbulence=sample_turbulence(rng))
        alphas = []
        for _ in range(500):
            state = step(parameters, state, action, rng)
            alphas.append(air_data(parameters, state)[2])
        return state, np.array(alphas)

    a, alphas = fly(0)
    b, _ = fly(0)
    c, _ = fly(1)
    assert np.allclose(a.position, b.position)  # reproducible with the same seed
    assert not np.allclose(a.position, c.position)
    assert np.rad2deg(alphas.std()) > 0.3  # the air data now fluctuate
