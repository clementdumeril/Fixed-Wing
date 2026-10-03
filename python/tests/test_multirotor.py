import numpy as np

from fixed_wing.multirotor.operations import hovering_throttle, initial_state, scale_action, step
from fixed_wing.multirotor.parameters import arpl
from fixed_wing.quaternion_helper import rotate_vector_by_quaternion


def hover_action(parameters):
    low, high = parameters.dynamics.action_limit
    return np.full(4, (hovering_throttle(parameters) - low) / (high - low) * 2 - 1)


def test_hover_stays_in_place():
    parameters = arpl()
    state = initial_state(parameters)
    action = hover_action(parameters)
    assert np.allclose(scale_action(parameters, action), hovering_throttle(parameters))
    for _ in range(500):
        state = step(parameters, state, action)
    assert np.allclose(state.position, 0, atol=1e-9)
    assert np.allclose(state.linear_velocity, 0, atol=1e-9)
    assert np.allclose(state.orientation, [1, 0, 0, 0])


def test_zero_thrust_falls():
    parameters = arpl()
    state = initial_state(parameters)
    action = hover_action(parameters)
    parameters.dynamics.rotor_thrust_coefficients[:] = 0
    for _ in range(100):
        state = step(parameters, state, action)
    assert np.isclose(state.linear_velocity[2], -9.81, atol=1e-9)
    assert np.isclose(state.position[2], -9.81 / 2, atol=1e-9)


def test_more_thrust_on_left_rotors_rolls_right():
    parameters = arpl()
    state = initial_state(parameters)
    left = parameters.dynamics.rotor_positions[:, 1] > 0
    action = hover_action(parameters) + np.where(left, 0.05, -0.05)
    for _ in range(20):
        state = step(parameters, state, action)
    assert state.angular_velocity[0] > 0  # positive rotation around x (forward): left side goes up
    body_y_in_world = rotate_vector_by_quaternion(state.orientation, np.array([0, 1.0, 0]))
    assert body_y_in_world[2] > 0
    assert np.isclose(np.linalg.norm(state.orientation), 1)
