"""Steady straight flight: find the pitch, elevator and throttle (or glide angle) with zero acceleration."""
import numpy as np

from .airplane import ELEVATOR, THROTTLE, Parameters, State
from .operations import airplane_dynamics, wind


def _state(parameters, airspeed, pitch, flight_path_angle, elevator, throttle):
    actuators = np.zeros(4)
    actuators[ELEVATOR] = elevator
    actuators[THROTTLE] = throttle
    position = np.zeros(3)
    air_relative_velocity = airspeed * np.array([np.cos(flight_path_angle), 0.0, np.sin(flight_path_angle)])
    return State(
        position=position,
        # nose up is a negative rotation around the body y axis (y points left)
        orientation=np.array([np.cos(pitch / 2), 0.0, -np.sin(pitch / 2), 0.0]),
        linear_velocity=air_relative_velocity + wind(parameters, position),
        angular_velocity=np.zeros(3),
        actuators=actuators,
    )


def trim(parameters: Parameters, airspeed, flight_path_angle=0.0, glide=False, tolerance=1e-10, max_iterations=50):
    """Returns the trimmed state, heading along world x. The trimmed action is state.actuators.

    Powered (glide=False): solves for pitch, elevator, throttle at the given flight path angle.
    Glide (glide=True): throttle is zero, solves for pitch, elevator, flight path angle.
    flight_path_angle is relative to the air mass, positive up.
    """
    def unpack(x):
        if glide:
            return _state(parameters, airspeed, pitch=x[0], flight_path_angle=x[2], elevator=x[1], throttle=0.0)
        return _state(parameters, airspeed, pitch=x[0], flight_path_angle=flight_path_angle, elevator=x[1], throttle=x[2])

    def residual(x):
        state = unpack(x)
        change = airplane_dynamics(parameters, state, state.actuators)
        return np.array([change.linear_velocity[0], change.linear_velocity[2], change.angular_velocity[1]])

    x = np.array([0.05, 0.0, -0.1 if glide else 0.5])
    for _ in range(max_iterations):
        r = residual(x)
        if np.max(np.abs(r)) < tolerance:
            return unpack(x)
        epsilon = 1e-6
        jacobian = np.column_stack([(residual(x + epsilon * e) - r) / epsilon for e in np.eye(3)])
        x = x - np.linalg.solve(jacobian, r)
    raise RuntimeError(f"trim did not converge, residual {r}")
