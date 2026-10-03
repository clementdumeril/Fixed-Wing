"""Port of rl_tools/utils/integrators and l2f/operations_generic/50_state_algebra.h

States are dataclasses of numpy arrays, a state change has the same type as the state.
"""
from dataclasses import fields


def add_scaled(state, state_change, scale):
    # state + scale * state_change
    return type(state)(**{
        f.name: getattr(state, f.name) + scale * getattr(state_change, f.name)
        for f in fields(state)
    })


def euler(dynamics, parameters, state, action, dt):
    return add_scaled(state, dynamics(parameters, state, action), dt)


def rk4(dynamics, parameters, state, action, dt):
    k1 = dynamics(parameters, state, action)
    k2 = dynamics(parameters, add_scaled(state, k1, dt / 2), action)
    k3 = dynamics(parameters, add_scaled(state, k2, dt / 2), action)
    k4 = dynamics(parameters, add_scaled(state, k3, dt), action)
    next_state = state
    for k, weight in ((k1, 1), (k2, 2), (k3, 2), (k4, 1)):
        next_state = add_scaled(next_state, k, weight * dt / 6)
    return next_state
