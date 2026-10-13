"""Simulation speed of the Python reference implementation, single airplane on one core."""
import time

from fixed_wing.airplane.operations import normalize_action, step
from fixed_wing.airplane.parameters import aerosonde
from fixed_wing.airplane.trim import trim

STEPS = 5000


def steps_per_second(parameters):
    state = trim(parameters, 25.0)
    action = normalize_action(parameters, state.actuators)
    start = time.perf_counter()
    for _ in range(STEPS):
        state = step(parameters, state, action)
    return STEPS / (time.perf_counter() - start)


for n_substeps, delay in [(1, 0), (2, 0), (1, 2)]:
    parameters = aerosonde()
    parameters.integration.n_substeps = n_substeps
    parameters.latency.action_steps = delay
    rate = steps_per_second(parameters)
    simulated = rate * parameters.integration.dt
    print(f"n_substeps={n_substeps} delay={delay}: {rate:8.0f} steps/s, {simulated:6.1f} s of flight per second")
