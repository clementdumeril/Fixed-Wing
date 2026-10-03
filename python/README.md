# fixed_wing (Python reference)

Minimal fixed-wing simulator for RL, written as a readable NumPy port of the structure of the
[L2F simulator](https://github.com/rl-tools/rl-tools/tree/master/include/rl_tools/rl/environments/l2f)
(`Parameters` / `State` / `dynamics` -> RK4 -> `post_integration`).

- `fixed_wing/multirotor/`: port of the L2F quadrotor core, used to check the skeleton (quaternions, RK4, actuator lag).
- `fixed_wing/airplane/`: same skeleton with fixed-wing aerodynamics, wind, actuator lag and a trim solver.

Conventions (same as L2F): world z up, body x forward / y left / z up, quaternion `[w, x, y, z]` body to world,
linear velocity in the world frame, angular velocity in the body frame, RK4 at 100 Hz.
The aerodynamic coefficients keep the aerospace convention (y right, z down); the conversion is done in
`airplane_aerodynamics` only.

## Setup

```bash
python -m venv ~/.venvs/fixed-wing
~/.venvs/fixed-wing/Scripts/python -m pip install -e ".[dev]"
~/.venvs/fixed-wing/Scripts/python -m pytest
```

## Example

```python
from fixed_wing.airplane.parameters import aerosonde
from fixed_wing.airplane.trim import trim
from fixed_wing.airplane.operations import normalize_action, step

parameters = aerosonde()
state = trim(parameters, airspeed=25.0)
action = normalize_action(parameters, state.actuators)  # [aileron, elevator, rudder, throttle] in [-1, 1]
for _ in range(1000):
    state = step(parameters, state, action)
```
