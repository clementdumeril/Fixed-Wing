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

## Visualizer

Uses the rl-tools [ui-server](https://github.com/rl-tools/ui-server), with the same protocol as L2F: the simulator
sends its own render module (`fixed_wing/airplane/ui.js`), then the parameters and the states over a websocket.

```bash
~/.venvs/fixed-wing/Scripts/python -m pip install -e ".[ui]"
~/.venvs/fixed-wing/Scripts/ui-server          # then open http://localhost:13337
~/.venvs/fixed-wing/Scripts/python examples/ui.py
```

Reload the page after changing `ui.js`, the browser only loads the render module once.

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
