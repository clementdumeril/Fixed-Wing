# fixed_wing (Python reference)

A NumPy port of the structure of the
[L2F simulator](https://github.com/rl-tools/rl-tools/tree/master/include/rl_tools/rl/environments/l2f)
with fixed-wing physics in `fixed_wing/airplane/` (aerodynamics, actuator lag, trim, wind with gusts, thermals and turbulence). `fixed_wing/multirotor/` is the L2F quadrotor core, which I used to check the skeleton.

`fixed_wing/airplane/environment.py` is the RL task (hold an airspeed, an altitude and a course) and `pid.py` its PID baseline.
`examples/train_sac.py` trains SAC with an asymmetric actor-critic and the L2F reward curriculum; [savio/README.md](savio/README.md) runs it on Savio.
`examples/plot_training.py`, `compare_flights.py` (RL against PID over time), `make_videos.py` (GIFs) and `ui_policy.py` (3D) show the results.
Results of the first run: [results/sac_run1](results/sac_run1/README.md).

[DIFFERENCES_L2F.md](DIFFERENCES_L2F.md) lists what changed from L2F.
[PARAMETERS.md](PARAMETERS.md) gives the source of every parameter.

Conventions are the same as L2F (world z up, body x forward / y left / z up, quaternion `[w, x, y, z]`, RK4 at 100 Hz).
The aerodynamic coefficients keep the aerospace convention, converted in `airplane_aerodynamics` only.

## Setup

```bash
python -m venv ~/.venvs/fixed-wing
~/.venvs/fixed-wing/Scripts/python -m pip install -e ".[dev]"
~/.venvs/fixed-wing/Scripts/python -m pytest
~/.venvs/fixed-wing/Scripts/python examples/validate.py   # flight performance vs the Strix numbers from the lab papers
~/.venvs/fixed-wing/Scripts/python examples/benchmark.py  # simulation speed
```

## Visualizer

Uses the rl-tools [ui-server](https://github.com/rl-tools/ui-server) with the same protocol as L2F.

```bash
~/.venvs/fixed-wing/Scripts/python -m pip install -e ".[ui]"
~/.venvs/fixed-wing/Scripts/ui-server          # then open http://localhost:13337
~/.venvs/fixed-wing/Scripts/python examples/ui.py
```

After changing `fixed_wing/airplane/ui.js`, reload the page.

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
