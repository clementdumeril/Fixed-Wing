# Fixed-Wing

A small fixed-wing airplane simulator for reinforcement learning, written in Python with NumPy. It follows the structure of the [L2F simulator](https://github.com/rl-tools/rl-tools/tree/master/include/rl_tools/rl/environments/l2f) from RLtools (`Parameters` / `State` / `dynamics` -> RK4 -> `post_integration`) and replaces the quadrotor physics with fixed-wing aerodynamics.

The code is in [`python/`](python/):

- `fixed_wing/airplane/`: the airplane model (linear aerodynamics, wind and "1 - cos" gusts, actuator lag, command latency, a trim solver, and checks against published flight data)
- `fixed_wing/multirotor/`: a port of the L2F quadrotor core, used to check the shared skeleton before building the airplane on it
- `examples/` and `tests/`

The aircraft parameters are still the Aerosonde from Beard & McLain, used as a placeholder. The RL environment side of L2F (observation, reward, termination, randomization) is not ported yet, so there is no closed-loop controller: the airplane either flies at trim or follows scripted commands.

Setup, examples and the visualizer are in [python/README.md](python/README.md). What was taken from L2F and what changed is in [python/DIFFERENCES_L2F.md](python/DIFFERENCES_L2F.md), and where each parameter comes from is in [python/PARAMETERS.md](python/PARAMETERS.md).

MIT license.
