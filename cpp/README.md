# fixed_wing (C++, RLtools)

The fixed-wing airplane and the low-level RL task of [`python/`](../python), ported to C++ in the MDP form of
[RLtools](https://github.com/rl-tools/rl-tools) (same structure as its L2F environment and as
[rl-tools/example](https://github.com/rl-tools/example)), so that training runs at C++ speed with RLtools' SAC and TD3.

- `include/fixed_wing/fixed_wing.h`: parameters, state, task configuration, observation sizes.
- `include/fixed_wing/dynamics.h`: the physics and the task, independent of RLtools (6-DOF rigid body, linear
  aerodynamics, actuator lag, latency, RK4 substeps, Newton trim, observations, reward, termination).
- `include/fixed_wing/operations_generic.h`: the RLtools interface (`sample_initial_parameters`,
  `sample_initial_state`, `step`, `reward`, `observe` for the actor and the critic, `terminated`).
- `src/train_sac.cpp`: SAC with the asymmetric actor-critic and the gated reward curriculum, with the recomputation of
  the replay buffer, then an evaluation on the 100 test scenarios of the Python environment.

Calm air only for now: the wind is a world-frame vector (zero), and the aerodynamics already use the air-relative
velocity. Gusts, turbulence, thermals and sensor noise are not ported yet.

## How I checked it

`tools/compare_with_python.py` records flights with the Python environment (seeds 0 to 9 with random commands around
trim, and a full nose-up flight that stalls), replays the same commands from the same starts in C++ (`src/check.cpp`)
and compares every step: 11 flights, 7136 steps, states within 7e-13, rewards within 8e-15, the same terminations, and
observations within 1.2e-7 (Python stores them as float32). First trainings: [`results/`](results/README.md); network
and batch size sweep on Savio: [`results/sweep`](results/sweep/README.md). The best calm-air policy so far comes from
`train_sac` (critic 256, batch 256); `train_sac_c256_b64` is half the cost for a small loss.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release        # fetches the tested RLtools commit; finds Intel MKL or OpenBLAS
cmake --build build --config Release
python tools/compare_with_python.py build/check        # needs the Python package (pip install -e ../python)
python tools/export_scenarios.py build/scenarios_calm.txt
./build/train_sac build/scenarios_calm.txt runs/sac 0 1000000
```

`-DRL_TOOLS_DIR=<checkout>` uses a local RLtools instead. Without a BLAS library RLtools falls back to plain loops and
the networks dominate the run time (under 20 steps/s here); with Intel MKL on one laptop core the training runs at
about 60 steps/s with the Python network sizes and 600 steps/s with the L2F ones (`train_sac_small`: critic 64,
batch 64). On Windows, `pip install mkl-devel` and `-DMKL_DIR=<venv>/Library/lib/cmake/mkl -DMKL_THREADING=sequential`
work.

## On Savio

From `Fixed-Wing/cpp`, with the Python venv of [`python/savio`](../python/savio/README.md):

```bash
source savio/setup.sh          # login node: modules, Intel MKL in the venv, configure (downloads RLtools)
sbatch savio/build.slurm       # compute node: compile, export the test scenarios, check against Python
sbatch savio/sweep.slurm       # 6 critic x batch sizes x 3 seeds, one core each, 1 M steps in calm air
sbatch --array=0-2 savio/train.slurm train_sac_c128_b64 3000000   # one variant, 3 seeds, any number of steps
sbatch savio/alpha_margin.slurm   # critic 256 x batch 64 with a margin to the stall in the reward, weights 0, 0.3, 1
```

Each run writes `runs/<variant>_<job>_seed<k>/` with `progress.csv` and `evaluation.json`.
