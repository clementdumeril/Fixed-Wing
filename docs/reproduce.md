# Components, usage and figures

Detail moved out of the top-level README. Reward, curriculum and checks: [method.md](method.md).

## What I built

| Component | Path | What it is | Origin |
| --- | --- | --- | --- |
| Simulator | `python/fixed_wing/airplane/` | 6-DOF rigid body, linear aerodynamics (28 coefficients), actuator lag, command latency, RK4 substeps, Newton trim | L2F skeleton, fixed-wing physics mine |
| Wind | `airplane/operations.py` | constant wind, 1-cos gust, Allen thermals, Dryden turbulence with exact discretization | models from Beard and McLain, Allen (2006) |
| RL task | `airplane/environment.py` | Gymnasium task, reward, gated curriculum with replay-buffer recomputation, one `evaluate` for every pilot | curriculum idea from L2F |
| Baseline | `airplane/pid.py` | cascaded PID on the same measurements as the actor, bank limited to 30° | mine |
| Training | `examples/train_sac.py`, `airplane/asymmetric_sac.py` | SAC with an asymmetric actor-critic | stable-baselines3 |
| C++ port | `cpp/include/fixed_wing/`, `cpp/src/train_sac.cpp` | same airplane and task in the RLtools MDP form; PX4 `advanced_plane`; tracking task (C++ only) | RLtools SAC |
| PX4 deployment | `cpp/deploy/ros2/fw_policy/`, `cpp/tools/export_policy.py`, `cpp/src/compare_sitl.cpp` | ROS 2 node that flies an exported actor on PX4 SITL over uXRCE-DDS, and a replay of SITL logs in this simulator | mine |
| Diagnostics | `examples/diagnose_flights.py`, `examples/reward_saturation.py` | attitude in straight flight, cause of every failure, reward flatness under exploration noise | mine |
| Tests | `python/tests/` | 51 tests: trim, control signs, wind, Allen and Dryden statistics, Gymnasium checker, reward recomputation | mine |

## Asymmetric actor-critic

- Actor (60 inputs): airspeed, altitude and course errors, attitude, body rates, body velocity, airspeed, last 10
  actions. Only what PX4 estimates. Outputs aileron, elevator, rudder, throttle in [-1, 1] at 50 Hz with one step of
  latency.
- Critics: the actor input plus angle of attack, sideslip, true wind and actuator positions.
- The same task is implemented in `python/fixed_wing/airplane/environment.py` and `cpp/include/fixed_wing/dynamics.h`.

## Usage

From `python/`:

```bash
python examples/pid_baseline.py          # PID on the 100 test scenarios, about 6 min
                                         # -> "success_rate": 1.0, "mean_return": 1321.8...
python examples/train_sac.py --steps 1000000 --n-envs 8 --out runs/sac        # about 4 h on 8 cores
python examples/train_sac.py --steps 1000000 --n-envs 8 --wind-max 5 --turbulence light --noise 1 --out runs/sac_windy
```

Each run writes `runs/<name>/` with `config.json`, `progress.csv`, `model.zip` and `evaluation.json` (same fields
and test scenarios as the PID). Slurm job files: [`python/slurm/README.md`](../python/slurm/README.md),
[`cpp/README.md`](../cpp/README.md#on-a-slurm-cluster).

## Reproducing figures

From `python/`, on a committed run directory:

| Output | Command |
| --- | --- |
| README table | `python examples/results_table.py` |
| `training_calm.png`, `training_windy.png` | `python examples/plot_training.py results/sac_gated_curriculum/calm_seed* --out training_calm.png` |
| `flights_windy_seed2.png` | `python examples/compare_flights.py --model results/sac_gated_curriculum/windy_seed2/model.zip --wind-max 5 --turbulence light --noise 1 --out flights.png` |
| `rl_vs_pid_windy_scenario1.gif` | `python examples/make_videos.py results/sac_gated_curriculum/windy_seed2 --seed 1 --comparison-only` |
| `diagnosis.json` | `python examples/diagnose_flights.py results/sac_gated_curriculum/windy_seed2 --out diagnosis.json` |
| `cpp/results/sweep/sweep.png` | `python tools/plot_sweep.py results/sweep` (from `cpp/`) |
| `docs/figures/overview.png` | `pdflatex overview.tex && pdftoppm -png -r 275 -singlefile overview.pdf overview` (from `docs/figures/`; TikZ with `paperfig.sty` and the GIF-frame crops `ov_*.png`) |

`model.zip` is committed for `calm_seed0` and `windy_seed2` only.

## More findings

- Of the 53 early terminations over the 6 Python policies, 41 happen in the first 5 s and 49 on the angle of attack:
  from a perturbed start and a large course change the policy banks 50 to 80° and pulls, where the PID limits its
  bank to 30° (`*/diagnosis.json`).
- In wind, SAC commands change 15 to 23 times more per step than the PID's.
- In C++ the critic size decides. With 256 units the policy learns in 115 to 158 min per seed on one core; with the
  L2F size (64) it never learns the course ([`cpp/results/sweep`](../cpp/results/sweep/README.md)).
- I have not isolated why the C++ runs learn better than the Python runs with the same settings.
- Aerosonde placeholder against the published Strix numbers: glide ratio 15.7 against 7.9, so about half the Strix
  drag ([`python/PARAMETERS.md`](../python/PARAMETERS.md)).
