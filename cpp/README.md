# fixed_wing (C++, RLtools)

The fixed-wing airplane and the low-level RL task of [`python/`](../python), ported to C++ in the MDP form of
[RLtools](https://github.com/rl-tools/rl-tools) (same structure as its L2F environment and as
[rl-tools/example](https://github.com/rl-tools/example)), so that training runs at C++ speed with RLtools' SAC.

- `include/fixed_wing/fixed_wing.h`: parameters, state, task configuration, reward configuration, observation sizes.
- `include/fixed_wing/dynamics.h`: the physics and the task, independent of RLtools (6-DOF rigid body, the three
  airframes, actuator lag, latency, RK4 substeps, wind, Dryden turbulence, gusts, Allen thermals, Newton trim,
  observations, rewards, termination).
- `include/fixed_wing/operations_generic.h`: the RLtools interface (`sample_initial_parameters`,
  `sample_initial_state`, `step`, `reward`, `observe` for the actor and the critic, `terminated`) and the sampling of
  the disturbances.
- `include/fixed_wing/reference_motion.h`: reference points that turn or change speed (circles, helices,
  figure-eights, speed oscillations).
- `src/train_sac.cpp`: SAC with the asymmetric actor-critic and the reward curriculum, with the recomputation of the
  replay buffer, the best checkpoint on validation flights, then the tests.

The main recipe is the [simple reward in wind and thermals](#simple-reward-in-wind-and-thermals), the one of the
[top-level README](../README.md). The [precision recipe](#the-precision-recipe) and the
[Eratix recipe](#the-labs-e-flite-umx-eratix) still build; their results are in the
[archive](../docs/archive/README.md).

This branch keeps the recipes that worked. Every variant tried along the way (other network and batch sizes, TD3, the
L2F update intervals alone, the error integral, other reward forms, near starts, profiling) is in tag
`experiments-2026-11`, with its targets, options, Slurm files and results.

## Simple reward in wind and thermals

`train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine_wind` (`FW_DISTURBANCES=1`) is the main recipe: the
advanced_plane follows a moving point (the tracking task, [below](#targets)) in a disturbed world, with
`--reward simple`, the reward of RLtools' L2F environment in meters, without rescaling:

$$r = 35 - \left( \lVert p - p_{\text{ref}} \rVert + \lVert a_t - a_{t-1} \rVert + w_E \left( T / T_{\max} \right)^{3/2} \right)$$

and $r = -100$ on the step that ends a flight; $a$ are the four normalized commands, $T / T_{\max}$ the thrust of the
rotor model over its maximum ($(\omega / \omega_{\max})^2$, from the rotor speed state). `--simple-constant C`
replaces the 35: with 0.5, as in RLtools, every step more than 0.5 m from the point costs, so ending the flight pays
better than flying on and the policy learns to crash. `--energy-weight W` ramps $w_E$ linearly from 0 to W between
the steps of `--energy-ramp START END` (default 1M and 3M), with the stored rewards recomputed from the critic
observations at every curriculum check (`--curriculum-period`). `slurm/tracking_px4plane_simple.slurm` also passes
`--distance-max 30 --near-start 1 5 --reference-heading-max 20` (the point starts within 5 m, its heading within 20°
of the airplane's course, and the flight ends 30 m from it). Savio job 41645214 (energy weights 0, 0.1 and 0.3, 4
seeds each, 8M steps): 84-90 % of the 120 s disturbed test flights completed, 99-100 % in calm air
([`results/simple_px4plane`](results/simple_px4plane/README.md), [`../docs/RESULTS.md`](../docs/RESULTS.md)).

Every episode draws (`include/fixed_wing/operations_generic.h`): a constant horizontal wind of 0-6 m/s in any
direction (`--wind-max`); Dryden turbulence from none to the low-altitude moderate preset (`--turbulence-max`, a
fraction of it); "1 - cos" gusts of up to 3 m/s in any direction lasting 1-3 s, one per minute on average
(`--gust-rate`); in half the episodes (`--thermal-fraction`) an Allen thermal field, as many updrafts as Allen's
Eq. (20) gives for the field, each with a core of 0.5-3 m/s and a radius of 30-80 m at 90 m of altitude, a sinking
ring around it (the downdraft ring of Allen's Eqs. (17)-(18) at full strength, which his model keeps above half the
mixing layer only), the centres drifting with the constant wind; the field covers the flight relative to the air,
with one updraft where the airplane is at mid-flight. The point flies straight in half the episodes and turns at a
constant rate (equivalent bank 5-30°) in the other half (`--turn-fraction`, `reference_motion.h`). The airplane's mass
is randomized by ±10 %, its maximum thrust by ±15 %, each advanced_plane coefficient by ±10 % (`advanced_aerodynamics`,
through the 28 factors of `Parameters::randomization`), and the command latency is 0-2 steps. `--calm` removes the
air disturbances. The actor observes the same 65 inputs as without disturbances; the critic also sees the local wind (constant, gust,
thermal and turbulence together), the randomization factors, the latency and the acceleration of the point.

The tests fly 100 flights of `--eval-duration` seconds (default 120) from fixed starts, airplanes and air:
`evaluation_disturbed.json` (success rate, position RMSE, energy per completed flight from the shaft power of the
rotor model, $Q \omega$ with $Q = 0.01 T$, and mean power), split with and without thermals, and
`evaluation_calm.json` (the same task in calm air); the best checkpoint is chosen on 100 disturbed validation flights
of 30 s. `tools/compare_wind_with_python.py build/check_wind` checks the C++ wind against the Python model
(`python/fixed_wing/airplane/operations.py`) value by value: 1440 local winds and Allen updrafts within 2e-15 m/s,
2700 Dryden filter states within 1.6e-14 with the same normal draws. For the Dryden states the script uses the C++
square root of the 2×2 noise covariance in the Python filter, and checks it against NumPy's (`np.linalg.eigh`) as
$G G^\top$, which does not depend on the sign and order conventions of the LAPACK build. `check_wind` links
`RLtools::Core` only, without the `-ffast-math -march=native` that `RLtools::RLtools` adds with GCC: on steps of
hundreds of meters the two noise eigenvalues are nearly equal, and the reordered arithmetic picks another square root
of the same covariance (Dryden states 0.21 apart, both with $G G^\top$ within 1e-15). Steps up to 1.25 m (training
steps are below 0.6 m) differ by at most 7.5e-9 between the two builds. With GCC and glibc, the Dryden states match
within 1.1e-8, as `exp` may differ from NumPy's by one unit in the last place; the script accepts 1e-7 for them and
1e-9 for the other values.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release        # fetches the tested RLtools commit; finds Intel MKL or OpenBLAS
cmake --build build --config Release
python tools/compare_with_python.py build/check        # needs the Python package (pip install -e ../python)
python tools/compare_wind_with_python.py build/check_wind
python tools/export_scenarios.py build/scenarios_calm.txt
./build/train_sac build/scenarios_calm.txt runs/sac 0 1000000          # first task, Aerosonde
```

The main recipe runs with the options of `slurm/tracking_px4plane_simple.slurm`:
`./build/train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine_wind build/scenarios_calm.txt runs/simple 0 8000000 <options>`.

`-DRL_TOOLS_DIR=<checkout>` uses a local RLtools instead. Without a BLAS library RLtools falls back to plain loops and
the networks dominate the run time; with Intel MKL on one laptop core the first task trains at about 60 steps/s
(critic 256, an update at every step) and the tracking recipes at about 3000 steps/s (critic 128, updates every 10
steps). On Windows, `pip install mkl-devel` and `-DMKL_DIR=<venv>/Library/lib/cmake/mkl -DMKL_THREADING=sequential`
work.

## How I checked it

`tools/compare_with_python.py` records flights with the Python environment (seeds 0 to 9 with random commands around
trim, and a full nose-up flight that stalls), replays the same commands from the same starts in C++ (`src/check.cpp`)
and compares every step: 11 flights, 7136 steps, states within 7e-13, rewards within 8e-15, the same terminations, and
observations within 1.2e-7 (Python stores them as float32). `tools/compare_wind_with_python.py` does the same for the
wind models (`src/check_wind.cpp`, see above). Results of every recipe: [`results/`](results/README.md).

## Targets

| Target | What it trains or does | Job file |
| --- | --- | --- |
| `train_sac` | first task (airspeed, altitude, course) on the Aerosonde with the settings of `python/examples/train_sac.py`: critic 256, batch 256, discount 0.99, an update at every step | |
| `train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine` | following a moving point on PX4's `advanced_plane`, with the position error also seen on a 2 m scale (65 inputs): critic 128, batch 256, discount 0.995, the critics updated every 10 steps and the actor every 20 (L2F paper). The precision recipe (archived results) | `slurm/tracking_px4plane_reward.slurm` |
| `train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine_wind` | the same in wind, turbulence, gusts and thermals, with randomized airplanes. **The main recipe** | `slurm/tracking_px4plane_simple.slurm` |
| `train_sac_c128_b256_tracking_g995_eratix_i10a20` | following a moving point on the lab's E-flite UMX Eratix, without the fine observation (62 inputs) (archived results) | `slurm/tracking_eratix.slurm` |
| `check`, `check_wind` | the C++ port against the Python environment and wind models | `slurm/build.slurm` |
| `evaluate_reference`, `evaluate_reference_fine` | an `advanced_plane` tracking actor when the point accelerates (62 or 65 inputs) | |
| `evaluate_smoothness` | a `_wind` actor on the 100 disturbed and 100 calm test flights, with attitude, sideslip, body rates, command rates, airspeed and along/cross-track errors | `slurm/tracking_px4plane_smooth.slurm` |
| `evaluate_limits` | a `_wind` actor under scripted events past the training range: motor cut and restart, wind, gusts, turbulence, thermals, mass, thrust, control power, latency, sensor noise, point jumps ([results](../docs/limit_tests/README.md)) | |
| `compare_sitl` | a PX4 SITL flight of the deployed policy against this simulator | |
| `eratix_trim` | trim table, stall speed, linear modes and a hands-off flight of the Eratix model | |

`train_sac <scenarios.txt> <out_dir> [seed] [steps] [options]`, with the step count in airplane steps. The first task
reads the 100 Python test scenarios from `scenarios.txt` (`tools/export_scenarios.py`); the tracking targets do not
use them. Every run writes `config.json`, `progress.csv`, `reward_check.txt` (the stored rewards against the same
rewards recomputed from the critic observations, ~1e-7) and the tests of the last actor:

- first task: `evaluation.json` (the 100 Python test scenarios) and `evaluation_random.json` (100 random starts);
- tracking: `evaluation.json` (test A, 100 normal flights), `evaluation_random.json` (test B, 100 random starts) and
  `evaluation_ahead.json` (test B with the point within 45° ahead), with `position_rmse` [m] and `velocity_rmse` [m/s];
- `_wind` target: `evaluation_disturbed.json` and `evaluation_calm.json` (below);

with the first 10 flights of each test step by step (`flights*.csv`) and the trained actor as C++ code (`actor.h`).

Options after the step count:

- `--start trimmed|neutral|random`: how the training flights start. `trimmed` (default) is the Python setup;
  `neutral` is the same flight with the surfaces at 0 and half throttle; `random` is any heading, bank ±60°,
  pitch ±20°, a wide airspeed range, angle of attack −4° to 10° and sideslip ±8° (never stalled), random commands.
- `--no-trim-reward`: no trim term in the reward, so that the policy finds trim on its own.
- `--alpha-weight W`: a cost on the angle of attack within 4° of the stall limits (W at the limit). In the tracking
  task it is outside the saturation of the reward, since far from the point the position cost saturates it.
- `--position-scale S`, `--position-linear W`, `--velocity-guidance K` (tracking): the scale of the position cost
  (default 50 m); a cost linear in the distance, outside the saturation of the tanh reward (the program refuses
  weights that could make the reward of a step negative); the velocity cost on the error to the reference velocity
  plus K times the position error toward the point (at most 5 m/s), so that closing the gap no longer costs velocity
  error.
- `--curriculum-period P`: airplane steps between two checks of the curriculum (default 100k). With the default reward
  a stage is taken only if the mean task error over the last period is below the gate.
- `--checkpoint-interval N`: every N steps, the actor flies 100 validation flights (random starts, other than the
  tests) and the best one is kept: `best_actor.h`, its tests in `evaluation_best*.json` and the scores in
  `checkpoints.csv` (RMS position error + 10 m per % of failed flights). The scores of a run can swing by 10 m between
  checkpoints 50k steps apart, so the last actor is often not the best one.
- `--checkpoint-precise P N`: the best checkpoint is the most precise one with at least P (fraction) of the validation
  flights completed after N steps.
- `--reward tanh|l2f-norm|simple`: the reward form. `tanh` (default): survival minus a capped cost and a linear far
  term ([`../docs/method.md`](../docs/method.md)). `l2f-norm` (tracking): the form and schedule of Learning to Fly in Seconds with the norms of
  the errors, as the L2F reward of RLtools at b32d9985: 2 minus weighted norms of the scaled position, velocity,
  sideslip, command change and stall margin, clipped at −2, a termination penalty of 2 / (1 − γ), the weights grown
  every `--curriculum-period` without a gate. `simple` (tracking): below.
- `--position-bonus W` (`l2f-norm`): a bonus W exp(−(d / s)²) for being close to the point; s starts at 20 m and
  follows half the mean training distance to the point, never growing and never below 1 m.
- `--near-start F D` (tracking): a share F of the training starts puts the point within D m instead of 50 m.
- `--distance-max D` (tracking): the flight ends more than D m from the point (3D). `--reference-heading-max DEG`: the
  spread of the point's heading (or of the course setpoint in the first task) around the airplane's course (default
  90°).
- `--energy-weight W`, `--energy-ramp START END`, `--simple-constant C`, `--eval-duration S`, `--calm`, `--wind-max V`,
  `--turbulence-max K`, `--gust-rate R`, `--thermal-fraction F`, `--turn-fraction F`: the simple reward and the
  disturbances, below.
- `--action-weight W`, `--rate-weight W`, `--velocity-weight W`, `--sideslip-weight W`, `--smooth-ramp START END`,
  `--reference-air-relative`: smoother flight with the simple reward, below.

The tracking task exists in C++ only. A reference point starts up to 50 m from the airplane and flies straight within
90° of the airplane's heading, climbing or descending at up to 1 m/s; the airplane has to reach it and follow it. The
policy sees the position and velocity errors in the body frame (62 inputs, 65 with the fine scale), and the reward
penalizes the distances to the reference position and velocity.

## On a Slurm cluster

From `Fixed-Wing/cpp`, with the Python venv of [`python/slurm`](../python/slurm/README.md):

```bash
source slurm/setup.sh          # login node: modules, Intel MKL in the venv, configure (downloads RLtools)
sbatch slurm/build.slurm       # compute node: compile, export the test scenarios, check against Python
sbatch slurm/tracking_px4plane_simple.slurm   # main: the simple reward in wind and thermals, energy weights 0, 0.1, 0.3, 8M steps, 4 seeds each
sbatch slurm/tracking_px4plane_smooth.slurm   # next: the same with the smoothness terms and the air-relative point, 8 variants, 8M steps, 4 seeds each
sbatch slurm/tracking_px4plane_reward.slurm   # archive: the precision recipe on the advanced_plane, calm air, 4M steps, 10 seeds
sbatch slurm/tracking_eratix.slurm            # archive: the Eratix model, tanh reward, calm air, 4M steps, 10 seeds
```

Each run writes `runs/<name>_<job>_seed<k>/` with `progress.csv`, the evaluations and the files above.

## Smoother flight

The policies of job 41645214 complete their flights, but their commands chatter and their bank swings by tens of
degrees, in calm air as much as in turbulence (`evaluate_smoothness`). Five options of `--reward simple`, each off by
default (the defaults give the same runs, byte for byte):

$$r = 35 - \left( \lVert p - p_{\text{ref}} \rVert + w_a \lVert a_t - a_{t-1} \rVert + w_\omega \lVert \omega \rVert + w_v \lVert v - v_{\text{ref}} \rVert + w_\beta \lvert \beta \rvert + w_E \left( T / T_{\max} \right)^{3/2} \right)$$

- `--action-weight W`: $w_a$ (default 1), the weight of the command change;
- `--rate-weight W`: $w_\omega$ (default 0), on the norm of the body rates in rad/s;
- `--velocity-weight W`: $w_v$ (default 0), on the error to the velocity of the point in m/s;
- `--sideslip-weight W`: $w_\beta$ (default 0), on the sideslip angle in rad. The policies of job 41645214 fly with 10 to
  30° of sideslip, in our simulator and in PX4 SITL + Gazebo alike: AdvancedLiftDrag computes lift and drag from
  $V \cos\beta$ only, so holding 30° of sideslip at 15 m/s needs 22% less thrust, and nothing in the reward stops it.
  Applied from the first step ($w_\beta$ = 5 or 10, 1M steps, one seed), it cut the success rate on the disturbed test
  flights from 73% to about 60% and doubled the body rates, so the Slurm array ramps it;
- `--smooth-ramp START END`: the four weights go linearly from their defaults (1, 0, 0, 0) to the given values between
  the two steps, updated at every curriculum check with the stored rewards recomputed (as `--energy-ramp`); without
  it they apply from the first step;
- `--reference-air-relative`: the point flies at its speed relative to the mean wind and drifts with it (straight
  lines and helices alike, `ReferenceMotion::drift`). By default the point moves relative to the ground, so in wind
  the airplane must change its airspeed by up to twice the wind speed on every circle of a helix.

`evaluate_smoothness <best_actor.h> <out_prefix> [--reference-air-relative]` flies an actor of the `_wind` target on
the test flights of `slurm/tracking_px4plane_simple.slurm` (seed 12345, 120 s) and writes `<out_prefix>.json` (means
over the completed flights: all flights, helices, straight lines, with and without thermals) and
`<out_prefix>_flights.csv` (one row per flight). `slurm/tracking_px4plane_smooth.slurm` trains eight variants (base,
$w_a = 10$, $w_\omega = 2$ ramped from 1M to 3M steps, $w_v = 0.5$, the air-relative point, $w_a = 10$ with
$w_\omega = 2$ and the air-relative point, $w_\beta = 5$ ramped from 1M to 3M steps, and the same with the air-relative
point), 4 seeds each,
and runs `evaluate_smoothness` on each best actor, on both the ground-fixed and the air-relative point.

## PX4's advanced_plane and flying the policy in Gazebo

`FW_AIRPLANE=1` (the `_px4plane` targets) replaces the Aerosonde with the `advanced_plane` of PX4's Gazebo models, as
Gazebo simulates it: the gz-sim AdvancedLiftDrag system with the coefficients of its `model.sdf`, its puller propeller
(MulticopterMotorModel), servos that follow their command with about 0.1 s of lag, and PX4's SITL output mapping
(motor 10 to 1600 rad/s, servos ±1 = ±45°). It is a 1 kg airplane: it trims at 15 m/s with 24.8% throttle (PX4's
`FW_THR_TRIM` for it is 25%), so the tracking task uses 12-18 m/s for the point and the stall limits move to −15°/18°.
C++ only.

`tools/export_policy.py <run>/actor.h <out.h>` writes a trained actor as plain C++ (weights and a 30-line forward
pass), identical to RLtools' evaluation. `deploy/ros2/fw_policy` is a ROS 2 node that flies it on PX4: at 50 Hz it reads
PX4's odometry over uXRCE-DDS, builds the training observation with `observe_actor`, and sends the servo and motor
commands offboard with direct actuator control; it hands back to PX4 (loiter) after 60 s, below 15 m or beyond 300 m
from the point. The node is built without the fine observation (62 inputs). On Ubuntu 24.04 with ROS 2 Jazzy, after
PX4's `Tools/setup/ubuntu.sh` and `make px4_sitl`:

```bash
python tools/export_policy.py <run>/actor.h deploy/ros2/fw_policy/policy/policy_weights.h
bash deploy/ros2/setup_wsl.sh ~/PX4-Autopilot <RLtools include directory>  # DDS agent, px4_msgs, the node (no sudo)
bash deploy/ros2/fly_sitl.sh                       # PX4 + Gazebo headless, PX4 takes off, the policy flies above 40 m
```

`fly_sitl.sh` writes the flight (state, point and commands at each step) to `~/fw_flights/<date>/flight.csv`. The node
takes over once PX4 is armed above 40 m and below 20 m/s (`max_start_speed`; `-p wait_for_loiter:=true` also waits for
PX4's loiter), and it moves the point on PX4's clock, the simulation's in SITL: when the PC freezes for a few seconds,
the simulation stops and the point with it. `compare_sitl <flight.csv>` flies the same policy in this simulator from
the same state and point, and predicts each logged step with this simulator's model from Gazebo's state and command:
the mean and RMS residuals show which part of the model disagrees (`residuals.csv`).

`IDENTIFY=1 bash deploy/ros2/fly_sitl.sh` is an identification flight: the policy flies, with doublets of each command
on top (fw_policy_node.cpp), and Gazebo's joint states are recorded (it adds Gazebo's JointStatePublisher to the
`advanced_plane` model of the PX4 checkout; it only publishes). `python tools/sitl_truth.py <flight directory>` turns
them into `truth.csv`, a flight log of Gazebo's exact state and measured surface positions, and `compare_sitl truth.csv`
then compares the two models without PX4's state estimate. On the first identification flight, the servos follow
their command with 0.1 s of lag as modelled, and the step-by-step residuals have no bias (accelerations within
0.2 m/s² and 0.2 rad/s²): the residual biases measured from PX4's odometry came from its estimate and timing.

The policy in `policy_weights.h` is seed 9 of the 400k-step critic-128 runs on this airplane (`tracking_px4plane`, in
tag `experiments-2026-11`): in its SITL flight it followed the point at 20 m RMSE over the last 30 s, and this
simulator flies the same flight at 20 m, 6 m below the point in both. Seed 3, the first policy flown, followed it at
15 m in Gazebo but 9 m in this simulator: its commands changed 10 times more from step to step, and amplified the noise
of PX4's estimate.

`evaluate_reference <run>/best_actor.h <out_dir>` (advanced_plane tracking actors without the fine observation;
`evaluate_reference_fine` for the `_fine` targets; the weights are read from the file, so one build serves every run)
flies an actor when the point does not fly straight: circles, helices, the figure-eight of L2F and speed oscillations
(`include/fixed_wing/reference_motion.h`), each with its acceleration swept, from the starts of test A and test B. It
writes `reference_suite.json`, one row per flight in `flights_summary.csv` and a few flights per level in `flights/`;
`tools/plot_reference_suite.py` averages runs and draws the figures, `tools/render_flight_video.py` turns recorded
flights into a 3D animation. Training and the calm-air tests move the point in a straight line.

## Archived recipes

Trained and tested in calm air with a nominal airplane. Results and discussion: [docs/archive](../docs/archive/README.md).

### The precision recipe

`slurm/tracking_px4plane_reward.slurm`: `train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine`, 4M steps,

```bash
--reward l2f-norm --start random --position-scale 20 --velocity-guidance 0.3 --alpha-weight 1 --position-bonus 4 \
--curriculum-period 100000 --checkpoint-interval 50000 --checkpoint-precise 0.97 1000000
```

1.4 ± 0.45 m RMS position error on test A, 98 % of the flights completed, over 10 seeds, 16.7 min per run on a Savio
core ([`results/reward_px4plane`](results/reward_px4plane/README.md), `norm_bonus4`).

### The lab's E-flite UMX Eratix

`FW_AIRPLANE=2` flies the E-flite UMX Eratix 3D FF 450 mm (83.5 g, flat foam), a first model from its specs, product
photos and textbook aerodynamics, before any flight data: [`../docs/eratix_model.md`](../docs/eratix_model.md) gives
the sources, geometry, every coefficient with its derivation and range, the checks and the identification flights to
fly next. It reuses the structure of the advanced_plane's AdvancedLiftDrag model (attached-flow coefficients blended
into a flat-plate model past a 12° stall) with the 28 linear coefficients in `Parameters::aerodynamics`, so that they
are randomized, and a motor-propeller model whose thrust falls with the airspeed (T/W 3.65 static). It trims level from
6 to 15.5 m/s, cruising near 8 m/s at half throttle, so the tracking task is scaled to it (point at 7-10 m/s, up to
20 m away, 15-60 m of altitude; the table is in the document). `slurm/tracking_eratix.slurm`:
`train_sac_c128_b256_tracking_g995_eratix_i10a20`, 4M steps,

```bash
--start random --no-trim-reward --position-scale 20 --position-linear 0.3 --velocity-guidance 0.1 --alpha-weight 1 \
--curriculum-period 40000 --checkpoint-interval 50000
```

99.2 % of the test flights completed by the best checkpoints of 10 seeds, 18 min per run
([`results/eratix`](results/eratix/README.md)). `eratix_trim [out_dir]` prints the trim table, stall speed, linear
modes and a hands-off flight.

