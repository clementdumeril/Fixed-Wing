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
and batch size sweep on a Slurm cluster: [`results/sweep`](results/sweep/README.md). Batch 256 on the advanced_plane: [`results/batch_px4plane`](results/batch_px4plane/README.md). 3 M steps with the best checkpoint, and the curves in the format of Learning to Fly in Seconds: [`results/long_px4plane`](results/long_px4plane/README.md). The update rate of L2F (critics every 10 steps, actor every 20): [`results/l2f_intervals_px4plane`](results/l2f_intervals_px4plane/README.md). With the L2F critic size and with TD3: [`results/l2f_td3_px4plane`](results/l2f_td3_px4plane/README.md). Two attempts to smooth TD3's commands: [`results/td3_smooth_px4plane`](results/td3_smooth_px4plane/README.md). The best calm-air policy so far comes from
`train_sac` (critic 256, batch 256); `train_sac_c256_b64` is half the cost for a small loss.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release        # fetches the tested RLtools commit; finds Intel MKL or OpenBLAS
cmake --build build --config Release
python tools/compare_with_python.py build/check        # needs the Python package (pip install -e ../python)
python tools/export_scenarios.py build/scenarios_calm.txt
./build/train_sac build/scenarios_calm.txt runs/sac 0 1000000
```

Options after the step count:

- `--start trimmed|neutral|random`: how the training flights start. `trimmed` (default) is the Python setup;
  `neutral` is the same flight with the surfaces at 0 and half throttle; `random` is any heading, bank ±60°,
  pitch ±20°, airspeed 18–32 m/s, angle of attack −4° to 10° and sideslip ±8° (never stalled), random commands.
- `--no-trim-reward`: no trim term in the reward, so that the policy finds trim on its own.
- `--huber D`: Huber instead of square on the scaled airspeed, altitude and course errors.
- `--alpha-weight W`: a cost on the angle of attack within 4° of the stall limits (W at the limit). In the tracking
  task it is outside the saturation of the reward, since far from the point the position cost saturates it.

The `train_sac_c256_b64_integral` target also gives the policy the time integrals of the three errors (63 inputs).

The `train_sac_c256_b64_tracking` target learns another task: position and velocity tracking. A reference point
starts up to 50 m from the airplane and flies straight at 20–28 m/s, within 90° of the airplane's heading, climbing or
descending at up to 1 m/s; the airplane has to reach it and follow it. The policy sees the position and velocity errors
in the body frame (62 inputs), and the reward penalizes the distances to the reference position and velocity. Its test
A is 100 normal flights (neutral start) instead of the Python scenarios, and its evaluations add `position_rmse` [m]
and `velocity_rmse` [m/s]. `--position-scale S` sets the scale of the position cost (default 50 m) and
`--position-linear W` adds a cost linear in the distance, outside the saturation of the reward (the program refuses
weights that could make the reward of a step negative); `--velocity-guidance K` puts the velocity cost on the error to
the reference velocity plus K times the position error toward the point (at most 5 m/s), so that closing the gap no
longer costs velocity error. `train_sac_c256_b64_tracking_g995_integral` also gives the policy the time integral of
the position error, with a forgetting time of 10 s so that the approach fades out of it (65 inputs). `--reference-bearing-max DEG` starts the point within DEG of the airplane's course
(default 180: anywhere around it, so about half the starts put it behind). Tracking runs also write
`evaluation_ahead.json`: test B with the point within 45° ahead. This task exists in C++ only.
`--curriculum-period P` checks the curriculum gate every P steps (default 100k) and `--curriculum-stage S` trains
with the weights of stage S from the start, without curriculum. `--profile` times each part of a training step and
writes `profile.json`. `--checkpoint-interval N` flies the actor every N steps on 100 validation flights (random
starts, other than the tests) and keeps the best: `best_actor.h`, its tests in `evaluation_best*.json` and the scores
in `checkpoints.csv` (RMS position error + 10 m per % of failed flights). The scores of a run can swing by 10 m
between checkpoints 50k steps apart, so the last actor is often not the best one. The `train_sac_*_e<N>` targets fly N airplanes in parallel, with one network update per step of
all of them.
The `train_sac_*_i<K>` targets update the networks every K steps only, as the L2F configurations of RLtools: each
step is about K times cheaper. `train_sac_c128_b256_tracking_g995_px4plane_i10a20` uses the intervals of the L2F paper: the critics every
10 steps and the actor every 20 (`FW_ACTOR_INTERVAL_FACTOR=2`).
Each run writes `evaluation.json` (the 100 Python test flights) and `evaluation_random.json` (100 random starts, the
same for every run), the first 10 flights of each step by step (`flights.csv`, `flights_random.csv`) and the trained
actor as C++ code (`actor.h`).

`-DRL_TOOLS_DIR=<checkout>` uses a local RLtools instead. Without a BLAS library RLtools falls back to plain loops and
the networks dominate the run time (under 20 steps/s here); with Intel MKL on one laptop core the training runs at
about 60 steps/s with the Python network sizes and 600 steps/s with the L2F ones (`train_sac_small`: critic 64,
batch 64). On Windows, `pip install mkl-devel` and `-DMKL_DIR=<venv>/Library/lib/cmake/mkl -DMKL_THREADING=sequential`
work.

## On a Slurm cluster

From `Fixed-Wing/cpp`, with the Python venv of [`python/slurm`](../python/slurm/README.md):

```bash
source slurm/setup.sh          # login node: modules, Intel MKL in the venv, configure (downloads RLtools)
sbatch slurm/build.slurm       # compute node: compile, export the test scenarios, check against Python
sbatch slurm/sweep.slurm       # 6 critic x batch sizes x 3 seeds, one core each, 1 M steps in calm air
sbatch --array=0-2 slurm/train.slurm train_sac_c128_b64 3000000   # one variant, 3 seeds, any number of steps
sbatch slurm/alpha_margin.slurm   # critic 256 x batch 64 with a margin to the stall in the reward, weights 0, 0.3, 1
sbatch slurm/random_start.slurm   # random start, no trim term, 10 seeds, with the actors and test flights saved
sbatch slurm/tracking.slurm       # position and velocity tracking, 3 reward settings x 3 seeds
sbatch slurm/tracking_gamma.slurm # the same task with a discount factor of 0.995 and 0.998, 3 seeds each
sbatch --export=ALL,STEPS=3000000 slurm/tracking_gamma.slurm   # the same with 3 M steps
sbatch slurm/tracking_guidance.slurm  # gamma 0.995, velocity cost toward the point, gains 0.1 and 0.2 /s
sbatch slurm/tracking_seeds.slurm     # the two best settings (gain 0 and 0.1) on 10 seeds each
sbatch slurm/tracking_ahead.slurm     # gain 0.1 with the point starting within 45 deg ahead, 10 seeds
sbatch slurm/tracking_alpha.slurm     # gain 0.1 with a margin to the stall, weights 0.3 and 1, 10 seeds each
sbatch slurm/tracking_integral.slurm  # gain 0.1, margin weight 1, with the integral of the position error, 10 seeds
sbatch slurm/tracking_critic.slurm    # critics of 256, 128 and 64, 400k steps, curriculum check every 40k, 10 seeds
sbatch slurm/profile.slurm            # time of each part of a training step for the three critics
sbatch slurm/tracking_px4plane.slurm  # tracking on PX4's advanced_plane, critic 128, 400k steps, 10 seeds
sbatch slurm/tracking_px4plane_long.slurm  # the same for 3M steps, keeping the best checkpoint, 10 seeds
sbatch slurm/tracking_px4plane_batch.slurm # batch 256: updates at every step, and with the L2F intervals for 400k and 4M steps, 10 seeds each
sbatch slurm/tracking_px4plane_l2f.slurm   # L2F intervals and batch, 4M steps: SAC critic 64, TD3, and SAC critic 128 as reference, 10 seeds each
sbatch slurm/tracking_px4plane_td3_smooth.slurm # TD3 with 10x the command-change cost, half the exploration noise, and both, 10 seeds each
```

Each run writes `runs/<variant>_<job>_seed<k>/` with `progress.csv`, the evaluations and the files above.

## PX4's advanced_plane and flying the policy in Gazebo

`FW_AIRPLANE=1` (targets `train_sac_c{256,128}_b64_tracking_g995_px4plane`) replaces the Aerosonde with the
`advanced_plane` of PX4's Gazebo models, as Gazebo simulates it: the gz-sim AdvancedLiftDrag system with the
coefficients of its `model.sdf`, its puller propeller (MulticopterMotorModel), servos that follow their command with
about 0.1 s of lag, and PX4's SITL output mapping (motor 10 to 1600 rad/s, servos ±1 = ±45°). It is a 1 kg airplane:
it trims at 15 m/s with 24.8% throttle (PX4's `FW_THR_TRIM` for it is 25%), so the tracking task uses 12-18 m/s for
the point and the stall limits move to −15°/18°. C++ only.

`tools/export_policy.py <run>/actor.h <out.h>` writes a trained actor as plain C++ (weights and a 30-line forward
pass), identical to RLtools' evaluation. `deploy/ros2/fw_policy` is a ROS 2 node that flies it on PX4: at 50 Hz it reads
PX4's odometry over uXRCE-DDS, builds the training observation with `observe_actor`, and sends the servo and motor
commands offboard with direct actuator control; it hands back to PX4 (loiter) after 60 s, below 15 m or beyond 300 m
from the point. On Ubuntu 24.04 with ROS 2 Jazzy, after PX4's `Tools/setup/ubuntu.sh` and `make px4_sitl`:

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

The policy in `policy_weights.h` is seed 9 of the 400k-step critic-128 runs on this airplane (`tracking_px4plane`):
in its SITL flight it followed the point at 20 m RMSE over the last 30 s, and this simulator flies the same
flight at 20 m, 6 m below the point in both. Seed 3, the first policy flown, followed it at 15 m in Gazebo but 9 m in
this simulator: its commands changed 10 times more from step to step, and amplified the noise of PX4's estimate.

## The lab's E-flite UMX Eratix

`FW_AIRPLANE=2` flies the E-flite UMX Eratix 3D FF 450 mm (83.5 g, flat foam), a first model from its specs, product
photos and textbook aerodynamics, before any flight data: [`../docs/eratix_model.md`](../docs/eratix_model.md) gives
the sources, geometry, every coefficient with its derivation and range, the checks and the identification flights to
fly next. It reuses the structure of the advanced_plane's AdvancedLiftDrag model (attached-flow coefficients blended
into a flat-plate model past a 12° stall) with the 28 linear coefficients in `Parameters::aerodynamics`, so that they
are randomized, and a motor-propeller model whose thrust falls with the airspeed (T/W 3.65 static). It trims level from
6 to 15.5 m/s, cruising near 8 m/s at half throttle, so the tracking task is scaled to it (point at 7-10 m/s, up to
20 m away, 15-60 m of altitude; the table is in the document). Targets: `train_sac_c128_b256_tracking_g995_eratix_i10a20`
(the current best recipe of the advanced_plane) and `eratix_trim` (trim table, stall speed, linear modes, hands-off
flight: `eratix_trim [out_dir]`).
