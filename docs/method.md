# Method

Details behind the [README](../README.md). The method follows *Learning to Fly in Seconds* (L2F, Eschmann, Albani,
Loianno, RA-L 2024) and its code in [RLtools](https://github.com/rl-tools/rl-tools). Everything here is the C++
target `train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine_wind`, run by `cpp/slurm/tracking_px4plane_simple.slurm`.
Options and file formats: [cpp/README.md](../cpp/README.md#simple-reward-in-wind-and-thermals).

## Simulator

- **Airplane.** PX4's `advanced_plane` as Gazebo simulates it: the AdvancedLiftDrag coefficients of its `model.sdf`,
  its propeller model, servos with about 0.1 s of lag, PX4's SITL output mapping. 1 kg; it trims at 15 m/s with
  24.8 % throttle.
- **Physics.** 6-DOF rigid body, RK4, control at 50 Hz, command latency of 0–2 steps.
- **Structure.** The L2F environment of RLtools: parameters, state, `sample_initial_parameters`,
  `sample_initial_state`, `step`, `observe`, `reward`, `terminated`. Code: `cpp/include/fixed_wing/`.
- **Checks.** The C++ airplane matches the Python one (`python/fixed_wing/airplane/`) to 7e-13 in state over 11
  flights and 7,136 steps (`cpp/tools/compare_with_python.py`). The wind models match value by value: local winds and
  Allen updrafts within 2e-15 m/s, Dryden filter states within 1.6e-14 (`cpp/tools/compare_wind_with_python.py`).

## Task

A point starts within 5 m of the airplane, its heading within 20° of the airplane's course. It flies at 12–18 m/s,
climbing or descending at up to 1 m/s, straight in half the episodes and turning at a constant rate (5–30° of
equivalent bank) in the other half. The airplane starts trimmed. The flight ends more than 30 m from the point, with
the angle of attack outside −15° to 18°, the airspeed outside 7–30 m/s, or below 20 m.

## Observations: asymmetric actor-critic

- **Actor (65 inputs).** Position and velocity errors to the point in the body frame, the position error again on a
  fine scale ($\tanh(e / 2 \text{ m})$), rotation matrix, body rates, body velocity, airspeed, and the last 10
  commands. Only quantities PX4 estimates, without sensor noise for now.
- **Critics.** The actor's inputs plus angle of attack, sideslip, the local wind (constant, gust, thermal and
  turbulence together), the actuator positions, the randomization factors, the latency, the acceleration of the
  point and the trim commands.
- **Actions.** Aileron, elevator, rudder and throttle in $[-1, 1]$ at 50 Hz.

## Reward

The reward of RLtools' L2F environment, in meters, without its orientation term (the airplane must bank to turn),
plus an energy term:

$$r = 35 - \left( \lVert p - p_{\text{ref}} \rVert + \lVert a_t - a_{t-1} \rVert + w_E \left( \frac{T}{T_{\max}} \right)^{3/2} \right)$$

and $r = -100$ on the step that ends a flight. $p$ and $p_{\text{ref}}$ are the positions of the airplane and the
point in meters, $a_t$ the four normalized commands, $T / T_{\max}$ the thrust over its maximum.

**Why 35 and not L2F's 0.5.** The constant must exceed the largest cost in flight. In flight the distance is at most
30 m, the command change at most 4 and the energy term at most $w_E$, so with 35 every step in flight earns more than
0. With 0.5, every step more than 0.5 m from the point costs: at 4 m a step is worth about −3.9, flying on is worth
about −700 with $\gamma = 0.995$, and ending the flight costs −100 once. In a 1 M-step test on a laptop, the policy
learned to end its flights on purpose, diving to the angle-of-attack limit (laptop run, not committed). L2F does not
need a larger constant because its quadrotor stays within about half a meter of its target.

**Energy term.** $w_E$ ramps linearly from 0 to its final value between 1 M and 3 M steps. Each time it changes, the
rewards stored in the replay buffer are recomputed from the critic observations, as L2F does for its curriculum
(`reward_check.txt` in each run: stored and recomputed rewards agree within 2e-6 at step 20 000).

## Domain randomization

Drawn at the start of every episode (`cpp/include/fixed_wing/operations_generic.h`):

| What | Range |
| --- | --- |
| Constant wind | 0–6 m/s, any horizontal direction |
| Dryden turbulence | from none to the low-altitude moderate preset |
| Gusts | "1 − cos", up to 3 m/s in any direction, 1–3 s long, one per minute on average |
| Allen thermals | in half the episodes: a field of updrafts with cores of 0.5–3 m/s and radii of 30–80 m, a sinking ring around each, drifting with the wind, one updraft where the airplane is at mid-flight |
| Mass | ±10 % |
| Maximum thrust | ±15 % |
| Each aerodynamic coefficient | ±10 % (28 factors) |
| Command latency | 0–2 steps |

## Training loop

RLtools' SAC on one CPU core:

| Setting | Value |
| --- | --- |
| Actor | 2 hidden layers of 64, ReLU |
| Critics | 2 hidden layers of 128, ReLU |
| Batch | 256 |
| Discount $\gamma$ | 0.995 |
| Updates | critics every 10 steps, actor every 20 (L2F paper) |
| Replay buffer | 1 M transitions, 10 k steps of random actions first |
| Episode | 30 s (1,500 steps) |
| Training | 8 M steps, 33–36 min per run on one Savio core |

## Checkpoints

Every 50 k steps, the actor flies 100 validation flights of 30 s in disturbed air, with their own random draws (the
training is the same as without them). We keep the most precise actor, by position RMSE, among those that complete
at least 97 % of the validation flights after 1 M steps. If none reaches 97 %, the score is the RMSE plus 10 m per %
of failed flights; this happened for one run of 12 (energy weight 0, seed 1). Files: `best_actor.h` and
`checkpoints.csv`.

## Evaluation protocol

The kept actor flies two test sets of 100 flights, each 120 s long, four times the training episode. The starts,
airplanes and air of the test flights are fixed, so every run flies the same flights.

- **Disturbed** (`evaluation_best_disturbed.json`): wind, turbulence and gusts in every flight, thermals in 42 of
  them, split into `with_thermals` and `without_thermals`.
- **Calm** (`evaluation_best_calm.json`): the same task in calm air.

Each file gives the share of flights completed, the position RMSE of the completed flights, the energy per completed
flight (shaft power of the rotor model, $Q \omega$ with $Q = 0.01 T$), the mean power, the mean command change per
step and the causes of the failed flights. `evaluation_disturbed.json` and `evaluation_calm.json` are the same tests
for the last actor.
