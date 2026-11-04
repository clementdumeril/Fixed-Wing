# Reward, curriculum and checks

Details behind the top-level README. Code: `python/fixed_wing/airplane/environment.py` (Python) and
`cpp/include/fixed_wing/dynamics.h` (C++), which agree step by step (`cpp/results/compare_with_python.txt`).

## Task and observations

- Control at 50 Hz, 2 RK4 substeps per period (simulation at 100 Hz), one control period of command latency.
- Episode: 30 s. Setpoint drawn once per episode: airspeed 20 to 30 m/s, altitude within ±20 m of the start (kept in
  50 to 120 m), course within ±90° of the start. Start: trim at a random airspeed, then up to 20° of roll, 10° of
  pitch and 0.3 rad/s on each rate.
- Actor (60 inputs): airspeed, altitude and course errors (the course as sin and cos), rotation matrix, body rates,
  body velocity, airspeed, and the last 10 actions. Only what PX4 estimates.
- Critic: the noise-free actor input plus angle of attack, sideslip, the true wind, the actuator positions, the
  randomization factors and the trim commands.

## Reward

Each step earns

`r = 1 - 0.5 tanh(J / 0.5) - 0.1 Σ min(|e| / f, 1)`

`J` sums squared, scaled errors on airspeed (2 m/s), altitude (10 m), course (30°) and sideslip (5°), the command
change between steps and the distance of the commands to trim. The last term is linear in the airspeed, altitude and
course errors (scales `f`: 10 m/s, 50 m, 90°), outside the `tanh`, so getting closer always pays even where `J` is
saturated. A step in flight always earns at least 0.2, so ending the episode never pays.

Episodes end below 20 m, outside 15 to 40 m/s, 50 m from the altitude setpoint, or with the angle of attack outside
[-10°, 15°], since the linear aerodynamic model has no stall.

## Gated curriculum

As in L2F, the reward weights grow in stages and the stored rewards of the replay buffer are recomputed from the
observations with the new weights. Unlike L2F's fixed schedule, a stage advances only when the mean course error
over the last 100 000 steps is below 15°.

## Three reward fixes

1. The first run ([`python/results/sac_run1`](../python/results/sac_run1/README.md)) completed 98 of 100 flights but
   flew 25° banked in a 9° sideslip, because the reward penalized neither. Added: sideslip and distance-to-trim
   terms.
2. With a command-change weight of 50, exploration noise alone saturated the cost and the reward went flat. The cap
   is now a smooth `tanh` and the weight stays low.
3. With a fixed schedule, the weights reached their target before the policy could turn, the capped reward was flat
   beyond 18° of course error, and the policy unlearned the course. Added: the linear far-field term and the gate on
   the course error.

`python/examples/reward_saturation.py` measures, before a long run, how often the reward is nearly flat under
exploration noise for pilots that do not track yet. It is the check that would have caught fix 3.

## Checks

- 51 tests (`cd python && pytest`): trim equilibrium, control signs, a constant wind that only shifts the
  trajectory, the Allen check case (outer radius 79.4 m, 5 updrafts per km²), Dryden variance and correlations,
  latency, substeps, the Gymnasium environment checker, reward recomputation matching the environment reward, and
  the gated curriculum.
- C++ against Python: 11 flights, 7136 steps, states within 7e-13, rewards within 8e-15, identical terminations,
  observations within 1.2e-7 (Python stores them as float32).
- Flight performance against the published Strix numbers ([`python/PARAMETERS.md`](../python/PARAMETERS.md)): the
  Aerosonde placeholder glides at 15.7 against 7.9.
- The PID completes every calm test flight, which shows the task is feasible before any training.
