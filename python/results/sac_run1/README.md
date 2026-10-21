# First SAC run (Savio job 40252558, seed 0)

SAC + MLP (actor 64-64, critics 256-256), asymmetric actor-critic, constant reward weights (no curriculum),
1 000 000 steps in 2 h 42 on 8 cores. Aerosonde placeholder, no wind, no randomization.

## Test scenarios (seeds 0 to 99), same reward as the PID baseline

| | PID | SAC |
| --- | --- | --- |
| Successful flights | 100 / 100 | 98 / 100 (1 alpha limit, 1 below 20 m) |
| Mean return (max 1500) | 1340 | 1358 |
| Airspeed RMS error | 0.37 m/s | 0.14 m/s |
| Altitude RMS error | 0.68 m | 0.87 m |
| Course RMS error | ~0 deg | 8.2 deg |
| Mean command change per step | 0.0009 | 0.0086 |

RMS errors over the second half of each flight. Raw numbers: `evaluation.json`.

## What the flights show

- SAC reaches the setpoint faster and with less overshoot than the PID (`flights_scenarios_0_1_2.png`).
- It ends in a sideslip: aileron about −0.77, rudder about +0.94, 25° of bank, 9° of sideslip, while flying straight.
  The course error comes from this crab, which also costs throttle (about 51 % against 30 % for the PID). On
  scenario 2 it keeps a 2.5 s oscillation in altitude and airspeed.
- Cause: the reward penalizes neither sideslip nor the commands themselves, only their changes. Next run: a sideslip
  term and a penalty on the distance to the trim commands (as L2F penalizes actions), with the curriculum.

## Over training (100 test scenarios, `swarm_3d.gif`)

| Pilot | Complete flights | Ended early |
| --- | --- | --- |
| PID | 100 | 0 |
| SAC after 100 000 steps | 87 | 13 |
| SAC after 500 000 steps | 85 | 15 |
| SAC after 1 000 000 steps (checkpoint) | 97 | 3 |

Not monotonic: the 500 000-step policy is slightly less robust than the 100 000-step one. The 1 000 000-step
checkpoint and the final `model.zip` (98 / 100) are saved a few steps apart.

## Files

- `training.png`: episode length and training return.
- `flights_scenarios_0_1_2.png`: SAC and PID over time on test scenarios 0, 1, 2.
- `learning_scenario1.gif`: the policy saved every 100 000 steps flying test scenario 1, earlier flights in gray.
- `rl_vs_pid_scenario1.gif`, `rl_vs_pid_scenario2.gif`: SAC and PID at the same time, seen from above.
- `swarm_3d.gif`: the 100 test scenarios at once in 3D, for the PID and SAC at 100 000, 500 000 and 1 000 000 steps.
- `flight_3d_scenario1.gif`: SAC and PID side by side in 3D, airplane drawn 15 times larger to show its attitude.
- `model.zip`: the final policy (`SAC.load`), `progress.csv`, `config.json`.

Made with `examples/plot_training.py`, `compare_flights.py`, `make_videos.py` and `make_3d_videos.py`. Live in the
web viewer: `examples/ui_policy.py` (one airplane) and `examples/ui_fleet.py` (36 airplanes at once, Isaac Lab style).
