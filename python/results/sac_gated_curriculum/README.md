# Far-field reward and gated curriculum (Savio jobs 40384633 and 40384634, 3 seeds each)

SAC + MLP (actor 64-64, critics 256-256), asymmetric actor-critic, 1 000 000 steps per seed on 8 cores of Savio
(3.4 to 4.2 h). Reward with sideslip, distance-to-trim and far-field terms; the L2F curriculum advances a stage only
when the mean course error over the last 100 000 steps is below 15°. Aerosonde placeholder, nominal airplane.

- **Calm air** (40384633): no wind, exact sensors.
- **Windy** (40384634): constant wind up to 5 m/s in a random direction, light low-altitude Dryden turbulence,
  sensor noise (`--wind-max 5 --turbulence light --noise 1`).

## Test scenarios (seeds 0 to 99), same reward for every pilot

| | PID, calm | SAC, calm | PID, windy | SAC, windy |
| --- | --- | --- | --- | --- |
| Complete flights | 100 | 90 ± 3 (92, 92, 86) | 97 | 92 ± 7 (93, 85, 99) |
| Mean return (max 1500) | 1322 | 1163 ± 21 | 1238 | 1197 ± 70 (best 1268) |
| Airspeed RMS error | 0.37 m/s | 0.55 ± 0.22 m/s | 0.78 m/s | 0.59 ± 0.05 m/s |
| Altitude RMS error | 0.68 m | 2.33 ± 0.15 m | 1.34 m | 1.73 ± 0.45 m |
| Course RMS error | 0.0° | 4.4 ± 2.1° | 0.1° | 3.6 ± 0.9° |
| Mean command change per step | 0.0009 | 0.0062 | 0.0057 | 0.101 |

SAC: mean ± standard deviation over the 3 seeds. RMS errors over the second half of the complete flights. Raw numbers:
`calm_seed*/evaluation.json`, `windy_seed*/evaluation.json`, `pid/evaluation_*.json`.

## What the curves and flights show

- **The task is learned by every seed** (`training_calm.png`, `training_windy.png`): full 30 s flights by 200 000
  steps, course error under 15° by 200 000 to 300 000 steps, final curriculum stage by 700 000 to 800 000 steps. In
  the previous series, with a fixed schedule, no seed learned the course (67 to 136° of error).
- **The crab of the first run is gone.** Over the last 10 s of the complete flights:

  | Straight flight, mean of the last 10 s | Bank | Sideslip | Aileron from trim | Rudder from trim |
  | --- | --- | --- | --- | --- |
  | First run ([`sac_run1`](../sac_run1/README.md)) | 25° | 9° | | |
  | PID, calm | 0.0° | 0.0° | 0.000 | 0.000 |
  | SAC, calm, 3 seeds | 1.0 to 1.9° | 0.3 to 0.6° | 0.026 to 0.055 | 0.032 to 0.061 |
  | PID, windy | 0.2° | 0.4° | 0.009 | 0.000 |
  | SAC, windy, 3 seeds | 1.6 to 1.9° | 0.6 to 0.7° | 0.057 to 0.081 | 0.062 to 0.070 |

  Commands in normalized units ([−1, 1]). Source: `*/diagnosis.json`.

- **In wind, SAC holds airspeed better than the PID** on all 3 seeds, and the best seed completes 99 flights against
  97 with a higher return. On average SAC stays below the PID.
- **Steady offsets** (`flights_calm_seed0.png`): after a faster response with less overshoot, SAC settles about 2 to
  4° off the course and 2 m off the altitude. It has no integral action.
- **Chattering commands in wind** (`flights_windy_seed2.png`): 15 to 23 times the PID's command change per step.

## Cause of the failures

Over the 6 policies, 53 of 600 flights end early: 49 on the angle of attack, 41 within the first 5 s and 47 within 8 s (`*/diagnosis.json`). The PID's 3 windy failures are also stalls, 2 of them after 16 s at low airspeed.

From a perturbed start (up to 20° of roll) and a course change of up to 90°, the policy banks to 50 to 80° (past
90° in 4 flights), pulls the elevator, often at idle throttle, and exceeds the 15° angle of attack. The PID bounds its bank
command at 30°. The reward only sees the stall through the termination, too late to teach a margin.

## Training collapse

Calm seed 1 collapses at 938 000 steps: the critic loss rises 15-fold, the actor loss drops from −67 to −95 and the
training episodes halve in length. The deterministic policy held: 92 complete flights at the end, against 82 for the
checkpoint at 900 000 steps. Success varies by about 10 flights between neighboring checkpoints.

## Next

A soft penalty on the angle of attack above about 11° (it is in the critic observation, so the buffer recomputation
still works), the integral of the tracking errors in the actor observation, then a larger command-change weight.

## Files

- `training_calm.png`, `training_windy.png`: episode length, return, course error and curriculum stage over training.
- `flights_calm_seed0.png`, `flights_windy_seed2.png`: SAC and PID over time on test scenarios 0, 1, 2.
- `rl_vs_pid_windy_scenario1.gif`: the best windy policy and the PID in the same wind, seen from above.
- `*_seed*/`: `evaluation.json`, `diagnosis.json` (attitude in straight flight, each failure), `progress.csv`,
  `config.json`; `model.zip` for `calm_seed0` and `windy_seed2`.
- `pid/`: the PID on the same scenarios, both conditions.

Made with `examples/plot_training.py`, `compare_flights.py` (with `--wind-max 5 --turbulence light --noise 1` for the
windy one), `make_videos.py --comparison-only` and `diagnose_flights.py`.
