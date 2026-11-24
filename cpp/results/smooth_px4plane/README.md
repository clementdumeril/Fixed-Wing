# Smoothness terms of the simple reward (PX4 advanced_plane, tracking, wind and thermals)

Savio job 41761071, `slurm/tracking_px4plane_smooth.slurm` at commit 2719842: the simple reward of
[`simple_px4plane/`](../simple_px4plane/README.md) (constant 35, energy weight 0) with the smoothness options of
[cpp/README.md](../../README.md#smoother-flight), 8 variants, 4 seeds each, 8 M steps, 31 to 45 min per run on one
core. Each best actor flies the 100 disturbed and the 100 calm test flights of 120 s with `evaluate_smoothness`, once
with the ground-fixed point and once with the point that drifts with the mean wind. Tables made with
`python tools/summarize_smooth.py results/smooth_px4plane`. The stored rewards agree with the recomputed ones within
2.6e-6 in all 32 runs, and the ramped weights reach their final values.

| Variant | Options |
| --- | --- |
| `base` | none: the recipe of job 41645214 |
| `act10` | `--action-weight 10` |
| `rate2ramp` | `--rate-weight 2 --smooth-ramp 1000000 3000000` |
| `vel05` | `--velocity-weight 0.5` |
| `air` | `--reference-air-relative` |
| `combo` | `--action-weight 10 --rate-weight 2 --reference-air-relative` |
| `beta5ramp` | `--sideslip-weight 5 --smooth-ramp 1000000 3000000` |
| `beta5ramp_air` | `--sideslip-weight 5 --smooth-ramp 1000000 3000000 --reference-air-relative` |

## Disturbed test flights, mean of the 4 seeds

Ground-fixed point:

| Variant | Success | RMSE | Mean sideslip | Max sideslip | Body rates | Command rate | Roll std | Failures (of 400) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base` | 88% | 5.14 m | 11.3° | 30° | 62°/s | 0.123 | 13.5° | airspeed 27, alpha 20, position 3 |
| `act10` | 84% | 5.79 m | 22.1° | 41° | 100°/s | 0.197 | 15.8° | alpha 34, airspeed 24, position 4 |
| `rate2ramp` | 87% | 5.18 m | 12.2° | 31° | 44°/s | 0.153 | 14.4° | airspeed 23, alpha 15, position 13 |
| `vel05` | 89% | 6.59 m | 17.1° | 37° | 76°/s | 0.145 | 14.1° | alpha 21, airspeed 16, position 7 |
| `air` | 87% | 4.39 m | 19.3° | 36° | 80°/s | 0.134 | 13.7° | airspeed 27, alpha 24, position 1 |
| `combo` | 82% | 6.04 m | 37.7° | 47° | 29°/s | 0.043 | 8.2° | alpha 48, airspeed 21, position 4 |
| `beta5ramp` | 88% | 4.93 m | 11.7° | 29° | 56°/s | 0.165 | 11.8° | airspeed 23, alpha 18, position 5 |
| `beta5ramp_air` | 84% | 3.34 m | 7.7° | 24° | 65°/s | 0.113 | 14.0° | alpha 36, airspeed 29, position 1 |

Air-relative point (the task `air`, `combo` and `beta5ramp_air` are trained on):

| Variant | Success | RMSE | Mean sideslip | Max sideslip | Body rates | Command rate | Roll std | Failures (of 400) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base` | 96% | 5.01 m | 11.4° | 30° | 57°/s | 0.112 | 13.6° | alpha 10, airspeed 7 |
| `act10` | 91% | 5.91 m | 22.2° | 41° | 95°/s | 0.184 | 15.6° | alpha 31, airspeed 4 |
| `rate2ramp` | 96% | 5.03 m | 12.1° | 31° | 42°/s | 0.141 | 14.1° | alpha 11, airspeed 4, position 1 |
| `vel05` | 98% | 6.27 m | 16.9° | 37° | 73°/s | 0.133 | 14.1° | alpha 6, airspeed 4 |
| `air` | 94% | 4.21 m | 19.4° | 36° | 71°/s | 0.116 | 12.7° | alpha 13, airspeed 10, position 2 |
| `combo` | 92% | 5.73 m | 37.7° | 47° | 28°/s | 0.035 | 8.0° | alpha 19, airspeed 10, position 2 |
| `beta5ramp` | 96% | 4.79 m | 12.1° | 29° | 53°/s | 0.154 | 11.9° | alpha 14, airspeed 2 |
| `beta5ramp_air` | 94% | 3.21 m | 7.8° | 24° | 60°/s | 0.090 | 13.9° | alpha 20, airspeed 3 |

In calm air the two points are the same (no mean wind) and every variant completes 99 to 100% of the flights;
`beta5ramp_air` has the lowest RMSE (3.07 m) and sideslip (7.4° mean, 19° max), `base` 4.85 m and 11.1°.

- `beta5ramp_air` is the most precise (35% less RMSE than `base`) and has the least sideslip (32% less mean, 24°
  instead of 30° max), but stalls more often: 36 alpha failures of 400 instead of 20 on the ground-fixed point.
  Its sideslip is also the most even across seeds (6.0, 6.4, 6.2 and 12.0°).
- `rate2ramp` cuts the body rates by 29% with the same success and RMSE, but keeps the sideslip and its commands
  change a little more.
- `beta5ramp` alone does not lower the mean sideslip; it depends on the seed (5.4 to 18.5°).
- `combo` has the smoothest commands and body rates, and flies at 38° of sideslip with the most stalls: penalizing the
  rotations and command changes pushes the policy further onto the free sideslip of AdvancedLiftDrag.
- `act10` is worse on every measure; `vel05` and `air` raise the sideslip.
- With 4 seeds, differences of 2 to 4 points of success are within the spread between seeds.

Next: `slurm/tracking_px4plane_smooth2.slurm` combines the body-rate and sideslip terms on the air-relative point and
raises the sideslip weight.

## Files

`<variant>/seed<k>/`, for each of the 32 runs:

| File | Content |
| --- | --- |
| `config.json` | every setting of the run |
| `smoothness_best.json`, `smoothness_best_air.json` | `evaluate_smoothness` of the best checkpoint, ground-fixed and air-relative point: `disturbed` and `calm`, each with `all`, `helix`, `straight`, `with_thermals`, `without_thermals` |
| `smoothness_best_flights.csv`, `smoothness_best_air_flights.csv` | the same, one row per flight |
| `evaluation_best_disturbed.json`, `evaluation_best_calm.json` | the evaluation of `train_sac` for the best checkpoint, as in `simple_px4plane/` |
| `evaluation_disturbed.json`, `evaluation_calm.json` | the same for the last actor |
| `checkpoints.csv` | validation success, RMSE and score every 50 k steps |
| `progress.csv` | training curves, with the four smoothness weights during the ramp |
| `reward_check.txt` | stored against recomputed rewards of the replay buffer |

Not committed (size): the actors (`actor.h`, `best_actor.h`) and the recorded flights (`flights_*.csv`).
