# Body rates and sideslip on the air-relative point (PX4 advanced_plane, tracking, wind and thermals)

Savio job 41775052, `slurm/tracking_px4plane_smooth2.slurm` at commit 2c96b36: the simple reward of
[`smooth_px4plane/`](../smooth_px4plane/README.md), every variant on the point that drifts with the mean wind and with
its weights ramped from 1M to 3M steps, 4 variants, 4 seeds each, 8 M steps, 29 to 35 min per run on one core. Each
best actor flies the 100 disturbed and the 100 calm test flights of 120 s with `evaluate_smoothness`, on the
ground-fixed and on the air-relative point. Tables made with
`python tools/summarize_smooth.py results/smooth2_px4plane`. The stored rewards agree with the recomputed ones within
1.9e-6 in all 16 runs, and the ramped weights reach their final values.

| Variant | Options |
| --- | --- |
| `rate2_beta5_air` | `--rate-weight 2 --sideslip-weight 5 --smooth-ramp 1000000 3000000 --reference-air-relative` |
| `beta10_air` | `--sideslip-weight 10 --smooth-ramp 1000000 3000000 --reference-air-relative` |
| `rate2_beta10_air` | `--rate-weight 2 --sideslip-weight 10 --smooth-ramp 1000000 3000000 --reference-air-relative` |
| `beta20_air` | `--sideslip-weight 20 --smooth-ramp 1000000 3000000 --reference-air-relative` |

## Disturbed test flights, mean of the 4 seeds

Ground-fixed point, with three variants of job 41761071 for comparison:

| Variant | Success | RMSE | Mean sideslip | Max sideslip | Body rates | Command rate | Roll std | Failures (of 400) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base` (41761071) | 88% | 5.14 m | 11.3° | 30° | 62°/s | 0.123 | 13.5° | airspeed 27, alpha 20, position 3 |
| `rate2ramp` (41761071) | 87% | 5.18 m | 12.2° | 31° | 44°/s | 0.153 | 14.4° | airspeed 23, alpha 15, position 13 |
| `beta5ramp_air` (41761071) | 84% | 3.34 m | 7.7° | 24° | 65°/s | 0.113 | 14.0° | alpha 36, airspeed 29, position 1 |
| `rate2_beta5_air` | 87% | 4.19 m | 10.6° | 28° | 39°/s | 0.189 | 10.4° | alpha 28, airspeed 19, position 6 |
| `beta10_air` | 85% | 3.21 m | 6.5° | 20° | 72°/s | 0.125 | 14.6° | alpha 27, airspeed 25, position 8 |
| `rate2_beta10_air` | 87% | 4.44 m | 9.2° | 25° | 41°/s | 0.172 | 11.8° | alpha 31, airspeed 15, position 5 |
| `beta20_air` | 85% | 3.71 m | 4.7° | 16° | 59°/s | 0.146 | 12.4° | airspeed 39, alpha 20, position 2 |

Air-relative point (the task they are trained on):

| Variant | Success | RMSE | Mean sideslip | Max sideslip | Body rates | Command rate | Roll std | Failures (of 400) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rate2_beta5_air` | 95% | 4.05 m | 10.5° | 28° | 36°/s | 0.166 | 10.4° | alpha 12, airspeed 7, position 1 |
| `beta10_air` | 94% | 3.19 m | 6.4° | 20° | 68°/s | 0.102 | 14.6° | alpha 13, airspeed 7, position 3 |
| `rate2_beta10_air` | 95% | 4.31 m | 9.2° | 25° | 38°/s | 0.153 | 11.4° | alpha 17, position 2, airspeed 1 |
| `beta20_air` | 94% | 3.54 m | 4.8° | 15° | 56°/s | 0.123 | 12.5° | alpha 13, airspeed 8, position 1 |

In calm air every variant completes 99 to 100% of the flights; `beta10_air` has the lowest RMSE (2.84 m), `beta20_air`
the lowest sideslip (4.4° mean, 12° max).

- `beta20_air` flies almost without sideslip: 4.7° mean and 16° max instead of 11.3° and 30° for `base`, the same on
  every seed (3.2 to 7.5°). It stalls no more than `base` (20 alpha failures of 400, against 36 for `beta5ramp_air`),
  with a little less body rate, for 3.7 m of RMSE.
- `beta10_air` is the most precise (3.2 m), with the highest body rates (72°/s).
- The body-rate term works against the sideslip term: with $w_\omega = 2$ the body rates drop by a third (39 to 41°/s),
  but the sideslip comes back to 9 to 11°, the RMSE to 4.2 to 4.4 m, and the commands change more (0.17 to 0.19).
- No variant is best on every measure. The spread between seeds is large (1.9 to 4.8 m of RMSE for `beta10_air`), so
  differences of a few tenths of a meter or a few points of success are within it.
- The failures of the disturbed flights (about 13%) are airspeed and stall failures in the strongest gusts, and change
  little from one variant to the other.

Next: `slurm/tracking_px4plane_smooth3.slurm` keeps $w_\beta = 20$ and adds $w_\omega = 1$, or the stall margin
`--alpha-weight` 10 or 30, and tries $w_\beta = 40$.

## Files

`<variant>/seed<k>/`, for each of the 16 runs, the same files as in [`smooth_px4plane/`](../smooth_px4plane/README.md#files).
Not committed (size): the actors (`actor.h`, `best_actor.h`) and the recorded flights (`flights_*.csv`).
