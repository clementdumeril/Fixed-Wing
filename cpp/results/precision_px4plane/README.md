# Closing the steady lag: a linear distance cost and a stronger velocity correction (Savio, 6 November 2026)

`slurm/tracking_px4plane_precision.slurm` (job 40986502): SAC, critic 128, L2F intervals and batch, `--reward
l2f-penalty`, 4 M steps, 10 seeds, best checkpoint every 50 k steps. The reference is
[`l2f_reward_px4plane`](../l2f_reward_px4plane/README.md) (same recipe, velocity correction K = 0.1 /s, no linear
term), whose policies sit about 6 m behind the point and 7 m to the side.

| 10 seeds, best checkpoint | Test A | Test B | Position RMS | Velocity RMS | Behind the point | Lateral RMS | Flights under 5 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reference (K 0.1) | 98.8 % | 96.4 % | 9.8 ± 2.3 m | 0.46 m/s | 5.9 m | 6.8 m | 9 % |
| `l1`: + 0.1 per meter (`--position-l1 0.1`) | 96.9 % | 95.5 % | 16.5 ± 16.2 m | 1.20 m/s | 8.7 m | 15.2 m | 6 % |
| **`guidance03`: K = 0.3 /s** | **98.7 %** | **97.2 %** | **6.6 ± 1.6 m** | **0.28 m/s** | **3.2 m** | **4.6 m** | **31 %** |
| `l1_guidance03`: both | 97.0 % | 95.8 % | 13.9 ± 7.8 m | 2.69 m/s | 2.9 m | 10.1 m | 19 % |

Behind / lateral: in the frame of the point, over the second half of the completed flights of test A.

- **The stronger velocity correction is the lever**: a third less position error (6.6 against 9.8 m), the lag halved
  (3.2 against 5.9 m), a velocity error 40 % smaller, the same success. Still far from the target of near zero for a
  constant-velocity reference.
- **The linear distance cost makes it worse** and less regular across seeds (16.5 ± 16.2 m): with the floor at −2 the
  reward of a step far from the point hits the floor sooner, and some seeds learn worse.
- Next: a stronger correction still (K = 0.5 and 1.0 /s), then the integral of the position error in the observation.

`<variant>/seed<k>/`: `config.json`, `progress.csv`, `checkpoints.csv`, `reward_check.txt`, `evaluation*.json`.

## Second run (job 41004772, `slurm/tracking_px4plane_precision2.slurm`)

| 10 seeds, best checkpoint | Test A | Test B | Position RMS | Behind the point | Lateral RMS | Flights under 5 m |
| --- | --- | --- | --- | --- | --- | --- |
| K 0.3 (first run, best so far) | 98.7 % | 97.2 % | **6.6 ± 1.6 m** | 3.2 m | 4.6 m | 31 % |
| `guidance05`: K 0.5 | 98.6 % | 98.0 % | 9.1 ± 5.6 m | 5.6 m | 6.7 m | 26 % |
| `bonus_guidance01`: closeness bonus, K 0.1 | 99.1 % | 97.2 % | 10.1 ± 2.5 m | 4.7 m | 6.4 m | 13 % |
| `bonus_guidance03`: closeness bonus, K 0.3 | 99.1 % | 97.0 % | 8.3 ± 2.4 m | 5.2 m | 5.9 m | 25 % |
| `bonus_guidance03_integral`: + error integral | 98.6 % | 97.3 % | 8.0 ± 1.3 m | 4.2 m | 5.1 m | 17 % |

- **No variant beats K 0.3**; K 0.5 is worse and more spread across seeds.
- **The closeness bonus was not really tested**: its scale shrank on a fixed schedule (x0.85 per 100 k steps, 1 m from
  1.4 M steps) while the training flights were still 10–40 m from the point on average, so exp(−(d / s)²) was about 0
  for the whole training (`progress.csv`: train/position_error_m against curriculum/stage). The scale has to follow the
  distance the policy actually reaches, or stay fixed.
- The position-error integral does not remove the offset either (4.2 m behind against 5.2 m without it).
