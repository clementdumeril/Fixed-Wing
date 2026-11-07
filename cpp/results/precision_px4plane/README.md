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
