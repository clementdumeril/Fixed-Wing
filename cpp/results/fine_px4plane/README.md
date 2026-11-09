# Position error observed on a fine scale (PX4 advanced_plane, tracking)

Job 41035244, `slurm/tracking_px4plane_fine.slurm`, target `train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine`.
The error left after the reward work is a steady offset of 3-4 m per axis (`results/near_px4plane`): 4 m is only 0.2 in
the observation scaled by 20 m. Here the actor and the critic also see tanh(error / S) in the body frame (65 values
instead of 62). SAC, critic 128, L2F intervals and batch, `--reward l2f-penalty`, K = 0.3 /s, 4M steps, 10 seeds.
Each folder holds per seed the configuration, the validation checkpoints, the training progress and the test results.

| Seeds | Best: test A, B | Best: position | Best: flights under 5 m, 2 m | Last: test A, B | Last: position | Last: flights under 5 m, 2 m | Validation, most precise checkpoint with >= 97 % success after 1M steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K 0.3, main scale only (`precision_px4plane/guidance03`), 10 | 98.7, 97.2 % | 6.6 ± 1.6 m | 31, 3 % | 97.0, 95.6 % | 7.7 ± 1.9 m | 23, 2 % | 5.1 m |
| `fine2m`: S = 2 m, 9 (see below) | 99.6, 97.3 % | 6.4 ± 1.8 m | 37, 3 % | 97.6, 94.0 % | 6.2 ± 1.1 m | 40, 2 % | 4.7 m |
| `fine5m`: S = 5 m, 10 | 99.3, 97.9 % | 6.0 ± 1.0 m | 41, 4 % | 95.7, 94.5 % | 7.2 ± 0.9 m | 33, 5 % | 4.7 m |
| `fine2m_bonus_adaptive`: S = 2 m and the adaptive closeness bonus, 10 | 98.0, 98.4 % | **4.9 ± 1.0 m** | **62, 12 %** | 97.4, 93.6 % | **4.9 ± 1.3 m** | **59, 14 %** | **3.3 m** |

- **The fine observation alone helps a little** (6.0-6.4 m for the kept actor instead of 6.6 m).
- **With the closeness bonus, the error drops to 4.9 m** for both the kept and the last actor, with 62 % of the test
  flights under 5 m and 12 % under 2 m, and smoother commands (mean command change 0.016 instead of 0.029). Neither
  change did this alone: the bonus without the fine observation gave 6.7-10.9 m (`precision_px4plane`), the fine
  observation without the bonus 6.0-6.4 m.
- The steady offset per flight drops from 3.7, 4.2, 3.9 m (behind, sideways, altitude; K 0.3, last actors) to 2.4, 2.4,
  2.0 m; the motion around it stays at 0.3-0.4 m. It is still the main part of the error.
- `fine2m` seed 5 stopped at 1M steps (`progress.csv`), about twice slower than the others and with a training return
  below zero: no test results, left out of the table.

## Second run (job 41037545, `slurm/tracking_px4plane_fine2.slurm`, folder `second_run/`)

Every variant has the fine observation and the adaptive closeness bonus, and keeps as best checkpoint the most precise
one with at least 97 % validation success after 1M steps (`--checkpoint-precise 0.97 1000000`).

| 10 seeds | Best: test A, B | Best: position | Best: flights under 5 m, 2 m | Best: median flight | Last: test A | Last: position |
| --- | --- | --- | --- | --- | --- | --- |
| `fine2m_bonus1`: S = 2 m, bonus weight 1 (the first run's best variant again) | 98.8, 96.8 % | 3.7 ± 1.0 m | 79, 15 % | 3.2 m | 97.7 % | 4.1 ± 1.3 m |
| `fine1m_bonus1`: S = 1 m | 99.4, 97.3 % | 3.4 ± 0.8 m | 85, 22 % | 3.2 m | 98.9 % | 4.3 ± 0.6 m |
| `fine2m_bonus2`: bonus weight 2 | 97.9, 95.8 % | **2.4 ± 0.4 m** | **96, 54 %** | **1.9 m** | 95.0 % | 3.5 ± 1.5 m |
| `fine2m_bonus1_quarter`: bonus scale a quarter of the training distance | 98.8, 97.0 % | 3.5 ± 1.2 m | 80, 26 % | 3.6 m | 95.4 % | 5.5 ± 1.6 m |

- **A bonus weight of 2 halves the error**: 2.4 ± 0.4 m on test A for the kept actors, every seed between 1.75 and
  3.0 m, 54 % of the flights under 2 m. The cost is a little safety: 97.9 % on test A, 95.8 % on test B (failures:
  airspeed too low and angle of attack beyond the limits).
- The precise checkpoint choice holds on the test flights: validation 2.5 m, test A 2.4 m for `fine2m_bonus2`.
- The steady offset per flight drops to 1.2, 1.0, 1.0 m (behind, sideways, altitude) for `fine2m_bonus2`, with
  0.2 m of motion around it.
- Runs with the same seed and options are not identical from one job to the other (`fine2m_bonus1` against the first
  run's `fine2m_bonus_adaptive`: the training progress differs from 20k steps on), so differences of about 0.5 m
  between variants are within what a rerun changes.
