# Training starts near the point (PX4 advanced_plane, tracking)

Job 41030913, `slurm/tracking_px4plane_near.slurm`. Question: is the 6-8 m tracking error left after the reward work
(`results/precision_px4plane`) due to training flights that start up to 50 m from the point, so that most of the
experience is the approach? `--near-start F 5`: a share F of the training starts puts the point within 5 m of the
airplane (random airplane state); validation and test flights keep the 0-50 m starts. SAC, critic 128, L2F intervals and
batch, `--reward l2f-penalty`, K = 0.3 /s, 4M steps, 10 seeds. Each folder holds per seed the configuration, the
validation checkpoints, the training progress and the test results (`evaluation*.json`).

| 10 seeds | Best: test A, B | Best: position | Last: test A, B | Last: position | Last: flights under 5 m | Validation, most precise checkpoint with >= 97 % success after 1M steps |
| --- | --- | --- | --- | --- | --- | --- |
| K 0.3, no near starts (`precision_px4plane/guidance03`) | 98.7, 97.2 % | 6.6 ± 1.6 m | 97.0, 95.6 % | 7.7 ± 1.9 m | 23 % | 5.1 m |
| `near50`: half of the starts near | 98.6, 96.6 % | 9.7 ± 2.6 m | 94.1, 91.8 % | 8.7 ± 2.1 m | 14 % | 6.7 m |
| `near80`: 80 % of the starts near | 98.4, 97.3 % | 11.4 ± 6.7 m | 93.3, 89.7 % | 7.6 ± 1.0 m | 29 % | 7.6 m |
| `near50_bonus_adaptive`: half near, adaptive closeness bonus | 98.5, 97.5 % | 11.5 ± 8.1 m | 94.1, 92.1 % | 7.0 ± 1.7 m | 28 % | 6.3 m |

- **Near starts do not make tracking more precise**, and the last actors fail more test flights (93-94 % instead of
  97 %): less practice at joining the point.
- Even with 80 % of the starts within 5 m, the mean training distance at the end is 7.6 m (`progress.csv`): from a near
  start, the airplane settles a few meters away. The lack of experience close to the point is not the cause.
- **The error left is a steady offset, not an oscillation.** Second half of the test flights, last actors (K 0.3 and
  `near80`): the mean offset of each flight is about 3-4 m along, across and in altitude, while its standard deviation
  within a flight is 0.4-0.5 m. The offset is specific to each trained actor (for example +5.0 m in altitude and -3.7 m
  sideways on average for `near80` seed 0) and hardly depends on the reference speed or climb. The environment is
  deterministic here (no wind, no randomization, observations without noise), so the offset is the actor's own bias:
  near the point, a few meters of error (0.15-0.25 in the observation, scaled by 20 m) changes the commands too little
  to cancel it.
