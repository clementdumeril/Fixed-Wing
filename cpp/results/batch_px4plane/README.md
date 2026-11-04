# Batch 256 on PX4's advanced_plane (lab PC, 3 November 2026)

Tracking task on the advanced_plane (`train_sac_c*_b*_tracking_g995_px4plane`), the setting of
`slurm/tracking_px4plane.slurm`: 400k steps, a curriculum check every 40k, networks updated at every step. Only the
critic hidden size and the batch size change; seeds 0 to 3 on the same PC for every variant. The batch 256 runs also
kept their best checkpoint on 100 validation flights every 50k steps (`--checkpoint-interval 50000`). Test A: 100
normal flights, B: 100 random starts, ahead: B with the point within 45° ahead; errors over the second half of the
completed flights of test A.

| Critic, batch | Test A | Test B | Ahead | Position RMS | Velocity RMS | Command change | Time per run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| critic 128, batch 64 | 93 % (99, 86, 87, 100) | 88 % | 94 % | 25.3 m | 5.1 m/s | 0.051 | 19 min, 4 at once |
| **critic 128, batch 256** | **94 %** (92, 88, 96, 99) | **95 %** | **96 %** | **14.4 m** | **1.5 m/s** | **0.029** | 75 min, 8 at once |
| critic 128, batch 256, best checkpoint | 97 % (92, 99, 100, 98) | 99 % | 99 % | 17.5 m | 3.2 m/s | 0.035 | |
| critic 64, batch 256 | 83 % (76, 81, 92, 85) | 82 % | 88 % | 34.0 m | 4.1 m/s | 0.108 | 42 min, 8 at once |
| critic 64, batch 256, best checkpoint | 92 % (93, 90, 95, 91) | 91 % | 93 % | 23.9 m | 3.0 m/s | 0.084 | |

Mean over the 4 seeds.

## What it shows

- **Batch 256 with the 128 critic tracks much better at the same number of steps**: 14 instead of 25 m of position
  error, a third of the velocity error and commands twice as smooth, with no seed below 88 %. As in the calm-air
  sweep on the Aerosonde (`results/sweep`), the larger batch helps.
- **The best checkpoint is often not the last one**: 3 of the 4 batch-256 runs with the 128 critic peaked before
  400k, seed 1 at 350k with a validation score of 22 against 125 at 400k (`checkpoints.csv`). Kept, it reaches
  97-99 % on the three tests.
- **The critic of Learning to Fly in Seconds (64) is too small for this task**, even with its batch of 256.
- **Not at equal compute.** A batch of 256 costs about 4 times as much per update, and the 8 runs shared the PC's
  8 threads where the batch-64 ones ran 4 at once. The comparison at equal compute is against batch 64 for about
  1.6M steps: `slurm/tracking_px4plane_long.slurm` records the same validation score every 50k steps up to 3M.
- 4 seeds per variant, with a spread of 51 to 100 % between seeds at batch 64 (10 seeds): a strong trend, to
  be confirmed on 10 seeds.
