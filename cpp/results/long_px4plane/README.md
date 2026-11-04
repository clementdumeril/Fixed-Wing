# 3 M steps on PX4's advanced_plane, with the best checkpoint (Savio, 3 November 2026)

`slurm/tracking_px4plane_long.slurm` (job 40821853): the tracking task on the advanced_plane, critic 128, batch 64,
networks updated at every step, 3 M steps, seeds 0 to 9. Every 50 k steps the actor flies 100 validation flights from
random starts (seed 54321, not the test flights) and is kept if its score, position RMSE + 10 m per % of failed flights,
is the best so far (`--checkpoint-interval 50000`). At the end both the last and the kept actor fly the tests. Test A:
100 normal flights, B: 100 random starts, ahead: B with the point within 45° ahead; errors over the second half of the
completed flights of test A.

| 10 seeds, mean | Test A | Test B | Ahead | Position RMS | Velocity RMS | Command change |
| --- | --- | --- | --- | --- | --- | --- |
| 400 k steps (job 40803749, no checkpoints) | 84 % | | | 24.9 m | | |
| 3 M steps, last actor | 96.6 % (91–99) | 94.2 % | 98.2 % | 17.1 m | 1.5 m/s | 0.060 |
| **3 M steps, best checkpoint** | **99.0 %** (96–100) | **98.3 %** | **99.4 %** | **11.1 m** | **1.2 m/s** | 0.051 |

Validation during training, mean of the 10 seeds:

| Steps | 400 k | 600 k | 1 M | 1.6 M | 2 M | 2.4 M | 2.8 M | 3 M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Completed flights | 83 % | 95 % | 96 % | 92 % | 93 % | 87 % | 95 % | 95 % |
| Position RMS, median | 28 m | 20 m | 22 m | 15 m | 19 m | 12 m | 11 m | 17 m |
| Seeds at 95 % or more | 3 | 6 | 7 | 6 | 5 | 4 | 6 | 6 |

## What it shows

- **Longer training helps**: 84 % and 25 m at 400 k steps, 97 % and 17 m at 3 M.
- **Keeping the best checkpoint helps more**: 99 % on test A, no seed below 96 %, and 11 m of position error. It is
  chosen on validation flights, not on the tests, so the gain is not a selection on the test flights.
- **The success rate stops rising after about 600 k steps and then oscillates** (87 % at 2.4 M), while the position
  error keeps falling slowly. The best checkpoints sit between 0.9 and 2.8 M steps, most of them between 1.6 and 2.3 M.
- **Batch 64 against batch 256 at equal computation** (an update with 256 samples costs about 4 times one with 64): at
  1.6 M steps batch 64 validates at 92 % and 15 m, batch 256 at 400 k steps ([`batch_px4plane`](../batch_px4plane/README.md),
  4 seeds, lab PC) at 95 % and 15 m. About the same; `slurm/tracking_px4plane_batch.slurm` runs batch 256 on Savio,
  10 seeds, for the same-machine comparison.

## Figures, in the format of Learning to Fly in Seconds

Made with `tools/plot_l2f_style.py`: mean and standard deviation over the seeds, the worst runs (smallest area under
the curve) in grey, the wall-clock time under the steps, as in Fig. 3 and 4 of the paper.

- `figures/episode_length.png`, `figures/episode_length_early.png`: episode length (max 1500 steps, 30 s). The mean
  reaches 90 % of the maximum after 178 k steps, 2.4 min on one Savio core in the 400 k run (job 40803749, the same
  trainings without the validation flights, which double the time here).
- `figures/episode_return.png`, `figures/validation.png`: return, and the validation flights every 50 k steps.
- `figures/compare_episode_length.png`, `figures/compare_position_error.png`: batch 64 (job 40803749, 10 seeds) against
  batch 256 ([`batch_px4plane`](../batch_px4plane/README.md), 4 seeds) over the first 400 k steps.
- `figures/test_flights.png`: as Fig. 5 of the paper, in the frame of the moving point (the black dot): 5 test flights of
  seed 8's best checkpoint, and the S9 policy flown in PX4 SITL with Gazebo.

## Files

`seed<k>/`: `config.json`, `progress.csv`, `checkpoints.csv` (step, validation success rate, position RMSE, score),
`evaluation{,_random,_ahead}.json` for the last actor and `evaluation_best{,_random,_ahead}.json` for the kept one.
