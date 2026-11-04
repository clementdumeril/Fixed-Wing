# The update rate of Learning to Fly in Seconds on PX4's advanced_plane (Savio, 4 November 2026)

`slurm/tracking_px4plane_batch.slurm` (job 40855038): the tracking task on the advanced_plane, critic 128, batch 256,
seeds 0 to 9, a validation every 50 k steps and the best checkpoint kept (as in [`long_px4plane`](../long_px4plane/README.md)).
Three settings in the same job:

- `b256_400k`: the networks updated at every step, 400 k steps;
- `b256_i10a20_400k`: the update rate of the L2F paper, the critics every 10 steps and the actor every 20, 400 k steps;
- `b256_i10a20_4M`: the same rate for 4 M steps, so the same number of network updates as `b256_400k` (400 k for the
  critics).

| 10 seeds, mean | Training time | Test A | Test B | Ahead | Position RMS | Best checkpoint: A, B, ahead, position |
| --- | --- | --- | --- | --- | --- | --- |
| every step, 400 k | 20.4 min | 94.1 % (77–100) | 93.6 % | 95.6 % | 21.8 m | 98.2 %, 97.0 %, 98.8 %, 23.0 m |
| L2F rate, 400 k | 1.8 min | 68.2 % (35–93) | 65.2 % | 66.0 % | 56.4 m | 70.8 %, 70.0 %, 71.5 %, 53.2 m |
| **L2F rate, 4 M** | **19.8 min** | 94.9 % (89–99) | 93.6 % | 95.7 % | 19.9 m | **99.7 %, 98.3 %, 99.4 %, 12.9 m** |
| *batch 64, every step, 3 M ([`long_px4plane`](../long_px4plane/README.md), job 40821853)* | *76.7 min* | *96.6 %* | *94.2 %* | *98.2 %* | *17.1 m* | *99.0 %, 98.3 %, 99.4 %, 11.1 m* |

Training time: mean wall-clock time of one run on one Savio core, validation flights included.

## What it shows

- **The L2F rate makes a step 11 times cheaper**: 400 k steps in 1.8 min instead of 20.4, same job.
- **Alone, at the same number of steps, it is not enough**: 68 % of the test flights at 400 k steps, a tenth of the
  updates.
- **At the same computation, it gives a better policy**: with 10 times more steps, the same number of updates and about
  the same time (20 min), the best checkpoints complete 99.7 % of test A (99–100 % for every seed) with 12.9 m of
  position error, against 98.2 % and 23 m when updating at every step. Same updates, ten times more flights to learn
  from.
- **The same quality as the 3 M batch-64 run in about a quarter of the time** (20 against 77 min, best checkpoints
  99.7 % / 12.9 m against 99.0 % / 11.1 m). The two ran in different jobs: the same 400 k batch-64 trainings took
  5.5 min in job 40803749 and 10.9 min in job 40821853, so times across jobs are only good to about a factor 2.
- **It flies later in steps but not in time**: the mean episode length reaches 90 % of the maximum after 515 k steps
  (2.5 min), against 162 k steps (7.9 min) at every step in the same job.
- **Batch 256 at every step, 10 seeds**: 94 % and 22 m at 400 k steps in 20 min, against 84 % and 25 m for batch 64
  in 5.5 min (job 40803749); at the same time, batch 64 for 800 k steps validates at 94 % and 20 m. As on the lab PC
  ([`batch_px4plane`](../batch_px4plane/README.md)): fewer steps, not less computation.

## Figures

Made with `tools/plot_l2f_style.py` (format of Fig. 3 and 4 of the L2F paper) for `b256_i10a20_4M`:
`figures/episode_length.png`, `episode_length_early.png`, `episode_return.png`, `validation.png`, and
`figures/compare_time.png`: the validation flights of the three long settings against the training time.

## Files

`<setting>/seed<k>/`: `config.json`, `progress.csv`, `checkpoints.csv`, `evaluation{,_random,_ahead}.json` (last
actor) and `evaluation_best{,_random,_ahead}.json` (kept actor).
